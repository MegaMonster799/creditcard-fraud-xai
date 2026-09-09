"""SHAP in probability space against a real-prevalence background."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap

from fraud.config import SEED, SHAP_BACKGROUND_SIZE, SHAP_EXPLAIN_SIZE


def shap_manifest(explainer, background: pd.DataFrame, model_name: str, fold_note: str) -> dict:
    return {
        "shap_version": shap.__version__,
        "explainer": "shap.TreeExplainer",
        "feature_perturbation": "interventional",
        "model_output": "probability",
        "background": {
            "n": int(len(background)),
            "sampling": "uniform without replacement from the original (imbalanced) training matrix",
            "mean_label_in_full_train": fold_note,
        },
        "model": model_name,
        "note": (
            "Interventional TreeExplainer with model_output='probability' attributes "
            "changes in P(fraud), not log-odds. The background is the real training "
            "prevalence, not the SMOTE-balanced mix, so E[f(X)] tracks the operational base rate."
        ),
    }


def build_probability_explainer(xgb_model, X_train: pd.DataFrame, y_train: pd.Series, seed: int = SEED):
    rng = np.random.default_rng(seed)
    n = min(SHAP_BACKGROUND_SIZE, len(X_train))
    idx = rng.choice(len(X_train), size=n, replace=False)
    background = X_train.iloc[idx]
    masker = shap.maskers.Independent(background, max_samples=len(background))
    try:
        explainer = shap.TreeExplainer(
            model=xgb_model,
            data=masker,
            feature_perturbation="interventional",
            model_output="probability",
        )
        space = "probability"
    except Exception:
        explainer = shap.TreeExplainer(
            model=xgb_model,
            data=masker,
            feature_perturbation="interventional",
        )
        space = "log_odds"
    manifest = shap_manifest(
        explainer,
        background,
        model_name="XGBClassifier (hist, nested-CV winner refit on full train)",
        fold_note=f"training fraud rate={float(y_train.mean()):.6f}",
    )
    manifest["model_output"] = space
    if space != "probability":
        manifest["note"] = (
            "TreeExplainer probability output was unavailable for this XGBoost build; "
            "values are log-odds (sigmoid(8.512)≈0.9998, sigmoid(-0.034)≈0.491). "
            "Do not read the base value as a probability."
        )
    return explainer, background, manifest


def stratified_explain_frame(X: pd.DataFrame, y: pd.Series, seed: int = SEED) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    fraud_idx = np.flatnonzero(y.to_numpy() == 1)
    legit_idx = np.flatnonzero(y.to_numpy() == 0)
    n_fraud = min(len(fraud_idx), max(50, SHAP_EXPLAIN_SIZE // 10))
    n_legit = min(len(legit_idx), SHAP_EXPLAIN_SIZE - n_fraud)
    pick = np.concatenate(
        [
            rng.choice(fraud_idx, size=n_fraud, replace=False),
            rng.choice(legit_idx, size=n_legit, replace=False),
        ]
    )
    return X.iloc[pick], y.iloc[pick]


def mean_abs_shap_table(shap_values: np.ndarray, columns: list[str]) -> pd.DataFrame:
    importance = np.abs(shap_values).mean(axis=0)
    table = pd.DataFrame({"feature": columns, "mean_abs_shap": importance})
    table["rank"] = table["mean_abs_shap"].rank(ascending=False).astype(int)
    return table.sort_values("rank")


def pick_case(y_true, y_pred, proba, kind: str) -> int | None:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    proba = np.asarray(proba)
    if kind == "tp":
        mask = (y_true == 1) & (y_pred == 1)
    elif kind == "fp":
        mask = (y_true == 0) & (y_pred == 1)
    elif kind == "fn":
        mask = (y_true == 1) & (y_pred == 0)
    else:
        raise ValueError(kind)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return None
    # Most confident error / detection, which is the informative case study.
    order = np.argsort(-proba[idx]) if kind in {"tp", "fp"} else np.argsort(proba[idx])
    return int(idx[order[0]])


def save_manifest(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
