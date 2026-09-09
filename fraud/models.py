"""Matched-budget model families: same preprocessor, same HPO iterations."""

from __future__ import annotations

import numpy as np
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy.stats import loguniform, randint, uniform
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from fraud.config import SEED, SMOTE_K_NEIGHBORS, SMOTE_SAMPLING_STRATEGY
from fraud.data import AmountWinsorizer, TimeAmountScaler


def _smote(seed: int = SEED) -> SMOTE:
    return SMOTE(
        sampling_strategy=SMOTE_SAMPLING_STRATEGY,
        k_neighbors=SMOTE_K_NEIGHBORS,
        random_state=seed,
    )


def _xgb(seed: int = SEED, scale_pos_weight: float | None = None) -> XGBClassifier:
    kwargs = dict(
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        n_estimators=200,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=2,
        gamma=0.0,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=1,
        verbosity=0,
    )
    if scale_pos_weight is not None:
        kwargs["scale_pos_weight"] = scale_pos_weight
    return XGBClassifier(**kwargs)


def build_pipeline(kind: str, imbalance: str = "smote", seed: int = SEED, n_neg: int | None = None, n_pos: int | None = None) -> ImbPipeline:
    """kind in {logreg, xgb}; imbalance in {smote, adasyn, class_weight}."""
    steps = [
        ("winsor", AmountWinsorizer()),
        ("scale", TimeAmountScaler()),
    ]
    if kind == "logreg":
        if imbalance == "smote":
            steps.append(("imbalance", _smote(seed)))
            clf = LogisticRegression(max_iter=2000, solver="liblinear", random_state=seed)
        elif imbalance == "adasyn":
            steps.append(("imbalance", ADASYN(random_state=seed, n_neighbors=SMOTE_K_NEIGHBORS)))
            clf = LogisticRegression(max_iter=2000, solver="liblinear", random_state=seed)
        else:
            clf = LogisticRegression(
                max_iter=2000, solver="liblinear", class_weight="balanced", random_state=seed
            )
        steps.append(("model", clf))
        return ImbPipeline(steps)

    if kind != "xgb":
        raise ValueError(kind)

    if imbalance == "smote":
        steps.append(("imbalance", _smote(seed)))
        clf = _xgb(seed)
    elif imbalance == "adasyn":
        steps.append(("imbalance", ADASYN(random_state=seed, n_neighbors=SMOTE_K_NEIGHBORS)))
        clf = _xgb(seed)
    else:
        spw = (n_neg / n_pos) if n_neg and n_pos else 1.0
        clf = _xgb(seed, scale_pos_weight=spw)
    steps.append(("model", clf))
    return ImbPipeline(steps)


def search_space(kind: str) -> dict:
    """Identical cardinality of random draws is enforced by HPO_ITERATIONS, not by this dict."""
    if kind == "logreg":
        return {
            "model__C": loguniform(1e-3, 1e2),
        }
    return {
        "model__n_estimators": randint(80, 280),
        "model__max_depth": randint(3, 8),
        "model__learning_rate": loguniform(0.03, 0.2),
        "model__subsample": uniform(0.6, 0.4),
        "model__colsample_bytree": uniform(0.6, 0.4),
        "model__min_child_weight": randint(1, 8),
        "model__gamma": uniform(0.0, 1.5),
        "model__reg_alpha": loguniform(1e-4, 1.0),
        "model__reg_lambda": loguniform(0.1, 5.0),
    }


def describe_xgb_defaults() -> dict:
    return {
        "n_estimators": "searched in [80, 280); not early-stopped; selected by inner PR-AUC",
        "max_depth": "searched in {3,...,7}",
        "learning_rate": "log-uniform [0.03, 0.2]",
        "subsample": "uniform [0.6, 1.0]",
        "colsample_bytree": "uniform [0.6, 1.0]",
        "min_child_weight": "searched in {1,...,7}",
        "gamma": "uniform [0, 1.5]",
        "reg_alpha": "log-uniform [1e-4, 1]",
        "reg_lambda": "log-uniform [0.1, 5]",
        "tree_method": "hist",
        "objective": "binary:logistic",
        "early_stopping": False,
        "selection": "RandomizedSearchCV, identical n_iter for both families",
    }
