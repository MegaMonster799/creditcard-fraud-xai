"""Nested CV: inner randomized HPO + inner-OOF threshold, outer evaluation."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict

from fraud.config import (
    HPO_ITERATIONS,
    INNER_FOLDS,
    OUTER_FOLDS,
    SEED,
)
from fraud.metrics import metrics_at_threshold, tune_threshold
from fraud.models import build_pipeline, search_space


@dataclass
class FoldResult:
    fold: int
    n_train: int
    n_val: int
    best_params: dict
    threshold: float
    inner_pr_auc: float
    metrics: dict
    fit_seconds: float


@dataclass
class NestedCVResult:
    name: str
    folds: list[FoldResult] = field(default_factory=list)

    def metric_vector(self, key: str) -> np.ndarray:
        return np.array([f.metrics[key] for f in self.folds], dtype=float)


def _oof_proba(pipeline, X, y, cv, seed: int) -> np.ndarray:
    """Out-of-fold scores on the outer-train split, used only to lock a threshold."""
    return cross_val_predict(
        clone(pipeline),
        X,
        y,
        cv=cv,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]


def run_nested_cv(
    X: pd.DataFrame,
    y: pd.Series,
    kind: str,
    imbalance: str = "smote",
    outer_folds: int = OUTER_FOLDS,
    inner_folds: int = INNER_FOLDS,
    n_iter: int = HPO_ITERATIONS,
    seed: int = SEED,
) -> NestedCVResult:
    outer = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=seed)
    inner = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    result = NestedCVResult(name=f"{kind}_{imbalance}")
    n_neg, n_pos = int((y == 0).sum()), int((y == 1).sum())

    for fold, (tr_idx, va_idx) in enumerate(outer.split(X, y), start=1):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]
        pipe = build_pipeline(kind, imbalance=imbalance, seed=seed, n_neg=n_neg, n_pos=n_pos)
        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=search_space(kind),
            n_iter=n_iter,
            scoring="average_precision",
            cv=inner,
            random_state=seed,
            n_jobs=1,
            refit=True,
            verbose=0,
        )
        t0 = time.perf_counter()
        search.fit(X_tr, y_tr)
        fit_s = time.perf_counter() - t0
        best = search.best_estimator_
        # Threshold from inner OOF of the *selected* pipeline, never from outer val or holdout.
        oof = _oof_proba(best, X_tr, y_tr, cv=inner, seed=seed)
        threshold = tune_threshold(y_tr, oof)
        proba_va = best.predict_proba(X_va)[:, 1]
        metrics = metrics_at_threshold(y_va, proba_va, threshold)
        result.folds.append(
            FoldResult(
                fold=fold,
                n_train=int(len(X_tr)),
                n_val=int(len(X_va)),
                best_params={k: _jsonable(v) for k, v in search.best_params_.items()},
                threshold=float(threshold),
                inner_pr_auc=float(search.best_score_),
                metrics=metrics,
                fit_seconds=float(fit_s),
            )
        )
        print(
            f"  {result.name} fold {fold}/{outer_folds}  "
            f"PR-AUC={metrics['pr_auc']:.4f}  F1={metrics['f1']:.4f}  "
            f"thr={threshold:.4f}  inner_PR-AUC={search.best_score_:.4f}  "
            f"({fit_s:.1f}s)"
        )
    return result


def fit_locked_model(X, y, kind, imbalance, params: dict, seed: int = SEED):
    n_neg, n_pos = int((y == 0).sum()), int((y == 1).sum())
    pipe = build_pipeline(kind, imbalance=imbalance, seed=seed, n_neg=n_neg, n_pos=n_pos)
    pipe.set_params(**params)
    pipe.fit(X, y)
    return pipe


def _jsonable(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value
