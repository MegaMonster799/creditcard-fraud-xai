"""Threshold-free and thresholded metrics; FPR operating points."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from fraud.config import THRESHOLD_OBJECTIVE


def predict_at(proba: np.ndarray, threshold: float) -> np.ndarray:
    return (np.asarray(proba) >= threshold).astype(int)


def f_beta(precision: np.ndarray, recall: np.ndarray, beta: float = 1.0) -> np.ndarray:
    b2 = beta * beta
    return (1 + b2) * precision * recall / (b2 * precision + recall + 1e-12)


def tune_threshold(y_true, proba, objective: str = THRESHOLD_OBJECTIVE) -> float:
    """Lock a single threshold from labeled scores that are *not* the outer test fold."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    if thresholds.size == 0:
        return 0.5
    if objective == "f1":
        scores = f_beta(precision[:-1], recall[:-1], beta=1.0)
        return float(thresholds[int(np.argmax(scores))])
    raise ValueError(objective)


def false_positive_rate(y_true, y_pred) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    denom = tn + fp
    return float(fp / denom) if denom else 0.0


def metrics_at_threshold(y_true, proba, threshold: float) -> dict:
    y_pred = predict_at(proba, threshold)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "fpr": false_positive_rate(y_true, y_pred),
        "brier": float(brier_score_loss(y_true, proba)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def metrics_at_fpr(y_true, proba, target_fpr: float) -> dict:
    """Score at the largest threshold whose empirical FPR is <= target (holdout only)."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    order = np.argsort(-proba)
    y_sorted = y_true[order]
    n_neg = int((y_true == 0).sum())
    fp = 0
    tp = 0
    chosen = None
    for i, lab in enumerate(y_sorted):
        if lab == 0:
            fp += 1
        else:
            tp += 1
        fpr = fp / n_neg if n_neg else 0.0
        if fpr > target_fpr:
            break
        thresh = float(proba[order[i]])
        chosen = (thresh, tp, fp)
    if chosen is None:
        thresh = float(np.max(proba) + 1.0)
        pred = np.zeros_like(y_true)
    else:
        thresh, _, _ = chosen
        pred = predict_at(proba, thresh)
    out = metrics_at_threshold(y_true, proba, thresh)
    out["target_fpr"] = float(target_fpr)
    return out
