"""Uncertainty that respects overlapping CV folds (Nadeau & Bengio 2003)."""

from __future__ import annotations

import numpy as np
from scipy.stats import t as student_t


def nadeau_bengio_corrected_ttest(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_train: int,
    n_test: int,
    n_splits: int,
) -> dict:
    """Paired corrected resampled t-test of mean(A - B).

    Five CV folds are not independent replicates. The Nadeau–Bengio correction
    inflates the variance by (1/k + n_test/n_train) to account for training-set
    overlap of about 75% under 5-fold CV.
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    diffs = a - b
    k = int(n_splits)
    mean_diff = float(np.mean(diffs))
    var_diff = float(np.var(diffs, ddof=1)) if k > 1 else 0.0
    correction = (1.0 / k) + (n_test / n_train)
    corrected_var = var_diff * correction
    se = float(np.sqrt(corrected_var)) if corrected_var > 0 else 0.0
    df = k - 1
    t_stat = mean_diff / se if se > 0 else 0.0
    p_value = float(2.0 * (1.0 - student_t.cdf(abs(t_stat), df))) if se > 0 else 1.0
    cohens_d = mean_diff / np.sqrt(var_diff) if var_diff > 0 else 0.0
    return {
        "a": a.tolist(),
        "b": b.tolist(),
        "diffs": diffs.tolist(),
        "mean_diff": mean_diff,
        "var_diff": var_diff,
        "correction_factor": correction,
        "t_stat": float(t_stat),
        "df": df,
        "p_value": p_value,
        "cohens_d": float(cohens_d),
        "n_train": int(n_train),
        "n_test": int(n_test),
    }


def holm_adjusted(p_values: dict[str, float], alpha: float) -> dict[str, dict]:
    """Holm–Bonferroni control of the family-wise error over several metrics."""
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    out = {}
    already_stop = False
    for i, (name, p) in enumerate(items):
        thresh = alpha / (m - i)
        reject = (not already_stop) and (p <= thresh)
        if not reject:
            already_stop = True
        out[name] = {
            "p_raw": p,
            "holm_threshold": thresh,
            "reject_h0": bool(reject),
        }
    return out


def bootstrap_mean_ci(values: np.ndarray, alpha: float, seed: int, n_boot: int = 10_000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    draws = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)
