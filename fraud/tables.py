"""Paper Tables 1–4: performance, Holm tests, SHAP ranks, and paired fold scores."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from fraud import OUTPUT_DIR
from fraud.config import ALPHA, METRICS_FOR_MULTIPLICITY, SEED
from fraud.metrics import metrics_at_threshold
from fraud.stats import bootstrap_mean_ci, holm_adjusted, nadeau_bengio_corrected_ttest

CORE_METRICS = METRICS_FOR_MULTIPLICITY

METRIC_LABELS = {
    "pr_auc": "PR-AUC",
    "f1": "F1",
    "precision": "Precision",
    "recall": "Recall",
    "fpr": "FPR",
    "balanced_accuracy": "Balanced Accuracy",
}

# FPR is ~10^-4; four decimals would collapse holdout XGBoost to 0.0000.
METRIC_DECIMALS = {
    "pr_auc": 4,
    "f1": 4,
    "precision": 4,
    "recall": 4,
    "fpr": 6,
    "balanced_accuracy": 4,
}


def _pretty_model(name: str) -> str:
    raw = str(name).lower()
    if "xgb" in raw:
        return "XGBoost"
    if "log" in raw:
        return "Logistic regression"
    return str(name)


def _fmt(value: float, key: str) -> str:
    return f"{value:.{METRIC_DECIMALS[key]}f}"


def _mean_sd(values: np.ndarray, alpha: float = ALPHA) -> tuple[float, float, float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    n = len(values)
    if n < 2 or sd == 0:
        return mean, sd, mean, mean
    se = sd / np.sqrt(n)
    tcrit = float(student_t.ppf(1 - alpha / 2, n - 1))
    return mean, sd, mean - tcrit * se, mean + tcrit * se


def cv_summary(fold_df: pd.DataFrame) -> pd.DataFrame:
    n_folds = int(fold_df["fold"].nunique()) if "fold" in fold_df.columns else 5
    eval_name = f"Nested {n_folds}-fold CV"
    rows = []
    for model, grp in fold_df.groupby("model"):
        row: dict = {"model": _pretty_model(model), "evaluation": eval_name}
        for key in CORE_METRICS:
            mean, sd, lo, hi = _mean_sd(grp[key].to_numpy())
            row[key] = mean
            row[f"{key}_sd"] = sd
            row[f"{key}_ci_lo"] = lo
            row[f"{key}_ci_hi"] = hi
            row[f"{key}_cell"] = f"{_fmt(mean, key)} ± {_fmt(sd, key)}"
        rows.append(row)
    return pd.DataFrame(rows)


def holdout_summary(holdout_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, rec in holdout_df.iterrows():
        row: dict = {"model": _pretty_model(rec["model"]), "evaluation": "Holdout"}
        for key in CORE_METRICS:
            val = float(rec[key])
            row[key] = val
            row[f"{key}_cell"] = _fmt(val, key)
        rows.append(row)
    return pd.DataFrame(rows)


def holdout_cells_with_ci(y_true, proba, threshold, n_boot: int = 2000, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    point = metrics_at_threshold(y_true, proba, threshold)
    n = len(y_true)
    draws = {k: np.empty(n_boot) for k in CORE_METRICS}
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        m = metrics_at_threshold(y_true[idx], proba[idx], threshold)
        for k in CORE_METRICS:
            draws[k][i] = m[k]
    row = {"evaluation": "Holdout"}
    for k in CORE_METRICS:
        lo, hi = np.quantile(draws[k], [0.025, 0.975])
        row[k] = point[k]
        row[f"{k}_ci_lo"] = float(lo)
        row[f"{k}_ci_hi"] = float(hi)
        row[f"{k}_cell"] = f"{_fmt(point[k], k)} [{_fmt(lo, k)}, {_fmt(hi, k)}]"
    return row


def build_table1(fold_df: pd.DataFrame, holdout_df: pd.DataFrame) -> pd.DataFrame:
    table = pd.concat([cv_summary(fold_df), holdout_summary(holdout_df)], ignore_index=True)
    order = {"Logistic regression": 0, "XGBoost": 1}
    table["_m"] = table["model"].map(order).fillna(9)
    table["_e"] = table["evaluation"].str.contains("Nested").astype(int).rsub(1)
    table = table.sort_values(["_m", "_e"]).drop(columns=["_m", "_e"]).reset_index(drop=True)
    return table


def display_frame(table: pd.DataFrame) -> pd.DataFrame:
    shown = {
        "Model": table["model"],
        "Evaluation": table["evaluation"],
    }
    for key in CORE_METRICS:
        shown[METRIC_LABELS[key]] = table[f"{key}_cell"]
    return pd.DataFrame(shown)


def write_table1(table: pd.DataFrame) -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    shown = display_frame(table)
    shown.to_csv(OUTPUT_DIR / "Table_1_performance.csv", index=False)
    table[["model", "evaluation", *CORE_METRICS]].to_csv(
        OUTPUT_DIR / "Table_1_performance_numeric.csv", index=False
    )
    headers = list(shown.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for _, row in shown.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in headers) + " |")
    (OUTPUT_DIR / "Table_1_performance.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return shown


def print_table1(shown: pd.DataFrame) -> None:
    print("\n" + "=" * 120)
    print(
        "Table 1. Fraud-class performance (PR-AUC is threshold-free; "
        "F1/precision/recall/FPR/balanced accuracy use the locked F1 threshold)"
    )
    print("=" * 120)
    print(shown.to_string(index=False))
    print("=" * 120)
    print("Nested CV cells are mean ± sd across outer folds. Holdout cells are a single untouched test split.")


def infer_nested_sizes(fold_df: pd.DataFrame) -> tuple[int, int]:
    """Outer-train and outer-val sizes from fold confusion counts (equal 5-fold)."""
    n_val = int((fold_df["tn"] + fold_df["fp"] + fold_df["fn"] + fold_df["tp"]).iloc[0])
    n_folds = int(fold_df["fold"].nunique())
    n_train = n_val * (n_folds - 1)
    return n_train, n_val


def _metric_vectors(fold_df: pd.DataFrame, metric: str) -> tuple[np.ndarray, np.ndarray]:
    xgb = fold_df[fold_df["model"].astype(str).str.contains("xgb", case=False)].sort_values("fold")
    lr = fold_df[fold_df["model"].astype(str).str.contains("log", case=False)].sort_values("fold")
    return xgb[metric].to_numpy(dtype=float), lr[metric].to_numpy(dtype=float)


def build_holm_table(
    fold_df: pd.DataFrame,
    n_train: int | None = None,
    n_val: int | None = None,
) -> pd.DataFrame:
    if n_train is None or n_val is None:
        n_train, n_val = infer_nested_sizes(fold_df)
    n_splits = int(fold_df["fold"].nunique())
    rows = []
    raw_p = {}
    for metric in CORE_METRICS:
        a, b = _metric_vectors(fold_df, metric)
        test = nadeau_bengio_corrected_ttest(
            a, b, n_train=n_train, n_test=n_val, n_splits=n_splits
        )
        lo, hi = bootstrap_mean_ci(np.array(test["diffs"]), alpha=ALPHA, seed=SEED)
        raw_p[metric] = test["p_value"]
        rows.append(
            {
                "metric": metric,
                "xgb_folds": test["a"],
                "logreg_folds": test["b"],
                "paired_diffs": test["diffs"],
                "mean_diff_xgb_minus_logreg": test["mean_diff"],
                "t_stat": test["t_stat"],
                "df": test["df"],
                "p_raw": test["p_value"],
                "cohens_d": test["cohens_d"],
                "bootstrap_ci_lo": lo,
                "bootstrap_ci_hi": hi,
                "correction_factor": test["correction_factor"],
            }
        )
    holm = holm_adjusted(raw_p, ALPHA)
    for row in rows:
        adj = holm[row["metric"]]
        row["holm_threshold"] = adj["holm_threshold"]
        row["significant_after_holm"] = adj["reject_h0"]
        row["claim"] = (
            "reject equal performance" if adj["reject_h0"] else "do not claim significance"
        )
    return pd.DataFrame(rows)


def display_holm(stats_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, rec in stats_df.iterrows():
        key = rec["metric"]
        dec = METRIC_DECIMALS[key]
        rows.append(
            {
                "Metric": METRIC_LABELS[key],
                "Mean diff": f"{rec['mean_diff_xgb_minus_logreg']:.{dec}f}",
                "t": f"{rec['t_stat']:.2f}",
                "p (raw)": f"{rec['p_raw']:.4f}",
                "Cohen’s d": f"{rec['cohens_d']:.2f}",
                "95% CI": f"[{rec['bootstrap_ci_lo']:.{dec}f}, {rec['bootstrap_ci_hi']:.{dec}f}]",
                "Holm threshold": f"{rec['holm_threshold']:.4f}",
                "Significant after Holm": "Yes" if rec["significant_after_holm"] else "No",
            }
        )
    return pd.DataFrame(rows)


def write_table2(stats_df: pd.DataFrame) -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    shown = display_holm(stats_df)
    shown.to_csv(OUTPUT_DIR / "Table_2_nadeau_bengio_holm.csv", index=False)
    headers = list(shown.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for _, row in shown.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in headers) + " |")
    (OUTPUT_DIR / "Table_2_nadeau_bengio_holm.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return shown


def print_table2(shown: pd.DataFrame) -> None:
    print("\n" + "=" * 120)
    print(
        "Table 2. Nadeau–Bengio corrected paired tests of XGBoost minus logistic regression, "
        "Holm-adjusted across six metrics (df = 4)"
    )
    print("=" * 120)
    print(shown.to_string(index=False))
    print("=" * 120)


def display_shap_importance(importance: pd.DataFrame) -> pd.DataFrame:
    table = importance.copy()
    if "rank" not in table.columns:
        table["rank"] = table["mean_abs_shap"].rank(ascending=False).astype(int)
    table = table.sort_values("rank")
    total = float(table["mean_abs_shap"].sum())
    share = 100.0 * table["mean_abs_shap"] / total if total else 0.0
    return pd.DataFrame(
        {
            "Rank": table["rank"].astype(int).to_numpy(),
            "Feature": table["feature"].to_numpy(),
            "Mean |SHAP|": [f"{v:.6f}" for v in table["mean_abs_shap"]],
            "Share (%)": [f"{v:.2f}" for v in share],
        }
    )


def write_table3(importance: pd.DataFrame) -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    importance.to_csv(OUTPUT_DIR / "Table_3_SHAP_feature_importance.csv", index=False)
    shown = display_shap_importance(importance)
    headers = list(shown.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for _, row in shown.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in headers) + " |")
    (OUTPUT_DIR / "Table_3_SHAP_feature_importance.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return shown


def print_table3(shown: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print(
        "Table 3. Global feature importance from interventional TreeExplainer "
        "(mean |SHAP| in probability space)"
    )
    print("=" * 80)
    print(shown.to_string(index=False))
    print("=" * 80)


def build_paired_fold_table(fold_df: pd.DataFrame) -> pd.DataFrame:
    """One row per metric × series (XGBoost, logistic, difference) across outer folds."""
    xgb = fold_df[fold_df["model"].astype(str).str.contains("xgb", case=False)].sort_values("fold")
    lr = fold_df[fold_df["model"].astype(str).str.contains("log", case=False)].sort_values("fold")
    folds = [int(f) for f in xgb["fold"].tolist()]
    rows = []
    for metric in CORE_METRICS:
        a = xgb[metric].to_numpy(dtype=float)
        b = lr[metric].to_numpy(dtype=float)
        for series, values in (
            ("XGBoost", a),
            ("Logistic regression", b),
            ("Difference (XGB − LR)", a - b),
        ):
            row = {"metric": metric, "series": series}
            for fold, val in zip(folds, values):
                row[f"fold_{fold}"] = float(val)
            rows.append(row)
    return pd.DataFrame(rows)


def display_paired_folds(paired: pd.DataFrame) -> pd.DataFrame:
    shown_rows = []
    fold_cols = [c for c in paired.columns if c.startswith("fold_")]
    for _, rec in paired.iterrows():
        key = rec["metric"]
        row = {
            "Metric": METRIC_LABELS[key],
            "Series": rec["series"],
        }
        for col in fold_cols:
            fold_n = col.split("_", 1)[1]
            row[f"Fold {fold_n}"] = _fmt(float(rec[col]), key)
        shown_rows.append(row)
    return pd.DataFrame(shown_rows)


def write_table4(fold_df: pd.DataFrame) -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paired = build_paired_fold_table(fold_df)
    paired.to_csv(OUTPUT_DIR / "Table_4_paired_fold_scores.csv", index=False)
    shown = display_paired_folds(paired)
    headers = list(shown.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for _, row in shown.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in headers) + " |")
    (OUTPUT_DIR / "Table_4_paired_fold_scores.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return shown


def print_table4(shown: pd.DataFrame) -> None:
    print("\n" + "=" * 120)
    print(
        "Table 4. Outer-fold paired scores and differences (XGBoost − logistic regression)"
    )
    print("=" * 120)
    print(shown.to_string(index=False))
    print("=" * 120)
