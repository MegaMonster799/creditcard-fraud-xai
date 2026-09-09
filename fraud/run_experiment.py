"""End-to-end nested-CV experiment. Run: python -m fraud.run_experiment"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from fraud import OUTPUT_DIR
from fraud.config import (
    FPR_OPERATING_POINTS,
    HPO_ITERATIONS,
    INNER_FOLDS,
    INFERENCE_BENCH_N,
    LOCKED_IMBALANCE,
    OUTER_FOLDS,
    SEED,
    SMOTE_K_NEIGHBORS,
    SMOTE_SAMPLING_STRATEGY,
    TEST_SIZE,
)
from fraud.data import load_and_split
from fraud.explain import (
    build_probability_explainer,
    mean_abs_shap_table,
    pick_case,
    save_manifest,
    stratified_explain_frame,
)
from fraud.metrics import metrics_at_fpr, metrics_at_threshold, tune_threshold
from fraud.models import build_pipeline, describe_xgb_defaults
from fraud.nested import run_nested_cv
from fraud.plots import (
    combine_side_by_side,
    plot_calibration,
    plot_confusion_and_pr,
    plot_pr_curves,
    plot_results_table,
    plot_shap_dependence,
    plot_shap_summary,
    plot_waterfall,
)
from fraud.tables import (
    build_holm_table,
    build_table1,
    print_table1,
    print_table2,
    print_table3,
    print_table4,
    write_table1,
    write_table2,
    write_table3,
    write_table4,
)


def _versions() -> dict:
    import sklearn
    import xgboost

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "sklearn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "shap": shap.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }


def _transform_for_trees(pipeline, X: pd.DataFrame) -> pd.DataFrame:
    Xt = pipeline.named_steps["winsor"].transform(X)
    return pipeline.named_steps["scale"].transform(Xt)


def _fold_table(result) -> pd.DataFrame:
    rows = []
    for fold in result.folds:
        row = {"model": result.name, "fold": fold.fold, "threshold": fold.threshold, **fold.metrics}
        rows.append(row)
    return pd.DataFrame(rows)


def refit_locked(X_train, y_train, X_test, y_test, kind, imbalance, cv_result, inner_folds, seed):
    """Refit on all training data using nested-CV hyperparameters (no second search)."""
    best_fold = max(cv_result.folds, key=lambda f: f.inner_pr_auc)
    n_neg, n_pos = int((y_train == 0).sum()), int((y_train == 1).sum())
    pipe = build_pipeline(kind, imbalance=imbalance, seed=seed, n_neg=n_neg, n_pos=n_pos)
    pipe.set_params(**best_fold.best_params)
    t0 = time.perf_counter()
    pipe.fit(X_train, y_train)
    train_s = time.perf_counter() - t0
    inner = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    oof = cross_val_predict(clone(pipe), X_train, y_train, cv=inner, method="predict_proba", n_jobs=1)[:, 1]
    threshold = tune_threshold(y_train, oof)
    t1 = time.perf_counter()
    proba = pipe.predict_proba(X_test)[:, 1]
    infer_s = time.perf_counter() - t1
    n_bench = min(INFERENCE_BENCH_N, len(X_test))
    t2 = time.perf_counter()
    _ = pipe.predict_proba(X_test.iloc[:n_bench])
    bench_s = time.perf_counter() - t2
    metrics = metrics_at_threshold(y_test, proba, threshold)
    fpr_rows = [metrics_at_fpr(y_test, proba, t) for t in FPR_OPERATING_POINTS]
    return {
        "kind": kind,
        "best_params": best_fold.best_params,
        "threshold": threshold,
        "inner_pr_auc": best_fold.inner_pr_auc,
        "metrics": metrics,
        "fpr_points": fpr_rows,
        "train_seconds": train_s,
        "holdout_infer_seconds": infer_s,
        "us_per_transaction": 1e6 * bench_s / n_bench,
        "pipeline": pipe,
        "proba": proba,
        "pred": (proba >= threshold).astype(int),
    }


def run(quick: bool = False) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outer_folds = 2 if quick else OUTER_FOLDS
    inner_folds = 2 if quick else INNER_FOLDS
    n_iter = 4 if quick else HPO_ITERATIONS
    print("=== Data audit and leak-free split ===")
    X_train, X_test, y_train, y_test, audit = load_and_split()
    if quick:
        rng = np.random.default_rng(SEED)
        fraud = np.flatnonzero(y_train.to_numpy() == 1)
        legit = np.flatnonzero(y_train.to_numpy() == 0)
        take_legit = rng.choice(legit, size=min(8_000, len(legit)), replace=False)
        take = np.concatenate([fraud, take_legit])
        X_train, y_train = X_train.iloc[take], y_train.iloc[take]
        print(f"[quick] nested CV train slice n={len(X_train)} fraud={int(y_train.sum())}")
    print(json.dumps(audit.__dict__, indent=2))
    print(
        f"Split: stratified {int((1 - TEST_SIZE) * 100)}/{int(TEST_SIZE * 100)}, "
        f"seed={SEED}. Train {len(X_train)} (fraud={int(y_train.sum())}), "
        f"holdout {len(X_test)} (fraud={int(y_test.sum())})."
    )
    print("Preprocessing (fit on train only): Amount winsorized at 0.5/99.5 percentiles; Time and Amount standardized. PCA features V1–V28 are left unchanged. No missing-value imputation (zero missing). Exact duplicate rows dropped before the split.")

    winner = LOCKED_IMBALANCE
    print(f"Imbalance method (predeclared): {winner}")

    print("\n=== Nested CV (matched HPO budget, inner-OOF F1 threshold) ===")
    xgb_cv = run_nested_cv(
        X_train, y_train, "xgb", imbalance=winner,
        outer_folds=outer_folds, inner_folds=inner_folds, n_iter=n_iter,
    )
    lr_cv = run_nested_cv(
        X_train, y_train, "logreg", imbalance=winner,
        outer_folds=outer_folds, inner_folds=inner_folds, n_iter=n_iter,
    )
    fold_df = pd.concat([_fold_table(xgb_cv), _fold_table(lr_cv)], ignore_index=True)
    fold_df.to_csv(OUTPUT_DIR / "Table_nested_cv_folds.csv", index=False)
    shown4 = write_table4(fold_df)
    print_table4(shown4)
    plot_results_table(
        shown4,
        path_name="Table_4_paired_fold_scores.png",
        title="Table 4. Outer-fold paired scores and differences (XGBoost − logistic regression)",
        figsize=(12.8, 6.6),
        fontsize=8,
    )

    n_tr = xgb_cv.folds[0].n_train
    n_va = xgb_cv.folds[0].n_val
    stats_df = build_holm_table(fold_df, n_train=n_tr, n_val=n_va)
    shown2 = write_table2(stats_df)
    print_table2(shown2)
    plot_results_table(
        shown2,
        path_name="Table_2_nadeau_bengio_holm.png",
        title="Table 2. Nadeau–Bengio / Holm tests (XGBoost − logistic regression)",
        figsize=(13.6, 3.4),
        fontsize=8,
    )

    print("\n=== Final holdout (nested-CV hyperparameters locked; threshold from train OOF) ===")
    xgb_final = refit_locked(X_train, y_train, X_test, y_test, "xgb", winner, xgb_cv, inner_folds, SEED)
    lr_final = refit_locked(X_train, y_train, X_test, y_test, "logreg", winner, lr_cv, inner_folds, SEED)
    holdout_rows = []
    for name, pack in (("xgboost", xgb_final), ("logreg", lr_final)):
        row = {"model": name, **pack["metrics"], "train_seconds": pack["train_seconds"], "us_per_transaction": pack["us_per_transaction"]}
        holdout_rows.append(row)
        print(f"{name}: PR-AUC={pack['metrics']['pr_auc']:.4f} F1={pack['metrics']['f1']:.4f} "
              f"Prec={pack['metrics']['precision']:.4f} Rec={pack['metrics']['recall']:.4f} "
              f"Brier={pack['metrics']['brier']:.4f} thr={pack['threshold']:.4f} "
              f"CM=[tn={pack['metrics']['tn']}, fp={pack['metrics']['fp']}, "
              f"fn={pack['metrics']['fn']}, tp={pack['metrics']['tp']}]")
    holdout_df = pd.DataFrame(holdout_rows)
    holdout_df.to_csv(OUTPUT_DIR / "Table_holdout_metrics.csv", index=False)
    table1 = build_table1(fold_df, holdout_df)
    shown = write_table1(table1)
    print_table1(shown)
    plot_results_table(
        shown,
        path_name="Table_1_performance.png",
        title="Table 1. Fraud-class performance: PR-AUC, F1, precision, recall, FPR, and balanced accuracy",
        figsize=(15.4, 3.1),
        fontsize=8,
    )
    plot_confusion_and_pr(
        {
            "XGBoost": xgb_final["metrics"],
            "Logistic regression": lr_final["metrics"],
        },
        y_test,
        {"XGBoost": xgb_final["proba"], "Logistic regression": lr_final["proba"]},
        path_name="Figure_confusion_matrices.png",
    )
    fpr_table = pd.DataFrame(
        [{"model": "xgboost", **r} for r in xgb_final["fpr_points"]]
        + [{"model": "logreg", **r} for r in lr_final["fpr_points"]]
    )
    fpr_table.to_csv(OUTPUT_DIR / "Table_holdout_fpr_operating_points.csv", index=False)

    plot_pr_curves(y_test, {"XGBoost": xgb_final["proba"], "Logistic regression": lr_final["proba"]})
    plot_calibration(y_test, {"XGBoost": xgb_final["proba"], "Logistic regression": lr_final["proba"]})

    print("\n=== SHAP (probability, interventional, real-prevalence background) ===")
    xgb_model = xgb_final["pipeline"].named_steps["model"]
    X_train_t = _transform_for_trees(xgb_final["pipeline"], X_train)
    X_test_t = _transform_for_trees(xgb_final["pipeline"], X_test)
    explainer, background, manifest = build_probability_explainer(xgb_model, X_train_t, y_train)
    X_exp, y_exp = stratified_explain_frame(X_test_t, y_test)
    shap_values = explainer.shap_values(X_exp)
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:
        shap_values = shap_values[1]
    importance = mean_abs_shap_table(np.asarray(shap_values), list(X_exp.columns))
    shown3 = write_table3(importance)
    print_table3(shown3)
    plot_results_table(
        shown3,
        path_name="Table_3_SHAP_feature_importance.png",
        title="Table 3. Global SHAP feature importance (probability space, interventional)",
        figsize=(8.4, 9.2),
        fontsize=8,
    )
    plot_shap_summary(shap_values, X_exp)
    plot_shap_dependence(shap_values, X_exp, feature="V14", interaction_index="V7")
    pred = xgb_final["pred"]
    proba = xgb_final["proba"]
    for kind, title, fname in (
        ("tp", "Figure 5. SHAP waterfall — true positive (probability space)", "Figure_SHAP_waterfall_TP.png"),
        ("fp", "Figure 6. SHAP waterfall — false positive (probability space)", "Figure_SHAP_waterfall_FP.png"),
        ("fn", "Figure 7. SHAP waterfall — false negative (probability space)", "Figure_SHAP_waterfall_FN.png"),
    ):
        idx = pick_case(y_test, pred, proba, kind)
        if idx is None:
            print(f"  no {kind} case on holdout")
            continue
        row = X_test_t.iloc[idx]
        explanation = shap.Explanation(
            values=np.asarray(explainer.shap_values(row.to_frame().T)).reshape(-1),
            base_values=float(np.asarray(explainer.expected_value).reshape(-1)[0]),
            data=row.values,
            feature_names=list(X_test_t.columns),
        )
        plot_waterfall(explanation, title, fname)
        print(f"  {kind} holdout index {idx}  p={proba[idx]:.4f}  y={int(y_test.iloc[idx])}")
    combined = combine_side_by_side(
        "Figure_SHAP_waterfall_FP.png",
        "Figure_SHAP_waterfall_FN.png",
        "Figure_SHAP_waterfall_FP_FN.png",
    )
    print(f"  combined Figure 6 and Figure 7 -> {combined.name}")

    manifest.update(
        {
            "seed": SEED,
            "imbalance": winner,
            "smote": {
                "k_neighbors": SMOTE_K_NEIGHBORS,
                "sampling_strategy": SMOTE_SAMPLING_STRATEGY,
                "random_state": SEED,
            },
            "xgb_search": describe_xgb_defaults(),
            "xgb_selected_params": xgb_final["best_params"],
            "threshold_objective": "maximize F1 on inner out-of-fold scores",
            "locked_threshold": xgb_final["threshold"],
            "model_used_for_shap": "final holdout XGBoost refit on the full training split",
            "hardware_and_timing": {
                **_versions(),
                "xgb_train_seconds": xgb_final["train_seconds"],
                "logreg_train_seconds": lr_final["train_seconds"],
                "xgb_us_per_transaction": xgb_final["us_per_transaction"],
                "logreg_us_per_transaction": lr_final["us_per_transaction"],
            },
            "data_audit": audit.__dict__,
        }
    )
    save_manifest(OUTPUT_DIR / "methods_manifest.json", manifest)
    print(f"\nWrote artifacts to {OUTPUT_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Nested-CV credit-card fraud experiment")
    parser.add_argument("--quick", action="store_true", help="2 outer folds, 4 HPO draws (debug)")
    args = parser.parse_args()
    run(quick=args.quick)


if __name__ == "__main__":
    main()
