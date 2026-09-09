"""Rebuild Table 1 and Table 2 from saved nested-CV and holdout CSVs."""

from __future__ import annotations

import pandas as pd

from fraud import OUTPUT_DIR
from fraud.plots import plot_results_table
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


def main() -> None:
    fold_df = pd.read_csv(OUTPUT_DIR / "Table_nested_cv_folds.csv")
    holdout_df = pd.read_csv(OUTPUT_DIR / "Table_holdout_metrics.csv")
    table1 = build_table1(fold_df, holdout_df)
    shown = write_table1(table1)
    plot_results_table(
        shown,
        path_name="Table_1_performance.png",
        title="Table 1. Fraud-class performance: PR-AUC, F1, precision, recall, FPR, and balanced accuracy",
        figsize=(15.4, 3.1),
        fontsize=8,
    )
    print_table1(shown)

    stats_df = build_holm_table(fold_df)
    shown2 = write_table2(stats_df)
    plot_results_table(
        shown2,
        path_name="Table_2_nadeau_bengio_holm.png",
        title="Table 2. Nadeau–Bengio / Holm tests (XGBoost − logistic regression)",
        figsize=(13.6, 3.4),
        fontsize=8,
    )
    print_table2(shown2)
    shap_path = OUTPUT_DIR / "Table_3_SHAP_feature_importance.csv"
    if not shap_path.exists():
        shap_path = OUTPUT_DIR / "Table_1_SHAP_feature_importance.csv"
    if shap_path.exists():
        shown3 = write_table3(pd.read_csv(shap_path))
        plot_results_table(
            shown3,
            path_name="Table_3_SHAP_feature_importance.png",
            title="Table 3. Global SHAP feature importance (probability space, interventional)",
            figsize=(8.4, 9.2),
            fontsize=8,
        )
        print_table3(shown3)

    shown4 = write_table4(fold_df)
    plot_results_table(
        shown4,
        path_name="Table_4_paired_fold_scores.png",
        title="Table 4. Outer-fold paired scores and differences (XGBoost − logistic regression)",
        figsize=(12.8, 6.6),
        fontsize=8,
    )
    print_table4(shown4)

    print(f"Wrote Table 1 to {OUTPUT_DIR / 'Table_1_performance.csv'}")
    print(f"Wrote Table 2 to {OUTPUT_DIR / 'Table_2_nadeau_bengio_holm.csv'}")
    print(f"Wrote Table 3 to {OUTPUT_DIR / 'Table_3_SHAP_feature_importance.csv'}")
    print(f"Wrote Table 4 to {OUTPUT_DIR / 'Table_4_paired_fold_scores.csv'}")


if __name__ == "__main__":
    main()
