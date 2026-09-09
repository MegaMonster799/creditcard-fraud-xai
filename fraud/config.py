"""Locked experimental protocol. Change here, not in ad-hoc scripts."""

from __future__ import annotations

SEED = 42
TEST_SIZE = 0.20
OUTER_FOLDS = 5
INNER_FOLDS = 3
HPO_ITERATIONS = 10
THRESHOLD_OBJECTIVE = "f1"  # predeclared operational objective
SMOTE_K_NEIGHBORS = 5
SMOTE_SAMPLING_STRATEGY = 1.0  # 1:1 after oversampling
# Predeclared for the matched LR vs XGB comparison. Ablation is descriptive only.
LOCKED_IMBALANCE = "smote"
AMOUNT_WINSOR_LO = 0.005
AMOUNT_WINSOR_HI = 0.995
SHAP_BACKGROUND_SIZE = 256
SHAP_EXPLAIN_SIZE = 1500
INFERENCE_BENCH_N = 20_000
ALPHA = 0.05
METRICS_FOR_MULTIPLICITY = (
    "pr_auc",
    "f1",
    "precision",
    "recall",
    "fpr",
    "balanced_accuracy",
)
FPR_OPERATING_POINTS = (0.001, 0.005, 0.01, 0.02)
