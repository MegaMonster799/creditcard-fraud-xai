# How to answer each reviewer comment

Code-side items are implemented in `python -m fraud.run_experiment` and `python -m fraud.eda`.
Paste the corresponding numbers from `fraud/output/` into the manuscript. Writing-only items are drafted below.

## Major (must fix)

### 1. Expand the literature review to 25–30 peer-reviewed papers

Replace the short Introduction survey with a critical review that names a gap, not a list. Suggested structure and sources (all primary research / journal or proceedings papers):

**Problem and data.** Dal Pozzolo et al. on concept drift and alert aggregation in credit-card fraud; the ULB 2013 PCA-anonymized corpus that this paper uses; Carcillo et al. on scalable fraud monitoring.

**Classical imbalance.** Chawla et al. (SMOTE, JAIR 2002); He et al. (ADASYN); Seiffert et al. (RUSBoost); Liu et al. (EasyEnsemble / BalanceCascade, TKDE); Elkan on cost-sensitive learning; Domingos (MetaCost).

**Ensembles on this task.** Chen & Guestrin (XGBoost, KDD 2016); Ke et al. (LightGBM); Prokhorenkova et al. (CatBoost); Bhattacharyya et al. and Whitrow et al. on aggregating transaction features.

**Anomaly detection.** Liu, Ting & Zhou (Isolation Forest); Schölkopf et al. (one-class SVM); Chalapathy & Chawla surveys of deep anomaly detection; Autoencoder-based fraud papers in Expert Systems with Applications.

**Graph-based fraud.** Dou et al. (CARE-GNN, CIKM); Liu et al. (GraphConsis; Pick and Choose); Wang et al. (SemiGNN); recent GNN fraud surveys in TKDE/CSUR. These methods need entity graphs this PCA dataset does not provide — that is a limitation, not a competitor you failed to beat.

**Transformers / tabular deep models.** Huang et al. (TabTransformer); Gorishniy et al. (FT-Transformer); Arik & Pfister (TabNet). They are data-hungry relative to 492 positives.

**Explainability.** Lundberg & Lee (SHAP, NeurIPS 2017); Lundberg et al. (TreeExplainer, Nature MI 2020); Ribeiro et al. (LIME). Do **not** claim “prior fraud work lacks SHAP.” The gap is narrower: few papers jointly (i) nest threshold selection inside resampling-aware CV, (ii) test XGBoost vs linear models with a fold-overlap-corrected test, and (iii) report TreeExplainer in probability space against a real-prevalence background.

Cite 25–30 of the above as peer-reviewed items. Drop blogs, Towards Data Science posts, and Kaggle kernels (Major issue 9).

### 2. Research questions, hypotheses, contributions

Insert at the end of the Introduction:

**RQ1.** After nested hyperparameter search, inner-OOF F1 thresholding, and identical SMOTE preprocessing, does XGBoost improve **threshold-free** ranking (PR-AUC) over L1/L2 logistic regression on ULB holdout data?

**RQ2.** Do F1/precision/FPR/balanced-accuracy gains survive a Nadeau–Bengio corrected paired test with Holm adjustment across six metrics?

**RQ3.** Which PCA components dominate P(fraud) under interventional TreeExplainer with a real-prevalence background, and do TP/FP/FN cases share those drivers?

**H1.** XGBoost raises PR-AUC relative to logistic regression (threshold-free).

**H2.** Any F1 gap at a locked threshold is smaller, and may not be significant, once both models receive the same threshold protocol.

**H3.** A small subset of latent components (typically V4, V14, V12, V10 in prior ULB studies) accounts for most mean |SHAP|.

**Contributions.** (1) A leak-free nested protocol with matched HPO budget and a predeclared F1 operating point. (2) Corrected inference for overlapping CV folds plus Holm control. (3) Probability-space SHAP with documented background, plus TP/FP/FN waterfalls and a numerical importance table.

### 3. SMOTE details and justification

Already in code / `methods_manifest.json`:

- `k_neighbors = 5`
- `sampling_strategy = 1.0` (synthetic minority until 1:1 on each training fold)
- `random_state = 42`
- Applied **inside** `imblearn.pipeline.Pipeline`, so validation/holdout folds are never resampled.

Justification to write: SMOTE (`k=5`, 1:1, seed 42) is predeclared and applied only inside training folds. There is no ablation. Keep the conceptual why-SMOTE paragraph; do not mention ADASYN/class-weight scores.

### 4. XGBoost hyperparameters and search

Do not list a single hand-tuned vector as if it were given. Report:

- Search: `RandomizedSearchCV`, `n_iter=10`, inner 3-fold, scoring = average precision, **same n_iter for logistic regression**.
- Ranges: see `describe_xgb_defaults()` / `methods_manifest.json` (`subsample`, `colsample_bytree`, `gamma`, `min_child_weight`, `reg_alpha`, `reg_lambda`, `n_estimators`, `max_depth`, `learning_rate`).
- Early stopping: **not used**. Tree count is selected by inner PR-AUC so the validation fold used for thresholding is not also used to stop boosting.
- Selected values: `fraud/output/methods_manifest.json` → `xgb_selected_params`.

### 5. Split, missingness, duplicates, outliers

Write the protocol (now true in code):

1. Drop exact duplicate rows (count in the audit JSON).
2. Stratified 80/20 split, `random_state=42`, **before** any scaling.
3. Outer 5-fold stratified CV on the 80% train set (shuffle=True, seed=42). Each outer training fold is ~4/5 of train; the outer validation fold is the remaining 1/5.
4. Inner 3-fold randomized search on that outer training fold.
5. Amount winsorized at the 0.5th/99.5th **training** percentiles; Time and Amount standardized on train only. V1–V28 untouched.
6. Missing values: none in ULB; no imputation.

The old script scaled Time/Amount on the full frame before splitting — that leak is gone.

### 6. Threshold, PR-AUC, calibration, modest claims

Operational objective (predeclared): **maximize F1 of the fraud class**.

Threshold is chosen on **inner out-of-fold** scores of the selected pipeline, then frozen for the outer fold and for the holdout. Both models get the same preprocessor, the same `n_iter`, and the same threshold rule.

Report from `Table_holdout_metrics.csv` and `Table_holdout_fpr_operating_points.csv`: PR-AUC, ROC-AUC, F1, precision, recall, balanced accuracy, FPR, Brier, confusion counts (TN/FP/FN/TP). Figures: PR curves, reliability diagram.

If Holm-adjusted tests are non-significant, **soften the Abstract/Discussion**: XGBoost is not a proven statistically significant upgrade; ROC-AUC near 0.98 for both models means ranking is similar; F1 at one threshold is not a deployment metric by itself.

### 7. Invalid paired t-test

Stop using a student t-test on five folds as if they were independent.

The code reports, for each of PR-AUC, F1, precision, recall, FPR, and balanced accuracy:

- the five paired scores and differences
- Nadeau–Bengio corrected t, df = 4, correction factor `(1/k + n_val/n_train)`
- Cohen’s d
- bootstrap CI of the mean difference
- Holm thresholds for six simultaneous tests
- seeds

If `significant_after_holm` is false, withdraw “p < 0.001 / highly significant.”

### 8. SHAP output space

The old manuscript treated a log-odds base value as a probability. The new figures are **interventional TreeExplainer, `model_output='probability'`**, background = 256 rows drawn from the **imbalanced** training matrix (not SMOTE). `methods_manifest.json` records SHAP version, perturbation mode, background, model, and that explanations are additive in P(fraud).

Waterfalls: Figure 4 TP (`Figure_SHAP_waterfall_TP.png`). Figures 5 (FP) and 6 (FN) are labeled individually but saved together as `Figure_SHAP_waterfall_FP_FN.png`.

### 9. Replace non-peer-reviewed references

Remove Kaggle blogs, Medium, documentation-only URLs except as footnotes for software. Replace with the journal/conference papers in item 1. Software: cite the XGBoost, SHAP, and scikit-learn papers, not only websites.

---

## Major (recommended)

1. **Gap.** Not “nobody uses SHAP.” Gap = nested threshold + corrected tests + probability-space SHAP on ULB latent features.
2. **2013 data.** Acknowledge ULB is dated; IEEE-CIS 2019 and more recent card datasets exist. This revision stays on ULB so results remain comparable to the large ULB literature; name that as a limitation.
3. **FP/FN SHAP.** Implemented.
4. **Compute.** `methods_manifest.json` has train seconds, µs/transaction, OS/CPU, library versions.
5. **Imbalance metrics.** PR-AUC, balanced accuracy, FPR, Brier, FPR-targeted operating points.
6. **Table 3 importance.** `Table_3_SHAP_feature_importance.csv` (mean |SHAP| ranks in probability space).

## Minor

1. Discuss SHAP in Methods + Results only; one sentence in the Introduction.
2. Fix “intepretability” → interpretability; “Logisitic” → Logistic.
3. Point reviewers to this repository and `python -m fraud.run_experiment`.
4. Captions: Figure 1 EDA, Figure 2 SHAP summary, Figure 3 PR, Figure 4 TP waterfall, Figures 5–6 combined FP/FN waterfalls (`Figure_SHAP_waterfall_FP_FN.png`), Figure 7 calibration, Figure 8 confusion. Tables: Table 1 performance, Table 2 Holm tests, Table 3 global SHAP importance, Table 4 paired outer-fold scores.

## Formatting

Times New Roman 12 pt, bold headings, stay within the page cap. Do not dump all 30 papers as a bullet list — write them as an argument.

## Commands

```bash
python -m fraud.eda
python -m fraud.run_experiment          # full nested CV
python -m fraud.run_experiment --quick  # debug
```
