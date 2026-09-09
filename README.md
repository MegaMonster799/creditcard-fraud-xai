# Explainable AI for Extreme Class Imbalance

Code and tables for the manuscript **Explainable AI for Extreme Class Imbalance: Interpreting Latent Variables in European Credit Card Transactions**.

The experiment compares **XGBoost** to **logistic regression** on the ULB 2013 European credit-card dataset after a leak-free nested protocol: stratified 80/20 split, outer 5-fold / inner 3-fold `RandomizedSearchCV` (matched search budget, scoring = PR-AUC), SMOTE only on training folds, and an F1 threshold locked on inner out-of-fold scores. Differences are tested with a Nadeau–Bengio corrected paired *t*-test and Holm adjustment. Explanations use interventional TreeExplainer in **probability space** against a real-prevalence background.

## Dataset

The ULB file is **not** stored in this repository (about 150 MB).

1. Download [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (`creditcard.csv`).
2. Place it at the repository root:

```text
creditcard-fraud-xai/creditcard.csv
```

Public release: 284,807 transactions, 492 frauds (0.172%). This pipeline drops 1,081 exact duplicate rows before the split, leaving **283,726** rows and **473** frauds (0.167%). Cite Dal Pozzolo et al. (IEEE SSCI 2015) and Chawla et al. (JAIR 2002) for the data and SMOTE, not only Kaggle.

## Setup

Python 3.12 recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run commands from the repository root so `python -m fraud.…` can import the package and find `creditcard.csv`.

## Reproduce the paper

```bash
# Figure 1 (cleaned data: 283,253 legitimate / 473 fraud)
python -m fraud.eda

# Nested CV, holdout, Tables 1–4, SHAP figures (slow; SMOTE + search)
python -m fraud.run_experiment

# Rebuild tables from saved fold / holdout CSVs (no retraining)
python -m fraud.make_table
```

`python -m fraud.run_experiment --quick` uses fewer folds and HPO draws for a smoke test. Do not report `--quick` numbers in the manuscript.

## Protocol (locked)

| Choice | Value |
|---|---|
| Split | Stratified 80/20, `random_state=42` |
| Nested CV | 5 outer folds, 3 inner folds, `n_iter=10`, scoring = average precision |
| Imbalance | SMOTE, `k_neighbors=5`, `sampling_strategy=1.0`, seed 42, **training folds only** |
| Amount | Winsorized at training 0.5th / 99.5th percentiles; Time and Amount scaled on train only |
| Threshold | Maximize fraud-class F1 on inner out-of-fold scores; freeze for outer fold and holdout |
| Tests | Nadeau–Bengio corrected paired *t* (df = 4), Holm over PR-AUC, F1, precision, recall, FPR, balanced accuracy |
| SHAP | TreeExplainer, interventional, `model_output='probability'`, background *n* = 256 from the imbalanced training matrix |

SMOTE is predeclared. There is no ablation that chooses the imbalance method.

## Paper artifacts

All generated files are written to [`fraud/output/`](fraud/output/).

| File | Contents |
|---|---|
| `Table_1_performance.*` | Nested-CV and holdout PR-AUC, F1, precision, recall, FPR, balanced accuracy |
| `Table_2_nadeau_bengio_holm.*` | Corrected tests and Holm decisions |
| `Table_3_SHAP_feature_importance.*` | Global mean \|SHAP\| ranks |
| `Table_4_paired_fold_scores.*` | Five outer-fold paired scores and XGBoost − logistic differences |
| `Figure_1_EDA.png` | Class counts, Amount boxplots, 48-hour Time KDEs |
| `Figure_PR_curves.png` / `Figure_calibration.png` | Holdout PR curves and reliability diagram |
| `Figure_SHAP_summary.png` | Global SHAP summary (probability space) |
| `Figure_SHAP_waterfall_TP.png` | Figure 4, true positive |
| `Figure_SHAP_waterfall_FP_FN.png` | Figures 5–6, false positive and false negative |
| `methods_manifest.json` | Selected hyperparameters, locked threshold, library versions, timing |

On the saved SMOTE run, only **PR-AUC** is significant after Holm (mean difference +0.1091, *p* = 0.0022). Holdout recall is 0.726 (XGBoost) vs 0.758 (logistic regression). Do not claim a significant F1 or FPR gain.

## Layout

```text
creditcard.csv          # you download this
requirements.txt
fraud/
  config.py             # seeds, folds, SMOTE, locked method
  data.py               # audit, split, winsorize, scale
  models.py             # pipelines and search spaces
  nested.py             # nested CV
  metrics.py / stats.py / tables.py / plots.py / explain.py
  run_experiment.py
  eda.py
  make_table.py
  output/               # tables and figures
  REVISION_GUIDE.md     # notes for the manuscript revision
```

## License and data terms

Code in this repository is provided for reproducing the paper. The credit-card file remains subject to the ULB / Kaggle terms. Do not commit `creditcard.csv`.
