"""Publication figures with sequential numbers and captions in the filenames."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve

from fraud import OUTPUT_DIR


def _save(fig, name: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_pr_curves(y_true, curves: dict[str, np.ndarray], path_name: str = "Figure_PR_curves.png") -> Path:
    from sklearn.metrics import average_precision_score

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    prevalence = float(np.mean(y_true))
    ax.axhline(prevalence, ls=":", c="0.5", label=f"No-skill (PR-AUC = {prevalence:.4f})")
    for label, proba in curves.items():
        p, r, _ = precision_recall_curve(y_true, proba)
        ap = average_precision_score(y_true, proba)
        ax.plot(r, p, lw=2, label=f"{label} (PR-AUC = {ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Figure 3. Precision–recall curves on the untouched holdout set")
    ax.legend(loc="lower left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    return _save(fig, path_name)


def plot_calibration(y_true, curves: dict[str, np.ndarray], path_name: str = "Figure_calibration.png") -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.plot([0, 1], [0, 1], ls="--", c="gray", label="Perfect calibration")
    for label, proba in curves.items():
        frac, mean_p = calibration_curve(y_true, proba, n_bins=12, strategy="quantile")
        ax.plot(mean_p, frac, marker="o", lw=2, label=label)
    ax.set_xlabel("Predicted P(fraud)")
    ax.set_ylabel("Observed fraud frequency")
    ax.set_title("Figure 7. Reliability diagram on the holdout set")
    ax.legend()
    fig.tight_layout()
    return _save(fig, path_name)


def plot_shap_summary(shap_values, X, path_name: str = "Figure_SHAP_summary.png") -> Path:
    fig = plt.figure(figsize=(8.2, 6.4))
    shap.summary_plot(shap_values, X, show=False)
    plt.title("Figure 2. SHAP summary (probability space, interventional)")
    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / path_name
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


def plot_waterfall(explanation, title: str, path_name: str) -> Path:
    shap.plots.waterfall(explanation, max_display=12, show=False)
    plt.title(title)
    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / path_name
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


def combine_side_by_side(
    left_path: str | Path,
    right_path: str | Path,
    out_name: str,
    gap: int = 36,
) -> Path:
    """Place two labeled figures in one file without changing their titles."""
    left = Image.open(OUTPUT_DIR / left_path).convert("RGBA")
    right = Image.open(OUTPUT_DIR / right_path).convert("RGBA")
    height = max(left.height, right.height)
    if left.height != height:
        canvas = Image.new("RGBA", (left.width, height), (255, 255, 255, 255))
        canvas.paste(left, (0, (height - left.height) // 2))
        left = canvas
    if right.height != height:
        canvas = Image.new("RGBA", (right.width, height), (255, 255, 255, 255))
        canvas.paste(right, (0, (height - right.height) // 2))
        right = canvas
    combined = Image.new("RGBA", (left.width + gap + right.width, height), (255, 255, 255, 255))
    combined.paste(left, (0, 0))
    combined.paste(right, (left.width + gap, 0))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / out_name
    combined.save(path)
    return path


def plot_eda(df: pd.DataFrame, path_name: str = "Figure_1_EDA.png") -> Path:
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.22)

    ax1 = fig.add_subplot(gs[0, 0])
    sns.countplot(x="Class", data=df, hue="Class", palette="Set2", legend=False, ax=ax1)
    ax1.set_title("(a) Distribution of Legitimate vs. Fraudulent Transactions", fontsize=12)
    ax1.set_xlabel("Class (0 = Legitimate, 1 = Fraudulent)", fontsize=10)
    ax1.set_ylabel("Number of Transactions", fontsize=10)
    ax1.set_yscale("log")
    for patch in ax1.patches:
        height = patch.get_height()
        ax1.annotate(
            f"{int(height):,}",
            (patch.get_x() + patch.get_width() / 2.0, height),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax2 = fig.add_subplot(gs[0, 1])
    sns.boxplot(x="Class", y="Amount", data=df, hue="Class", palette="Set2", legend=False, ax=ax2)
    ax2.set_title("(b) Transaction Amount by Class", fontsize=12)
    ax2.set_xlabel("Class (0 = Legitimate, 1 = Fraudulent)", fontsize=10)
    ax2.set_ylabel("Transaction Amount", fontsize=10)
    ax2.set_yscale("symlog")

    ax3 = fig.add_subplot(gs[1, :])
    sns.kdeplot(
        data=df.loc[df["Class"] == 0],
        x="Time",
        label="Legitimate",
        fill=True,
        color="#66c2a5",
        alpha=0.35,
        ax=ax3,
    )
    sns.kdeplot(
        data=df.loc[df["Class"] == 1],
        x="Time",
        label="Fraudulent",
        fill=True,
        color="#fc8d62",
        alpha=0.45,
        ax=ax3,
    )
    ax3.set_title("(c) Temporal Density of Transactions over 48 Hours", fontsize=12)
    ax3.set_xlabel("Time (seconds from first transaction)", fontsize=10)
    ax3.set_ylabel("Density", fontsize=10)
    ax3.set_xlim(0, float(df["Time"].max()))
    ax3.legend()

    fig.suptitle("Figure 1. Exploratory Data Analysis of the ULB Dataset", fontsize=13, y=0.98)
    fig.subplots_adjust(top=0.91)
    path = _save(fig, path_name)
    sns.reset_defaults()
    matplotlib.use("Agg")
    return path


def plot_confusion_matrices(results: dict[str, dict], path_name: str = "Figure_confusion_matrices.png") -> Path:
    fig, axes = plt.subplots(1, len(results), figsize=(4.6 * len(results), 4.0))
    if len(results) == 1:
        axes = [axes]
    for ax, (name, m) in zip(axes, results.items()):
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]], dtype=float)
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1], ["Pred 0", "Pred 1"])
        ax.set_yticks([0, 1], ["True 0", "True 1"])
        ax.set_title(name)
        for (i, j), val in np.ndenumerate(cm):
            ax.text(j, i, f"{int(val):,}", ha="center", va="center", color="black", fontsize=12)
    fig.suptitle("Figure 8. Holdout confusion counts at the locked F1 threshold")
    fig.tight_layout()
    return _save(fig, path_name)


def plot_results_table(
    shown: pd.DataFrame,
    path_name: str = "Table_1_performance.png",
    title: str = "Table 1. Fraud-class performance: PR-AUC, F1, precision, recall, FPR, and balanced accuracy",
    figsize: tuple[float, float] | None = None,
    fontsize: int = 8,
) -> Path:
    fig, ax = plt.subplots(figsize=figsize or (15.4, 2.8 + 0.28 * len(shown)))
    ax.axis("off")
    ax.set_title(
        title,
        fontsize=12,
        pad=12,
        loc="left",
        fontweight="bold",
    )
    cell_text = shown.values.tolist()
    col_labels = list(shown.columns)
    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="upper center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1.05, 1.55)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if row == 0:
            cell.set_facecolor("#1f4e79")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f4f7fb")
        else:
            cell.set_facecolor("white")
    fig.tight_layout()
    return _save(fig, path_name)
