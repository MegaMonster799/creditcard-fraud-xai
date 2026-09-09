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
    ax.set_title("Figure 9. Precision–recall curves on the untouched holdout set")
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
    ax.set_title("Figure 8. Reliability diagram on the holdout set")
    ax.legend()
    fig.tight_layout()
    return _save(fig, path_name)


def plot_shap_summary(shap_values, X, path_name: str = "Figure_SHAP_summary.png") -> Path:
    fig = plt.figure(figsize=(8.2, 6.4))
    shap.summary_plot(shap_values, X, show=False)
    plt.title("Figure 3. SHAP summary (probability space, interventional)")
    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / path_name
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


def plot_shap_dependence(
    shap_values,
    X,
    feature: str = "V14",
    interaction_index: str = "V7",
    path_name: str = "Figure_SHAP_dependence_V14.png",
) -> Path:
    shap.dependence_plot(
        feature,
        np.asarray(shap_values),
        X,
        interaction_index=interaction_index,
        show=False,
    )
    plt.title(
        f"Figure 4. SHAP dependence of {feature} (interaction: {interaction_index})",
        pad=10,
    )
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


_XGB_COLOR = "#1f77b4"
_LR_COLOR = "#ff7f0e"


def _model_color(name: str) -> str:
    return _XGB_COLOR if "xgb" in name.lower() else _LR_COLOR


def _holdout_maps(holdout_df: pd.DataFrame) -> dict[str, dict]:
    maps: dict[str, dict] = {}
    for _, rec in holdout_df.iterrows():
        name = "XGBoost" if "xgb" in str(rec["model"]).lower() else "Logistic regression"
        maps[name] = rec.to_dict()
    return maps


def _metric_footnote(metrics: dict) -> str:
    f1 = float(metrics["f1"])
    precision = float(metrics["precision"])
    recall = float(metrics["recall"])
    threshold = float(metrics["threshold"])
    thr_txt = f"{threshold:.6f}" if threshold >= 0.999 else f"{threshold:.3f}"
    return (
        f"Locked F1 threshold = {thr_txt}\n"
        f"F1 = {f1:.3f}   Precision = {precision:.3f}   Recall = {recall:.3f}"
    )


def _draw_confusion(ax, metrics: dict, title: str, vmax: float, title_color: str | None = None):
    from matplotlib.colors import LogNorm

    cm = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]], dtype=float)
    im = ax.imshow(cm, cmap="Blues", norm=LogNorm(vmin=1, vmax=max(vmax, 1)), aspect="equal")
    ax.set_xticks([0, 1], ["Predicted\nlegitimate", "Predicted\nfraud"])
    ax.set_yticks([0, 1], ["Actual\nlegitimate", "Actual\nfraud"])
    ax.tick_params(length=0)
    ax.set_title(title, fontsize=12, color=title_color or "black", pad=10, fontweight="semibold")
    labels = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            val = int(cm[i, j])
            ax.text(
                j,
                i,
                f"{labels[i][j]}\n{val:,}",
                ha="center",
                va="center",
                color="white" if val > 100 else "black",
                fontsize=11,
                fontweight="bold",
            )
    ax.text(
        0.5,
        -0.22,
        _metric_footnote(metrics),
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        color="#333333",
        linespacing=1.35,
        clip_on=False,
    )
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#4a4a4a")
    return im


def plot_confusion_matrices(results: dict[str, dict], path_name: str = "Figure_confusion_matrices.png") -> Path:
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(11.4, 4.8))
    if n == 1:
        axes = [axes]
    vmax = max(max(m["tn"], m["fp"], m["fn"], m["tp"]) for m in results.values())
    im = None
    for ax, (name, m), letter in zip(axes, results.items(), "abcdefghijklmnopqrstuvwxyz"):
        im = _draw_confusion(ax, m, f"({letter}) {name}", vmax)
    fig.suptitle("Figure 2. Holdout confusion matrices at the locked F1 threshold", fontsize=13)
    fig.subplots_adjust(top=0.82, wspace=0.32, left=0.10, right=0.78)
    cax = fig.add_axes([0.88, 0.18, 0.025, 0.55])
    fig.colorbar(im, cax=cax, label="Count (log scale)")
    return _save(fig, path_name)


def _draw_pr_from_scores(ax, y_true, curves: dict[str, np.ndarray], results: dict[str, dict]) -> None:
    from sklearn.metrics import average_precision_score, precision_score, recall_score

    y_true = np.asarray(y_true)
    prevalence = float(np.mean(y_true))
    ax.axhline(prevalence, ls=":", c="0.55", lw=1.4, label=f"No-skill (PR-AUC = {prevalence:.4f})")
    marked = False
    for label, proba in curves.items():
        precision, recall, _ = precision_recall_curve(y_true, proba)
        ap = average_precision_score(y_true, proba)
        color = _model_color(label)
        ax.plot(recall, precision, lw=2.2, color=color, label=f"{label} (PR-AUC = {ap:.3f})")
        metrics = results.get(label)
        if metrics is not None and "threshold" in metrics:
            pred = (np.asarray(proba) >= float(metrics["threshold"])).astype(int)
            ax.scatter(
                recall_score(y_true, pred, zero_division=0),
                precision_score(y_true, pred, zero_division=0),
                s=48,
                zorder=5,
                color=color,
                edgecolors="black",
                linewidths=0.7,
                label="Locked F1 threshold" if not marked else None,
            )
            marked = True
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle="--", alpha=0.28)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", frameon=True, fancybox=False, edgecolor="#dddddd")


def _draw_pr_from_image(ax, image_path: Path) -> None:
    """Reuse the published PR-curve PNG so holdout scores are not recomputed."""
    image = Image.open(image_path).convert("RGBA")
    arr = np.asarray(image)
    rgb = arr[:, :, :3]
    ink = (rgb.min(axis=2) <= 250).sum(axis=1)
    cols = np.where((rgb.min(axis=2) <= 250).any(axis=0))[0]
    # Title is the first ink blob; the axes start after the following quiet band.
    active = ink > 40
    blob_starts = np.where(active & ~np.r_[False, active[:-1]])[0]
    crop_top = int(blob_starts[1]) - 12 if len(blob_starts) >= 2 else int(np.argmax(active))
    crop_top = max(crop_top, 0)
    crop = arr[crop_top : int(np.where(ink > 0)[0][-1]) + 8, int(cols[0]) : int(cols[-1]) + 1]
    ax.imshow(crop, aspect="equal")
    ax.set_axis_off()


def plot_confusion_and_pr(
    results: dict[str, dict],
    y_true=None,
    curves: dict[str, np.ndarray] | None = None,
    path_name: str = "Figure_confusion_matrices.png",
    pr_image: str | Path | None = None,
) -> Path:
    """(a) XGBoost CM, (b) logistic CM, (c) holdout precision–recall curves."""
    names = list(results.keys())
    fig = plt.figure(figsize=(12.0, 11.0))
    gs = fig.add_gridspec(
        2,
        3,
        height_ratios=[0.92, 1.42],
        width_ratios=[1.0, 1.0, 0.045],
        hspace=0.38,
        wspace=0.28,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[0, 2])
    ax_c = fig.add_subplot(gs[1, :])

    vmax = max(max(m["tn"], m["fp"], m["fn"], m["tp"]) for m in results.values())
    im = _draw_confusion(
        ax_a, results[names[0]], f"(a) {names[0]}", vmax, title_color=_model_color(names[0])
    )
    _draw_confusion(
        ax_b, results[names[1]], f"(b) {names[1]}", vmax, title_color=_model_color(names[1])
    )
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Count (log scale)", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    ax_c.set_title("(c) Precision–recall curves on the untouched holdout set", fontsize=12, pad=8)
    if curves is not None and y_true is not None:
        _draw_pr_from_scores(ax_c, y_true, curves, results)
    elif pr_image is not None:
        _draw_pr_from_image(ax_c, Path(pr_image))
    else:
        raise ValueError("Provide holdout scores or pr_image")

    fig.suptitle(
        "Figure 2. Holdout confusion matrices at the locked F1 threshold and precision–recall curves",
        fontsize=13,
        y=0.995,
    )
    fig.subplots_adjust(top=0.93, left=0.08, right=0.96, bottom=0.04)
    return _save(fig, path_name)


def compose_figure2_from_saved(
    path_name: str = "Figure_confusion_matrices.png",
) -> Path:
    """Build Figure 2 from Table 1 holdout counts and the existing PR-curve PNG."""
    holdout = pd.read_csv(OUTPUT_DIR / "Table_holdout_metrics.csv")
    results = _holdout_maps(holdout)
    return plot_confusion_and_pr(
        results,
        pr_image=OUTPUT_DIR / "Figure_PR_curves.png",
        path_name=path_name,
    )


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


if __name__ == "__main__":
    out = compose_figure2_from_saved()
    print(f"Wrote {out}")
