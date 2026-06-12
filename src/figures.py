"""Figure-generation utilities for the manuscript."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def annotate_counts(ax, x, y, counts, dy_px: int = 8) -> None:
    """Annotate points with excluded counts."""
    for xi, yi, ci in zip(x, y, counts):
        if np.isfinite(yi):
            ax.annotate(
                str(int(ci)),
                xy=(xi, yi),
                xytext=(0, dy_px),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85),
            )


def plot_threshold_curve(
    curve_df: pd.DataFrame,
    xlabel: str,
    title: str,
    save_path: str | Path | None = None,
    baseline_accuracy: float | None = None,
) -> None:
    """Plot retained-case accuracy vs uncertainty threshold."""
    fig, ax = plt.subplots(figsize=(4.2, 3.2), dpi=300)
    ax.plot(curve_df["threshold"], curve_df["accuracy_retained"], "-o", label="Accuracy")
    if baseline_accuracy is not None:
        ax.axhline(baseline_accuracy, ls="--", color="0.6", lw=1, label="Baseline")
    annotate_counts(ax, curve_df["threshold"], curve_df["accuracy_retained"], curve_df["excluded_count"])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Accuracy of retained cases")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def infer_dataset_name(path: str) -> str:
    """Infer dataset source from image path/filename."""
    s = str(path).lower()
    if "drishti" in s or "drishtigs" in s:
        return "Drishti"
    if "origa" in s:
        return "ORIGA"
    if "rim-one" in s or "rim_one" in s or "rimone" in s or "/rim" in s:
        return "RIM-ONE"
    if "lag" in s:
        return "LAG"
    return "UNKNOWN"


def plot_uncertainty_by_dataset(
    filepaths: Iterable[str],
    uncertainty: np.ndarray,
    ylabel: str,
    title: str,
    save_path: str | Path | None = None,
    approximate_counts: Dict[str, int] | None = None,
) -> None:
    """Scatter uncertainty values in dataset blocks.

    If dataset names cannot be inferred from paths, approximate_counts is used.
    Example: {'Drishti': 330, 'ORIGA': 220, 'RIM-ONE': 130}; LAG is the remainder.
    """
    filepaths = list(filepaths)
    uncertainty = np.asarray(uncertainty, dtype=float).reshape(-1)
    n = uncertainty.size
    order = ["Drishti", "ORIGA", "RIM-ONE", "LAG"]
    colors = {"Drishti": "#E64B35", "ORIGA": "#4DBBD5", "RIM-ONE": "#3C5488", "LAG": "#00A087"}

    labels = np.array([infer_dataset_name(p) for p in filepaths[:n]])
    unknown_ratio = float(np.mean(labels == "UNKNOWN")) if labels.size else 1.0

    if unknown_ratio <= 0.20:
        segments = {}
        cursor = 0
        x = np.zeros(n, dtype=int)
        y_sorted = np.zeros(n, dtype=float)
        for name in order:
            idx = np.where(labels == name)[0]
            idx = idx[np.argsort(idx)]
            x_block = np.arange(cursor, cursor + idx.size)
            x[idx] = x_block
            y_sorted[x_block] = uncertainty[idx]
            segments[name] = (cursor, cursor + idx.size)
            cursor += idx.size
    else:
        if approximate_counts is None:
            approximate_counts = {"Drishti": 330, "ORIGA": 220, "RIM-ONE": 130}
        counts = dict(approximate_counts)
        counts["LAG"] = max(0, n - sum(counts.values()))
        segments = {}
        cursor = 0
        for name in order:
            right = min(n, cursor + counts.get(name, 0))
            segments[name] = (cursor, right)
            cursor = right
        y_sorted = uncertainty.copy()

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    for name in order:
        l, r = segments.get(name, (0, 0))
        if r <= l:
            continue
        idx = np.arange(l, r)
        ax.scatter(idx, y_sorted[idx], s=12, alpha=0.80, color=colors[name], label=name, edgecolor="none")
        ax.hlines(float(np.nanmedian(y_sorted[idx])), l, r - 1, colors=colors[name], lw=2, alpha=0.9)

    for name in order[:-1]:
        if name in segments:
            _, right = segments[name]
            ax.axvline(right - 0.5, ls="--", color="0.7", lw=1)

    y_top = float(np.nanmax(y_sorted)) * 1.06 if np.nanmax(y_sorted) > 0 else 0.05
    for name, (l, r) in segments.items():
        if r > l:
            mid = (l + r - 1) / 2
            ax.text(mid, y_top, name, ha="center", va="bottom", color=colors.get(name, "black"), fontsize=11, fontweight="bold")

    ax.set_xlabel("Image index (dataset blocks)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
