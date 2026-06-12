"""Evaluation utilities for retained-case analyses and sensitivity studies."""
from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score


def classification_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Compute binary classification metrics."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = roc_auc_score(y_true, y_score)
        metrics["average_precision"] = average_precision_score(y_true, y_score)
    else:
        metrics["roc_auc"] = np.nan
        metrics["average_precision"] = np.nan
    return metrics


def threshold_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    uncertainty: np.ndarray,
    thresholds: Iterable[float],
    decision_threshold: float = 0.5,
) -> pd.DataFrame:
    """Accuracy after retaining cases with uncertainty <= threshold.

    The returned `excluded_count` is the number of images referred/flagged as uncertain.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    rows = []
    n = y_true.size
    for thr in thresholds:
        keep = np.isfinite(uncertainty) & (uncertainty <= thr)
        used = int(keep.sum())
        excluded = int(n - used)
        if used > 0:
            y_pred = (y_score[keep] >= decision_threshold).astype(int)
            acc = float(np.mean(y_pred == y_true[keep]))
            auc = float(roc_auc_score(y_true[keep], y_score[keep])) if len(np.unique(y_true[keep])) == 2 else np.nan
        else:
            acc = np.nan
            auc = np.nan
        rows.append({
            "threshold": float(thr),
            "retained_count": used,
            "excluded_count": excluded,
            "retained_fraction": used / n,
            "accuracy_retained": acc,
            "auc_retained": auc,
        })
    return pd.DataFrame(rows)


def quantile_thresholds(values: np.ndarray, n: int = 10, q_low: float = 0.10, q_high: float = 1.00) -> np.ndarray:
    """Return n quantile-based thresholds from an uncertainty vector."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("Cannot build thresholds from an empty array.")
    qs = np.linspace(q_low, q_high, n)
    thresholds = np.quantile(values, qs)
    thresholds[-1] = np.nextafter(float(np.nanmax(values)), np.inf)
    return thresholds


def top_uncertain_overlap(u_reference: np.ndarray, u_candidate: np.ndarray, top_fraction: float = 0.10) -> float:
    """Overlap between top uncertain cases under two uncertainty vectors."""
    n = len(u_reference)
    k = max(1, int(round(top_fraction * n)))
    top_ref = set(np.argsort(u_reference)[-k:])
    top_cand = set(np.argsort(u_candidate)[-k:])
    return len(top_ref & top_cand) / k


def rankdata_simple(x: np.ndarray) -> np.ndarray:
    """Simple rankdata implementation without SciPy ties handling."""
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, x.size + 1)
    return ranks


def spearman_simple(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation without requiring scipy."""
    rx = rankdata_simple(np.asarray(x))
    ry = rankdata_simple(np.asarray(y))
    return float(np.corrcoef(rx, ry)[0, 1])
