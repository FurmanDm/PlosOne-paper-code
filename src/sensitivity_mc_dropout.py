"""Sensitivity analysis for the number of MC-Dropout samples."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from evaluation import spearman_simple, top_uncertain_overlap
from uncertainty import compute_uncertainty_metrics, mc_dropout_probabilities


def run_mc_sensitivity(
    model,
    X: np.ndarray,
    output_dir: str | Path,
    mc_values: Iterable[int] = (20, 50, 100, 200),
    reference_T: int = 200,
    batch_size: int = 32,
    seed: int = 1337,
) -> pd.DataFrame:
    """Compute uncertainty stability for T = 20/50/100/200.

    The reference is T=200 by default. For each metric and each T, the function
    calculates Spearman rank correlation, mean absolute difference and top-10% most
    uncertain overlap relative to the reference.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    mc_values = list(mc_values)
    if reference_T not in mc_values:
        mc_values.append(reference_T)
        mc_values = sorted(mc_values)

    metrics_by_T = {}
    for T in mc_values:
        P = mc_dropout_probabilities(model, X, T=T, batch_size=batch_size)
        np.savez(output / f"mc_probs_seed{seed}_T{T}.npz", P=P)
        metrics_by_T[T] = compute_uncertainty_metrics(P)

    metric_names = [
        "TU_E_bits",
        "AU_E_bits",
        "EU_E_bits",
        "EU_V",
        "IQR_prob",
        "IQR_logit",
        "SD_logit",
    ]

    ref = metrics_by_T[reference_T]
    rows = []
    for T in mc_values:
        for metric in metric_names:
            cand = metrics_by_T[T][metric]
            ref_values = ref[metric]
            rows.append({
                "T": T,
                "metric": metric,
                "spearman_vs_reference": spearman_simple(cand, ref_values),
                "mean_absolute_difference": float(np.mean(np.abs(cand - ref_values))),
                "top10_overlap": top_uncertain_overlap(ref_values, cand, top_fraction=0.10),
                "reference_T": reference_T,
            })
    df = pd.DataFrame(rows)
    df.to_csv(output / "mc_sensitivity_summary.csv", index=False)
    return df
