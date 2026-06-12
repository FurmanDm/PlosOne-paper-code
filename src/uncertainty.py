"""Monte-Carlo Dropout and uncertainty metrics."""
from __future__ import annotations

import gc
from typing import Dict

import numpy as np
import tensorflow as tf


def set_global_seed(seed: int = 1337) -> None:
    """Set NumPy and TensorFlow seeds for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)


def mc_dropout_probabilities(model, X: np.ndarray, T: int = 50, batch_size: int = 32) -> np.ndarray:
    """Run Monte-Carlo Dropout and return probabilities.

    Dropout is activated at inference by calling `model(..., training=True)`.

    Returns
    -------
    np.ndarray
        Array of shape (T, N), where each row is one stochastic forward pass.
    """
    n = X.shape[0]
    P = np.empty((T, n), dtype=np.float32)
    for t in range(T):
        out = []
        for i in range(0, n, batch_size):
            probs = model(X[i:i + batch_size], training=True).numpy().reshape(-1)
            out.append(probs)
        P[t] = np.clip(np.concatenate(out, axis=0), 1e-12, 1.0 - 1e-12)
        gc.collect()
    return P


def deterministic_probabilities(model, X: np.ndarray, batch_size: int = 32) -> np.ndarray:
    """Return deterministic probabilities with dropout disabled."""
    probs = model.predict(X, batch_size=batch_size, verbose=0).reshape(-1)
    return np.clip(probs, 1e-12, 1.0 - 1e-12)


def binary_entropy_bits(p, eps: float = 1e-12) -> np.ndarray:
    """Binary Shannon entropy in bits."""
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))


def probabilities_to_logits(P: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Convert probabilities to logits with numerical clipping."""
    P = np.clip(np.asarray(P, dtype=np.float64), eps, 1.0 - eps)
    return np.log(P) - np.log1p(-P)


def sample_skewness_fisher(x: np.ndarray) -> float:
    """Bias-corrected Fisher-Pearson sample skewness for one vector."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n < 3:
        return 0.0
    mu = x.mean()
    s = x.std(ddof=1)
    if not np.isfinite(s) or s <= 1e-12:
        return 0.0
    g1 = np.mean(((x - mu) / s) ** 3)
    return float(np.sqrt(n * (n - 1)) / (n - 2) * g1)


def compute_uncertainty_metrics(P: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute uncertainty metrics from MC probabilities P with shape (T, N).

    Entropy metrics follow:
      AU_E = E_t[H(p_t)]
      TU_E = H(E_t[p_t])
      EU_E = TU_E - AU_E

    Variance metrics follow:
      EU_V = Var_t[p_t]
      TU_V = p_bar * (1 - p_bar)
      AU_V = TU_V - EU_V

    IQR and logit SD are also returned.
    """
    P = np.clip(np.asarray(P, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    pbar = P.mean(axis=0)

    TU_E = binary_entropy_bits(pbar)
    AU_E = binary_entropy_bits(P).mean(axis=0)
    EU_E = np.maximum(TU_E - AU_E, 0.0)

    EU_V = P.var(axis=0, ddof=0)
    TU_V = pbar * (1.0 - pbar)
    AU_V = np.maximum(TU_V - EU_V, 0.0)

    IQR = np.percentile(P, 75, axis=0) - np.percentile(P, 25, axis=0)

    logits = probabilities_to_logits(P)
    logit_iqr = np.percentile(logits, 75, axis=0) - np.percentile(logits, 25, axis=0)
    logit_sd = logits.std(axis=0, ddof=1)
    skewness = np.apply_along_axis(sample_skewness_fisher, 0, P)

    return {
        "p_mc_mean": pbar,
        "AU_E_bits": AU_E,
        "TU_E_bits": TU_E,
        "EU_E_bits": EU_E,
        "AU_V": AU_V,
        "TU_V": TU_V,
        "EU_V": EU_V,
        "IQR_prob": IQR,
        "IQR_logit": logit_iqr,
        "SD_logit": logit_sd,
        "skewness": skewness,
    }
