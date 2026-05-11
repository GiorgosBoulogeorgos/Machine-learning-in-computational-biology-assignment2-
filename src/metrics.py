"""
Metric computation utilities for the rnCV pipeline.

This module provides:
  - `compute_metrics`  : computes the full battery of clinical-context metrics
                         from a single (y_true, y_pred, y_proba) triple.
  - `bootstrap_median_ci` : non-parametric 95% CI for the median of a sample.
"""

from __future__ import annotations
from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import (
    matthews_corrcoef, roc_auc_score, balanced_accuracy_score,
    f1_score, recall_score, precision_score, confusion_matrix,
    average_precision_score,
)


# Ordered tuple of metric names — used as DataFrame column order.
METRIC_NAMES: Tuple[str, ...] = (
    'mcc', 'auc', 'pr_auc', 'balanced_acc', 'f1',
    'recall', 'specificity', 'precision',
)


def compute_metrics(
    y_true:  np.ndarray,
    y_pred:  np.ndarray,
    y_proba: np.ndarray | None = None,
) -> Dict[str, float]:
    """Compute the full metric battery for one (y_true, y_pred, y_proba) triple.

    Parameters
    ----------
    y_true : 1-D array of {0, 1}
    y_pred : 1-D array of {0, 1}    -- hard predictions
    y_proba : 1-D array of [0,1]    -- probability of the positive class.
        Required for AUC and PR-AUC; if None those metrics return NaN.

    Returns
    -------
    Dict[str, float]
        Keys: 'mcc', 'auc', 'pr_auc', 'balanced_acc', 'f1',
              'recall', 'specificity', 'precision'.

    Notes
    -----
    Specificity (TN / (TN + FP)) is computed manually since sklearn lacks a
    direct function. `zero_division=0` is used in precision/recall/F1 to
    avoid warnings when a fold contains no positive predictions.
    """
    # Confusion matrix → specificity
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    metrics = {
        'mcc':          matthews_corrcoef(y_true, y_pred),
        'balanced_acc': balanced_accuracy_score(y_true, y_pred),
        'f1':           f1_score(y_true, y_pred, zero_division=0),
        'recall':       recall_score(y_true, y_pred, zero_division=0),
        'precision':    precision_score(y_true, y_pred, zero_division=0),
        'specificity':  specificity,
    }

    # Probability-based metrics (AUC, PR-AUC).
    if y_proba is not None:
        metrics['auc']    = roc_auc_score(y_true, y_proba)
        metrics['pr_auc'] = average_precision_score(y_true, y_proba)
    else:
        metrics['auc']    = np.nan
        metrics['pr_auc'] = np.nan
    return metrics


def bootstrap_median_ci(
    values:    np.ndarray,
    n_resamples: int = 10_000,
    confidence:  float = 0.95,
    seed:        int   = 42,
) -> Tuple[float, float, float]:
    """Non-parametric bootstrap CI for the median of a 1-D sample.

    The percentile method is used: resample with replacement `n_resamples`
    times, compute the median of each resample, and take the empirical
    [α/2, 1-α/2] quantiles of that distribution.

    Parameters
    ----------
    values : 1-D array of metric scores (e.g. 50 outer-fold MCC values).
    n_resamples : int
        Number of bootstrap resamples. 10 000 is standard for stable CI
        estimates; reduce only if runtime becomes a concern.
    confidence : float
        Confidence level (default 0.95 → 95% CI).
    seed : int
        Seed for the resampling RNG (ensures reproducibility).

    Returns
    -------
    (median, ci_low, ci_high)
    """
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)
    # Vectorised resampling: draw an (n_resamples × n) index matrix at once.
    n = len(values)
    idx = rng.integers(0, n, size=(n_resamples, n))
    medians = np.median(values[idx], axis=1)

    alpha = 1.0 - confidence
    lo, hi = np.quantile(medians, [alpha / 2, 1 - alpha / 2])
    return float(np.median(values)), float(lo), float(hi)
