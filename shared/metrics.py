"""Shared evaluation metrics for cross-dataset hypnosis-depth experiments.

Provides compute_all_metrics (per-seed scalar dict) and aggregate_seeds
(mean/std across seeds), used by run_exp104_eegnet_reproducible.py and
run_exp104_v2_focal.py.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    recall_score,
)


def compute_all_metrics(y_true, y_pred, y_proba=None):
    """Compute per-seed classification metrics.

    Args:
        y_true: 1-D array of true labels.
        y_pred: 1-D array of predicted labels.
        y_proba: optional [n, n_classes] probability matrix (unused for the
            scalar metrics but accepted for API symmetry).

    Returns:
        dict with keys: accuracy, balanced_accuracy, cohen_kappa, f1_macro,
        recall_per_class (list), confusion_matrix (list of lists).
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_per_class": [
            float(x)
            for x in recall_score(y_true, y_pred, average=None, zero_division=0)
        ],
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    return metrics


def aggregate_seeds(metrics_list):
    """Aggregate a list of per-seed metric dicts into mean/std summaries.

    Numeric scalar keys (accuracy, balanced_accuracy, cohen_kappa, f1_macro)
    become ``{key}_mean`` / ``{key}_std``. The ``recall_per_class`` list is
    aggregated element-wise as ``recall_per_class_{i}_mean`` / ``_std``.
    Non-numeric keys (confusion_matrix) are dropped.

    Returns:
        dict, e.g. {"balanced_accuracy_mean": ..., "balanced_accuracy_std": ...}.
    """
    agg: dict = {}
    if not metrics_list:
        return agg
    scalar_keys = [
        k
        for k, v in metrics_list[0].items()
        if k not in ("recall_per_class", "confusion_matrix")
        and isinstance(v, (int, float))
    ]
    for k in scalar_keys:
        vals = np.array([float(m[k]) for m in metrics_list], dtype=float)
        agg[f"{k}_mean"] = float(vals.mean())
        agg[f"{k}_std"] = float(vals.std(ddof=0))
    if "recall_per_class" in metrics_list[0]:
        n_c = len(metrics_list[0]["recall_per_class"])
        for ci in range(n_c):
            vals = np.array(
                [float(m["recall_per_class"][ci]) for m in metrics_list], dtype=float
            )
            agg[f"recall_per_class_{ci}_mean"] = float(vals.mean())
            agg[f"recall_per_class_{ci}_std"] = float(vals.std(ddof=0))
    return agg
