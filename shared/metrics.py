"""
metrics.py — Comprehensive metrics computation and statistical testing.

Required metrics per evaluation:
  - Accuracy, Balanced Accuracy, Macro-F1, Weighted-F1
  - Per-class Precision / Recall / F1 (Awake / Light / Deep)
  - Confusion Matrix (3x3)
  - Cohen's Kappa

Statistical tests:
  - Paired t-test (parametric)
  - Wilcoxon signed-rank test (non-parametric, preferred)
  - Bootstrap 95% CI (distribution-free)
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    f1_score, precision_score, recall_score,
    confusion_matrix, cohen_kappa_score,
    classification_report
)


CLASS_NAMES = ['Awake', 'Light Hypnosis', 'Deep Hypnosis']
CLASS_NAMES_CN = ['清醒', '浅催眠', '深催眠']


def compute_all_metrics(y_true, y_pred, y_proba=None):
    """
    Compute the full set of evaluation metrics.

    Args:
        y_true: ndarray (n_samples,) — true labels (0, 1, 2)
        y_pred: ndarray (n_samples,) — predicted labels
        y_proba: ndarray (n_samples, n_classes), optional — predicted probabilities

    Returns:
        dict: all metrics
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics = {}

    # Core metrics
    metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
    metrics['balanced_accuracy'] = float(balanced_accuracy_score(y_true, y_pred))
    metrics['macro_f1'] = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    metrics['weighted_f1'] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))

    # Cohen's Kappa
    metrics['cohens_kappa'] = float(cohen_kappa_score(y_true, y_pred))

    # Confusion Matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()

    # Per-class metrics
    for i, class_name in enumerate(CLASS_NAMES):
        metrics[f'{class_name}_precision'] = float(precision_score(
            y_true, y_pred, labels=[i], average='macro', zero_division=0))
        metrics[f'{class_name}_recall'] = float(recall_score(
            y_true, y_pred, labels=[i], average='macro', zero_division=0))
        metrics[f'{class_name}_f1'] = float(f1_score(
            y_true, y_pred, labels=[i], average='macro', zero_division=0))

    # Probabilistic metrics (if available)
    if y_proba is not None:
        metrics['y_proba'] = y_proba

    return metrics


def print_metrics(metrics, title="Results"):
    """Pretty-print evaluation metrics."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

    print(f"\n  Accuracy:          {metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"  Macro-F1:          {metrics['macro_f1']:.4f}")
    print(f"  Weighted-F1:       {metrics['weighted_f1']:.4f}")
    print(f"  Cohen's Kappa:     {metrics['cohens_kappa']:.4f}")

    print(f"\n  Per-class metrics:")
    print(f"  {'Class':<20} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'-'*50}")
    for class_name in CLASS_NAMES:
        p = metrics.get(f'{class_name}_precision', 0)
        r = metrics.get(f'{class_name}_recall', 0)
        f = metrics.get(f'{class_name}_f1', 0)
        print(f"  {class_name:<20} {p:>10.4f} {r:>10.4f} {f:>10.4f}")

    print(f"\n  Confusion Matrix:")
    cm = np.array(metrics['confusion_matrix'])
    print(f"  {'':>12}", end='')
    for name in CLASS_NAMES:
        print(f" {name:>12}", end='')
    print()
    for i, row in enumerate(cm):
        print(f"  {CLASS_NAMES[i]:>12}", end='')
        for val in row:
            print(f" {val:>12}", end='')
        print()
    print(f"{'='*60}\n")


def aggregate_seeds(results_list):
    """
    Aggregate metrics across multiple seeds.

    Args:
        results_list: list of dict, each from compute_all_metrics

    Returns:
        dict: mean and std for each metric
    """
    if not results_list:
        return {}

    metric_keys = [
        'accuracy', 'balanced_accuracy', 'macro_f1', 'weighted_f1', 'cohens_kappa'
    ]
    per_class_keys = [f'{cn}_{m}' for cn in CLASS_NAMES for m in ['precision', 'recall', 'f1']]
    all_keys = metric_keys + per_class_keys

    aggregated = {}
    for key in all_keys:
        values = [r[key] for r in results_list if key in r]
        if values:
            aggregated[f'{key}_mean'] = float(np.mean(values))
            aggregated[f'{key}_std'] = float(np.std(values, ddof=1))

    # Aggregate confusion matrices
    if 'confusion_matrix' in results_list[0]:
        cms = [np.array(r['confusion_matrix']) for r in results_list]
        aggregated['confusion_matrix_mean'] = np.mean(cms, axis=0).tolist()

    return aggregated


def paired_ttest(method_a_results, method_b_results, metric='balanced_accuracy'):
    """
    Perform paired t-test between two methods across seeds.

    Args:
        method_a_results: list of dict, metrics from method A per seed
        method_b_results: list of dict, metrics from method B per seed
        metric: str, metric name to compare

    Returns:
        dict: t_statistic, p_value, mean_diff, significant, n_seeds
    """
    from scipy import stats

    a_values = [r[metric] for r in method_a_results if metric in r]
    b_values = [r[metric] for r in method_b_results if metric in r]

    min_len = min(len(a_values), len(b_values))
    a_values = a_values[:min_len]
    b_values = b_values[:min_len]

    if min_len < 2:
        return {
            't_statistic': 0, 'p_value': 1.0, 'mean_diff': 0,
            'significant': False, 'n_seeds': min_len,
            'note': 'Insufficient samples for t-test'
        }

    t_stat, p_value = stats.ttest_rel(a_values, b_values)
    mean_diff = np.mean(a_values) - np.mean(b_values)

    # Effect size (Cohen's d)
    diff = np.array(a_values) - np.array(b_values)
    cohens_d = np.mean(diff) / (np.std(diff, ddof=1) + 1e-10)

    return {
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'mean_diff': float(mean_diff),
        'cohens_d': float(cohens_d),
        'significant': bool(p_value < 0.05),
        'n_seeds': min_len,
    }


def wilcoxon_test(method_a_results, method_b_results, metric='balanced_accuracy'):
    """
    Perform Wilcoxon signed-rank test (non-parametric).

    Args:
        method_a_results: list of dict, metrics from method A per seed
        method_b_results: list of dict, metrics from method B per seed
        metric: str, metric name to compare

    Returns:
        dict: W_statistic, p_value, significant, n_seeds
    """
    from scipy import stats

    a_values = [r[metric] for r in method_a_results if metric in r]
    b_values = [r[metric] for r in method_b_results if metric in r]

    min_len = min(len(a_values), len(b_values))
    a_values = a_values[:min_len]
    b_values = b_values[:min_len]

    if min_len < 5:
        return {
            'W_statistic': 0, 'p_value': 1.0, 'significant': False,
            'n_seeds': min_len,
            'note': 'Insufficient samples for Wilcoxon (need >= 5)'
        }

    try:
        W_stat, p_value = stats.wilcoxon(a_values, b_values, alternative='two-sided')
    except ValueError:
        # All differences are zero
        W_stat, p_value = 0, 1.0

    # Effect size (r = Z / sqrt(N))
    diff = np.array(a_values) - np.array(b_values)
    n = len(diff)
    mean_rank = np.mean(np.abs(np.sort(diff)))
    r_effect = (W_stat - n * (n + 1) / 4) / (n * (n + 1) / 4 + 1e-10)

    return {
        'W_statistic': float(W_stat),
        'p_value': float(p_value),
        'effect_size_r': float(abs(r_effect)),
        'significant': bool(p_value < 0.05),
        'n_seeds': min_len,
    }


def bootstrap_ci(values, n_bootstrap=10000, alpha=0.05, metric_name=''):
    """
    Compute bootstrap 95% confidence interval.

    Args:
        values: array-like, metric values across seeds
        n_bootstrap: int, number of bootstrap iterations
        alpha: float, significance level (default 0.05 for 95% CI)
        metric_name: str, optional label

    Returns:
        dict: mean, std, ci_lower, ci_upper, ci_width
    """
    values = np.asarray(values)
    rng = np.random.RandomState(42)

    boot_means = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        boot_means[i] = np.mean(sample)

    ci_lower = float(np.percentile(boot_means, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))

    return {
        'metric': metric_name,
        'mean': float(np.mean(values)),
        'std': float(np.std(values, ddof=1)),
        'n': len(values),
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'ci_width': float(ci_upper - ci_lower),
    }
