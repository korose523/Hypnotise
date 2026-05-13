#!/usr/bin/env python3
"""
exp107_stats_tests_bootstrap_wilcoxon.py — Statistical tests and bootstrap CI.

Experiment 107 (Paper 1, Statistical Rigor):
  - Loads results from exp101-exp106
  - Computes bootstrap 95% CI for all metrics
  - Performs Wilcoxon signed-rank tests between methods
  - Generates publication-ready tables (LaTeX and CSV format)
  - Effect size reporting (Cohen's d for t-test, r for Wilcoxon)

Output: results/exp107_stats/exp107_bootstrap_ci.json
        results/exp107_stats/exp107_wilcoxon_tests.json
        results/exp107_stats/exp107_table.csv
        results/exp107_stats/exp107_table.tex
"""

import sys
import json
import csv
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.config_loader import load_config
from shared.metrics import (
    aggregate_seeds, bootstrap_ci, wilcoxon_test, paired_ttest,
    CLASS_NAMES
)
from shared.logger import setup_logger


def load_experiment_results(exp_name, results_dir):
    """Load results from a previous experiment."""
    results_path = results_dir / exp_name / f'{exp_name}_results.json'
    if not results_path.exists():
        return None

    with open(results_path) as f:
        data = json.load(f)
    return data


def extract_seed_metrics(results_data, metric='balanced_accuracy'):
    """Extract metric values per seed from experiment results."""
    values = []
    if 'per_seed' in results_data:
        for target, seed_results in results_data['per_seed'].items():
            if isinstance(seed_results, dict):
                for seed, seed_data in seed_results.items():
                    if isinstance(seed_data, dict):
                        for method, metrics in seed_data.items():
                            if isinstance(metrics, dict) and metric in metrics:
                                values.append({
                                    'target': target,
                                    'seed': seed,
                                    'method': method,
                                    'value': metrics[metric],
                                })
    return values


def extract_method_seed_metrics(results_data, method_name, metric='balanced_accuracy'):
    """Extract metric values for a specific method across seeds."""
    values = []
    if 'per_seed' in results_data:
        for target, seed_results in results_data['per_seed'].items():
            if isinstance(seed_results, dict):
                for seed, seed_data in seed_results.items():
                    if isinstance(seed_data, dict) and method_name in seed_data:
                        m = seed_data[method_name]
                        if isinstance(m, dict) and metric in m:
                            values.append(m[metric])
    return values


def main():
    config = load_config(str(PROJECT_ROOT / 'config.yaml'))
    logger = setup_logger('exp107', str(PROJECT_ROOT / config['logs_dir']))

    results_dir = Path(PROJECT_ROOT / config['output_dir'])
    out_dir = Path(PROJECT_ROOT / config['output_dir'] / 'exp107_stats')
    out_dir.mkdir(parents=True, exist_ok=True)

    n_bootstrap = config['experiment']['n_bootstrap']
    alpha = config['experiment']['significance_level']
    metrics_to_test = ['balanced_accuracy', 'macro_f1', 'accuracy', 'cohens_kappa']

    # Load results from key experiments
    experiments = {
        'exp101_lodo_loso': 'exp101_lodo_loso',
        'exp102_calib_sweep': 'exp102_calib_sweep',
        'exp103_mahal_vs_fixed': 'exp103_mahal_vs_fixed',
        'exp104_eegnet': 'exp104_eegnet',
        'exp105_da_baselines': 'exp105_da_baselines',
    }

    loaded = {}
    for exp_key, exp_dir in experiments.items():
        results_path = results_dir / exp_dir / f'{exp_key}_results.json'
        if results_path.exists():
            with open(results_path) as f:
                loaded[exp_key] = json.load(f)
            logger.info(f"Loaded {exp_key}")
        else:
            logger.warning(f"Results not found: {results_path}")

    # ==================================================================
    # Bootstrap 95% CI
    # ==================================================================
    bootstrap_results = {}
    logger.info("\n" + "=" * 60)
    logger.info("Computing Bootstrap 95% Confidence Intervals...")
    logger.info("=" * 60)

    for exp_key, data in loaded.items():
        bootstrap_results[exp_key] = {}
        if 'per_seed' not in data:
            continue

        for target, seed_results in data['per_seed'].items():
            if not isinstance(seed_results, dict):
                continue

            for method in ['wfsc_mahalanobis', 'wfsc_fixed', 'zero_shot',
                           'CORAL', 'TCA', 'AdaBN']:
                values = []
                for seed, seed_data in seed_results.items():
                    if isinstance(seed_data, dict) and method in seed_data:
                        m = seed_data[method]
                        if isinstance(m, dict) and 'balanced_accuracy' in m:
                            values.append(m['balanced_accuracy'])

                if len(values) >= 3:
                    for metric in metrics_to_test:
                        vals = []
                        for seed, seed_data in seed_results.items():
                            if isinstance(seed_data, dict) and method in seed_data:
                                m = seed_data[method]
                                if isinstance(m, dict) and metric in m:
                                    vals.append(m[metric])

                        if len(vals) >= 3:
                            ci = bootstrap_ci(vals, n_bootstrap=n_bootstrap, alpha=alpha,
                                             metric_name=f'{target}_{method}_{metric}')
                            key = f'{target}_{method}_{metric}'
                            bootstrap_results[exp_key][key] = ci

    # ==================================================================
    # Wilcoxon signed-rank tests (pairwise method comparisons)
    # ==================================================================
    wilcoxon_results = []
    logger.info("\n" + "=" * 60)
    logger.info("Computing Wilcoxon Signed-Rank Tests...")
    logger.info("=" * 60)

    # Compare methods within each experiment
    method_pairs = [
        ('wfsc_mahalanobis', 'wfsc_fixed'),
        ('wfsc_mahalanobis', 'zero_shot'),
        ('wfsc_fixed', 'zero_shot'),
    ]

    for exp_key, data in loaded.items():
        if 'per_seed' not in data:
            continue

        for target, seed_results in data['per_seed'].items():
            if not isinstance(seed_results, dict):
                continue

            for method_a, method_b in method_pairs:
                a_values = []
                b_values = []
                for seed, seed_data in seed_results.items():
                    if isinstance(seed_data, dict):
                        m_a = seed_data.get(method_a)
                        m_b = seed_data.get(method_b)
                        if (isinstance(m_a, dict) and 'balanced_accuracy' in m_a and
                                isinstance(m_b, dict) and 'balanced_accuracy' in m_b):
                            a_values.append(m_a)
                            b_values.append(m_b)

                if len(a_values) >= 5:
                    test = wilcoxon_test(a_values, b_values)
                    test['experiment'] = exp_key
                    test['target'] = target
                    test['method_a'] = method_a
                    test['method_b'] = method_b
                    wilcoxon_results.append(test)

                    sig = "***" if test['p_value'] < 0.001 else \
                          "**" if test['p_value'] < 0.01 else \
                          "*" if test['p_value'] < 0.05 else "n.s."
                    logger.info(
                        f"  {exp_key} | {target}: {method_a} vs {method_b} | "
                        f"p={test['p_value']:.4f} ({sig}) | "
                        f"r={test['effect_size_r']:.3f}"
                    )

    # ==================================================================
    # Generate publication table (CSV + LaTeX)
    # ==================================================================
    logger.info("\nGenerating publication tables...")

    # Collect summary stats
    table_rows = []
    for exp_key, data in loaded.items():
        if 'summary' not in data:
            continue

        summary = data['summary']
        for key, agg in summary.items():
            if isinstance(agg, dict):
                table_rows.append({
                    'experiment': exp_key,
                    'key': key,
                    'BAcc_mean': agg.get('balanced_accuracy_mean', 0),
                    'BAcc_std': agg.get('balanced_accuracy_std', 0),
                    'MF1_mean': agg.get('macro_f1_mean', 0),
                    'MF1_std': agg.get('macro_f1_std', 0),
                })

    # CSV
    csv_path = out_dir / 'exp107_table.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['experiment', 'key', 'BAcc_mean', 'BAcc_std',
                                                'MF1_mean', 'MF1_std'])
        writer.writeheader()
        for row in table_rows:
            writer.writerow({k: f"{v:.4f}" if isinstance(v, float) else v
                            for k, v in row.items()})
    logger.info(f"  CSV saved: {csv_path}")

    # LaTeX
    tex_path = out_dir / 'exp107_table.tex'
    with open(tex_path, 'w') as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Cross-dataset hypnosis classification results (mean $\\pm$ std)}\n")
        f.write("\\label{tab:main_results}\n")
        f.write("\\begin{tabular}{lcccc}\n")
        f.write("\\hline\n")
        f.write("Method & Target & BAcc & Macro-F1 & Kappa \\\\\n")
        f.write("\\hline\n")

        for row in table_rows:
            f.write(f"{row['key']} & {row['BAcc_mean']:.3f}$\\pm${row['BAcc_std']:.3f} & "
                    f"{row['MF1_mean']:.3f}$\\pm${row['MF1_std']:.3f} \\\\\n")

        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    logger.info(f"  LaTeX saved: {tex_path}")

    # Save all results
    with open(out_dir / 'exp107_bootstrap_ci.json', 'w') as f:
        json.dump(bootstrap_results, f, indent=2, default=str)

    with open(out_dir / 'exp107_wilcoxon_tests.json', 'w') as f:
        json.dump(wilcoxon_results, f, indent=2, default=str)

    # Summary of significant findings
    n_sig = sum(1 for r in wilcoxon_results if r['significant'])
    n_total = len(wilcoxon_results)
    logger.info(f"\n{'='*60}")
    logger.info(f"exp107 complete.")
    logger.info(f"  Bootstrap CI computed for {len(bootstrap_results)} metric combinations")
    logger.info(f"  Wilcoxon tests: {n_sig}/{n_total} significant at p<0.05")
    logger.info(f"  Tables saved to: {out_dir}")


if __name__ == '__main__':
    main()
