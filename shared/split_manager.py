"""
split_manager.py — Centralized data split management.

Guarantees ALL methods/baselines use EXACTLY the same train/calib/test split.
Three split types:
  1. LODO (Leave-One-Domain-Out): rotate which dataset is target (8-fold)
  2. LOSO (Leave-One-Subject-Out): within-target subject-level split
  3. LOO (Leave-One-Out): fallback when no subject info (by trial/session)

CRITICAL: No experiment script may perform its own random split.
          All splits must be loaded from files managed by this module.
"""

import json
import numpy as np
from pathlib import Path


ALL_DATASETS = [
    'DREAMER', 'DEAP', 'MAHNOB', 'SEED', 'SEED_IV',
    'FACED', 'ds004572', 'ds006437'
]


class SplitManager:
    """
    Centralized split management with LODO + inner LOSO/LOO.

    Usage:
        sm = SplitManager("splits")

        # LODO splits (deterministic, no randomness)
        lodo = sm.load_lodo_splits()

        # LOSO: subject-level split within a target domain
        sm.generate_loso_split("DREAMER", subject_ids, labels_per_subject, seed=42)

        # LOO: trial-level split when no subject info
        sm.generate_loo_split("DEAP", n_trials, labels, seed=42)
    """

    def __init__(self, splits_dir="splits"):
        self.splits_dir = Path(splits_dir)
        self.splits_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # LODO splits (deterministic — no randomness)
    # ------------------------------------------------------------------
    def generate_lodo_splits(self):
        """
        Generate LODO splits: each dataset takes turns as target domain.

        Returns:
            dict: {target_domain: {source_domains: [...], target_domain: str}}
        """
        splits = {}
        for target in ALL_DATASETS:
            source = [d for d in ALL_DATASETS if d != target]
            splits[target] = {
                'source_domains': source,
                'target_domain': target
            }

        path = self.splits_dir / "lodo_splits.json"
        with open(path, 'w') as f:
            json.dump(splits, f, indent=2)

        return splits

    def load_lodo_splits(self):
        """Load LODO splits from file, or generate if not exists."""
        path = self.splits_dir / "lodo_splits.json"
        if not path.exists():
            return self.generate_lodo_splits()
        with open(path) as f:
            return json.load(f)

    # ------------------------------------------------------------------
    # LOSO: Leave-One-Subject-Out within target domain
    # ------------------------------------------------------------------
    def generate_loso_split(self, dataset_name, subject_ids, labels_per_subject,
                            seed, calib_ratio=0.2):
        """
        Split subjects within a target domain into calibration/test sets.

        Args:
            dataset_name: str, name of target domain
            subject_ids: list of subject identifiers
            labels_per_subject: dict {subject_id: majority_label} for stratification
            seed: int, random seed
            calib_ratio: float, proportion of subjects for calibration (default 0.2)

        Returns:
            dict with keys: dataset, seed, split_type, calib_subjects, test_subjects,
                           calib_ratio, class_distribution_calib, class_distribution_test
        """
        subject_ids = np.array(subject_ids)
        labels = np.array([labels_per_subject[s] for s in subject_ids])

        np.random.seed(seed)

        unique_labels = np.unique(labels)
        calib_indices = []
        test_indices = []

        for label in unique_labels:
            label_indices = np.where(labels == label)[0]
            np.random.shuffle(label_indices)
            n_calib_label = max(1, int(len(label_indices) * calib_ratio))
            calib_indices.extend(label_indices[:n_calib_label].tolist())
            test_indices.extend(label_indices[n_calib_label:].tolist())

        calib_idx = np.array(calib_indices)
        test_idx = np.array(test_indices)

        calib_subjects = subject_ids[calib_idx].tolist()
        test_subjects = subject_ids[test_idx].tolist()

        split = {
            'dataset': dataset_name,
            'seed': seed,
            'split_type': 'LOSO',
            'calib_subjects': calib_subjects,
            'test_subjects': test_subjects,
            'calib_ratio': calib_ratio,
            'n_calib_subjects': len(calib_subjects),
            'n_test_subjects': len(test_subjects),
        }

        path = self.splits_dir / f"loso_{dataset_name}_seed{seed}.json"
        with open(path, 'w') as f:
            json.dump(split, f, indent=2)

        return split

    # ------------------------------------------------------------------
    # LOO: Leave-One-Out (by trial) — fallback when no subject info
    # ------------------------------------------------------------------
    def generate_loo_split(self, dataset_name, n_trials, labels, seed,
                           calib_ratio=0.2):
        """
        Split trials within a target domain into calibration/test sets.

        Args:
            dataset_name: str, name of target domain
            n_trials: int, total number of trials
            labels: array-like, trial-level labels for stratification
            seed: int, random seed
            calib_ratio: float, proportion of trials for calibration

        Returns:
            dict with keys: dataset, seed, split_type, calib_indices, test_indices
        """
        labels = np.asarray(labels)
        n = len(labels)
        np.random.seed(seed)

        unique_labels = np.unique(labels)
        calib_idx = []
        test_idx = []

        for label in unique_labels:
            label_indices = np.where(labels == label)[0]
            np.random.shuffle(label_indices)
            n_calib_label = max(1, int(len(label_indices) * calib_ratio))
            calib_idx.extend(label_indices[:n_calib_label].tolist())
            test_idx.extend(label_indices[n_calib_label:].tolist())

        split = {
            'dataset': dataset_name,
            'seed': seed,
            'split_type': 'LOO',
            'calib_indices': calib_idx,
            'test_indices': test_idx,
            'calib_ratio': calib_ratio,
            'n_calib': len(calib_idx),
            'n_test': len(test_idx),
        }

        path = self.splits_dir / f"loo_{dataset_name}_seed{seed}.json"
        with open(path, 'w') as f:
            json.dump(split, f, indent=2)

        return split

    # ------------------------------------------------------------------
    # Generate splits for all calibration ratios (for exp102)
    # ------------------------------------------------------------------
    def generate_calib_ratio_splits(self, dataset_name, subject_ids,
                                    labels_per_subject, seed, calib_ratios):
        """
        Generate splits for multiple calibration ratios.

        Args:
            dataset_name: str
            subject_ids: list of subject identifiers
            labels_per_subject: dict
            seed: int
            calib_ratios: list of float, e.g. [0, 0.01, 0.02, 0.05, 0.10, 0.20]

        Returns:
            dict {calib_ratio: split_dict}
        """
        splits = {}
        for cr in calib_ratios:
            if cr == 0:
                # Zero-shot: empty calibration set, all subjects are test
                splits[cr] = {
                    'dataset': dataset_name,
                    'seed': seed,
                    'split_type': 'zero_shot',
                    'calib_subjects': [],
                    'test_subjects': list(subject_ids),
                    'calib_ratio': 0.0,
                    'n_calib_subjects': 0,
                    'n_test_subjects': len(subject_ids),
                }
            else:
                splits[cr] = self.generate_loso_split(
                    dataset_name, subject_ids, labels_per_subject, seed, cr
                )
        return splits

    # ------------------------------------------------------------------
    # Load existing splits
    # ------------------------------------------------------------------
    def load_subject_split(self, dataset_name, seed):
        """Load a previously generated LOSO/LOO split."""
        for prefix in ['loso', 'loo']:
            path = self.splits_dir / f"{prefix}_{dataset_name}_seed{seed}.json"
            if path.exists():
                with open(path) as f:
                    return json.load(f)

        raise FileNotFoundError(
            f"No split found for {dataset_name} seed {seed}. "
            f"Run generate_loso_split() or generate_loo_split() first."
        )
