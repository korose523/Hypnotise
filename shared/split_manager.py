"""
split_manager.py — Centralized data split management.

Guarantees ALL methods/baselines use EXACTLY the same train/calib/test split.
Two split types:
  1. LODO (Leave-One-Domain-Out): rotate which dataset is target
  2. Within-domain subject split: for target domain calibration/test

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
    Centralized split management.

    Usage:
        sm = SplitManager("splits")

        # Generate and load LODO splits
        lodo = sm.load_lodo_splits()

        # Generate within-domain subject splits
        sm.generate_subject_splits("DREAMER", subject_ids, labels_per_subject, seed=42)

        # Load existing subject splits
        split = sm.load_subject_split("DREAMER", seed=42)
    """

    def __init__(self, splits_dir="splits"):
        """
        Args:
            splits_dir: str or Path, directory to store split JSON files
        """
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

    # ------------------------------------------------------------------
    # Within-domain subject-level splits (for calibration/test)
    # ------------------------------------------------------------------
    def generate_subject_splits(self, dataset_name, subject_ids, labels_per_subject,
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
            dict with keys: dataset, seed, calib_subjects, test_subjects, calib_ratio
        """
        subject_ids = np.array(subject_ids)
        labels = np.array([labels_per_subject[s] for s in subject_ids])

        n_calib = max(3, int(len(subject_ids) * calib_ratio))

        # Stratified split at subject level
        np.random.seed(seed)

        # Group indices by label for stratification
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

        split = {
            'dataset': dataset_name,
            'seed': seed,
            'calib_subjects': subject_ids[calib_idx].tolist(),
            'test_subjects': subject_ids[test_idx].tolist(),
            'calib_ratio': calib_ratio
        }

        path = self.splits_dir / f"subject_split_{dataset_name}_seed{seed}.json"
        with open(path, 'w') as f:
            json.dump(split, f, indent=2)

        return split

    # ------------------------------------------------------------------
    # Load existing splits
    # ------------------------------------------------------------------
    def load_lodo_splits(self):
        """
        Load LODO splits from file, or generate if not exists.

        Returns:
            dict: LODO split configuration
        """
        path = self.splits_dir / "lodo_splits.json"
        if not path.exists():
            return self.generate_lodo_splits()
        with open(path) as f:
            return json.load(f)

    def load_subject_split(self, dataset_name, seed):
        """
        Load a previously generated subject split.

        Args:
            dataset_name: str, target domain name
            seed: int, random seed used to generate the split

        Returns:
            dict: split configuration

        Raises:
            FileNotFoundError: if split has not been generated yet
        """
        path = self.splits_dir / f"subject_split_{dataset_name}_seed{seed}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"Subject split not found: {path}. "
                f"Run generate_subject_splits() first."
            )
        with open(path) as f:
            return json.load(f)
