"""split_manager.py — load pre-generated LOSO / LODO split partitions.

Reconstructed minimal version. The ORIGINAL split_manager also contained the
logic that *generated* splits/loso_<DS>_seed<S>.json (via sklearn
GroupShuffleSplit); that generation step has already been run and its outputs
are committed under splits/. This module re-implements only the *read* side
used by run_exp101_reproducible.py, so the loaded partitions are byte-for-byte
the same JSON objects that were generated.

Interface (as consumed by run_exp101_reproducible.py):
    SplitManager(splits_dir)
    sm.load_subject_split(target, seed)  -> dict (calib_subjects, test_subjects, ...)
    sm.load_lodo_splits()                -> dict
    ALL_DATASETS                        -> list[str]
"""
import json
import os


ALL_DATASETS = [
    "DEAP",
    "DREAMER",
    "MAHNOB",
    "SEED",
    "SEED_IV",
    "FACED",
    "ds004572",
    "ds006437",
]


class SplitManager:
    def __init__(self, splits_dir):
        self.splits_dir = splits_dir

    def load_subject_split(self, target, seed):
        """Load the inner LOSO split for (target, seed)."""
        fname = f"loso_{target}_seed{seed}.json"
        path = os.path.join(self.splits_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_lodo_splits(self):
        """Load the multi-source LODO (leave-one-domain-out) splits, if present."""
        path = os.path.join(self.splits_dir, "lodo_splits.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}
