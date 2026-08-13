"""
seed_manager.py — Central random seed management.

Ensures ALL modules (numpy, random, os, torch) use the same seed
for each experimental run, guaranteeing full reproducibility.
"""

import numpy as np
import random
import os


class SeedManager:
    """
    Central random seed management.

    Ensures numpy, random, os, and torch (if available) all use the same
    seed for each experimental run.

    Usage:
        sm = SeedManager()
        for seed in sm:
            sm.set_seed(seed)
            # ... run experiment with this seed ...
    """

    DEFAULT_SEEDS = [
        42, 123, 256, 512, 1024, 2048, 3090, 4096, 5000, 6174,
        7777, 8192, 9001, 9999, 10007, 11111, 12345, 13579, 24680, 31415
    ]

    def __init__(self, seeds=None):
        """
        Args:
            seeds: list of int, or None to use DEFAULT_SEEDS (20 seeds)
        """
        self.seeds = seeds if seeds is not None else list(self.DEFAULT_SEEDS)
        self.current_seed = None

    def set_seed(self, seed):
        """
        Set seed for all random number generators.

        Args:
            seed: int, the random seed to apply
        """
        self.current_seed = seed
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

        # If torch is available, set its seeds too
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except ImportError:
            pass

    def __iter__(self):
        """Iterate over all seeds."""
        for s in self.seeds:
            yield s

    def __len__(self):
        """Return number of seeds."""
        return len(self.seeds)

    def __repr__(self):
        return f"SeedManager(n_seeds={len(self)}, current={self.current_seed})"
