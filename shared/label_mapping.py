"""
label_mapping.py — Dataset-specific labels to 3-class hypnosis depth mapping.

Maps each of the 8 datasets' native labels to a unified 3-class scheme:
  0 = Awake (清醒)
  1 = Light Hypnosis (浅催眠)
  2 = Deep Hypnosis (深催眠)
"""

import numpy as np


class LabelMapper:
    """
    Unified label mapping for all 8 datasets.

    Each dataset has its own mapping rules defined in config.yaml.
    This class applies those rules to convert raw labels to 3-class labels.
    """

    # Built-in mapping rules (fallback when config is not loaded)
    DEFAULT_RULES = {
        'DREAMER': {
            'source_field': 'arousal',
            'awake': lambda v: v >= 4,
            'light': lambda v: v == 3,
            'deep':  lambda v: v <= 2,
        },
        'DEAP': {
            'source_field': 'arousal',
            'awake': lambda v: v >= 7,
            'light': lambda v: 4 <= v <= 6,
            'deep':  lambda v: v <= 3,
        },
        'MAHNOB': {
            'source_field': 'arousal',
            'awake': lambda v: v >= 7,
            'light': lambda v: 4 <= v <= 6,
            'deep':  lambda v: v <= 3,
        },
        'SEED': {
            'source_field': 'emotion',
            'awake': lambda v: v == 1,
            'light': lambda v: v == 0,
            'deep':  lambda v: v == -1,
        },
        'SEED_IV': {
            'source_field': 'emotion',
            'awake': lambda v: v in [2, 3],
            'light': lambda v: v == 0,
            'deep':  lambda v: v == 1,
        },
        'FACED': {
            'source_field': 'arousal',
            'awake': lambda v: v >= 6.5,
            'light': lambda v: 3.5 <= v < 6.5,
            'deep':  lambda v: v < 3.5,
        },
        'ds004572': {
            'source_field': 'hypnosis_depth',
            'awake': lambda v: v <= 3,
            'light': lambda v: 4 <= v <= 6,
            'deep':  lambda v: v >= 7,
        },
        'ds006437': {
            'source_field': 'phase',
            'awake': lambda v: v in ['pre', 'post'],
            'light': lambda v: v == 'during',
            'deep':  None,  # May not have deep hypnosis
        },
    }

    CLASS_NAMES = ['Awake', 'Light Hypnosis', 'Deep Hypnosis']
    CLASS_NAMES_CN = ['清醒', '浅催眠', '深催眠']

    def __init__(self, config=None):
        """
        Args:
            config: dict, loaded from config.yaml. If None, uses DEFAULT_RULES.
        """
        self.config = config
        self._rules = {}

        if config and 'label_mapping' in config:
            self._parse_config_rules(config['label_mapping'])
        else:
            self._rules = self.DEFAULT_RULES

    def _parse_config_rules(self, label_mapping_cfg):
        """Parse label mapping rules from config dict."""
        for dataset_name, cfg in label_mapping_cfg.items():
            source_field = cfg.get('source_field', 'v')
            self._rules[dataset_name] = {'source_field': source_field}

            rules_str = cfg.get('rules', {})
            if 'awake' in rules_str and rules_str['awake'] is not None:
                self._rules[dataset_name]['awake'] = self._make_rule(rules_str['awake'], source_field)
            if 'light' in rules_str and rules_str['light'] is not None:
                self._rules[dataset_name]['light'] = self._make_rule(rules_str['light'], source_field)
            if 'deep' in rules_str and rules_str['deep'] is not None:
                self._rules[dataset_name]['deep'] = self._make_rule(rules_str['deep'], source_field)

    def _make_rule(self, rule_str, source_field='v'):
        """
        Convert a rule string like 'depth <= 3' to a lambda function.
        Supports: comparison operators, 'in' for list membership.
        """
        def rule_func(v):
            ns = {'v': v, source_field: v, '__builtins__': {}}
            return eval(rule_str, ns)
        return rule_func

    def map_labels(self, dataset_name, raw_values):
        """
        Convert raw label values to 3-class hypnosis depth labels.

        Args:
            dataset_name: str, one of the 8 dataset names
            raw_values: array-like of raw label values

        Returns:
            labels: ndarray of int (0, 1, 2) — mapped 3-class labels
            mask: ndarray of bool — True where mapping was successful
        """
        if dataset_name not in self._rules:
            raise ValueError(f"Unknown dataset: {dataset_name}. "
                             f"Available: {list(self._rules.keys())}")

        rules = self._rules[dataset_name]
        raw_values = np.asarray(raw_values)
        labels = np.full(len(raw_values), -1, dtype=int)
        mask = np.zeros(len(raw_values), dtype=bool)

        for i, v in enumerate(raw_values):
            if rules.get('awake') and rules['awake'](v):
                labels[i] = 0
                mask[i] = True
            elif rules.get('light') and rules['light'](v):
                labels[i] = 1
                mask[i] = True
            elif rules.get('deep') and rules['deep'](v):
                labels[i] = 2
                mask[i] = True

        return labels, mask

    def get_class_distribution(self, labels):
        """
        Compute class distribution statistics.

        Args:
            labels: array-like of int (0, 1, 2)

        Returns:
            dict: {class_name: count, ...}
        """
        labels = np.asarray(labels)
        dist = {}
        for i, name in enumerate(self.CLASS_NAMES):
            dist[name] = int(np.sum(labels == i))
        dist['unmapped'] = int(np.sum(labels == -1))
        return dist
