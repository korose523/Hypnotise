#!/usr/bin/env python3
"""
rt203_realtime_inference_public_pretrained.py — Real-time inference with publicly pretrained models.

Real-time Experiment 203 (Paper 2, Real-time Classification):
  - Loads pretrained WFSC models from offline experiments (exp101)
  - Performs real-time inference on incoming 63-dim feature vectors
  - Supports both WFSC-Mahalanobis and WFSC-Fixed ensemble modes
  - Implements prediction smoothing (sliding window averaging)
  - Outputs class predictions with confidence scores

Input:  63-dim feature stream (from rt202)
        Pretrained models (from models/ directory)
Output: results/rt203_inference/inference_results.json
        results/rt203_inference/prediction_trace.npz
"""

import sys
import json
import pickle
import numpy as np
from pathlib import Path
from collections import deque

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.config_loader import load_config
from shared.wfsc import WFSC_Mahalanobis, WFSC_Fixed
from shared.metrics import compute_all_metrics, CLASS_NAMES, CLASS_NAMES_CN
from shared.logger import setup_logger


class PredictionSmoother:
    """
    Sliding window smoother for real-time predictions.

    Reduces prediction jitter by averaging probabilities over a
    configurable time window.
    """

    def __init__(self, window_size=5):
        self.window_size = window_size
        self.probability_buffer = deque(maxlen=window_size)
        self.n_classes = 3

    def update(self, probabilities):
        """
        Add new prediction and return smoothed result.

        Args:
            probabilities: ndarray (n_classes,)

        Returns:
            smoothed_probabilities: ndarray (n_classes,)
            smoothed_class: int
        """
        self.probability_buffer.append(probabilities)

        # Average probabilities
        stacked = np.array(list(self.probability_buffer))
        smoothed = np.mean(stacked, axis=0)
        smoothed_class = np.argmax(smoothed)

        return smoothed, smoothed_class

    def get_confidence(self):
        """Get confidence of current prediction (max probability)."""
        if not self.probability_buffer:
            return 0.0
        smoothed = np.mean(list(self.probability_buffer), axis=0)
        return float(np.max(smoothed))

    def get_entropy(self):
        """Get entropy of current prediction distribution."""
        if not self.probability_buffer:
            return 0.0
        smoothed = np.mean(list(self.probability_buffer), axis=0)
        smoothed = smoothed / (np.sum(smoothed) + 1e-10)
        return float(-np.sum(smoothed * np.log(smoothed + 1e-10)))

    def reset(self):
        self.probability_buffer.clear()


class RealtimeClassifier:
    """
    Real-time hypnosis depth classifier.

    Loads pretrained models and performs inference on incoming feature vectors.
    """

    def __init__(self, model_dir=None, config=None, smoother_window=5):
        if config is None:
            config = load_config(str(PROJECT_ROOT / 'config.yaml'))

        self.config = config
        self.model_dir = Path(model_dir) if model_dir else \
            Path(PROJECT_ROOT / config['models_dir'])
        self.smoother = PredictionSmoother(window_size=smoother_window)

        self.wfsc = None
        self.model_loaded = False
        self.prediction_history = []
        self.class_names = CLASS_NAMES
        self.class_names_cn = CLASS_NAMES_CN

    def load_model(self, model_path=None):
        """
        Load a pretrained WFSC model.

        Args:
            model_path: Path to saved model (pickle), or None to auto-detect
        """
        if model_path is None:
            # Auto-detect the latest model
            model_files = sorted(self.model_dir.glob('wfsc_*.pkl'))
            if not model_files:
                model_files = sorted(self.model_dir.glob('*.pkl'))
            if not model_files:
                raise FileNotFoundError(
                    f"No pretrained model found in {self.model_dir}. "
                    f"Run exp101 first to generate models."
                )
            model_path = model_files[-1]

        logger = setup_logger('rt203', str(PROJECT_ROOT / self.config['logs_dir']))
        logger.info(f"Loading model from: {model_path}")

        with open(model_path, 'rb') as f:
            self.wfsc = pickle.load(f)

        self.model_loaded = True
        logger.info(f"Model loaded. Source domains: {list(self.wfsc.source_models.keys())}")
        return self

    def predict(self, feature_vector):
        """
        Predict hypnosis depth class from a single 63-dim feature vector.

        Args:
            feature_vector: ndarray (63,) or (1, 63)

        Returns:
            result: dict with class, confidence, probabilities, etc.
        """
        if not self.model_loaded:
            raise RuntimeError("No model loaded. Call load_model() first.")

        if feature_vector.ndim == 1:
            feature_vector = feature_vector.reshape(1, -1)

        # Get raw prediction
        proba = self.wfsc.predict_proba(feature_vector)[0]
        raw_class = int(np.argmax(proba))

        # Smoothed prediction
        smoothed_proba, smoothed_class = self.smoother.update(proba)

        confidence = float(np.max(smoothed_proba))
        entropy = self.smoother.get_entropy()

        result = {
            'raw_class': raw_class,
            'raw_class_name': self.class_names[raw_class],
            'raw_class_name_cn': self.class_names_cn[raw_class],
            'smoothed_class': smoothed_class,
            'smoothed_class_name': self.class_names[smoothed_class],
            'smoothed_class_name_cn': self.class_names_cn[smoothed_class],
            'confidence': confidence,
            'entropy': entropy,
            'probabilities': {
                self.class_names[i]: float(proba[i]) for i in range(3)
            },
            'smoothed_probabilities': {
                self.class_names[i]: float(smoothed_proba[i]) for i in range(3)
            },
        }

        self.prediction_history.append(result)
        return result

    def get_history_summary(self, last_n=None):
        """Get summary of prediction history."""
        history = self.prediction_history
        if last_n is not None:
            history = history[-last_n:]

        if not history:
            return {}

        classes = [h['smoothed_class'] for h in history]
        confidences = [h['confidence'] for h in history]

        unique, counts = np.unique(classes, return_counts=True)
        class_counts = {int(u): int(c) for u, c in zip(unique, counts)}

        return {
            'n_predictions': len(history),
            'class_distribution': class_counts,
            'mean_confidence': float(np.mean(confidences)),
            'min_confidence': float(np.min(confidences)),
            'dominant_class': int(unique[np.argmax(counts)]),
            'dominant_class_name': self.class_names[int(unique[np.argmax(counts)])],
        }

    def reset(self):
        """Reset prediction state."""
        self.smoother.reset()
        self.prediction_history = []


def simulate_realtime_inference(classifier, n_predictions=30):
    """
    Simulate real-time inference with synthetic features.

    Args:
        classifier: RealtimeClassifier instance
        n_predictions: int, number of predictions to simulate

    Returns:
        results: list of prediction dicts
    """
    np.random.seed(42)
    results = []

    for i in range(n_predictions):
        # Generate synthetic 63-dim features
        features = np.random.randn(63) * 0.5

        # Simulate changing brain state over time
        if i < n_predictions // 3:
            # Awake phase: higher DASM values
            features[:42] += np.random.randn(42) * 0.3
        elif i < 2 * n_predictions // 3:
            # Light hypnosis: moderate DASM, higher alpha
            features[7:14] += 0.5  # theta band enhancement
            features[14:21] += 0.3  # alpha band enhancement
        else:
            # Deep hypnosis: distinct pattern
            features[:7] -= 0.3  # delta increase
            features[7:14] += 0.8  # theta increase
            features[42:] += 0.2  # overall bandpower shift

        result = classifier.predict(features)
        results.append(result)

    return results


def main():
    config = load_config(str(PROJECT_ROOT / 'config.yaml'))
    logger = setup_logger('rt203', str(PROJECT_ROOT / config['logs_dir']))

    out_dir = Path(PROJECT_ROOT / config['output_dir'] / 'rt203_inference')
    out_dir.mkdir(parents=True, exist_ok=True)

    classifier = RealtimeClassifier(config=config, smoother_window=5)

    # Try to load pretrained model (fall back to simulation if not available)
    try:
        classifier.load_model()
        logger.info("Pretrained model loaded successfully.")
        use_pretrained = True
    except FileNotFoundError:
        logger.warning("No pretrained model found. Training a temporary model for demo.")
        # Create a temporary model for demo
        from shared.wfsc import WFSC_Mahalanobis

        temp_source = {
            'DREAMER': (np.random.randn(100, 63), np.random.randint(0, 3, 100)),
            'DEAP': (np.random.randn(100, 63), np.random.randint(0, 3, 100)),
        }
        classifier.wfsc = WFSC_Mahalanobis(random_state=42)
        classifier.wfsc.fit(temp_source)
        classifier.model_loaded = True
        use_pretrained = False

    # Simulate real-time inference
    logger.info("\nSimulating real-time inference (30 predictions)...")
    results = simulate_realtime_inference(classifier, n_predictions=30)

    # Print prediction trace
    logger.info(f"\n{'Time':>5} {'Raw':>5} {'Smooth':>8} {'Conf':>8} {'Class':>20}")
    logger.info("-" * 55)
    for i, r in enumerate(results):
        logger.info(f"{i * 2:5d} {r['raw_class']:>5d} "
                     f"{r['smoothed_class']:>8d} {r['confidence']:>8.3f} "
                     f"{r['smoothed_class_name']:>20}")

    # History summary
    summary = classifier.get_history_summary()
    logger.info(f"\nPrediction Summary:")
    logger.info(f"  Total predictions: {summary.get('n_predictions', 0)}")
    logger.info(f"  Class distribution: {summary.get('class_distribution', {})}")
    logger.info(f"  Mean confidence: {summary.get('mean_confidence', 0):.3f}")
    logger.info(f"  Dominant class: {summary.get('dominant_class_name', '?')}")

    # Save results
    pred_trace = {
        'predictions': results,
        'summary': summary,
        'model_loaded': use_pretrained,
    }

    with open(out_dir / 'inference_results.json', 'w', encoding='utf-8') as f:
        json.dump(pred_trace, f, indent=2, ensure_ascii=False, default=str)

    # Save prediction trace as numpy
    trace_classes = np.array([r['smoothed_class'] for r in results])
    trace_conf = np.array([r['confidence'] for r in results])
    trace_proba = np.array([list(r['probabilities'].values()) for r in results])

    np.savez_compressed(
        out_dir / 'prediction_trace.npz',
        classes=trace_classes,
        confidences=trace_conf,
        probabilities=trace_proba,
        raw_classes=np.array([r['raw_class'] for r in results]),
    )

    logger.info(f"\nResults saved to: {out_dir}")
    logger.info("rt203 complete.")


if __name__ == '__main__':
    main()
