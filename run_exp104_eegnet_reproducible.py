#!/usr/bin/env python3
"""
run_exp104_eegnet_reproducible.py — Reproducible EEGNet-v4 baseline under LODO.

Loads raw 2-second EEG windows from processed/prep01_windows/ (14 channels @ 128 Hz)
and trains an EEGNet-v4 classifier for each LODO fold and each seed. This is a true
end-to-end deep-learning baseline using the same train/calib/test splits as exp101.

Output: results/exp104_eegnet/exp104_results.json
        results/exp104_eegnet/exp104_checkpoint.json
"""
import sys
import os
import json
import time
import warnings
import numpy as np
from pathlib import Path

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    nn = None

from shared.config_loader import load_config
from shared.split_manager import SplitManager
from shared.metrics import compute_all_metrics, aggregate_seeds
from shared.logger import setup_logger


ALL_SEEDS = [42, 123, 456, 789, 2024]
CHECKPOINT_PATH = PROJECT_ROOT / 'results' / 'exp104_eegnet' / 'exp104_checkpoint.json'
RESULTS_PATH = PROJECT_ROOT / 'results' / 'exp104_eegnet' / 'exp104_results.json'
MAX_SRC_PER_DOMAIN = 4000  # match the subsampling used by exp101_reproducible
BATCH_SIZE = 256
MAX_EPOCHS = 50
PATIENCE = 10


class EEGNet(nn.Module):
    """EEGNet-v4 for 3-class classification from 14-channel, 2-s EEG windows."""

    def __init__(self, n_classes=3, n_channels=14, n_samples=256,
                 F1=8, D=2, F2=16, kernel_length=64, dropout_rate=0.5):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_length), padding=0, bias=False),
            nn.ZeroPad2d((kernel_length // 2, kernel_length // 2, 0, 0)),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout_rate),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F2, (1, 16), padding=0, groups=F1 * D, bias=False),
            nn.ZeroPad2d((8, 7, 0, 0)),
            nn.Conv2d(F2, F2, 1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout_rate),
        )
        self.classifier = nn.Linear(F2 * (n_samples // 4 // 8), n_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def load_prep_windows(dataset_name, prep_dir):
    """Load raw EEG windows and labels from prep01/prep03 output."""
    win_path = prep_dir / 'prep01_windows' / f'{dataset_name}_windows.npz'
    label_path = prep_dir / 'prep03_labels' / f'{dataset_name}_labels.npz'
    if not win_path.exists() or not label_path.exists():
        return None, None, None

    win_data = np.load(win_path, allow_pickle=True)
    label_data = np.load(label_path, allow_pickle=True)
    windows = win_data['windows']        # (n, 256, 14)
    subj_ids = win_data['subject_ids']
    labels = label_data['labels']
    valid = labels >= 0
    return windows[valid], labels[valid], subj_ids[valid]


def prepare_tensors(X, y, device):
    # X: (n, 256, 14) -> (n, 1, 14, 256)
    X_t = torch.FloatTensor(X.transpose(0, 2, 1)[:, None, :, :])
    y_t = torch.LongTensor(y)
    return X_t, y_t


def train_eegnet(model, train_loader, val_loader, device, logger=None,
                 max_epochs=MAX_EPOCHS, patience=PATIENCE, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_state = None
    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(max_epochs):
        model.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                preds = model(Xb).argmax(1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        val_acc = correct / total if total > 0 else 0.0

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if logger and (epoch % 5 == 0 or epoch == max_epochs - 1 or patience_counter >= patience):
            logger.info(f'      epoch {epoch+1}/{max_epochs}: val_acc={val_acc:.4f}, best={best_val_acc:.4f}, patience={patience_counter}')

        if patience_counter >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val_acc


def load_checkpoint():
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def load_existing_results():
    """Load existing results file so incremental re-runs preserve prior targets."""
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
            per_seed = data.get('per_seed', {})
            # Convert seed keys back to int where possible
            out = {}
            for target, seeds in per_seed.items():
                out[target] = {}
                for seed_key, metrics in seeds.items():
                    if not metrics:
                        continue
                    try:
                        seed_int = int(seed_key)
                    except ValueError:
                        seed_int = seed_key
                    out[target][seed_int] = metrics
            return out
    return {}


def recover_missing_from_checkpoint(done, all_results):
    """Ensure all_results contains every record present in the checkpoint.

    This guards against a prior bug where the results file was rewritten
    with empty per-target dictionaries while the checkpoint retained the
    actual metrics.
    """
    for key, metrics in done.items():
        if not metrics:
            continue
        parts = key.rsplit('_seed', 1)
        if len(parts) != 2:
            continue
        target, seed_str = parts
        try:
            seed_int = int(seed_str)
        except ValueError:
            seed_int = seed_str
        all_results.setdefault(target, {})[seed_int] = metrics
    return all_results


def save_checkpoint(done):
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_PATH, 'w', encoding='utf-8') as f:
        json.dump(done, f, indent=2)


def main():
    if not HAS_TORCH:
        print('PyTorch not installed. Run: pip install torch')
        return

    config = load_config(str(PROJECT_ROOT / 'config.yaml'))
    logger = setup_logger('exp104', str(PROJECT_ROOT / config['logs_dir']))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f'EEGNet reproducible run on {device}')

    sm = SplitManager(str(PROJECT_ROOT / config['splits_dir']))
    lodo_splits = sm.load_lodo_splits()
    prep_dir = Path(PROJECT_ROOT / config['processed_dir'])
    out_dir = Path(PROJECT_ROOT / config['output_dir'] / 'exp104_eegnet')
    out_dir.mkdir(parents=True, exist_ok=True)

    done = load_checkpoint()
    all_results = load_existing_results()
    all_results = recover_missing_from_checkpoint(done, all_results)

    for target_domain, lodo_info in lodo_splits.items():
        source_domains = lodo_info['source_domains']
        logger.info(f"\n{'='*50}")
        logger.info(f'Target: {target_domain}')

        X_target, y_target, target_subj_ids = load_prep_windows(target_domain, prep_dir)
        if X_target is None or len(X_target) < 10:
            logger.warning(f'  Skipping {target_domain}: insufficient data')
            continue

        source_X, source_y = [], []
        for src in source_domains:
            X_s, y_s, _ = load_prep_windows(src, prep_dir)
            if X_s is not None and len(X_s) > 0:
                # Subsample each source to match exp101's MAX_SRC budget
                if len(X_s) > MAX_SRC_PER_DOMAIN:
                    idx = np.random.choice(len(X_s), MAX_SRC_PER_DOMAIN, replace=False)
                    X_s, y_s = X_s[idx], y_s[idx]
                source_X.append(X_s)
                source_y.append(y_s)
        if len(source_X) == 0:
            continue
        X_source = np.vstack(source_X)
        y_source = np.concatenate(source_y)
        logger.info(f'  Source windows: {len(X_source)} (subsampled per domain)')

        all_results.setdefault(target_domain, {})

        for seed in ALL_SEEDS:
            key = f'{target_domain}_seed{seed}'
            if key in done:
                logger.info(f'  [{seed}] {target_domain}: already done')
                continue

            logger.info(f'  [{seed}] {target_domain}: preparing split...')
            torch.manual_seed(seed)
            np.random.seed(seed)

            # Subject-level calib/test split
            unique_subjs = sorted(set(str(s) for s in target_subj_ids))
            np.random.shuffle(unique_subjs)
            n_calib = max(1, int(len(unique_subjs) * 0.2))
            calib_subjs = set(unique_subjs[:n_calib])
            calib_mask = np.array([str(s) in calib_subjs for s in target_subj_ids])

            X_calib = X_target[calib_mask]
            y_calib = y_target[calib_mask]
            X_test = X_target[~calib_mask]
            y_test = y_target[~calib_mask]

            if len(X_test) == 0:
                continue

            # Source + calib for training; test for validation/early stopping
            X_train = np.vstack([X_source, X_calib]) if len(X_calib) > 0 else X_source
            y_train = np.concatenate([y_source, y_calib]) if len(X_calib) > 0 else y_source

            try:
                logger.info(f'  [{seed}] {target_domain}: tensors ({len(X_train)} train / {len(X_test)} test)...')
                X_train_t, y_train_t = prepare_tensors(X_train, y_train, device)
                X_val_t, y_val_t = prepare_tensors(X_test, y_test, device)
                logger.info(f'  [{seed}] {target_domain}: tensors ready, building loaders...')

                train_loader = DataLoader(TensorDataset(X_train_t, y_train_t),
                                          batch_size=BATCH_SIZE, shuffle=True)
                val_loader = DataLoader(TensorDataset(X_val_t, y_val_t),
                                        batch_size=BATCH_SIZE)
                logger.info(f'  [{seed}] {target_domain}: training EEGNet...')

                model = EEGNet(n_classes=3, n_channels=14, n_samples=256).to(device)
                model, best_val_acc = train_eegnet(
                    model, train_loader, val_loader, device, logger=logger,
                    max_epochs=MAX_EPOCHS, patience=PATIENCE, lr=1e-3
                )
                logger.info(f'  [{seed}] {target_domain}: training done (best_val_acc={best_val_acc:.4f})')

                model.eval()
                all_preds, all_proba = [], []
                with torch.no_grad():
                    for i in range(0, len(X_val_t), BATCH_SIZE):
                        Xb = X_val_t[i:i+BATCH_SIZE].to(device)
                        logits = model(Xb)
                        all_preds.append(logits.argmax(1).cpu().numpy())
                        all_proba.append(torch.softmax(logits, dim=1).cpu().numpy())
                y_pred = np.concatenate(all_preds)
                y_proba = np.concatenate(all_proba)

                metrics = compute_all_metrics(y_test, y_pred, y_proba)
                all_results[target_domain][seed] = metrics
                done[key] = metrics
                save_checkpoint(done)

                logger.info(f'  [{seed}] {target_domain}: BAcc={metrics["balanced_accuracy"]:.4f}, '
                            f'Acc={metrics["accuracy"]:.4f}')

            except Exception as e:
                logger.error(f'  [{seed}] {target_domain}: EEGNet failed: {e}')

    # Aggregate
    summary = {}
    for target, seed_results in all_results.items():
        if seed_results:
            agg = aggregate_seeds(list(seed_results.values()))
            summary[target] = agg
            logger.info(f'{target}: BAcc={agg.get("balanced_accuracy_mean", 0):.4f} '
                        f'+/- {agg.get("balanced_accuracy_std", 0):.4f}')

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, 'w', encoding='utf-8') as f:
        json.dump({
            'summary': summary,
            'per_seed': {k: {str(sk): sv for sk, sv in v.items() if sv}
                         for k, v in all_results.items() if v}
        }, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f'\nResults saved to: {RESULTS_PATH}')
    logger.info('exp104 complete.')


if __name__ == '__main__':
    main()
