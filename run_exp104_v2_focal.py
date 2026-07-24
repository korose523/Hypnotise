#!/usr/bin/env python3
"""
run_exp104_v2_focal.py — EEGNet with focal loss for label collapse mitigation.

Focal loss: FL(p_t) = -alpha_t * (1-p_t)^gamma * log(p_t)
Reduces the relative loss for well-classified examples, putting more focus on hard, misclassified examples.
"""
import sys, os, json, time, warnings
import numpy as np
from pathlib import Path
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

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

ALL_SEEDS = [42]
CHECKPOINT_PATH = PROJECT_ROOT / 'results' / 'exp104_v2_focal' / 'exp104_focal_checkpoint.json'
RESULTS_PATH = PROJECT_ROOT / 'results' / 'exp104_v2_focal' / 'exp104_focal_results.json'
MAX_SRC_PER_DOMAIN = 4000
BATCH_SIZE = 256
MAX_EPOCHS = 50
PATIENCE = 10


class FocalLoss(nn.Module):
    """Focal Loss for multi-class classification."""
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # class weights tensor or None
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(logits, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class EEGNet(nn.Module):
    """EEGNet-v4 — same architecture as original."""

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
    win_path = prep_dir / 'prep01_windows' / f'{dataset_name}_windows.npz'
    label_path = prep_dir / 'prep03_labels' / f'{dataset_name}_labels.npz'
    if not win_path.exists() or not label_path.exists():
        return None, None, None
    win_data = np.load(win_path, allow_pickle=True)
    label_data = np.load(label_path, allow_pickle=True)
    windows = win_data['windows']
    subj_ids = win_data['subject_ids']
    labels = label_data['labels']
    valid = labels >= 0
    return windows[valid], labels[valid], subj_ids[valid]


def train_eegnet(train_windows, train_labels, test_windows, test_labels,
                 n_classes=3, n_channels=14, n_samples=256, seed=42, use_focal=True, gamma=2.0):
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Compute class weights for focal loss
    class_counts = np.bincount(train_labels, minlength=n_classes)
    class_weights = 1.0 / (class_counts + 1)
    class_weights = class_weights / class_weights.sum() * n_classes
    alpha = torch.tensor(class_weights, dtype=torch.float32).to(device)

    model = EEGNet(n_classes=n_classes, n_channels=n_channels, n_samples=n_samples).to(device)

    if use_focal:
        criterion = FocalLoss(gamma=gamma, alpha=alpha)
    else:
        criterion = nn.CrossEntropyLoss(weight=alpha)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_dataset = TensorDataset(
        torch.tensor(train_windows, dtype=torch.float32).permute(0, 2, 1).unsqueeze(1),
        torch.tensor(train_labels, dtype=torch.long))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = TensorDataset(
        torch.tensor(test_windows, dtype=torch.float32).permute(0, 2, 1).unsqueeze(1),
        torch.tensor(test_labels, dtype=torch.long))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    best_acc = 0
    best_model = None
    patience_counter = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            outputs = model(Xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()

        # Evaluate
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for Xb, yb in test_loader:
                Xb = Xb.to(device)
                outputs = model(Xb)
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(yb.numpy())

        acc = (np.array(all_preds) == np.array(all_labels)).mean()
        if acc > best_acc:
            best_acc = acc
            best_model = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    # Final evaluation
    model.load_state_dict(best_model)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb = Xb.to(device)
            outputs = model(Xb)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(yb.numpy())

    from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
    acc = accuracy_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds, average=None, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2]).tolist()
    return acc, rec.tolist(), cm


def main():
    if not HAS_TORCH:
        print('PyTorch not available. Skipping.')
        return

    config = load_config(str(PROJECT_ROOT / 'config.yaml'))
    logger = setup_logger('exp104_focal', str(PROJECT_ROOT / config['logs_dir']))
    prep_dir = Path(PROJECT_ROOT / config['processed_dir'])
    splits_dir = Path(PROJECT_ROOT / config['splits_dir'])
    sm = SplitManager(str(splits_dir))
    datasets = ['DREAMER', 'DEAP', 'MAHNOB', 'ds004572']  # 4 representative targets

    out_dir = PROJECT_ROOT / 'results' / 'exp104_v2_focal'
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for target in datasets:
        sources = [d for d in ['DREAMER', 'DEAP', 'MAHNOB', 'SEED', 'SEED_IV', 'FACED', 'ds006437', 'ds004572']
                   if d != target]

        for seed in ALL_SEEDS:
            logger.info(f'[{target} s={seed}] Loading...')
            # Load source windows
            Xs_all, ys_all = [], []
            for src in sources:
                X, y, _ = load_prep_windows(src, prep_dir)
                if X is None: continue
                # Subsample
                if len(X) > MAX_SRC_PER_DOMAIN:
                    rng = np.random.RandomState(seed)
                    idx = rng.choice(len(X), MAX_SRC_PER_DOMAIN, replace=False)
                    X, y = X[idx], y[idx]
                Xs_all.append(X)
                ys_all.append(y)

            if not Xs_all: continue
            X_src = np.concatenate(Xs_all, axis=0)
            y_src = np.concatenate(ys_all, axis=0)

            # Load target
            X_tgt, y_tgt, tgt_sids = load_prep_windows(target, prep_dir)
            if X_tgt is None: continue

            split = sm.load_subject_split(target, seed)
            calib_subjs = set(str(s) for s in split.get('calib_subjects', []))
            test_subjs = set(str(s) for s in split.get('test_subjects', []))
            calib_mask = np.array([str(s) in calib_subjs for s in tgt_sids])
            test_mask = np.array([str(s) in test_subjs for s in tgt_sids])

            if calib_mask.sum() < 2 or test_mask.sum() < 2:
                continue

            # Train zero-shot with focal loss
            logger.info(f'  Training zero-shot (focal, gamma=2.0)...')
            zs_acc, zs_recall, zs_cm = train_eegnet(
                X_src, y_src, X_tgt[test_mask], y_tgt[test_mask],
                seed=seed, use_focal=True, gamma=2.0)

            # Calibration: append 20% target calibration
            X_calib = np.concatenate([X_src, X_tgt[calib_mask]], axis=0)
            y_calib = np.concatenate([y_src, y_tgt[calib_mask]], axis=0)
            logger.info(f'  Training calibration (focal, gamma=2.0)...')
            cal_acc, cal_recall, cal_cm = train_eegnet(
                X_calib, y_calib, X_tgt[test_mask], y_tgt[test_mask],
                seed=seed, use_focal=True, gamma=2.0)

            result = {
                'target': target, 'seed': seed,
                'zs_acc': round(zs_acc, 4), 'calib_acc': round(cal_acc, 4),
                'zs_per_class_recall': zs_recall, 'calib_per_class_recall': cal_recall,
                'zs_cm': zs_cm, 'calib_cm': cal_cm,
            }
            results.append(result)
            logger.info(f'[{target} s={seed}] ZS={zs_acc:.4f} Calib={cal_acc:.4f}')

    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=1)
    logger.info(f'Saved {len(results)} results to {RESULTS_PATH}')


if __name__ == '__main__':
    main()
