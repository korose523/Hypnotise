"""EEGNet-v4 baseline for cross-domain EEG classification.
Adapted from Lawhern et al. (2018) "EEGNet: A Compact Convolutional Network for EEG-based BCIs".

Paper: https://arxiv.org/abs/1611.08024
Reference implementation: https://github.com/vlawhern/arl-eegmodels

Usage:
  python eegnet_baseline.py
  (requires PyTorch: pip install torch)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time, json, os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '.')

class EEGNet(nn.Module):
    """
    EEGNet-v4 for 14-channel, 2-second (256 sample) EEG windows.
    
    Architecture:
    - Temporal convolution (F1 filters, kernel=64, input 14ch)
    - Depthwise spatial convolution (D*F1 filters)
    - Separable convolution (F2 filters)
    - Classification head
    """
    def __init__(self, n_channels=14, n_samples=256, n_classes=3,
                 F1=8, D=2, F2=16, dropout=0.5):
        super().__init__()
        
        # Block 1: Temporal + Spatial
        self.conv1 = nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.depthwise = nn.Conv2d(F1, D*F1, (n_channels, 1), groups=F1, bias=False)
        self.bn_depth = nn.BatchNorm2d(D*F1)
        self.act1 = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout)
        
        # Block 2: Separable convolution
        self.sep_conv = nn.Conv2d(D*F1, D*F1, (1, 16), padding=(0, 8), groups=D*F1, bias=False)
        self.sep_point = nn.Conv2d(D*F1, F2, (1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(F2)
        self.act2 = nn.ELU()
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout)
        
        # Classifier
        self.flatten = nn.Flatten()
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            out = self.pool2(self.act2(self.bn2(self.sep_point(self.sep_conv(
                self.drop1(self.pool1(self.act1(self.bn_depth(self.depthwise(
                self.act1(self.bn1(self.conv1(dummy)))))))))))))
            self.fc_dim = out.numel()
        self.fc = nn.Linear(self.fc_dim, n_classes)
    
    def forward(self, x):
        # x: (batch, 1, n_channels, n_samples)
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act1(self.bn_depth(self.depthwise(x)))
        x = self.drop1(self.pool1(x))
        x = self.act2(self.bn2(self.sep_point(self.sep_conv(x))))
        x = self.drop2(self.pool2(x))
        return self.fc(self.flatten(x))


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct += (model(X).argmax(1) == y).sum().item()
        total += len(y)
    return total_loss / total, correct / total

def eval_model(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            loss = criterion(model(X), y)
            total_loss += loss.item() * len(y)
            correct += (model(X).argmax(1) == y).sum().item()
            total += len(y)
    return total_loss / total, correct / total

def run_eegnet_lodo(target, source_datasets, X_dict, y_dict, epochs=100, batch_size=64, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'EEGNet on {device}')
    
    # Source data
    Xs = np.concatenate([X_dict[d] for d in source_datasets], axis=0)
    ys = np.concatenate([y_dict[d] for d in source_datasets], axis=0)
    # Reshape: (n, 256, 14) -> (n, 1, 14, 256)
    Xs = Xs.transpose(0, 2, 1)[:, None, :, :]
    ys = ys.astype(np.int64)
    
    Xt = X_dict[target].transpose(0, 2, 1)[:, None, :, :]
    yt = y_dict[target].astype(np.int64)
    
    # Train/test split
    n_test = len(yt) // 5
    X_train = torch.FloatTensor(Xs)
    y_train = torch.LongTensor(ys)
    X_test = torch.FloatTensor(Xt[-n_test:])
    y_test = torch.LongTensor(yt[-n_test:])
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)
    
    model = EEGNet(n_channels=14, n_samples=256, n_classes=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = eval_model(model, test_loader, criterion, device)
        if test_acc > best_acc:
            best_acc = test_acc
        if (epoch + 1) % 20 == 0:
            print(f'  Epoch {epoch+1}: train={train_acc:.3f} test={test_acc:.3f}', flush=True)
    
    return best_acc


if __name__ == '__main__':
    print('EEGNet baseline ready. See run_all_experiments.py for integration.')
    print('Quick test:')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EEGNet().to(device)
    x = torch.randn(4, 1, 14, 256).to(device)
    y = model(x)
    print(f'  Input: {x.shape}, Output: {y.shape}')
    print(f'  Params: {sum(p.numel() for p in model.parameters()):,}')
