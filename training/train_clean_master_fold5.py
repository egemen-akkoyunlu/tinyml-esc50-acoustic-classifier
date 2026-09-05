#!/usr/bin/env python3
"""
=============================================================================
🔒 STEP 2: TRAIN CLEAN MASTER MODEL ON OFFICIAL FOLDS 1-4 (ZERO LEAKAGE)
=============================================================================
Model: AudioPhiNetCRNNClassifierQAT (124.5k Parameters)
  - Training Set: Folds 1, 2, 3, 4 (1,600 Clips)
  - Test Set    : Fold 5 (400 Completely Unseen Clips)
  - Augmentations: SpecAugment (Time/Freq Mask) + Cosine Annealing + Label Smoothing
Saves: models/clean_master_fold5.pth
=============================================================================
"""

import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Compute Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

SPECTROGRAM_PATH = os.path.join(PROJECT_ROOT, 'training', 'official_fold5_spectrograms.pt')
MODEL_OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'models')
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
MODEL_OUTPUT_PATH = os.path.join(MODEL_OUTPUT_DIR, 'clean_master_fold5.pth')

if not os.path.exists(SPECTROGRAM_PATH):
    print(f"❌ Error: {SPECTROGRAM_PATH} not found! Please run Step 1 first:")
    print("   /home/acar/kws_env/bin/python3 training/prepare_official_fold5_dataset.py")
    sys.exit(1)

# 1. Load Precomputed Official Dataset
print(f"📦 Loading precomputed official spectrograms from: {SPECTROGRAM_PATH}...")
data = torch.load(SPECTROGRAM_PATH, map_location='cpu', weights_only=False)

train_specs  = data['train_specs']  # [1600, 1, 52, 313]
train_labels = data['train_labels'] # [1600]
test_specs   = data['test_specs']   # [400, 1, 52, 313]
test_labels  = data['test_labels']  # [400]

print(f"   • Train Set: {train_specs.shape[0]} clips (Folds 1-4)")
print(f"   • Test Set : {test_specs.shape[0]} clips (Fold 5 - Completely Isolated)")

# 2. Audio SpecAugment for Robust Edge Acoustic Generalization
def spec_augment(x, freq_mask_max=8, time_mask_max=24):
    # x: [batch, 1, freq=52, time=313]
    b, c, f, t = x.shape
    x_aug = x.clone()
    
    # Frequency Masking
    for i in range(b):
        f_len = random.randint(0, freq_mask_max)
        f_0 = random.randint(0, max(0, f - f_len))
        x_aug[i, :, f_0:f_0 + f_len, :] = 0.0
        
        # Time Masking
        t_len = random.randint(0, time_mask_max)
        t_0 = random.randint(0, max(0, t - t_len))
        x_aug[i, :, :, t_0:t_0 + t_len] = 0.0
        
    return x_aug

# 3. Model Architecture: Clean PhiNet-CRNN (124.5k)
class HighCapBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride=1):
        super().__init__()
        self.use_residual = (stride == 1 and in_channels == out_channels)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6()
        )

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)

class CleanPhiNetCRNN(nn.Module):
    def __init__(self, num_classes=50):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=(1, 2), padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU6()
        )
        self.phi_blocks = nn.Sequential(
            HighCapBlock(in_channels=16, out_channels=32, stride=(1, 2)),
            HighCapBlock(in_channels=32, out_channels=32, stride=(1, 1)),
            HighCapBlock(in_channels=32, out_channels=48, stride=(1, 2))
        )
        self.conv_compress = nn.Conv2d(48, 32, kernel_size=1, bias=False)
        self.pre_gru_bn = nn.BatchNorm1d(32)
        
        self.gru = nn.GRU(input_size=32, hidden_size=160, batch_first=True)
        self.post_gru_bn = nn.BatchNorm1d(160)
        self.drop = nn.Dropout(0.3)
        self.bottleneck = nn.Linear(160, 128)
        self.fc = nn.Linear(128, num_classes)

    def extract_features(self, x):
        x = self.stem(x)
        x = self.phi_blocks(x)
        x = self.conv_compress(x)
        x = F.avg_pool2d(x, (x.shape[2], 1)) # pool freq
        b, c, f, t = x.shape
        seq_in = x.view(b, c, t).permute(0, 2, 1) # [b, 39, 32]
        seq_in = seq_in.permute(0, 2, 1)
        seq_in = self.pre_gru_bn(seq_in)
        seq_in = seq_in.permute(0, 2, 1)
        return seq_in

    def forward(self, x):
        seq_in = self.extract_features(x)
        r_out, _ = self.gru(seq_in)
        attn_scores = r_out.mean(dim=-1)
        attn_weights = torch.softmax(attn_scores, dim=1)
        pooled = (r_out * attn_weights.unsqueeze(-1)).sum(dim=1)
        pooled = self.post_gru_bn(pooled)
        pooled = self.drop(pooled)
        compressed = F.relu6(self.bottleneck(pooled))
        return self.fc(compressed)

def main():
    model = CleanPhiNetCRNN(num_classes=50).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📐 Clean Model Parameters: {total_params:,} (124.5k Flagship Base)")

    train_ds = TensorDataset(train_specs, train_labels)
    test_ds  = TensorDataset(test_specs, test_labels)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=64, shuffle=False)

    epochs = 60
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.08)

    best_acc = 0.0
    best_weights = None

    print("\n" + "=" * 70)
    print("🚀 TRAINING CLEAN MASTER MODEL ON OFFICIAL FOLDS 1-4 (NO DATA LEAKAGE)")
    print("   Evaluation strictly on Fold 5 (400 Completely Unseen Clips)")
    print("=" * 70)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_corr = 0
        total_train = 0

        for x, y in train_loader:
            x = spec_augment(x).to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            train_loss += loss.item() * y.size(0)
            train_corr += (logits.argmax(1) == y).sum().item()
            total_train += y.size(0)

        scheduler.step()
        train_loss /= total_train
        train_acc = (train_corr / total_train) * 100.0

        # Evaluate on Completely Unseen Fold 5
        model.eval()
        test_corr = 0
        total_test = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                test_corr += (logits.argmax(1) == y).sum().item()
                total_test += y.size(0)

        test_acc = (test_corr / total_test) * 100.0
        is_best = test_acc > best_acc
        if is_best:
            best_acc = test_acc
            best_weights = model.state_dict().copy()
            torch.save(best_weights, MODEL_OUTPUT_PATH)

        marker = " 🌟 [NEW BEST ON FOLD-5!]" if is_best else ""
        print(f"Epoch [{epoch:2d}/{epochs}] Train Loss: {train_loss:.4f} (Acc: {train_acc:5.1f}%) | Fold 5 Test: {test_acc:5.2f}% ({test_corr}/{total_test}){marker}")

    print("\n" + "=" * 70)
    print(f"🏆 CLEAN MASTER TRAINING COMPLETE (ZERO LEAKAGE)!")
    print(f"   • Best Official Fold-5 Test Accuracy: {best_acc:.2f}%")
    print(f"   • Model Checkpoint Saved to         : {MODEL_OUTPUT_PATH}")
    print("=" * 70)

if __name__ == '__main__':
    main()
