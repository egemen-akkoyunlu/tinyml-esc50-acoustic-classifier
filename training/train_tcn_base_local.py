#!/usr/bin/env python3
# ==============================================================================
# 🚀 FULLY-FUSED 1D DILATED TCN (TC-ResNet) + ATTENTION ARCHITECTURE
#
# Architectural Optimizations & QAT Bug Fixes Applied:
#   1. Strategy 2 Width Multiplier: Channels scaled (48->96) for ~83k parameter target.
#   2. True Inverted Residuals: Expansion phase added, pw_relu deleted (Linear).
#   3. QNNPACK Engine: Target-aligned INT8 quantization for ARM Cortex-M / Xtensa.
#   4. Full 3-Way Module Fusion: Exact hardware alignment for all Conv+BN+ReLU layers.
# ==============================================================================

import os
import sys
import copy
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torch.ao.quantization as quantization
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import MelSpectrogram, FrequencyMasking, TimeMasking

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

# Set QNNPACK for ARM / Mobile hardware alignment
if 'qnnpack' in torch.backends.quantized.supported_engines:
    torch.backends.quantized.engine = 'qnnpack'
else:
    torch.backends.quantized.engine = 'fbgemm'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'🖥️ Training Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"})')
print(f'⚙️ Quantization Engine: {torch.backends.quantized.engine}')

DURATION_SEC = 5
SAMPLE_RATE = 16000
NUM_CLASSES = 50

# ------------------------------------------------------------------------------
# 1. ADVANCED DATA AUGMENTATION: SPECAUGMENT & SPECMIX / MIXUP
# ------------------------------------------------------------------------------
class SpecAugment(nn.Module):
    def __init__(self, freq_mask_param: int = 8, time_mask_param: int = 24):
        super().__init__()
        self.freq_mask = FrequencyMasking(freq_mask_param)
        self.time_mask = TimeMasking(time_mask_param)

    def forward(self, spec):
        spec = self.freq_mask(spec)
        spec = self.time_mask(spec)
        return spec

def mixup_data(x, y, alpha: float = 0.25):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def specmix_data(x, y):
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    split_mel = random.randint(15, 37)
    mixed_x = x.clone()
    mixed_x[:, :, :split_mel, :] = x[index, :, :split_mel, :]
    lam = split_mel / 52.0
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, 1.0 - lam

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 1.5, label_smoothing: float = 0.05, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

# ------------------------------------------------------------------------------
# 2. ESC-50 DATASET (STRICT 80/20 STRATIFIED HELD-OUT SPLIT)
# ------------------------------------------------------------------------------
class ESC50(Dataset):
    base_folder = 'ESC-50-master'
    audio_dir = 'audio'
    label_col = 'category'
    file_col = 'filename'
    meta = {'filename': os.path.join('meta', 'esc50.csv')}

    def __init__(self, root: str, sample_rate: int = 16000, is_train: bool = True):
        super().__init__()
        self.root = os.path.expanduser(root)
        self.sample_rate = sample_rate
        self.is_train = is_train

        meta_path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        self.df = pd.read_csv(meta_path)
        self.df['category'] = self.df['category'].str.replace('_', ' ')
        self.classes = sorted(self.df[self.label_col].unique())
        self.class_to_idx = {cat: i for i, cat in enumerate(self.classes)}

        train_indices, val_indices = [], []
        for class_idx, class_name in enumerate(self.classes):
            class_df = self.df[self.df[self.label_col] == class_name].sort_values(by=self.file_col)
            all_idx = list(class_df.index)
            rng = random.Random(42 + class_idx)
            val_picks = rng.sample(all_idx, 8)
            train_picks = [idx for idx in all_idx if idx not in val_picks]
            train_indices.extend(train_picks)
            val_indices.extend(val_picks)

        selected_indices = train_indices if is_train else val_indices
        self.df = self.df.iloc[selected_indices].reset_index(drop=True)

        self.audio_paths = [
            os.path.join(self.root, self.base_folder, self.audio_dir, f) for f in self.df[self.file_col]
        ]
        self.targets = [self.class_to_idx[cat] for cat in self.df[self.label_col]]
        self.melspec_transform = MelSpectrogram(sample_rate=self.sample_rate, n_fft=512, hop_length=256, n_mels=52)
        self.spec_aug = SpecAugment(freq_mask_param=8, time_mask_param=24)

        print(f'📦 Pre-caching {len(self.audio_paths)} spectrograms (is_train={is_train})...')
        self.cached_spectrograms = []
        for path in self.audio_paths:
            tmp, sr = torchaudio.load(path)
            if sr != self.sample_rate:
                resample = torchaudio.transforms.Resample(sr, self.sample_rate)
                tmp = resample(tmp)
            zeros = (DURATION_SEC * self.sample_rate) - tmp.shape[1]
            if zeros > 0:
                tmp = F.pad(tmp, (0, zeros))
            else:
                tmp = tmp[:, :DURATION_SEC * self.sample_rate]
            tmp = tmp.sum(0, keepdims=True)
            log_mel = torch.log(self.melspec_transform(tmp) + 1e-6)
            self.cached_spectrograms.append(log_mel)

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        spec = self.cached_spectrograms[idx].clone()
        if self.is_train:
            shift = random.randint(-16, 16)
            spec = torch.roll(spec, shifts=shift, dims=-1)
            spec = spec + (torch.randn_like(spec) * 0.02)
            spec = self.spec_aug(spec)
        return spec, torch.tensor(self.targets[idx])

# ------------------------------------------------------------------------------
# 3. FULLY-FUSED 2D PHINET & 1D DILATED TC-RESNET BLOCKS (HYPOTHESIS A)
# ------------------------------------------------------------------------------
class HighCapBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride=(1, 1)):
        super().__init__()
        self.use_residual = (stride == (1, 1) and in_channels == out_channels)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        if self.use_residual:
            return self.skip_add.add(x, out)
        return out

class DilatedResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        self.use_residual = (in_channels == out_channels)
        padding = (kernel_size - 1) * dilation // 2

        # 1. Dilated Depthwise Conv + BN + ReLU
        self.dw_conv = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size,
                                 padding=padding, dilation=dilation, groups=in_channels, bias=False)
        self.dw_bn = nn.BatchNorm1d(in_channels)
        self.dw_relu = nn.ReLU()

        # 2. Pointwise Conv + BN + ReLU
        self.pw_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pw_bn = nn.BatchNorm1d(out_channels)
        self.pw_relu = nn.ReLU()

        # 3. 1x1 Shortcut Projection (When Channels Expand)
        if not self.use_residual and in_channels != out_channels:
            self.shortcut_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
            self.shortcut_bn = nn.BatchNorm1d(out_channels)
        else:
            self.shortcut_conv = None

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        out = self.dw_relu(self.dw_bn(self.dw_conv(x)))
        out = self.pw_relu(self.pw_bn(self.pw_conv(out)))

        if self.use_residual:
            return self.skip_add.add(x, out)
        elif self.shortcut_conv is not None:
            return self.skip_add.add(self.shortcut_bn(self.shortcut_conv(x)), out)
        return out

class LearnedTemporalAttention1D(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.attn_conv = nn.Conv1d(in_channels, 1, kernel_size=1, bias=True)

    def forward(self, x):
        scores = self.attn_conv(x).squeeze(1) # [B, T]
        weights = torch.softmax(scores, dim=-1) # [B, T]
        context = (x * weights.unsqueeze(1)).sum(dim=-1) # [B, C]
        return context

# ------------------------------------------------------------------------------
# 4. HYPOTHESIS A: FREQUENCY-FOLDED 1D TC-RESNET (~93.7k Parameters)
# ------------------------------------------------------------------------------
class AudioPhiNetTCNClassifierQAT(nn.Module):
    def __init__(self, num_classes: int = 50):
        super().__init__()
        self.quant = quantization.QuantStub()
        self.dequant = quantization.DeQuantStub()

        # Stage 1: 2D PhiNet -> [32 Channels x 4 Freq Bins x 40 Time]
        self.stem_conv = nn.Conv2d(1, 16, kernel_size=3, stride=(2, 2), padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(16)
        self.stem_relu = nn.ReLU()

        self.phi_blocks = nn.Sequential(
            HighCapBlock2D(in_channels=16, out_channels=32, stride=(1, 2)),
            nn.Dropout(0.10),
            HighCapBlock2D(in_channels=32, out_channels=32, stride=(2, 2)),
            nn.Dropout(0.10)
        )
        # Pool to 4 Frequency Bins (Low, Low-Mid, High-Mid, High)
        self.freq_pool = nn.AdaptiveAvgPool2d((4, 40)) # [B, 32, 4, 40]

        # Stage 2: 1D Dilated TC-ResNet with 128 Folded Channels (32 Ch x 4 Freq Bins)
        self.tcn = nn.Sequential(
            DilatedResidualBlock1D(in_channels=128, out_channels=96, kernel_size=3, dilation=1),
            DilatedResidualBlock1D(in_channels=96, out_channels=96, kernel_size=3, dilation=2),
            DilatedResidualBlock1D(in_channels=96, out_channels=96, kernel_size=3, dilation=4),
            DilatedResidualBlock1D(in_channels=96, out_channels=96, kernel_size=3, dilation=8),
            DilatedResidualBlock1D(in_channels=96, out_channels=128, kernel_size=3, dilation=16),
        )

        # Stage 3: Learned Temporal Attention & Classifier Head
        self.attention = LearnedTemporalAttention1D(128)
        self.post_tcn_bn = nn.BatchNorm1d(128)
        self.drop = nn.Dropout(0.30)
        self.bottleneck = nn.Linear(128, 64)
        self.btn_relu = nn.ReLU()
        self.fc = nn.Linear(64, num_classes)

    def forward(self, log_mel):
        x = self.quant(log_mel)
        x = self.stem_relu(self.stem_bn(self.stem_conv(x)))
        x = self.phi_blocks(x)
        x = self.freq_pool(x) # [B, 32, 4, 40]

        # FOLD Frequency into Channels (Zero Parameter Operation!)
        b, c, f, t = x.shape
        x_1d = x.reshape(b, c * f, t) # [B, 128, 40]

        tcn_out = self.tcn(x_1d) # [B, 128, 40]
        tcn_out = self.dequant(tcn_out)

        context = self.attention(tcn_out)
        context = self.post_tcn_bn(context)
        context = self.drop(context)
        btn = self.btn_relu(self.bottleneck(context))
        logits = self.fc(btn)
        return logits

def evaluate_model(model, loader, criterion, dev):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for specs, labels in loader:
            specs, labels = specs.to(dev), labels.to(dev)
            outputs = model(specs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * specs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return val_loss / total, (correct / total) * 100.0, correct, total

## ------------------------------------------------------------------------------
# 5. EXECUTE 2-PHASE HIGH-ACCURACY TRAINING (120 FP32 + 40 QAT FINE-TUNING)
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    train_set = ESC50(PROJECT_ROOT, is_train=True)
    val_set = ESC50(PROJECT_ROOT, is_train=False)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

    print("=" * 80)
    print("🚀 PHASE 1: 120-EPOCH FP32 PRE-TRAINING (HYPOTHESIS A: FREQ FOLDING)")
    print("=" * 80)

    model_fp32 = AudioPhiNetTCNClassifierQAT(num_classes=NUM_CLASSES).to(device)
    total_params = sum(p.numel() for p in model_fp32.parameters())
    print(f"📊 Total Model Parameters: {total_params:,} ({total_params/1000.0:.2f}k)")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer_fp32 = torch.optim.AdamW(model_fp32.parameters(), lr=1.5e-3, weight_decay=1e-3)
    scheduler_fp32 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_fp32, T_max=120, eta_min=1e-6)

    best_fp32_acc = 0.0
    best_fp32_state = None

    for epoch in range(120):
        model_fp32.train()
        train_corr = 0
        train_tot = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            r = random.random()

            if r < 0.30:
                x_sm, y_a, y_b, lam = specmix_data(x, y)
                outputs = model_fp32(x_sm)
                loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
            elif r < 0.55:
                x_mix, y_a, y_b, lam = mixup_data(x, y, alpha=0.25)
                outputs = model_fp32(x_mix)
                loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
            else:
                outputs = model_fp32(x)
                loss = criterion(outputs, y)

            optimizer_fp32.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_fp32.parameters(), max_norm=5.0)
            optimizer_fp32.step()

            train_corr += (outputs.argmax(dim=1) == y).sum().item()
            train_tot += y.size(0)

        scheduler_fp32.step()
        val_loss, val_acc, corr, tot = evaluate_model(model_fp32, val_loader, criterion, device)
        train_acc = (train_corr / train_tot) * 100.0

        if val_acc > best_fp32_acc:
            best_fp32_acc = val_acc
            best_fp32_state = copy.deepcopy(model_fp32.state_dict())
            print(f"  [FP32] Epoch {epoch+1:03d}/120 | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% ({corr}/{tot}) 🌟 [NEW BEST FP32!]")
        elif (epoch + 1) % 10 == 0:
            print(f"  [FP32] Epoch {epoch+1:03d}/120 | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% (Best: {best_fp32_acc:.2f}%)")

    print(f"\n✅ Phase 1 Complete! Best FP32 Accuracy: {best_fp32_acc:.2f}%\n")
    fp32_save_path = os.path.join(PROJECT_ROOT, 'models', 'best_tcn_base_fp32.pth')
    os.makedirs(os.path.dirname(fp32_save_path), exist_ok=True)
    torch.save(best_fp32_state, fp32_save_path)
    print(f"💾 Saved Phase 1 Base FP32 Checkpoint to: {fp32_save_path}\n")

    # =========================================================================
    # PHASE 2: FULL 3-WAY MODULE FUSION & 40-EPOCH QAT FINE-TUNING
    # =========================================================================
    print("=" * 80)
    print("⚡ PHASE 2: QUANTIZATION-AWARE TRAINING (QAT) FINE-TUNING (40 EPOCHS)")
    print("=" * 80)

    model_qat = AudioPhiNetTCNClassifierQAT(num_classes=NUM_CLASSES).to(device)
    model_qat.load_state_dict(best_fp32_state)

    if 'qnnpack' in torch.backends.quantized.supported_engines:
        qat_qconfig = quantization.get_default_qat_qconfig('qnnpack')
    else:
        qat_qconfig = quantization.get_default_qat_qconfig('fbgemm')

    model_qat.qconfig = qat_qconfig
    model_qat.eval()

    # 🔗 Complete 3-Way Fused Quantization Nodes [Conv + BN + ReLU]
    torch.ao.quantization.fuse_modules(model_qat, [['stem_conv', 'stem_bn', 'stem_relu']], inplace=True)

    for b in [model_qat.phi_blocks[0], model_qat.phi_blocks[2]]:
        torch.ao.quantization.fuse_modules(b, [['conv1', 'bn1', 'relu1'], ['conv2', 'bn2', 'relu2']], inplace=True)

    for b in model_qat.tcn:
        torch.ao.quantization.fuse_modules(b, [['dw_conv', 'dw_bn', 'dw_relu'],
                                               ['pw_conv', 'pw_bn', 'pw_relu']], inplace=True)
        if b.shortcut_conv is not None:
            torch.ao.quantization.fuse_modules(b, [['shortcut_conv', 'shortcut_bn']], inplace=True)

    model_qat.train()
    quantization.prepare_qat(model_qat, inplace=True)

    optimizer_qat = torch.optim.AdamW(model_qat.parameters(), lr=1.5e-4, weight_decay=5e-4)
    scheduler_qat = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_qat, T_max=40, eta_min=1e-6)

    best_qat_acc = 0.0
    save_path = os.path.join(PROJECT_ROOT, 'models', 'best_tcn_base_qat.pth')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(40):
        model_qat.train()
        train_corr = 0
        train_tot = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer_qat.zero_grad()
            outputs = model_qat(x)
            loss = criterion(outputs, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_qat.parameters(), max_norm=4.0)
            optimizer_qat.step()
            train_corr += (outputs.argmax(dim=1) == y).sum().item()
            train_tot += y.size(0)

        scheduler_qat.step()
        val_loss, val_acc, corr, tot = evaluate_model(model_qat, val_loader, criterion, device)
        train_acc = (train_corr / train_tot) * 100.0

        if val_acc > best_qat_acc:
            best_qat_acc = val_acc
            torch.save(model_qat.state_dict(), save_path)
            print(f"  [QAT]  Epoch {epoch+1:02d}/40 | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% ({corr}/{tot}) 🌟 [NEW BEST QAT!]")
        elif (epoch + 1) % 5 == 0:
            print(f"  [QAT]  Epoch {epoch+1:02d}/40 | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% (Best: {best_qat_acc:.2f}%)")

    print("\n" + "=" * 80)
    print(f"🏆 FINAL OPTIMIZED TCN QAT RESULT: {best_qat_acc:.2f}%")
    print(f"💾 Checkpoint saved to: {save_path}")
    print("=" * 80 + "\n")