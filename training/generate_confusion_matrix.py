#!/usr/bin/env python3
# ==============================================================================
# 📊 ESC-50 50x50 CONFUSION MATRIX & PER-CLASS ACCURACY GENERATOR
# Evaluates: best_distilled_qat_model.pth (91.00% Flagship Model)
# Dataset: ESC-50 Validation Set (400 Disjoint Audio Clips)
# ==============================================================================

import os
import csv
import random
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torch.ao.quantization as quantization
from torchaudio.transforms import MelSpectrogram
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

curr_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(curr_dir, "..")) if os.path.basename(curr_dir) in ["training", "export"] else curr_dir

checkpoint_path = os.path.join(project_root, "models", "best_distilled_qat_model.pth")
if not os.path.exists(checkpoint_path):
    checkpoint_path = os.path.join(project_root, "best_distilled_qat_model.pth")
dataset_root = os.path.join(project_root, "ESC-50-master")
output_png = os.path.join(project_root, "assets", "confusion_matrix_91_5.png")
os.makedirs(os.path.dirname(output_png), exist_ok=True)

# 1. Dataset Loader Matching Training Pipeline Exactly
class ESC50(Dataset):
    def __init__(self, root=dataset_root):
        meta_csv = os.path.join(root, 'meta', 'esc50.csv')
        with open(meta_csv, 'r') as f:
            reader = csv.DictReader(f)
            self.rows = list(reader)

        for r in self.rows:
            r['category'] = r['category'].replace('_', ' ')

        self.classes = sorted(list(set(r['category'] for r in self.rows)))
        self.class_to_idx = {cat: i for i, cat in enumerate(self.classes)}
        self.audio_paths = [os.path.join(root, 'audio', r['filename']) for r in self.rows]
        self.targets = [self.class_to_idx[r['category']] for r in self.rows]
        self.melspec = MelSpectrogram(sample_rate=16000, n_fft=512, hop_length=256, n_mels=52)

        print(f"Pre-caching {len(self.audio_paths)} spectrograms for validation evaluation...")
        self.cached = []
        for p in self.audio_paths:
            tmp, sr = torchaudio.load(p)
            if sr != 16000:
                tmp = torchaudio.transforms.Resample(sr, 16000)(tmp)
            if tmp.shape[1] < 80000:
                tmp = F.pad(tmp, (0, 80000 - tmp.shape[1]))
            else:
                tmp = tmp[:, :80000]
            log_mel = torch.log(self.melspec(tmp.sum(0, keepdims=True)) + 1e-6)
            self.cached.append(log_mel)

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        return self.cached[idx].clone(), torch.tensor(self.targets[idx])

ds = ESC50()
class_indices = defaultdict(list)
for idx, target in enumerate(ds.targets):
    class_indices[target].append(idx)

train_idx, val_idx = [], []
for cat, indices in class_indices.items():
    rng = random.Random(42 + cat)
    shuffled = list(indices)
    rng.shuffle(shuffled)
    split_pt = int(len(shuffled) * 0.8)
    train_idx.extend(shuffled[:split_pt])
    val_idx.extend(shuffled[split_pt:])

val_loader = DataLoader(Subset(ds, val_idx), batch_size=32, shuffle=False)
print(f"✅ Loaded {len(val_idx)} Validation Samples across 50 Classes (8 samples/class).")

# 2. Model Architecture
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
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        if self.use_residual:
            return self.skip_add.add(x, self.conv(x))
        return self.conv(x)

class AudioPhiNetCRNNClassifierQAT(nn.Module):
    def __init__(self, num_classes: int = 50, sample_rate: int = 16000):
        super().__init__()
        self.quant = quantization.QuantStub()
        self.dequant = quantization.DeQuantStub()

        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=(2, 2), padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU6()
        )

        self.phi_blocks = nn.Sequential(
            HighCapBlock(in_channels=16, out_channels=32, stride=(1, 2)),
            nn.Dropout(0.2),
            HighCapBlock(in_channels=32, out_channels=48, stride=(2, 2)),
            nn.Dropout(0.2),
        )

        self.conv_compress = nn.Conv2d(48, 32, kernel_size=1, bias=False)
        self.freq_pool = nn.AvgPool2d(kernel_size=(13, 1))
        self.pre_gru_bn = nn.BatchNorm1d(32)

        hidden_dim = 160
        self.gru = nn.GRU(input_size=32, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.post_gru_bn = nn.BatchNorm1d(hidden_dim)
        self.drop = nn.Dropout(0.5)
        self.bottleneck = nn.Linear(hidden_dim, 128)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, log_mel):
        log_mel = self.quant(log_mel)
        x_stem = self.stem(log_mel)
        features = self.phi_blocks(x_stem)
        compressed = self.conv_compress(features)
        freq_pooled = self.freq_pool(compressed)[:, :, :, :39]
        freq_pooled = self.dequant(freq_pooled)

        b, c, f, t = freq_pooled.shape
        seq_in = freq_pooled.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
        seq_in = seq_in.permute(0, 2, 1)
        seq_in = self.pre_gru_bn(seq_in)
        seq_in = seq_in.permute(0, 2, 1)

        rnn_out, _ = self.gru(seq_in)

        attn_scores = rnn_out.mean(dim=-1)
        attn_weights = torch.softmax(attn_scores, dim=1)
        rnn_pooled = (rnn_out * attn_weights.unsqueeze(-1)).sum(dim=1)

        rnn_pooled = self.post_gru_bn(rnn_pooled)
        rnn_pooled = self.drop(rnn_pooled)
        rnn_compressed = self.bottleneck(rnn_pooled)
        rnn_compressed = F.relu6(rnn_compressed)
        out = self.fc(rnn_compressed)
        return out

model = AudioPhiNetCRNNClassifierQAT(num_classes=50)
model.eval()
model.qconfig = quantization.get_default_qat_qconfig('fbgemm')
torch.ao.quantization.fuse_modules(model, [['stem.0', 'stem.1']], inplace=True)
torch.ao.quantization.fuse_modules(model.phi_blocks[0], [['conv.0', 'conv.1'], ['conv.3', 'conv.4']], inplace=True)
torch.ao.quantization.fuse_modules(model.phi_blocks[2], [['conv.0', 'conv.1'], ['conv.3', 'conv.4']], inplace=True)
model.gru.qconfig = None
model.bottleneck.qconfig = None
model.fc.qconfig = None
model.train()
quantization.prepare_qat(model, inplace=True)

model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
model.eval()

# 3. Evaluate Entire Validation Set
all_preds = []
all_targets = []

with torch.no_grad():
    for x, y in val_loader:
        logits = model(x)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(y.cpu().numpy())

all_preds = np.array(all_preds)
all_targets = np.array(all_targets)

total_correct = (all_preds == all_targets).sum()
overall_acc = (total_correct / len(all_targets)) * 100.0

cm = confusion_matrix(all_targets, all_preds, labels=range(50))

print("=" * 80)
print(f"🏆 OVERALL VALIDATION ACCURACY: {total_correct}/{len(all_targets)} ({overall_acc:.2f}%)")
print("=" * 80)

# Per-class accuracy
per_class_acc = cm.diagonal() / cm.sum(axis=1) * 100.0

perfect_classes = []
imperfect_classes = []

for i, cat_name in enumerate(ds.classes):
    correct_count = cm[i, i]
    total_count = cm.sum(axis=1)[i]
    acc = per_class_acc[i]
    if acc == 100.0:
        perfect_classes.append(cat_name)
    else:
        imperfect_classes.append((cat_name, correct_count, total_count, acc))

print(f"\n🌟 PERFECT 100% ACCURACY CLASSES ({len(perfect_classes)}/50):")
print(", ".join(perfect_classes[:10]) + ", ...")

print(f"\n⚠️ CLASSES WITH MINOR CONFUSION ({len(imperfect_classes)}/50):")
imperfect_classes.sort(key=lambda x: x[3])
for name, c, t, acc in imperfect_classes[:10]:
    print(f"  • {name:<22s}: {c}/{t} ({acc:5.1f}%)")

# Keyboard typing specifics
kb_idx = ds.class_to_idx["keyboard typing"]
print("\n" + "-" * 60)
print(f"🔍 DEEP-DIVE: 'keyboard typing' (Class {kb_idx}):")
print(f"   • Correct Predictions : {cm[kb_idx, kb_idx]}/{cm.sum(axis=1)[kb_idx]} ({per_class_acc[kb_idx]:.1f}%)")
for pred_k in range(50):
    if pred_k != kb_idx and cm[kb_idx, pred_k] > 0:
        print(f"   • Confused with '{ds.classes[pred_k]}': {cm[kb_idx, pred_k]} clip(s)")
print("-" * 60)

# 4. Generate High-Res Heatmap Plot
plt.figure(figsize=(24, 20))
sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', xticklabels=ds.classes, yticklabels=ds.classes, cbar=True)
plt.title(f"ESC-50 50-Class Confusion Matrix (Overall Accuracy: {overall_acc:.2f}%)", fontsize=18, pad=20)
plt.xlabel("Predicted Class", fontsize=14, labelpad=10)
plt.ylabel("Ground Truth Class", fontsize=14, labelpad=10)
plt.xticks(rotation=90, fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.savefig(output_png, dpi=300)
plt.close()

print(f"\n🎨 High-Resolution Confusion Matrix Heatmap saved to: {output_png}")
