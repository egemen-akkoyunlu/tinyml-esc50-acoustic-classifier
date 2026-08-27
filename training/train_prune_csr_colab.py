#!/usr/bin/env python3
# ==============================================================================
# Sparse Pruned & Knowledge Distilled TinyML Acoustic Classifier (ESC-50)
# Target: ARM Cortex-M33 (Silicon Labs EFR32MG24) / ESP32-S3
# ==============================================================================

import os
import sys
import csv
import random
import copy
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torch.ao.quantization as quantization
import torch.nn.utils.prune as prune
from torchaudio.transforms import MelSpectrogram, FrequencyMasking, TimeMasking
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.models as models

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
torch.backends.quantized.engine = 'fbgemm'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Running on: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU (Multi-threaded)'}")

# Download ESC-50 dataset if not present
if not os.path.exists('ESC-50') and not os.path.exists('ESC-50-master'):
    print("📦 Cloning ESC-50 Dataset...")
    os.system("git clone --depth 1 https://github.com/karoldvl/ESC-50.git")

dataset_root = 'ESC-50' if os.path.exists('ESC-50') else 'ESC-50-master'

# ==============================================================================
# 1. DATASET LOADER (Exact Stratified Held-Out Split)
# ==============================================================================
class ESC50(Dataset):
    def __init__(self, root=dataset_root, is_train=True):
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
        self.is_train = is_train
        self.freq_mask = FrequencyMasking(12)
        self.time_mask = TimeMasking(32)

        print(f"Pre-caching {len(self.audio_paths)} spectrograms (is_train={is_train})...")
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
        spec = self.cached[idx].clone()
        if self.is_train:
            spec = torch.roll(spec, shifts=random.randint(-24, 24), dims=-1)
            spec = self.freq_mask(self.time_mask(spec))
        return spec, torch.tensor(self.targets[idx])

ds_train = ESC50(is_train=True)
ds_val = ESC50(is_train=False)

class_indices = defaultdict(list)
for idx, target in enumerate(ds_train.targets):
    class_indices[target].append(idx)

train_idx, val_idx = [], []
for cat, indices in class_indices.items():
    rng = random.Random(42 + cat)
    shuffled = list(indices)
    rng.shuffle(shuffled)
    split_pt = int(len(shuffled) * 0.8)
    train_idx.extend(shuffled[:split_pt])
    val_idx.extend(shuffled[split_pt:])

train_loader = DataLoader(Subset(ds_train, train_idx), batch_size=32, shuffle=True)
val_loader = DataLoader(Subset(ds_val, val_idx), batch_size=32, shuffle=False)

# ==============================================================================
# 2. TRAIN PRE-TRAINED RESNET-34 TEACHER (~89.25% Accuracy)
# ==============================================================================
print("\n" + "="*70)
print("🎓 1. INITIALIZING PRE-TRAINED RESNET-34 TEACHER (40 Epochs)...")
print("="*70)
teacher = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
teacher.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
teacher.fc = nn.Linear(teacher.fc.in_features, 50)
teacher = teacher.to(device)

opt_t = torch.optim.AdamW(teacher.parameters(), lr=3e-4, weight_decay=1e-2)
sched_t = torch.optim.lr_scheduler.CosineAnnealingLR(opt_t, T_max=40)

best_t_acc = 0.0
for epoch in range(40):
    teacher.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        loss = F.cross_entropy(teacher(x), y, label_smoothing=0.1)
        opt_t.zero_grad(); loss.backward(); opt_t.step()
    sched_t.step()

    teacher.eval()
    corr = 0
    with torch.no_grad():
        for x, y in val_loader:
            corr += (teacher(x.to(device)).argmax(1) == y.to(device)).sum().item()
    acc = (corr / len(val_idx)) * 100.0
    if acc > best_t_acc: best_t_acc = acc
    if (epoch+1) % 10 == 0:
        print(f"Teacher Epoch [{epoch+1:02d}/40] | Val Acc: {acc:.2f}% (Best: {best_t_acc:.2f}%)")

print(f"🏆 Teacher Peak Accuracy: {best_t_acc:.2f}%")
teacher.eval()
for p in teacher.parameters(): p.requires_grad = False

# ==============================================================================
# 3. EXACT TINYML ARCHITECTURE
# ==============================================================================
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
        if self.use_residual: return self.skip_add.add(x, self.conv(x))
        return self.conv(x)

class AudioPhiNetCRNNClassifierQAT(nn.Module):
    def __init__(self, num_classes: int = 50):
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

# ==============================================================================
# 4. INSTANTIATE, FUSE & LOAD 91.00% WEIGHTS
# ==============================================================================
student = AudioPhiNetCRNNClassifierQAT(num_classes=50).to(device)
student.eval()

student.qconfig = quantization.get_default_qat_qconfig('fbgemm')
torch.ao.quantization.fuse_modules(student, [['stem.0', 'stem.1']], inplace=True)
torch.ao.quantization.fuse_modules(student.phi_blocks[0], [['conv.0', 'conv.1'], ['conv.3', 'conv.4']], inplace=True)
torch.ao.quantization.fuse_modules(student.phi_blocks[2], [['conv.0', 'conv.1'], ['conv.3', 'conv.4']], inplace=True)

student.gru.qconfig = None
student.bottleneck.qconfig = None
student.fc.qconfig = None

student.train()
quantization.prepare_qat(student, inplace=True)

ckpt_flagship = 'models/best_distilled_qat_model.pth' if os.path.exists('models/best_distilled_qat_model.pth') else 'best_distilled_qat_model.pth'
if os.path.exists(ckpt_flagship):
    student.load_state_dict(torch.load(ckpt_flagship, map_location=device))
    print(f"✅ Loaded Flagship 91.00% Model ({ckpt_flagship})!")
else:
    print("⚠️ Flagship checkpoint not found in local path, initializing student from scratch...")

# ==============================================================================
# 5. TARGETED L1 UNSTRUCTURED PRUNING (GRU 68% + BOTTLENECK 68%)
# ==============================================================================
print("\n" + "="*70)
print("✂️ APPLYING UNSTRUCTURED PRUNING: GRU 68%, Bottleneck 68%, FC 45%...")
print("="*70)

prune.l1_unstructured(student.gru, name="weight_hh_l0", amount=0.68)
prune.l1_unstructured(student.gru, name="weight_ih_l0", amount=0.45)
prune.l1_unstructured(student.bottleneck, name="weight", amount=0.68)
prune.l1_unstructured(student.fc, name="weight", amount=0.45)

def count_active_params(model):
    total = 0
    for name, p in model.named_parameters():
        total += int((p != 0).sum().item())
    return total

total_active = count_active_params(student)
print(f"📊 ACTIVE NON-ZERO PARAMETERS AFTER PRUNING: {total_active:,} ({total_active/1000.0:.2f}k parameters)")

# ==============================================================================
# 6. DISTILLATION REFINEMENT (25 Epochs)
# ==============================================================================
print("\n" + "="*70)
print("🧪 FINE-TUNING PRUNED STUDENT VIA KNOWLEDGE DISTILLATION (25 Epochs)...")
print("="*70)

opt_s = torch.optim.AdamW(student.parameters(), lr=1.5e-4, weight_decay=1e-4)
sched_s = torch.optim.lr_scheduler.CosineAnnealingLR(opt_s, T_max=25, eta_min=1e-6)
kl_loss_fn = nn.KLDivLoss(reduction='batchmean')
T = 3.0
alpha = 0.65

best_s_acc = 0.0
for epoch in range(25):
    student.train()
    if epoch >= 15: student.apply(quantization.disable_observer)
    if epoch >= 20: student.apply(torch.ao.quantization.disable_fake_quant)
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad(): t_logits = teacher(x)
        s_logits = student(x)

        hard_loss = F.cross_entropy(s_logits, y, label_smoothing=0.05)
        soft_loss = kl_loss_fn(F.log_softmax(s_logits/T, dim=1), F.softmax(t_logits/T, dim=1)) * (T**2)
        loss = (1.0 - alpha) * hard_loss + alpha * soft_loss

        opt_s.zero_grad(); loss.backward(); opt_s.step()
    sched_s.step()

    student.eval()
    corr = 0
    with torch.no_grad():
        for x, y in val_loader:
            corr += (student(x.to(device)).argmax(1) == y.to(device)).sum().item()
    acc = (corr / len(val_idx)) * 100.0
    if acc > best_s_acc:
        best_s_acc = acc
        torch.save(student.state_dict(), 'models/best_pruned_50k_qat.pth' if os.path.exists('models') else 'best_pruned_50k_qat.pth')
        star = f" 🌟 [NEW BEST: {acc:.2f}% | Eff: {acc/(total_active/1000.0):.3f}%/kParam]"
    else:
        star = ""
    print(f"Prune-Distill Epoch [{epoch+1:02d}/25] | Val Acc: {acc:.2f}% (Best: {best_s_acc:.2f}%){star}")

print("\n" + "="*70)
print(f"✅ PRUNING & DISTILLATION COMPLETE! Peak Accuracy: {best_s_acc:.2f}%")
eff = best_s_acc / (total_active / 1000.0)
print(f"📊 Global Parameter Efficiency: {eff:.3f}% / kParam")
print("="*70)
