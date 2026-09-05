#!/usr/bin/env python3
"""
================================================================================
🔍 ZERO-LEAKAGE AUDIT: BASE TCN & SLIM TCN ON OFFICIAL FOLD-5 (400 CLIPS)
================================================================================
Audits the existing TCN checkpoints on Karol Piczak's official held-out Fold-5:
  1. Base 1D Dilated TC-ResNet (~125k params, ~92 KB Flash)
  2. Channel-Pruned Slim TC-ResNet (~48k params, <50 KB Flash)
================================================================================
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch.ao.quantization as quantization

PROJECT_ROOT = "/home/acar/new_task"
sys.path.insert(0, PROJECT_ROOT)

from training.train_tcn_base_local import AudioPhiNetTCNClassifierQAT, NUM_CLASSES
from training.train_tcn_channel_prune import AudioPhiNetSlimTCNClassifierQAT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 80)
print(f"🚀 RUNNING TCN AUDIT ON DEVICE: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
print("🔒 DATASET: OFFICIAL KAROL PICZAK FOLD-5 (ZERO DATA LEAKAGE)")
print("=" * 80)

# 1. Load Precomputed Official Fold-5 Test Set
SPECS_PATH = os.path.join(PROJECT_ROOT, "official_fold5_test_specs_400.npy")
LABELS_PATH = os.path.join(PROJECT_ROOT, "official_fold5_test_labels_400.npy")

if not os.path.exists(SPECS_PATH) or not os.path.exists(LABELS_PATH):
    print("❌ Error: Fold-5 test set files not found!")
    sys.exit(1)

test_specs = np.load(SPECS_PATH)   # [400, 52, 313]
test_labels = np.load(LABELS_PATH) # [400]

# Expand channel dimension: [400, 1, 52, 313]
test_specs_tensor = torch.from_numpy(test_specs).unsqueeze(1).float()
test_labels_tensor = torch.from_numpy(test_labels).long()

test_dataset = TensorDataset(test_specs_tensor, test_labels_tensor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"📊 Loaded {len(test_dataset)} official unseen Fold-5 test spectrograms.")

# Evaluation Helper
def evaluate_model(model, desc):
    model.eval()
    correct1, correct3, total = 0, 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred1 = out.argmax(dim=1)
            correct1 += (pred1 == y).sum().item()
            pred3 = out.topk(3, dim=1)[1]
            correct3 += (pred3 == y.unsqueeze(-1)).any(dim=1).sum().item()
            total += y.size(0)
    top1 = (correct1 / total) * 100.0
    top3 = (correct3 / total) * 100.0
    print(f"  • {desc:<35s}: Top-1 = {top1:6.2f}% ({correct1}/{total}) | Top-3 = {top3:6.2f}%")
    return top1, top3, correct1

# 2. Audit Model A: Base 1D Dilated TC-ResNet (~125k)
print("\n" + "=" * 80)
print("📦 AUDITING MODEL A: BASE 1D DILATED TC-RESNET (~125k / ~92 KB FLASH)")
print("=" * 80)

base_qat_path = os.path.join(PROJECT_ROOT, "models", "best_tcn_125k_base_qat.pth")
if os.path.exists(base_qat_path):
    base_model = AudioPhiNetTCNClassifierQAT(num_classes=NUM_CLASSES)
    base_model.eval()
    
    qat_qconfig = quantization.get_default_qat_qconfig('fbgemm')
    base_model.qconfig = qat_qconfig
    base_model.attention.qconfig = None
    base_model.post_tcn_bn.qconfig = None
    base_model.bottleneck.qconfig = None
    base_model.fc.qconfig = None

    torch.ao.quantization.fuse_modules(base_model, [['stem_conv', 'stem_bn', 'stem_relu']], inplace=True)
    for b in [base_model.phi_blocks[0], base_model.phi_blocks[2]]:
        torch.ao.quantization.fuse_modules(b, [['conv1', 'bn1', 'relu1'], ['conv2', 'bn2', 'relu2']], inplace=True)
    for b in base_model.tcn:
        torch.ao.quantization.fuse_modules(b, [['dw_conv', 'dw_bn', 'dw_relu'], ['pw_conv', 'pw_bn', 'pw_relu']], inplace=True)
        if b.shortcut_conv is not None:
            torch.ao.quantization.fuse_modules(b, [['shortcut_conv', 'shortcut_bn']], inplace=True)

    base_model.train()
    quantization.prepare_qat(base_model, inplace=True)
    sd = torch.load(base_qat_path, map_location="cpu", weights_only=False)
    base_model.load_state_dict(sd, strict=False)
    base_model.eval().to(device)

    # FP32 Float Mode
    base_model.apply(torch.ao.quantization.disable_fake_quant)
    base_model.apply(torch.ao.quantization.disable_observer)
    top1_base_fp32, top3_base_fp32, _ = evaluate_model(base_model, "Base TCN (Pre-Quant FP32)")

    # INT8 QAT Mode
    base_model.apply(torch.ao.quantization.enable_fake_quant)
    base_model.apply(torch.ao.quantization.disable_observer)
    top1_base_int8, top3_base_int8, _ = evaluate_model(base_model, "Base TCN (Simulated INT8 QAT)")
else:
    print(f"⚠️ Checkpoint not found: {base_qat_path}")
    top1_base_fp32, top1_base_int8 = 0.0, 0.0

# 3. Audit Model B: Channel-Pruned Slim TC-ResNet (<50 KB Flash)
print("\n" + "=" * 80)
print("📦 AUDITING MODEL B: SLIM CHANNEL-PRUNED TC-RESNET (<50 KB FLASH)")
print("=" * 80)

slim_qat_path = os.path.join(PROJECT_ROOT, "models", "best_tcn_slim_qat.pth")
if os.path.exists(slim_qat_path):
    slim_model = AudioPhiNetSlimTCNClassifierQAT(num_classes=NUM_CLASSES)
    slim_model.eval()

    qat_qconfig = quantization.get_default_qat_qconfig('fbgemm')
    slim_model.qconfig = qat_qconfig
    slim_model.attention.qconfig = None
    slim_model.post_tcn_bn.qconfig = None
    slim_model.bottleneck.qconfig = None
    slim_model.fc.qconfig = None

    torch.ao.quantization.fuse_modules(slim_model, [['stem_conv', 'stem_bn', 'stem_relu']], inplace=True)
    for b in [slim_model.phi_blocks[0], slim_model.phi_blocks[2]]:
        torch.ao.quantization.fuse_modules(b, [['conv1', 'bn1', 'relu1'], ['conv2', 'bn2', 'relu2']], inplace=True)
    for b in slim_model.tcn:
        torch.ao.quantization.fuse_modules(b, [['dw_conv', 'dw_bn', 'dw_relu'], ['pw_conv', 'pw_bn', 'pw_relu']], inplace=True)
        if b.shortcut_conv is not None:
            torch.ao.quantization.fuse_modules(b, [['shortcut_conv', 'shortcut_bn']], inplace=True)

    slim_model.train()
    quantization.prepare_qat(slim_model, inplace=True)
    sd = torch.load(slim_qat_path, map_location="cpu", weights_only=False)
    slim_model.load_state_dict(sd, strict=False)
    slim_model.eval().to(device)

    # FP32 Float Mode
    slim_model.apply(torch.ao.quantization.disable_fake_quant)
    slim_model.apply(torch.ao.quantization.disable_observer)
    top1_slim_fp32, top3_slim_fp32, _ = evaluate_model(slim_model, "Slim TCN (Pre-Quant FP32)")

    # INT8 QAT Mode
    slim_model.apply(torch.ao.quantization.enable_fake_quant)
    slim_model.apply(torch.ao.quantization.disable_observer)
    top1_slim_int8, top3_slim_int8, _ = evaluate_model(slim_model, "Slim TCN (Simulated INT8 QAT)")
else:
    print(f"⚠️ Checkpoint not found: {slim_qat_path}")
    top1_slim_fp32, top1_slim_int8 = 0.0, 0.0

print("\n" + "=" * 80)
print("📊 OFFICIAL ESC-50 FOLD-5 ZERO-LEAKAGE TCN BENCHMARK SUMMARY:")
print("=" * 80)
print(f"{'Model Architecture':<32} | {'Weights Flash':<15} | {'FP32 Acc (Fold-5)':<18} | {'INT8 Acc (Fold-5)'}")
print("-" * 80)
print(f"{'Base 1D Dilated TC-ResNet':<32} | {'~92.86 KB':<15} | {top1_base_fp32:6.2f}%{'':<11} | {top1_base_int8:6.2f}%")
print(f"{'Slim Channel-Pruned TC-ResNet':<32} | {'47.57 KB (<50KB)':<15} | {top1_slim_fp32:6.2f}%{'':<11} | {top1_slim_int8:6.2f}%")
print("=" * 80)
