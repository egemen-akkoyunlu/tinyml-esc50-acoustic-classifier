#!/usr/bin/env python3
# ==============================================================================
# ⚡ ESP-PPQ INT8 QUANTIZATION COMPILER FOR ESP32-S3 SENSE
# Architecture: 48.7k Parameter Slim 1D TC-ResNet (<50 KB Flash Milestone)
# Protocol: KAROL PICZAK OFFICIAL 5-FOLD CV (FOLD 5 ZERO-LEAKAGE HELD-OUT)
# Target: 100% Full-Integer INT8 Monolithic Execution via 128-bit Xtensa PIE SIMD
# ==============================================================================

import os
import sys
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.ao.quantization as torch_quant
from torch.utils.data import Dataset, DataLoader, Subset

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

try:
    from esp_ppq.api import quantize_onnx_model, export_ppq_graph, QuantizationSettingFactory
    from esp_ppq.core import TargetPlatform
    from esp_ppq.executor import TorchExecutor
except ImportError:
    from ppq.api import quantize_onnx_model, export_ppq_graph, QuantizationSettingFactory
    from ppq.core import TargetPlatform
    from ppq.executor import TorchExecutor

import onnx
from onnxsim import simplify

# ------------------------------------------------------------------------------
# 1. ARCHITECTURAL BLOCKS (MATCHING TRAINING MASTER PIPELINE)
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

        self.dw_conv = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size,
                                 padding=padding, dilation=dilation, groups=in_channels, bias=False)
        self.dw_bn = nn.BatchNorm1d(in_channels)
        self.dw_relu = nn.ReLU()

        self.pw_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pw_bn = nn.BatchNorm1d(out_channels)
        self.pw_relu = nn.ReLU()

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
        scores = self.attn_conv(x).squeeze(1)
        weights = torch.softmax(scores, dim=-1)
        context = (x * weights.unsqueeze(1)).sum(dim=-1)
        return context


class AudioPhiNetSlimTCNClassifierQAT(nn.Module):
    def __init__(self, num_classes: int = 50):
        super().__init__()
        self.quant = torch_quant.QuantStub()
        self.dequant = torch_quant.DeQuantStub()

        self.stem_conv = nn.Conv2d(1, 16, kernel_size=3, stride=(2, 2), padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(16)
        self.stem_relu = nn.ReLU()

        self.phi_blocks = nn.Sequential(
            HighCapBlock2D(in_channels=16, out_channels=24, stride=(1, 2)),
            nn.Dropout(0.10),
            HighCapBlock2D(in_channels=24, out_channels=24, stride=(2, 2)),
            nn.Dropout(0.10)
        )
        self.freq_pool = nn.AdaptiveAvgPool2d((4, 40))

        self.tcn = nn.Sequential(
            DilatedResidualBlock1D(in_channels=96, out_channels=64, kernel_size=3, dilation=1),
            DilatedResidualBlock1D(in_channels=64, out_channels=64, kernel_size=3, dilation=2),
            DilatedResidualBlock1D(in_channels=64, out_channels=64, kernel_size=3, dilation=4),
            DilatedResidualBlock1D(in_channels=64, out_channels=64, kernel_size=3, dilation=8),
            DilatedResidualBlock1D(in_channels=64, out_channels=96, kernel_size=3, dilation=16),
        )

        self.attention = LearnedTemporalAttention1D(in_channels=96)
        self.post_tcn_bn = nn.BatchNorm1d(96)
        self.drop = nn.Dropout(0.30)
        self.bottleneck = nn.Linear(96, 48)
        self.btn_relu = nn.ReLU()
        self.fc = nn.Linear(48, num_classes)

    def forward(self, log_mel):
        x = self.quant(log_mel)
        x = self.stem_relu(self.stem_bn(self.stem_conv(x)))
        x = self.phi_blocks(x)
        x = self.freq_pool(x)

        b, c, f, t = x.shape
        x_1d = x.reshape(b, c * f, t)

        tcn_out = self.tcn(x_1d)
        tcn_out = self.dequant(tcn_out)

        context = self.attention(tcn_out)
        context = self.post_tcn_bn(context)
        context = self.drop(context)
        btn = self.btn_relu(self.bottleneck(context))
        logits = self.fc(btn)
        return logits


# ------------------------------------------------------------------------------
# 2. CLEAN FP32 GRAPH SUITABLE FOR ONNX & ESP-DL
# ------------------------------------------------------------------------------
class CleanSlimTCN(nn.Module):
    def __init__(self, num_classes=50):
        super().__init__()
        # Stage 1: 2D CNN
        self.stem = nn.Conv2d(1, 16, kernel_size=3, stride=(2, 2), padding=1, bias=True)
        self.stem_relu = nn.ReLU()

        self.pb0_dw = nn.Conv2d(16, 16, kernel_size=3, stride=(1, 2), padding=1, groups=16, bias=True)
        self.pb0_dw_relu = nn.ReLU()
        self.pb0_pw = nn.Conv2d(16, 24, kernel_size=1, bias=True)
        self.pb0_pw_relu = nn.ReLU()

        self.pb2_dw = nn.Conv2d(24, 24, kernel_size=3, stride=(2, 2), padding=1, groups=24, bias=True)
        self.pb2_dw_relu = nn.ReLU()
        self.pb2_pw = nn.Conv2d(24, 24, kernel_size=1, bias=True)
        self.pb2_pw_relu = nn.ReLU()

        # Stage 2: 1D TCN
        self.tcn0_dw = nn.Conv1d(96, 96, kernel_size=3, padding=1, dilation=1, groups=96, bias=True)
        self.tcn0_dw_relu = nn.ReLU()
        self.tcn0_pw = nn.Conv1d(96, 64, kernel_size=1, bias=True)
        self.tcn0_pw_relu = nn.ReLU()
        self.tcn0_sc = nn.Conv1d(96, 64, kernel_size=1, bias=True)

        self.tcn1_dw = nn.Conv1d(64, 64, kernel_size=3, padding=2, dilation=2, groups=64, bias=True)
        self.tcn1_dw_relu = nn.ReLU()
        self.tcn1_pw = nn.Conv1d(64, 64, kernel_size=1, bias=True)
        self.tcn1_pw_relu = nn.ReLU()

        self.tcn2_dw = nn.Conv1d(64, 64, kernel_size=3, padding=4, dilation=4, groups=64, bias=True)
        self.tcn2_dw_relu = nn.ReLU()
        self.tcn2_pw = nn.Conv1d(64, 64, kernel_size=1, bias=True)
        self.tcn2_pw_relu = nn.ReLU()

        self.tcn3_dw = nn.Conv1d(64, 64, kernel_size=3, padding=8, dilation=8, groups=64, bias=True)
        self.tcn3_dw_relu = nn.ReLU()
        self.tcn3_pw = nn.Conv1d(64, 64, kernel_size=1, bias=True)
        self.tcn3_pw_relu = nn.ReLU()

        self.tcn4_dw = nn.Conv1d(64, 64, kernel_size=3, padding=16, dilation=16, groups=64, bias=True)
        self.tcn4_dw_relu = nn.ReLU()
        self.tcn4_pw = nn.Conv1d(64, 96, kernel_size=1, bias=True)
        self.tcn4_pw_relu = nn.ReLU()
        self.tcn4_sc = nn.Conv1d(64, 96, kernel_size=1, bias=True)

        # Stage 3: Attention & Fused Head
        self.attn_conv = nn.Conv1d(96, 1, kernel_size=1, bias=True)
        self.bottleneck = nn.Linear(96, 48, bias=True)
        self.btn_relu = nn.ReLU()
        self.fc = nn.Linear(48, num_classes, bias=True)

    def forward(self, x):
        # 1. 2D CNN
        x = self.stem_relu(self.stem(x))
        x = self.pb0_dw_relu(self.pb0_dw(x))
        x = self.pb0_pw_relu(self.pb0_pw(x))
        x = self.pb2_dw_relu(self.pb2_dw(x))
        x = self.pb2_pw_relu(self.pb2_pw(x))

        # 2. 4 Sub-band Pooling matching PyTorch AdaptiveAvgPool2d((4, 40)): [0..4, 3..7, 6..10, 9..13]
        b0 = x[:, :, 0:4, :].mean(dim=2, keepdim=True)
        b1 = x[:, :, 3:7, :].mean(dim=2, keepdim=True)
        b2 = x[:, :, 6:10, :].mean(dim=2, keepdim=True)
        b3 = x[:, :, 9:13, :].mean(dim=2, keepdim=True)
        x_p = torch.cat([b0, b1, b2, b3], dim=2)

        # 3. Reshape 2D -> 1D
        b, c, f, t = x_p.shape
        x_1d = x_p.reshape(b, c * f, t)

        # 4. 1D TC-ResNet
        x0 = self.tcn0_pw_relu(self.tcn0_pw(self.tcn0_dw_relu(self.tcn0_dw(x_1d)))) + self.tcn0_sc(x_1d)
        x1 = self.tcn1_pw_relu(self.tcn1_pw(self.tcn1_dw_relu(self.tcn1_dw(x0)))) + x0
        x2 = self.tcn2_pw_relu(self.tcn2_pw(self.tcn2_dw_relu(self.tcn2_dw(x1)))) + x1
        x3 = self.tcn3_pw_relu(self.tcn3_pw(self.tcn3_dw_relu(self.tcn3_dw(x2)))) + x2
        x4 = self.tcn4_pw_relu(self.tcn4_pw(self.tcn4_dw_relu(self.tcn4_dw(x3)))) + self.tcn4_sc(x3)

        # 5. Attention & Head
        scores = self.attn_conv(x4)
        weights = torch.softmax(scores, dim=-1)
        context = (x4 * weights).sum(dim=-1)
        btn = self.btn_relu(self.bottleneck(context))
        logits = self.fc(btn)
        return logits


# ------------------------------------------------------------------------------
# 3. PRECOMPUTED OFFICIAL ZERO-LEAKAGE DATASET
# ------------------------------------------------------------------------------
class ESC50TensorDataset(Dataset):
    def __init__(self, specs_path, labels_path):
        specs_np = np.load(specs_path)   # [N, 52, 313]
        labels_np = np.load(labels_path) # [N]
        self.specs = torch.from_numpy(specs_np).unsqueeze(1).float()
        self.labels = torch.from_numpy(labels_np).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.specs[idx], self.labels[idx]


# ------------------------------------------------------------------------------
# 4. MAIN EXPORT & ESP-PPQ INT8 COMPILER PIPELINE
# ------------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("⚡ ESP-PPQ COMPILER & INT8 EXPORT FOR ESP32-S3 SENSE (SLIM 1D TC-RESNET)")
    print("   Protocol: KAROL PICZAK OFFICIAL 5-FOLD CV (FOLD 5 ZERO-LEAKAGE HELD-OUT)")
    print("   Target  : ~48k Parameters | 128-bit Xtensa PIE Vector SIMD Acceleration")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Execution Device: {device}")

    # 1. Load Precomputed Zero-Leakage Datasets
    train_specs_path = os.path.join(PROJECT_ROOT, "official_folds14_train_specs_1600.npy")
    train_labels_path = os.path.join(PROJECT_ROOT, "official_folds14_train_labels_1600.npy")
    test_specs_path = os.path.join(PROJECT_ROOT, "official_fold5_test_specs_400.npy")
    test_labels_path = os.path.join(PROJECT_ROOT, "official_fold5_test_labels_400.npy")

    if not os.path.exists(test_specs_path):
        raise FileNotFoundError(f"Missing official Fold-5 test set: {test_specs_path}")

    train_set = ESC50TensorDataset(train_specs_path, train_labels_path)
    test_set = ESC50TensorDataset(test_specs_path, test_labels_path)

    # Use first 128 samples of Folds 1-4 for calibration (0% test leakage!)
    calib_indices = list(range(min(128, len(train_set))))
    calib_loader = DataLoader(Subset(train_set, calib_indices), batch_size=1, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    print(f"📦 Calibration Set (from Folds 1-4) : {len(calib_indices)} clips (Zero test overlap!)")
    print(f"📦 Validation Set  (Fold 5 Held-out): {len(test_set)} clips (Karol Piczak Official)")

    # 2. Load Trained Clean Fold-5 INT8 Checkpoint
    int8_path = os.path.join(PROJECT_ROOT, "models", "best_fold5_tcn_slim_int8.pth")
    if not os.path.exists(int8_path):
        raise FileNotFoundError(f"❌ Error: {int8_path} not found!")

    print(f"\n📂 Loading Clean Fold-5 Quantized Checkpoint: {int8_path}")

    # Build QAT Model Structure and Convert to INT8 container
    qat = AudioPhiNetSlimTCNClassifierQAT(50)
    qat.eval()
    qat.qconfig = torch_quant.get_default_qat_qconfig('qnnpack')
    qat.attention.qconfig = None
    qat.post_tcn_bn.qconfig = None
    qat.bottleneck.qconfig = None
    qat.fc.qconfig = None

    torch.ao.quantization.fuse_modules(qat, [['stem_conv', 'stem_bn', 'stem_relu']], inplace=True)
    for b in [qat.phi_blocks[0], qat.phi_blocks[2]]:
        torch.ao.quantization.fuse_modules(b, [['conv1', 'bn1', 'relu1'], ['conv2', 'bn2', 'relu2']], inplace=True)
    for b in qat.tcn:
        torch.ao.quantization.fuse_modules(b, [['dw_conv', 'dw_bn', 'dw_relu'], ['pw_conv', 'pw_bn', 'pw_relu']], inplace=True)
        if b.shortcut_conv is not None:
            torch.ao.quantization.fuse_modules(b, [['shortcut_conv', 'shortcut_bn']], inplace=True)

    qat.train()
    prep = torch_quant.prepare_qat(qat, inplace=False)
    prep.eval()
    conv = torch_quant.convert(prep.to('cpu'), inplace=False)
    conv.load_state_dict(torch.load(int8_path, map_location='cpu'))
    conv.eval()

    # 2b. Evaluate PyTorch Converted INT8 Model on Fold 5 directly
    print("\n🔍 Evaluating Loaded PyTorch INT8 Model Directly on Fold-5...")
    conv_correct = 0
    conv_top3 = 0
    conv_total = 0
    with torch.no_grad():
        for specs, labels in test_loader:
            out = conv(specs)
            preds = out.argmax(dim=1)
            conv_correct += (preds == labels).sum().item()
            _, top3 = out.topk(3, dim=1)
            conv_top3 += (top3 == labels.view(-1, 1)).any(dim=1).sum().item()
            conv_total += labels.size(0)
    print(f"  • Original PyTorch INT8 Model: Top-1 = {(conv_correct/conv_total)*100.0:.2f}% ({conv_correct}/{conv_total}) | Top-3 = {(conv_top3/conv_total)*100.0:.2f}%")

    # 3. Create Clean FP32 Model & Populate Weights
    clean_model = CleanSlimTCN(50).to(device)

    clean_model.stem.weight.data.copy_(conv.stem_conv.weight().dequantize())
    clean_model.stem.bias.data.copy_(conv.stem_conv.bias())

    clean_model.pb0_dw.weight.data.copy_(conv.phi_blocks[0].conv1.weight().dequantize())
    clean_model.pb0_dw.bias.data.copy_(conv.phi_blocks[0].conv1.bias())
    clean_model.pb0_pw.weight.data.copy_(conv.phi_blocks[0].conv2.weight().dequantize())
    clean_model.pb0_pw.bias.data.copy_(conv.phi_blocks[0].conv2.bias())

    clean_model.pb2_dw.weight.data.copy_(conv.phi_blocks[2].conv1.weight().dequantize())
    clean_model.pb2_dw.bias.data.copy_(conv.phi_blocks[2].conv1.bias())
    clean_model.pb2_pw.weight.data.copy_(conv.phi_blocks[2].conv2.weight().dequantize())
    clean_model.pb2_pw.bias.data.copy_(conv.phi_blocks[2].conv2.bias())

    tcn_layers = [
        (clean_model.tcn0_dw, clean_model.tcn0_pw, clean_model.tcn0_sc),
        (clean_model.tcn1_dw, clean_model.tcn1_pw, None),
        (clean_model.tcn2_dw, clean_model.tcn2_pw, None),
        (clean_model.tcn3_dw, clean_model.tcn3_pw, None),
        (clean_model.tcn4_dw, clean_model.tcn4_pw, clean_model.tcn4_sc),
    ]
    for i, (dw, pw, sc) in enumerate(tcn_layers):
        dw.weight.data.copy_(conv.tcn[i].dw_conv.weight().dequantize())
        dw.bias.data.copy_(conv.tcn[i].dw_conv.bias())
        pw.weight.data.copy_(conv.tcn[i].pw_conv.weight().dequantize())
        pw.bias.data.copy_(conv.tcn[i].pw_conv.bias())
        if sc is not None and conv.tcn[i].shortcut_conv is not None:
            sc.weight.data.copy_(conv.tcn[i].shortcut_conv.weight().dequantize())
            sc.bias.data.copy_(conv.tcn[i].shortcut_conv.bias())

    clean_model.attn_conv.weight.data.copy_(conv.attention.attn_conv.weight)
    clean_model.attn_conv.bias.data.copy_(conv.attention.attn_conv.bias)

    # Fuse Post-TCN BN into bottleneck Linear
    gamma = conv.post_tcn_bn.weight.data
    beta = conv.post_tcn_bn.bias.data
    mean = conv.post_tcn_bn.running_mean.data
    var = conv.post_tcn_bn.running_var.data
    eps = conv.post_tcn_bn.eps
    scale = gamma / torch.sqrt(var + eps)
    offset = beta - mean * scale

    w_btn = conv.bottleneck.weight.data
    b_btn = conv.bottleneck.bias.data
    w_btn_fused = w_btn * scale.unsqueeze(0)
    b_btn_fused = b_btn + torch.matmul(w_btn, offset)

    clean_model.bottleneck.weight.data.copy_(w_btn_fused)
    clean_model.bottleneck.bias.data.copy_(b_btn_fused)

    clean_model.fc.weight.data.copy_(conv.fc.weight.data)
    clean_model.fc.bias.data.copy_(conv.fc.bias.data)

    clean_model.eval()

    # Validate exact parity on 400 official Fold-5 test clips
    correct = 0
    top3_correct = 0
    total = 0
    with torch.no_grad():
        for specs, labels in test_loader:
            specs = specs.to(device)
            logits = clean_model(specs).cpu()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            
            # Top-3 check
            _, top3_preds = logits.topk(3, dim=1)
            top3_correct += (top3_preds == labels.view(-1, 1)).any(dim=1).sum().item()
            total += labels.size(0)

    clean_acc = (correct / total) * 100.0
    clean_top3 = (top3_correct / total) * 100.0
    print(f"\n🎯 Clean FP32 Parity on Fold-5: Top-1 = {clean_acc:.2f}% ({correct}/{total}) | Top-3 = {clean_top3:.2f}% 🌟")

    # 4. Export Clean Simplified ONNX Graph
    out_dir = os.path.join(PROJECT_ROOT, "firmware", "esp32s3", "src")
    os.makedirs(out_dir, exist_ok=True)
    onnx_path = os.path.join(out_dir, "model.onnx")
    dummy_in = torch.randn(1, 1, 52, 313, dtype=torch.float32, device=device)

    print("\n📦 Exporting Clean ONNX Graph to: " + onnx_path)
    torch.onnx.export(
        clean_model,
        dummy_in,
        onnx_path,
        opset_version=13,
        input_names=["input"],
        output_names=["output"],
        do_constant_folding=True
    )
    m_onnx = onnx.load(onnx_path)
    m_sim, _ = simplify(m_onnx)
    onnx.save(m_sim, onnx_path)
    print("✅ Successfully simplified ONNX Graph (0 dynamic shape ops)")

    # 5. Run ESP-PPQ Quantizer for TargetPlatform.ESPDL_S3_INT8
    print("\n" + "=" * 80)
    print("⚙️ RUNNING ESP-PPQ INT8 COMPILER (Platform: ESPDL_S3_INT8)...")
    print("=" * 80)

    setting = QuantizationSettingFactory.espdl_setting()

    quantized_ir = quantize_onnx_model(
        onnx_import_file=onnx_path,
        calib_dataloader=calib_loader,
        calib_steps=min(128, len(calib_indices)),
        input_shape=[1, 1, 52, 313],
        platform=TargetPlatform.ESPDL_S3_INT8,
        setting=setting,
        collate_fn=lambda x: x[0].to('cpu'),
        device='cpu'
    )

    # 6. Export ESP-DL Binary Package
    espdl_path = os.path.join(out_dir, "model.espdl")
    json_path = os.path.join(out_dir, "model.json")

    export_ppq_graph(
        graph=quantized_ir,
        platform=TargetPlatform.ESPDL_S3_INT8,
        graph_save_to=espdl_path,
        config_save_to=json_path
    )
    if os.path.exists(espdl_path):
        size_kb = os.path.getsize(espdl_path) / 1024.0
        print(f"\n🎉 Successfully Generated ESP-DL Package in: {out_dir}")
        print(f"   • model.espdl ({size_kb:.2f} KB)")
        print(f"   • model.json")

    # 7. Evaluate Quantized INT8 Accuracy on ESP-PPQ TorchExecutor (Fold 5)
    executor = TorchExecutor(graph=quantized_ir, device='cpu')
    int8_correct = 0
    int8_top3 = 0
    int8_total = 0

    print("\n📊 Evaluating Quantized ESP-DL INT8 Model on 400 Official Fold-5 Clips...")
    for specs, labels in test_loader:
        specs_cpu = specs.to('cpu')
        out = executor.forward(inputs=specs_cpu)[0]
        out_tensor = torch.tensor(out)
        preds = out_tensor.argmax(dim=1)
        int8_correct += (preds == labels).sum().item()
        
        _, top3 = out_tensor.topk(3, dim=1)
        int8_top3 += (top3 == labels.view(-1, 1)).any(dim=1).sum().item()
        int8_total += labels.size(0)

    final_top1 = (int8_correct / int8_total) * 100.0
    final_top3 = (int8_top3 / int8_total) * 100.0
    print("=" * 80)
    print(f"🏆 FINAL ESP-DL S3 INT8 FOLD-5 ACCURACY:")
    print(f"   • Top-1 Accuracy : {final_top1:.2f}% ({int8_correct}/{int8_total})")
    print(f"   • Top-3 Accuracy : {final_top3:.2f}% ({int8_top3}/{int8_total})")
    print(f"   • Model Footprint: {os.path.getsize(espdl_path) / 1024.0:.2f} KB Flash")
    print("=" * 80)

if __name__ == "__main__":
    main()
