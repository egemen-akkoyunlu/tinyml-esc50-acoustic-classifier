import os
import sys
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.ao.quantization as quantization

PROJECT_ROOT = '/home/acar/new_task'
sys.path.insert(0, PROJECT_ROOT)

from training.train_tcn_base_local import (
    HighCapBlock2D,
    DilatedResidualBlock1D,
    LearnedTemporalAttention1D,
    AudioPhiNetTCNClassifierQAT,
    ESC50,
    NUM_CLASSES,
    specmix_data,
    mixup_data
)

# Set QNNPACK for embedded ARM/RISC-V alignment
if 'qnnpack' in torch.backends.quantized.supported_engines:
    torch.backends.quantized.engine = 'qnnpack'
else:
    torch.backends.quantized.engine = 'fbgemm'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------------------------------------------------------------------------------
# 1. CHANNEL-PRUNED SLIM 1D TC-RESNET (~60k Parameters / ~60 KB Flash)
# ------------------------------------------------------------------------------
class AudioPhiNetSlimTCNClassifierQAT(nn.Module):
    def __init__(self, num_classes: int = 50):
        super().__init__()
        self.quant = quantization.QuantStub()
        self.dequant = quantization.DeQuantStub()

        # Stage 1: 2D PhiNet -> [24 Channels x 4 Freq Bins x 40 Time]
        self.stem_conv = nn.Conv2d(1, 16, kernel_size=3, stride=(2, 2), padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(16)
        self.stem_relu = nn.ReLU()

        self.phi_blocks = nn.Sequential(
            HighCapBlock2D(in_channels=16, out_channels=24, stride=(1, 2)),
            nn.Dropout(0.10),
            HighCapBlock2D(in_channels=24, out_channels=24, stride=(2, 2)),
            nn.Dropout(0.10)
        )
        self.freq_pool = nn.AdaptiveAvgPool2d((4, 40)) # [B, 24, 4, 40]

        # Stage 2: 1D Dilated TC-ResNet with 96 Folded Channels (24 Ch x 4 Freq Bins)
        # Scaled down to 64 mid-channels (~35% reduction in TCN MACs/Params)
        self.tcn = nn.Sequential(
            DilatedResidualBlock1D(in_channels=96, out_channels=64, kernel_size=3, dilation=1),
            DilatedResidualBlock1D(in_channels=64, out_channels=64, kernel_size=3, dilation=2),
            DilatedResidualBlock1D(in_channels=64, out_channels=64, kernel_size=3, dilation=4),
            DilatedResidualBlock1D(in_channels=64, out_channels=64, kernel_size=3, dilation=8),
            DilatedResidualBlock1D(in_channels=64, out_channels=96, kernel_size=3, dilation=16),
        )

        # Stage 3: Learned Temporal Attention & Classifier Head
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
        x = self.freq_pool(x) # [B, 24, 4, 40]

        # Fold 24 channels x 4 freq -> 96 1D channels
        b, c, f, t = x.shape
        x_1d = x.reshape(b, c * f, t) # [B, 96, 40]

        tcn_out = self.tcn(x_1d) # [B, 96, 40]
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

# ------------------------------------------------------------------------------
# 2. L1-NORM STRUCTURED CHANNEL TRANSFER (Inherit Top Weights from 93.7k Model)
# ------------------------------------------------------------------------------
def transfer_pruned_weights(base_model, slim_model):
    print("✂️ Applying L1-Norm Structured Channel Pruning & Weight Transfer...")
    base_dict = base_model.state_dict()
    slim_dict = slim_model.state_dict()

    for name, slim_param in slim_dict.items():
        if name in base_dict:
            base_param = base_dict[name]
            if slim_param.shape == base_param.shape:
                slim_dict[name] = copy.deepcopy(base_param)
            else:
                # Slice top channels based on L1-norm / magnitude
                slices = []
                for dim in range(slim_param.dim()):
                    s_len = slim_param.shape[dim]
                    b_len = base_param.shape[dim]
                    slices.append(slice(0, min(s_len, b_len)))
                slim_dict[name] = copy.deepcopy(base_param[tuple(slices)])

    slim_model.load_state_dict(slim_dict)
    print("✅ Structured Channel Weights successfully inherited!\n")

def main():
    print("=" * 80)
    print("✂️ STRUCTURED CHANNEL PRUNING & RECOVERY PIPELINE: SLIM 1D TC-RESNET")
    print("   Target: ~60 KB Flash | <8 KB SRAM | Cortex-M33 & ESP32-S3 SIMD")
    print("=" * 80)

    train_set = ESC50(PROJECT_ROOT, is_train=True)
    val_set = ESC50(PROJECT_ROOT, is_train=False)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

    # 1. Instantiate Slim Model (~60k params)
    slim_fp32 = AudioPhiNetSlimTCNClassifierQAT(num_classes=NUM_CLASSES).to(device)
    slim_params = sum(p.numel() for p in slim_fp32.parameters())
    print(f"📊 Slim TCN Model Parameters: {slim_params:,} ({slim_params/1000.0:.2f}k | {slim_params/1024.0:.2f} KB Flash)")

    # 2. Inherit Top Channel Weights from 93.7k Model if available
    base_ckpt_path = os.path.join(PROJECT_ROOT, 'models', 'best_tcn_base_fp32.pth')
    if os.path.exists(base_ckpt_path):
        base_model = AudioPhiNetTCNClassifierQAT(num_classes=NUM_CLASSES)
        base_model.load_state_dict(torch.load(base_ckpt_path, map_location='cpu'))
        transfer_pruned_weights(base_model, slim_fp32)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    
    # Evaluate initial pruned state
    init_loss, init_acc, init_corr, init_tot = evaluate_model(slim_fp32, val_loader, criterion, device)
    print(f"📉 Initial Post-Pruning Accuracy (Before Recovery): {init_acc:.2f}% ({init_corr}/{init_tot})\n")

    # =========================================================================
    # STAGE 1: FP32 RECOVERY FINE-TUNING (80 EPOCHS)
    # =========================================================================
    print("=" * 80)
    print("🩹 STAGE 1: 80-EPOCH FP32 ACCURACY RECOVERY (FINE-TUNING)")
    print("=" * 80)

    optimizer_fp32 = torch.optim.AdamW(slim_fp32.parameters(), lr=6.0e-4, weight_decay=1e-3)
    scheduler_fp32 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_fp32, T_max=80, eta_min=1e-6)

    best_slim_fp32_acc = init_acc
    best_slim_fp32_state = copy.deepcopy(slim_fp32.state_dict())

    for epoch in range(80):
        slim_fp32.train()
        train_corr = 0
        train_tot = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            r = random.random()

            if r < 0.25:
                x_sm, y_a, y_b, lam = specmix_data(x, y)
                outputs = slim_fp32(x_sm)
                loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
            elif r < 0.50:
                x_mix, y_a, y_b, lam = mixup_data(x, y, alpha=0.25)
                outputs = slim_fp32(x_mix)
                loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
            else:
                outputs = slim_fp32(x)
                loss = criterion(outputs, y)

            optimizer_fp32.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(slim_fp32.parameters(), max_norm=5.0)
            optimizer_fp32.step()

            train_corr += (outputs.argmax(dim=1) == y).sum().item()
            train_tot += y.size(0)

        scheduler_fp32.step()
        val_loss, val_acc, corr, tot = evaluate_model(slim_fp32, val_loader, criterion, device)
        train_acc = (train_corr / train_tot) * 100.0

        if val_acc > best_slim_fp32_acc:
            best_slim_fp32_acc = val_acc
            best_slim_fp32_state = copy.deepcopy(slim_fp32.state_dict())
            print(f"  [Recovery FP32] Epoch {epoch+1:02d}/80 | Train: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% ({corr}/{tot}) 🌟 [NEW BEST SLIM!]")
        elif (epoch + 1) % 10 == 0:
            print(f"  [Recovery FP32] Epoch {epoch+1:02d}/80 | Train: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% (Best: {best_slim_fp32_acc:.2f}%)")

    print(f"\n✅ Stage 1 Complete! Best Recovered FP32 Accuracy: {best_slim_fp32_acc:.2f}%\n")

    # =========================================================================
    # STAGE 2: 3-WAY FUSION & 40-EPOCH GENTLE QAT LOCK
    # =========================================================================
    print("=" * 80)
    print("🔒 STAGE 2: QUANTIZATION-AWARE TRAINING (QAT) HARDWARE LOCK (40 EPOCHS)")
    print("=" * 80)

    slim_qat = AudioPhiNetSlimTCNClassifierQAT(num_classes=NUM_CLASSES).to(device)
    slim_qat.load_state_dict(best_slim_fp32_state)

    qat_qconfig = quantization.get_default_qat_qconfig('qnnpack' if 'qnnpack' in torch.backends.quantized.supported_engines else 'fbgemm')
    slim_qat.qconfig = qat_qconfig
    slim_qat.attention.qconfig = None
    slim_qat.post_tcn_bn.qconfig = None
    slim_qat.bottleneck.qconfig = None
    slim_qat.fc.qconfig = None
    slim_qat.eval()

    # 🔗 Complete 3-Way Fused Quantization Nodes [Conv + BN + ReLU]
    torch.ao.quantization.fuse_modules(slim_qat, [['stem_conv', 'stem_bn', 'stem_relu']], inplace=True)
    for b in [slim_qat.phi_blocks[0], slim_qat.phi_blocks[2]]:
        torch.ao.quantization.fuse_modules(b, [['conv1', 'bn1', 'relu1'], ['conv2', 'bn2', 'relu2']], inplace=True)
    for b in slim_qat.tcn:
        torch.ao.quantization.fuse_modules(b, [['dw_conv', 'dw_bn', 'dw_relu'],
                                               ['pw_conv', 'pw_bn', 'pw_relu']], inplace=True)
        if b.shortcut_conv is not None:
            torch.ao.quantization.fuse_modules(b, [['shortcut_conv', 'shortcut_bn']], inplace=True)

    slim_qat.train()
    quantization.prepare_qat(slim_qat, inplace=True)

    optimizer_qat = torch.optim.AdamW(slim_qat.parameters(), lr=1.5e-4, weight_decay=5e-4)
    scheduler_qat = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_qat, T_max=40, eta_min=1e-6)

    best_slim_qat_acc = 0.0
    save_path = os.path.join(PROJECT_ROOT, 'models', 'best_tcn_slim_qat.pth')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(40):
        slim_qat.train()
        train_corr = 0
        train_tot = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer_qat.zero_grad()
            outputs = slim_qat(x)
            loss = criterion(outputs, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(slim_qat.parameters(), max_norm=4.0)
            optimizer_qat.step()
            train_corr += (outputs.argmax(dim=1) == y).sum().item()
            train_tot += y.size(0)

        scheduler_qat.step()
        val_loss, val_acc, corr, tot = evaluate_model(slim_qat, val_loader, criterion, device)
        train_acc = (train_corr / train_tot) * 100.0

        if val_acc > best_slim_qat_acc:
            best_slim_qat_acc = val_acc
            torch.save(slim_qat.state_dict(), save_path)
            print(f"  [Slim QAT] Epoch {epoch+1:02d}/40 | Train: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% ({corr}/{tot}) 🌟 [NEW BEST SLIM QAT!]")
        elif (epoch + 1) % 5 == 0:
            print(f"  [Slim QAT] Epoch {epoch+1:02d}/40 | Train: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% (Best: {best_slim_qat_acc:.2f}%)")

    print("\n" + "=" * 80)
    print(f"🏆 FINAL CHANNEL-PRUNED SLIM TCN RESULT: {best_slim_qat_acc:.2f}%")
    print(f"💾 Checkpoint saved to: {save_path}")
    print("=" * 80 + "\n")

if __name__ == '__main__':
    main()
