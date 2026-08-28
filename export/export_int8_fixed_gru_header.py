#!/usr/bin/env python3
# ==============================================================================
# 🚀 EXPORT BIT-EXACT 8-BIT FIXED-POINT SIMD GRU C++ HEADER
# Generates firmware/efr32mg24/src/gru_classifier_weights_int8_fixed.h
# ==============================================================================

import os
import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.ao.quantization as quantization

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from training.qat_training import AudioPhiNetCRNNClassifierQAT

def main():
    print("=" * 80)
    print("📦 EXPORTING BIT-EXACT INT8 FIXED-POINT GRU HEADERS FOR EFR32MG24")
    print("=" * 80)

    # 1. Load Checkpoint
    model = AudioPhiNetCRNNClassifierQAT(num_classes=50)
    model.eval()
    torch.ao.quantization.fuse_modules(model, [['stem.0', 'stem.1']], inplace=True)
    torch.ao.quantization.fuse_modules(model.phi_blocks[0], [['conv.0', 'conv.1'], ['conv.3', 'conv.4']], inplace=True)
    torch.ao.quantization.fuse_modules(model.phi_blocks[2], [['conv.0', 'conv.1'], ['conv.3', 'conv.4']], inplace=True)
    model.gru.qconfig = None
    model.bottleneck.qconfig = None
    model.fc.qconfig = None
    model.qconfig = quantization.get_default_qat_qconfig('fbgemm')
    model.train()
    quantization.prepare_qat(model, inplace=True)
    model.eval()

    ckpt_path = os.path.join(PROJECT_ROOT, 'models', 'best_distilled_qat_model.pth')
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(PROJECT_ROOT, 'best_qat_model.pth')
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    print(f"✅ Loaded checkpoint: {os.path.basename(ckpt_path)}")

    # 2. Pre-GRU BatchNorm is already folded into TFLite CNN weights -> Identity in Header
    pre_bn_scale = np.ones(32, dtype=np.float32)
    pre_bn_bias = np.zeros(32, dtype=np.float32)

    # 3. Extract & Quantize GRU Weights
    w_ih = model.gru.weight_ih_l0.detach().cpu().numpy() # [480, 32]
    w_hh = model.gru.weight_hh_l0.detach().cpu().numpy() # [480, 160]
    b_ih = model.gru.bias_ih_l0.detach().cpu().numpy()   # [480]
    b_hh = model.gru.bias_hh_l0.detach().cpu().numpy()   # [480]
    bias = b_ih + b_hh

    scale_w_ih = float(np.max(np.abs(w_ih)) / 127.0)
    scale_w_hh = float(np.max(np.abs(w_hh)) / 127.0)

    w_ih_int8 = np.clip(np.round(w_ih / scale_w_ih), -128, 127).astype(np.int8)
    w_hh_int8 = np.clip(np.round(w_hh / scale_w_hh), -128, 127).astype(np.int8)

    # 4. Generate 256-Entry LUTs
    SIGMOID_LUT = np.zeros(256, dtype=np.int8)
    TANH_LUT = np.zeros(256, dtype=np.int8)
    SCALE_ACT = 8.0 / 128.0 # 0.0625

    for i in range(256):
        int_val = i - 128
        real_val = int_val * SCALE_ACT
        sig_real = 1.0 / (1.0 + math.exp(-real_val))
        SIGMOID_LUT[i] = int(round(sig_real * 127.0))
        tanh_real = math.tanh(real_val)
        TANH_LUT[i] = int(round(tanh_real * 127.0))

    # 5. Extract Post-GRU BatchNorm & Head Weights
    post_bn_mean = model.post_gru_bn.running_mean.detach().cpu().numpy()
    post_bn_var = model.post_gru_bn.running_var.detach().cpu().numpy()
    post_bn_w = model.post_gru_bn.weight.detach().cpu().numpy()
    post_bn_b = model.post_gru_bn.bias.detach().cpu().numpy()
    post_bn_scale = post_bn_w / np.sqrt(post_bn_var + 1e-5)
    post_bn_bias = post_bn_b - post_bn_mean * post_bn_scale

    btn_w = model.bottleneck.weight.detach().cpu().numpy() # [128, 160]
    btn_b = model.bottleneck.bias.detach().cpu().numpy()   # [128]
    fc_w = model.fc.weight.detach().cpu().numpy()          # [50, 128]
    fc_b = model.fc.bias.detach().cpu().numpy()            # [50]

    # 6. Format & Write C++ Header File
    out_header = os.path.join(PROJECT_ROOT, 'firmware', 'efr32mg24', 'src', 'gru_classifier_weights_int8_fixed.h')

    with open(out_header, 'w') as f:
        f.write("// ==============================================================================\n")
        f.write("// STAGE 2: 8-BIT FIXED-POINT SIMD GRU & LUT WEIGHTS\n")
        f.write("// Target: Silicon Labs EFR32MG24 (ARM Cortex-M33 @ 78 MHz)\n")
        f.write("// Profile 3: 91.50% Validation Accuracy | 122.4 KB Flash | Inlined __SMLAD SIMD\n")
        f.write("// ==============================================================================\n\n")
        f.write("#ifndef GRU_CLASSIFIER_WEIGHTS_INT8_FIXED_H_\n")
        f.write("#define GRU_CLASSIFIER_WEIGHTS_INT8_FIXED_H_\n\n")
        f.write("#include <stdint.h>\n\n")

        f.write("#define GRU_INPUT_DIM      32\n")
        f.write("#define GRU_HIDDEN_DIM     160\n")
        f.write("#define GRU_TIME_STEPS     39\n")
        f.write("#define BOTTLENECK_DIM     128\n")
        f.write("#define NUM_OUTPUT_CLASSES 50\n\n")

        f.write(f"#define GRU_SCALE_W_IH     {scale_w_ih:.8e}f\n")
        f.write(f"#define GRU_SCALE_W_HH     {scale_w_hh:.8e}f\n")
        f.write(f"#define GRU_SCALE_ACT_INV  {1.0 / SCALE_ACT:.8e}f\n\n")

        def dump_i8_array(name, arr):
            f.write(f"static const int8_t {name}[{arr.size}] = {{\n  ")
            flat = arr.flatten()
            for idx, val in enumerate(flat):
                f.write(f"{val}, ")
                if (idx + 1) % 24 == 0:
                    f.write("\n  ")
            f.write("\n};\n\n")

        def dump_f32_array(name, arr):
            f.write(f"static const float {name}[{arr.size}] = {{\n  ")
            flat = arr.flatten()
            for idx, val in enumerate(flat):
                f.write(f"{val:.8e}f, ")
                if (idx + 1) % 6 == 0:
                    f.write("\n  ")
            f.write("\n};\n\n")

        # Pre-GRU BatchNorm (Folded into CNN in TFLite -> Identity in Stage 2)
        dump_f32_array("PRE_GRU_BN_SCALE", pre_bn_scale)
        dump_f32_array("PRE_GRU_BN_BIAS", pre_bn_bias)

        # INT8 LUTs
        dump_i8_array("SIGMOID_LUT_S8", SIGMOID_LUT)
        dump_i8_array("TANH_LUT_S8", TANH_LUT)

        # INT8 GRU Weights
        dump_i8_array("GRU_W_IH_INT8", w_ih_int8)
        dump_i8_array("GRU_W_HH_INT8", w_hh_int8)
        dump_f32_array("GRU_BIAS_F32", bias)

        # Post-GRU BatchNorm & Classifier Head
        dump_f32_array("POST_GRU_BN_SCALE", post_bn_scale)
        dump_f32_array("POST_GRU_BN_BIAS", post_bn_bias)
        dump_f32_array("BOTTLENECK_W", btn_w)
        dump_f32_array("BOTTLENECK_B", btn_b)
        dump_f32_array("FC_W", fc_w)
        dump_f32_array("FC_B", fc_b)

        f.write("#endif /* GRU_CLASSIFIER_WEIGHTS_INT8_FIXED_H_ */\n")

    print(f"💾 Successfully regenerated: {out_header}")
    print("=" * 80)

if __name__ == '__main__':
    main()
