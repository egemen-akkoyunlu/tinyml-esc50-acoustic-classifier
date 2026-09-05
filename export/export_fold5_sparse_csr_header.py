#!/usr/bin/env python3
import os
import sys
import torch
import numpy as np

PROJECT_ROOT = "/home/acar/new_task"
CKPT_PATH = os.path.join(PROJECT_ROOT, "best_distilled_qat_model.pth")
OUT_PATHS = [
    os.path.join(PROJECT_ROOT, "firmware", "efr32mg24", "src", "gru_classifier_weights_pruned_csr.h"),
    "/home/acar/zephyrproject/my_apps/silabs_ble_audio_peripheral/src/gru_classifier_weights_pruned_csr.h"
]

print("=" * 80)
print(f"📦 Loading Fold-5 checkpoint: {CKPT_PATH}")
print("=" * 80)

sd = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)

# 1. Pre-GRU BatchNorm
pre_w = sd["pre_gru_bn.weight"].numpy()
pre_b = sd["pre_gru_bn.bias"].numpy()
pre_m = sd["pre_gru_bn.running_mean"].numpy()
pre_v = sd["pre_gru_bn.running_var"].numpy()
pre_scale = pre_w / np.sqrt(pre_v + 1e-5)
pre_bias = pre_b - pre_m * pre_scale

# 2. GRU Weights & Biases
w_ih = sd["gru.weight_ih_l0"].numpy() # [480, 32]
w_hh = sd["gru.weight_hh_l0"].numpy() # [480, 160]
b_ih = sd["gru.bias_ih_l0"].numpy()   # [480]
b_hh = sd["gru.bias_hh_l0"].numpy()   # [480]

# 3. Post-GRU BatchNorm
post_w = sd["post_gru_bn.weight"].numpy()
post_b = sd["post_gru_bn.bias"].numpy()
post_m = sd["post_gru_bn.running_mean"].numpy()
post_v = sd["post_gru_bn.running_var"].numpy()
post_scale = post_w / np.sqrt(post_v + 1e-5)
post_bias = post_b - post_m * post_scale

# 4. Bottleneck
btn_w = sd["bottleneck.weight"].numpy() # [128, 160]
btn_b = sd["bottleneck.bias"].numpy()   # [128]

# 5. FC Classifier Head
fc_w = sd["fc.weight"].numpy() # [50, 128]
fc_b = sd["fc.bias"].numpy()   # [50]

# 6. Apply 50% Magnitude Pruning (L1 Unstructured)
def prune_50_percent(mat, name):
    th = np.percentile(np.abs(mat), 50.0)
    pruned = mat.copy()
    pruned[np.abs(pruned) < th] = 0.0
    nz = np.count_nonzero(pruned)
    tot = mat.size
    print(f"  • {name:<16s}: {nz:6d} / {tot:6d} Non-Zeros ({nz/tot*100:5.1f}%) [Threshold: {th:.6f}]")
    return pruned

print("\n⚡ Applying 50% Magnitude Pruning to Fold-5 Stage 2 Layers:")
p_w_ih = prune_50_percent(w_ih, "GRU_W_IH")
p_w_hh = prune_50_percent(w_hh, "GRU_W_HH")
p_btn_w = prune_50_percent(btn_w, "BOTTLENECK_W")
p_fc_w = prune_50_percent(fc_w, "FC_W")

total_dense = w_ih.size + w_hh.size + btn_w.size + fc_w.size
total_sparse = np.count_nonzero(p_w_ih) + np.count_nonzero(p_w_hh) + np.count_nonzero(p_btn_w) + np.count_nonzero(p_fc_w)
print(f"  🏆 Overall Sparsity: {total_sparse:,} non-zeros out of {total_dense:,} ({total_sparse/total_dense*100:.1f}%)")

# 7. Convert to Compressed Sparse Row (CSR) format
def to_csr(dense_matrix, name):
    rows, cols = dense_matrix.shape
    sparse_vals = []
    col_indices = []
    row_offsets = [0]

    for r in range(rows):
        for c in range(cols):
            val = dense_matrix[r, c]
            if abs(val) > 1e-9:
                sparse_vals.append(val)
                col_indices.append(c)
        row_offsets.append(len(sparse_vals))

    sparse_vals = np.array(sparse_vals, dtype=np.float32)
    col_indices = np.array(col_indices, dtype=np.uint8)
    row_offsets = np.array(row_offsets, dtype=np.uint32)
    return sparse_vals, col_indices, row_offsets

print("\n📊 Converting Pruned Matrices to Compressed Sparse Row (CSR):")
w_ih_val, w_ih_col, w_ih_row = to_csr(p_w_ih, "GRU_W_IH")
w_hh_val, w_hh_col, w_hh_row = to_csr(p_w_hh, "GRU_W_HH")
btn_val, btn_col, btn_row = to_csr(p_btn_w, "BOTTLENECK_W")
fc_val, fc_col, fc_row = to_csr(p_fc_w, "FC_W")

# 8. Format C Arrays
def format_c_array(arr, arr_type, name):
    code = f"static const {arr_type} {name}[{len(arr)}] = {{\n"
    for i in range(0, len(arr), 8):
        chunk = arr[i:i+8]
        if "float" in arr_type:
            items = ", ".join([f"{v:.8e}f" for v in chunk])
        else:
            items = ", ".join([str(int(v)) for v in chunk])
        code += f"    {items},\n"
    code += "};\n\n"
    return code

header_content = """// ==============================================================================
// STAGE 2: 50% PRUNED COMPRESSED SPARSE ROW (CSR) WEIGHTS (FOLD-5 ZERO-LEAKAGE)
// Target: ARM Cortex-M33 Hardware FPU (Branchless CSR Zero-Skipping Execution)
// Model: Official Karol Piczak Fold-5 Distilled PhiNet-CRNN (50% Magnitude Pruned)
// ==============================================================================

#ifndef GRU_CLASSIFIER_WEIGHTS_PRUNED_CSR_H
#define GRU_CLASSIFIER_WEIGHTS_PRUNED_CSR_H

#include <stdint.h>

#define GRU_INPUT_DIM       32
#define GRU_HIDDEN_DIM      160
#define BOTTLENECK_DIM      128
#define NUM_CLASSES         50
#define GRU_TIME_STEPS      39

"""

header_content += format_c_array(pre_scale, "float", "PRE_GRU_BN_SCALE")
header_content += format_c_array(pre_bias, "float", "PRE_GRU_BN_BIAS")
header_content += format_c_array(post_scale, "float", "POST_GRU_BN_SCALE")
header_content += format_c_array(post_bias, "float", "POST_GRU_BN_BIAS")

header_content += format_c_array(b_ih, "float", "GRU_B_IH")
header_content += format_c_array(b_hh, "float", "GRU_B_HH")
header_content += format_c_array(btn_b, "float", "BOTTLENECK_B")
header_content += format_c_array(fc_b, "float", "FC_B")

header_content += f"#define GRU_W_IH_NONZEROS {len(w_ih_val)}\n"
header_content += format_c_array(w_ih_val, "float", "GRU_W_IH_SPARSE")
header_content += format_c_array(w_ih_col, "uint8_t", "GRU_W_IH_COL_IDX")
header_content += format_c_array(w_ih_row, "uint32_t", "GRU_W_IH_ROW_OFFSETS")

header_content += f"#define GRU_W_HH_NONZEROS {len(w_hh_val)}\n"
header_content += format_c_array(w_hh_val, "float", "GRU_W_HH_SPARSE")
header_content += format_c_array(w_hh_col, "uint8_t", "GRU_W_HH_COL_IDX")
header_content += format_c_array(w_hh_row, "uint32_t", "GRU_W_HH_ROW_OFFSETS")

header_content += f"#define BOTTLENECK_W_NONZEROS {len(btn_val)}\n"
header_content += format_c_array(btn_val, "float", "BOTTLENECK_W_SPARSE")
header_content += format_c_array(btn_col, "uint8_t", "BOTTLENECK_W_COL_IDX")
header_content += format_c_array(btn_row, "uint32_t", "BOTTLENECK_W_ROW_OFFSETS")

header_content += f"#define FC_W_NONZEROS {len(fc_val)}\n"
header_content += format_c_array(fc_val, "float", "FC_W_SPARSE")
header_content += format_c_array(fc_col, "uint8_t", "FC_W_COL_IDX")
header_content += format_c_array(fc_row, "uint32_t", "FC_W_ROW_OFFSETS")

header_content += "#endif /* GRU_CLASSIFIER_WEIGHTS_PRUNED_CSR_H */\n"

for p in OUT_PATHS:
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        f.write(header_content)
    print(f"💾 Saved Clean Fold-5 CSR Header to: {p}")

print("\n🎉 Fold-5 50% Pruned CSR Header Export Complete!")
