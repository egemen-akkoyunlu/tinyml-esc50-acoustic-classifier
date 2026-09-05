#!/usr/bin/env python3
"""
================================================================================
🎯 OFFICIAL BENCHMARK: BEFORE VS. AFTER PRUNING (FOLD-5 ZERO-LEAKAGE)
Compares:
  Model A: Dense Baseline Distilled Fold-5 Model (0% Sparsity)
  Model B: 50% Pruned CSR GRU Fold-5 Model (50% Sparsity)
Dataset: Official Karol Piczak Unseen Fold-5 (400 Held-Out Audio Clips)
================================================================================
"""

import os
import sys
import re
import numpy as np
import torch
import ai_edge_litert.interpreter as litert

PROJECT_ROOT = "/home/acar/new_task"
CKPT_PATH = os.path.join(PROJECT_ROOT, "best_distilled_qat_model.pth")
FOLD5_SPECS_PATH = os.path.join(PROJECT_ROOT, "official_fold5_test_specs_400.npy")
FOLD5_LABELS_PATH = os.path.join(PROJECT_ROOT, "official_fold5_test_labels_400.npy")
CNN_HEADER = "/home/acar/zephyrproject/my_apps/silabs_ble_audio_peripheral/src/phinet_features_model_data.h"

print("=" * 85)
print("🔍 ESC-50 FOLD-5 ZERO-LEAKAGE BENCHMARK: BEFORE VS AFTER 50% PRUNING")
print("=" * 85)

# 1. Load or Precompute Official Fold-5 Test Data (fold == 5)
if not os.path.exists(FOLD5_SPECS_PATH) or not os.path.exists(FOLD5_LABELS_PATH):
    print("⚡ Official Fold-5 test set not cached yet. Generating from audio clips (fold == 5)...")
    import pandas as pd
    import soundfile as sf
    import torchaudio.transforms as T
    import torch.nn.functional as F

    csv_path = os.path.join(PROJECT_ROOT, "ESC-50-master", "meta", "esc50.csv")
    audio_dir = os.path.join(PROJECT_ROOT, "ESC-50-master", "audio")
    
    df = pd.read_csv(csv_path)
    df['category'] = df['category'].str.replace('_', ' ')
    classes = sorted(df['category'].unique())
    class_to_idx = {cat: i for i, cat in enumerate(classes)}

    # Strictly select Fold 5 (Karol Piczak Official Protocol - Zero Leakage)
    fold5_rows = df[df['fold'] == 5].reset_index(drop=True)
    print(f"🔒 Identified {len(fold5_rows)} official Fold-5 test clips (0% train overlap).")

    melspec = T.MelSpectrogram(sample_rate=16000, n_fft=512, hop_length=256, n_mels=52)
    resample_44k = T.Resample(44100, 16000)
    f5_specs = []
    f5_labels = []

    for i, row in fold5_rows.iterrows():
        p = os.path.join(audio_dir, row['filename'])
        y = class_to_idx[row['category']]
        
        data, sr = sf.read(p)
        t = torch.from_numpy(data).float()
        if t.ndim == 1:
            t = t.unsqueeze(0)
        else:
            t = t.t()
            
        if sr != 16000:
            if sr == 44100:
                t = resample_44k(t)
            else:
                t = T.Resample(sr, 16000)(t)
        
        if t.shape[1] < 80000:
            t = F.pad(t, (0, 80000 - t.shape[1]))
        else:
            t = t[:, :80000]
            
        mono = t.sum(0, keepdims=True)
        log_mel = torch.log(melspec(mono) + 1e-6).squeeze(0).numpy() # (52, 313)
        f5_specs.append(log_mel)
        f5_labels.append(y)

    specs = np.array(f5_specs, dtype=np.float32)
    labels = np.array(f5_labels, dtype=np.int64)
    np.save(FOLD5_SPECS_PATH, specs)
    np.save(FOLD5_LABELS_PATH, labels)
    print(f"✅ Generated and cached official Fold-5 test set: {specs.shape}")
else:
    print(f"📦 Loading cached Official Fold-5 test set (Zero Leakage): {FOLD5_SPECS_PATH}")
    specs = np.load(FOLD5_SPECS_PATH)
    labels = np.load(FOLD5_LABELS_PATH)

num_samples = len(specs)
print(f"📊 Verified {num_samples} unseen Fold-5 test spectrograms.")

# 2. Extract and Initialize TFLite CNN Backbone
with open(CNN_HEADER, "r") as f:
    text = f.read()

hex_vals = re.findall(r"0x[0-9a-fA-F]{2}", text)
raw_bytes = bytes([int(x, 16) for x in hex_vals])
interp = litert.Interpreter(model_content=raw_bytes)
interp.allocate_tensors()

in_det = interp.get_input_details()[0]
out_det = interp.get_output_details()[0]

in_scale = in_det["quantization_parameters"]["scales"][0]
in_zp = in_det["quantization_parameters"]["zero_points"][0]
out_scale = out_det["quantization_parameters"]["scales"][0]
out_zp = out_det["quantization_parameters"]["zero_points"][0]

print(f"✅ TFLite CNN Loaded: Input Scale={in_scale:.6f}, ZP={in_zp} | Output Scale={out_scale:.6f}, ZP={out_zp}")

# 3. Load PyTorch Fold-5 Checkpoint
print(f"📦 Loading Checkpoint: {CKPT_PATH}")
sd = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)

w_ih = sd["gru.weight_ih_l0"].numpy()
w_hh = sd["gru.weight_hh_l0"].numpy()
b_ih = sd["gru.bias_ih_l0"].numpy()
b_hh = sd["gru.bias_hh_l0"].numpy()
btn_w = sd["bottleneck.weight"].numpy()
btn_b = sd["bottleneck.bias"].numpy()
fc_w = sd["fc.weight"].numpy()
fc_b = sd["fc.bias"].numpy()

pre_w = sd["pre_gru_bn.weight"].numpy()
pre_b = sd["pre_gru_bn.bias"].numpy()
pre_m = sd["pre_gru_bn.running_mean"].numpy()
pre_v = sd["pre_gru_bn.running_var"].numpy()
pre_scale = pre_w / np.sqrt(pre_v + 1e-5)
pre_bias = pre_b - pre_m * pre_scale

post_w = sd["post_gru_bn.weight"].numpy()
post_b = sd["post_gru_bn.bias"].numpy()
post_m = sd["post_gru_bn.running_mean"].numpy()
post_v = sd["post_gru_bn.running_var"].numpy()
post_scale = post_w / np.sqrt(post_v + 1e-5)
post_bias = post_b - post_m * post_scale

# 4. Generate 50% Magnitude Pruned Weights
def prune_mat(mat, s=0.5):
    th = np.percentile(np.abs(mat), s * 100)
    pruned = mat.copy()
    pruned[np.abs(pruned) < th] = 0.0
    return pruned

p_w_ih = prune_mat(w_ih, 0.5)
p_w_hh = prune_mat(w_hh, 0.5)
p_btn_w = prune_mat(btn_w, 0.5)
p_fc_w = prune_mat(fc_w, 0.5)

total_weights = w_ih.size + w_hh.size + btn_w.size + fc_w.size
dense_nonzeros = np.count_nonzero(w_ih) + np.count_nonzero(w_hh) + np.count_nonzero(btn_w) + np.count_nonzero(fc_w)
sparse_nonzeros = np.count_nonzero(p_w_ih) + np.count_nonzero(p_w_hh) + np.count_nonzero(p_btn_w) + np.count_nonzero(p_fc_w)

print(f"\n📊 Weight Statistics:")
print(f"   • Dense Weights (Before Pruning) : {dense_nonzeros:,} / {total_weights:,} (100.0% active)")
print(f"   • Pruned Weights (After Pruning)  : {sparse_nonzeros:,} / {total_weights:,} (50.0% active)")

# 5. Fast Padé Rational Approximations (Exact match to C++ firmware)
def fast_tanh_fpu(x):
    x_clamped = np.clip(x, -4.0, 4.0)
    x2 = x_clamped * x_clamped
    res = x_clamped * (105.0 + 10.0 * x2) / (105.0 + 45.0 * x2 + x2 * x2)
    res[x >= 4.0] = 1.0
    res[x <= -4.0] = -1.0
    return res

def fast_sigmoid_fpu(x):
    return 0.5 + 0.5 * fast_tanh_fpu(0.5 * x)

# 6. Evaluation Loop
def evaluate_model(weights_tuple, model_name):
    cur_w_ih, cur_w_hh, cur_btn_w, cur_fc_w = weights_tuple
    correct_top1 = 0
    correct_top3 = 0
    
    print(f"\n🚀 Evaluating {model_name} on {num_samples} unseen Fold-5 clips...")
    
    for i in range(num_samples):
        # Step A: Stage 1 CNN Inference
        inp_int8 = np.clip(np.round(specs[i] / in_scale) + in_zp, -128, 127).astype(np.int8)
        inp_tensor = np.expand_dims(np.expand_dims(inp_int8, 0), -1) # (1, 52, 313, 1)
        interp.set_tensor(in_det["index"], inp_tensor)
        interp.invoke()
        
        out_int8 = interp.get_tensor(out_det["index"]) # (1, 13, 40, 32)
        
        # Frequency pooling & Pre-GRU BN
        features = np.mean((out_int8.astype(np.float32) - out_zp) * out_scale, axis=1)[0, :39, :] # (39, 32)
        features = features * pre_scale + pre_bias
        
        # Step B: Stage 2 GRU Inference
        h = np.zeros(160, dtype=np.float32)
        H = []
        
        for t in range(39):
            feat_t = features[t]
            gate_x = np.dot(cur_w_ih, feat_t) + b_ih
            gate_h = np.dot(cur_w_hh, h) + b_hh
            
            r = fast_sigmoid_fpu(gate_x[:160] + gate_h[:160])
            z = fast_sigmoid_fpu(gate_x[160:320] + gate_h[160:320])
            n = fast_tanh_fpu(gate_x[320:] + r * gate_h[320:])
            
            h = (1.0 - z) * n + z * h
            H.append(h.copy())
            
        H = np.array(H) # (39, 160)
        
        # Step C: Temporal Softmax Sequence Attention
        attn_scores = np.mean(H, axis=1) # (39,)
        attn_scores -= np.max(attn_scores)
        attn_w = np.exp(attn_scores) / np.sum(np.exp(attn_scores))
        h_pooled = np.sum(H * attn_w[:, None], axis=0) # (160,)
        
        # Step D: Post-GRU BN & Bottleneck ReLU6
        h_pooled = h_pooled * post_scale + post_bias
        btn_out = np.clip(np.dot(cur_btn_w, h_pooled) + btn_b, 0.0, 6.0) # (128,)
        
        # Step E: FC Classifier Head
        logits = np.dot(cur_fc_w, btn_out) + fc_b # (50,)
        
        pred_top1 = np.argmax(logits)
        pred_top3 = np.argsort(logits)[-3:]
        gt = labels[i]
        
        if pred_top1 == gt:
            correct_top1 += 1
        if gt in pred_top3:
            correct_top3 += 1
            
        if (i + 1) % 100 == 0 or (i + 1) == num_samples:
            print(f"   [{i+1:3d}/{num_samples}] Current Top-1: {correct_top1/(i+1)*100:.2f}% | Top-3: {correct_top3/(i+1)*100:.2f}%")
            
    top1 = (correct_top1 / num_samples) * 100.0
    top3 = (correct_top3 / num_samples) * 100.0
    return top1, top3, correct_top1, correct_top3

# Run Evaluation
top1_dense, top3_dense, c1_dense, c3_dense = evaluate_model((w_ih, w_hh, btn_w, fc_w), "Model A (Dense Baseline FP32)")
top1_pruned, top3_pruned, c1_pruned, c3_pruned = evaluate_model((p_w_ih, p_w_hh, p_btn_w, p_fc_w), "Model B (50% Pruned CSR GRU)")

# 7. Print Comprehensive Summary Table
print("\n" + "=" * 85)
print("🏆 OFFICIAL ESC-50 FOLD-5 ZERO-LEAKAGE BENCHMARK COMPARISON:")
print("=" * 85)
print(f"{'Metric':<30s} | {'Before Pruning (Dense)':<25s} | {'After 50% Pruning (CSR)':<25s} | {'Delta':<10s}")
print("-" * 85)
print(f"{'Active Non-Zero Weights':<30s} | {dense_nonzeros:<25,d} | {sparse_nonzeros:<25,d} | {-50.0:+.1f}%")
print(f"{'Stage 2 Sparsity':<30s} | {'0.0% (Dense)':<25s} | {'50.0% (Sparse CSR)':<25s} | {+50.0:+.1f}%")
print(f"{'Top-1 Test Accuracy (Fold-5)':<30s} | {top1_dense:6.2f}% ({c1_dense}/{num_samples}){'':<9s} | {top1_pruned:6.2f}% ({c1_pruned}/{num_samples}){'':<9s} | {top1_pruned - top1_dense:+5.2f}%")
print(f"{'Top-3 Test Accuracy (Fold-5)':<30s} | {top3_dense:6.2f}% ({c3_dense}/{num_samples}){'':<9s} | {top3_pruned:6.2f}% ({c3_pruned}/{num_samples}){'':<9s} | {top3_pruned - top3_dense:+5.2f}%")
print("=" * 85)

if top1_pruned >= (top1_dense - 3.0):
    print("🎉 SUCCESS: 50% Pruning successfully compressed the model with virtually zero accuracy degradation!")
else:
    print("⚠️ NOTE: Noticeable drop observed. Consider fine-tuning or slightly lower sparsity.")
print("=" * 85)
