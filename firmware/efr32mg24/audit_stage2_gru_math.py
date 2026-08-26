#!/usr/bin/env python3
"""
================================================================================
🔬 AUDIT SCRIPT 3: STAGE 2 C++ GRU MATH SIMULATION & PREDICTION AUDIT
================================================================================
Simulates the exact C++ Stage 2 logic (inference.cpp) using NumPy.
Prints intermediate values, Attention weights, Logits, and Top-5 predictions.
================================================================================
"""

import os
import sys
import numpy as np

FEAT_PATH = "/home/acar/zephyrproject/my_apps/silabs_ble_audio_peripheral/audit_tflite_features.npy"
WEIGHTS_PATH = "/home/acar/new_task/stage2_weights.npz"

ESC50_CLASSES = [
    "airplane", "breathing", "brushing teeth", "can opening", "car horn",
    "cat", "chainsaw", "chirping birds", "church bells", "clapping",
    "clock alarm", "clock tick", "coughing", "cow", "crackling fire",
    "crickets", "crow", "crying baby", "dog", "door wood creaks",
    "door wood knock", "drinking sipping", "engine", "fireworks", "footsteps",
    "frog", "glass breaking", "hand saw", "helicopter", "hen",
    "insects", "keyboard typing", "laughing", "mouse click", "pig",
    "pouring water", "rain", "rooster", "sea waves", "sheep",
    "siren", "sneezing", "snoring", "thunderstorm", "toilet flush",
    "train", "vacuum cleaner", "washing machine", "water drops", "wind"
]

def main():
    print("=" * 80)
    print("🔬 AUDIT 3: STAGE 2 C++ GRU & CLASSIFIER HEAD SIMULATION")
    print("=" * 80)

    if not os.path.exists(FEAT_PATH):
        print(f"❌ Error: Please run audit_tflite_backbone.py first to generate {FEAT_PATH}!")
        sys.exit(1)

    features = np.load(FEAT_PATH) # (39, 32)
    w = np.load(WEIGHTS_PATH)

    w_ih = w['w_ih'] # (480, 32)
    w_hh = w['w_hh'] # (480, 160)
    b_ih = w['b_ih'] # (480,)
    b_hh = w['b_hh'] # (480,)
    pre_scale = w['pre_scale'] # (32,)
    pre_bias = w['pre_bias'] # (32,)
    post_scale = w['post_scale'] # (160,)
    post_bias = w['post_bias'] # (160,)
    btn_w = w['btn_w'] # (128, 160)
    btn_b = w['btn_b'] # (128,)
    fc_w = w['fc_w'] # (50, 128)
    fc_b = w['fc_b'] # (50,)

    print(f"-> Loaded Feature Map: shape={features.shape}")
    print(f"   • Features Min/Max: [{features.min():.4f}, {features.max():.4f}]")

    # 1. Pre-GRU BatchNorm
    feat_norm = np.zeros_like(features)
    for t in range(39):
        for c in range(32):
            feat_norm[t, c] = features[t, c] * pre_scale[c] + pre_bias[c]

    # 2. Recurrent GRU Cell (39 steps)
    H = np.zeros((39, 160), dtype=np.float32)
    h = np.zeros(160, dtype=np.float32)

    for t in range(39):
        # Gate X: W_ih @ x + b_ih
        gate_x = np.dot(w_ih, feat_norm[t]) + b_ih
        # Gate H: W_hh @ h + b_hh
        gate_h = np.dot(w_hh, h) + b_hh

        for j in range(160):
            r_in = np.clip(gate_x[j] + gate_h[j], -30.0, 30.0)
            r = 1.0 / (1.0 + np.exp(-r_in))

            z_in = np.clip(gate_x[160 + j] + gate_h[160 + j], -30.0, 30.0)
            z = 1.0 / (1.0 + np.exp(-z_in))

            n = np.tanh(gate_x[320 + j] + r * gate_h[320 + j])

            h[j] = (1.0 - z) * n + z * h[j]
            H[t, j] = h[j]

    # 3. Softmax Sequence Attention Pooling
    attn_scores = H.mean(axis=-1) # (39,)
    max_attn = np.max(attn_scores)
    exp_attn = np.exp(attn_scores - max_attn)
    attn_weights = exp_attn / np.sum(exp_attn)

    h_pooled = np.zeros(160, dtype=np.float32)
    for t in range(39):
        h_pooled += H[t] * attn_weights[t]

    # 4. Post-GRU BatchNorm
    h_post = h_pooled * post_scale + post_bias

    # 5. Bottleneck Linear (160 -> 128) + ReLU6
    btn_out = np.clip(np.dot(btn_w, h_post) + btn_b, 0.0, 6.0)

    # 6. FC Head (128 -> 50) + Softmax
    logits = np.dot(fc_w, btn_out) + fc_b
    exp_l = np.exp(logits - np.max(logits))
    probs = exp_l / np.sum(exp_l)

    top5 = np.argsort(probs)[::-1][:5]

    print("\n🏆 STAGE 2 PREDICTION RESULTS (Ground Truth: 'siren'):")
    print("-" * 65)
    for rank, idx in enumerate(top5, 1):
        sound_name = ESC50_CLASSES[idx]
        conf = probs[idx] * 100.0
        status = "✅ TARGET" if sound_name == "siren" else ""
        print(f"   #{rank}: Class {idx:2d} -> {sound_name:<20} : {conf:6.2f}% {status}")
    print("-" * 65)

if __name__ == "__main__":
    main()
