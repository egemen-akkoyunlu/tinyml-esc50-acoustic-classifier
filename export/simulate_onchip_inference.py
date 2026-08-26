#!/usr/bin/env python3
import os
import numpy as np
try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite.python.interpreter import Interpreter
import torchaudio
import torch
import torch.nn.functional as F
from torchaudio.transforms import MelSpectrogram

root_dir = "/home/acar/new_task"
tflite_path = os.path.join(root_dir, "tflite_models", "cnn_backbone_out", "cnn_backbone_full_integer_quant.tflite")
stage2_npz_path = os.path.join(root_dir, "stage2_weights.npz")
audio_path = os.path.join(root_dir, "ESC-50-master", "audio", "1-94231-B-32.wav")

# 1. Load TFLite Model
interpreter = Interpreter(model_path=tflite_path)
interpreter.allocate_tensors()
in_details = interpreter.get_input_details()[0]
out_details = interpreter.get_output_details()[0]

in_scale, in_zp = in_details['quantization']
out_scale, out_zp = out_details['quantization']

# 2. Extract Spectrogram exactly as done in DSP
tmp, sr = torchaudio.load(audio_path)
if sr != 16000:
    tmp = torchaudio.transforms.Resample(sr, 16000)(tmp)
if tmp.shape[1] < 80000:
    tmp = F.pad(tmp, (0, 80000 - tmp.shape[1]))
else:
    tmp = tmp[:, :80000]

melspec = MelSpectrogram(sample_rate=16000, n_fft=512, hop_length=256, n_mels=52)
log_mel = torch.log(melspec(tmp.sum(0, keepdims=True)) + 1e-6).numpy() # (1, 52, 313)

# Quantize to INT8
log_mel_int8 = np.clip(np.round(log_mel / in_scale) + in_zp, -128, 127).astype(np.int8)
log_mel_in = np.expand_dims(np.transpose(log_mel_int8, (1, 2, 0)), axis=0) # (1, 52, 313, 1)

interpreter.set_tensor(in_details['index'], log_mel_in)
interpreter.invoke()
cnn_out_int8 = interpreter.get_tensor(out_details['index']) # (1, 13, 40, 32)

print("=" * 80)
print(f"🔬 BIT-EXACT SIMULATION OF EFR32MG24 ON {os.path.basename(audio_path)}")
print("=" * 80)
print(f"CNN Out Shape: {cnn_out_int8.shape}, Quant: scale={out_scale:.6f}, zp={out_zp}")

# 3. Stage 2 in Pure Python/NumPy exactly matching inference.cpp
stage2 = np.load(stage2_npz_path)
w_ih = stage2['w_ih']
w_hh = stage2['w_hh']
b_ih = stage2['b_ih']
b_hh = stage2['b_hh']
pre_scale = stage2['pre_scale']
pre_bias = stage2['pre_bias']
post_scale = stage2['post_scale']
post_bias = stage2['post_bias']
btn_w = stage2['btn_w']
btn_b = stage2['btn_b']
fc_w = stage2['fc_w']
fc_b = stage2['fc_b']

# Frequency pooling (13 -> 1) & Slice 39 steps
pooled = np.mean((cnn_out_int8[0, :, :39, :].astype(np.float32) - out_zp) * out_scale, axis=0) # (39, 32)
features = pooled * pre_scale + pre_bias # (39, 32)

print(f"Feat Norm t= 0 [0..4]: {features[0, :5]}")
print(f"Feat Norm t= 1 [0..4]: {features[1, :5]}")
print(f"Feat Norm t= 2 [0..4]: {features[2, :5]}")

# GRU Simulation
hidden = np.zeros(160, dtype=np.float32)
all_h = []

for t in range(39):
    x_t = features[t] # (32,)
    # PyTorch GRU gate ordering: r, z, n
    gates_x = np.dot(w_ih, x_t) + b_ih # (480,)
    gates_h = np.dot(w_hh, hidden) + b_hh # (480,)
    
    r = 1.0 / (1.0 + np.exp(-(gates_x[0:160] + gates_h[0:160])))
    z = 1.0 / (1.0 + np.exp(-(gates_x[160:320] + gates_h[160:320])))
    n = np.tanh(gates_x[320:480] + r * gates_h[320:480])
    
    hidden = (1.0 - z) * n + z * hidden
    all_h.append(hidden.copy())

all_h = np.array(all_h) # (39, 160)

# Softmax Attention Pooling
attn_scores = np.mean(all_h, axis=-1) # (39,)
attn_weights = np.exp(attn_scores - np.max(attn_scores))
attn_weights /= np.sum(attn_weights)
h_pooled = np.sum(all_h * np.expand_dims(attn_weights, -1), axis=0) # (160,)

h_norm = h_pooled * post_scale + post_bias
btn_out = np.clip(np.dot(btn_w, h_norm) + btn_b, 0.0, 6.0)
logits = np.dot(fc_w, btn_out) + fc_b

classes = [
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

probs = np.exp(logits - np.max(logits))
probs /= np.sum(probs)

top_indices = np.argsort(logits)[::-1][:5]
print("\n📊 INT8 TFLite + C++ Stage 2 Predictions:")
for rank, idx in enumerate(top_indices):
    print(f"  #{rank+1}: Class {idx:2d} ({classes[idx]:<20s}) -> {probs[idx]*100:5.2f}% (Logit: {logits[idx]:.4f})")
print("=" * 80)
