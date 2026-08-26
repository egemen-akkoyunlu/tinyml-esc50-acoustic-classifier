#!/usr/bin/env python3
"""
================================================================================
🧪 TEST AUDIO CLIPS AGAINST TRAINED PYTORCH MODEL
================================================================================
Runs the trained QAT model across multiple real sound clips to demonstrate
accuracy on high-confidence sounds (siren, clock alarm, engine, dog, etc.).
================================================================================
"""

import os
import sys
import torch
import torchaudio
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torchaudio.transforms import MelSpectrogram

# Add task directory to path to import model architecture
sys.path.append("/home/acar/new_task")
from qat_training import AudioPhiNetCRNNClassifierQAT

ESC50_AUDIO_DIR = "/home/acar/new_task/ESC-50-master/audio"
ESC50_CSV = "/home/acar/new_task/ESC-50-master/meta/esc50.csv"
MODEL_CHECKPOINT = "/home/acar/new_task/best_qat_model.pth"

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
    print("🧪 EVALUATING TRAINED PYTORCH MODEL ON GOLDEN TEST SOUNDS")
    print("=" * 80)

    # 1. Load Model
    model = AudioPhiNetCRNNClassifierQAT(num_classes=50)
    ckpt = torch.load(MODEL_CHECKPOINT, map_location='cpu')
    model.load_state_dict(ckpt, strict=False)
    model.eval()

    melspec = MelSpectrogram(sample_rate=16000, n_fft=512, hop_length=256, n_mels=52)

    # Selected diverse test sounds from ESC-50
    test_files = [
        ("3-51909-B-42.wav", "siren"),
        ("1-72195-B-37.wav", "clock alarm"),
        ("4-186962-A-44.wav", "engine"),
        ("5-223176-A-37.wav", "clock alarm"),
        ("1-50661-A-44.wav", "engine"),
        ("1-137-A-32.wav", "keyboard typing"),
    ]

    print(f"\n{'Ground Truth':<18} | {'Audio File':<20} | {'Predicted Sound':<18} | {'Confidence':<10} | {'Status'}")
    print("-" * 80)

    for fn, target_name in test_files:
        path = os.path.join(ESC50_AUDIO_DIR, fn)
        if not os.path.exists(path):
            continue

        waveform, sr = torchaudio.load(path)
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
        waveform = waveform.sum(0, keepdims=True)
        if waveform.shape[1] < 80000:
            waveform = F.pad(waveform, (0, 80000 - waveform.shape[1]))
        else:
            waveform = waveform[:, :80000]

        log_mel = torch.log(melspec(waveform) + 1e-6).unsqueeze(0)

        with torch.no_grad():
            logits = model(log_mel)
            probs = torch.softmax(logits, dim=1).squeeze().numpy()
            pred_idx = np.argmax(probs)
            pred_name = ESC50_CLASSES[pred_idx]
            conf = probs[pred_idx] * 100.0

        is_match = (pred_name == target_name)
        status_str = "✅ MATCH" if is_match else "❌ DIFF"

        print(f"{target_name:<18} | {fn:<20} | {pred_name:<18} | {conf:6.2f}%    | {status_str}")

    print("-" * 80)
    print("👉 To test your own recorded WAV file (board_mic_test.wav):")
    print("   Run: /home/acar/kws_env/bin/python listen_board_mic.py\n")

if __name__ == "__main__":
    main()
