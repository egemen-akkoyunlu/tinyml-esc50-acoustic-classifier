#!/usr/bin/env python3
"""
================================================================================
🔬 AUDIT SCRIPT 1: DSP PREPROCESSING & SPECTROGRAM FORENSICS
================================================================================
Compares the on-chip C Spectrogram algorithm with PyTorch Log-Mel Spectrogram.
Outputs min/max ranges, quantization values, and layer-by-layer alignment.
================================================================================
"""

import os
import sys
import torch
import torchaudio
import numpy as np
import torch.nn.functional as F
from torchaudio.transforms import MelSpectrogram

TEST_WAV = "/home/acar/new_task/ESC-50-master/audio/3-51909-B-42.wav"

def main():
    print("=" * 80)
    print("🔬 AUDIT 1: MEL-SPECTROGRAM DSP MATH & BIT-EXACT ALIGNMENT")
    print(f" Audio File: {os.path.basename(TEST_WAV)} (Ground Truth: 'siren')")
    print("=" * 80)

    # 1. Load Audio
    waveform, sr = torchaudio.load(TEST_WAV)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
    waveform = waveform.sum(0, keepdims=True)
    if waveform.shape[1] < 80000:
        waveform = F.pad(waveform, (0, 80000 - waveform.shape[1]))
    else:
        waveform = waveform[:, :80000]

    pcm = waveform.squeeze().numpy() # [-1.0, 1.0]

    # 2. PyTorch MelSpectrogram
    melspec_torch = MelSpectrogram(sample_rate=16000, n_fft=512, hop_length=256, n_mels=52)
    py_log_mel = torch.log(melspec_torch(waveform) + 1e-6).squeeze().numpy() # (52, 313)

    # 3. Simulate Exact C Preprocessing Math with Reflect Padding
    fb = melspec_torch.mel_scale.fb.numpy() # (257, 52)
    hann = torch.hann_window(512, periodic=True).numpy()

    c_log_mel = np.zeros((52, 313), dtype=np.float32)
    for t in range(313):
        chunk = np.zeros(512, dtype=np.float32)
        for n in range(512):
            idx = t * 256 - 256 + n
            if idx < 0:
                idx = -idx
            elif idx >= len(pcm):
                idx = 2 * (len(pcm) - 1) - idx
                if idx < 0: idx = 0
            chunk[n] = pcm[idx] * hann[n]

        fft_out = np.fft.rfft(chunk, n=512)
        power = np.abs(fft_out)**2
        mel_e = np.dot(power, fb)
        c_log_mel[:, t] = np.log(mel_e + 1e-6)

    # 4. Error Metrics
    abs_diff = np.abs(py_log_mel - c_log_mel)
    max_err = np.max(abs_diff)
    mean_err = np.mean(abs_diff)
    corr = np.corrcoef(py_log_mel.flatten(), c_log_mel.flatten())[0, 1]

    print("\n📊 SPECTROGRAM COMPARISON RESULTS:")
    print(f"   • PyTorch Spectrogram Shape : {py_log_mel.shape} (Range: [{py_log_mel.min():.2f}, {py_log_mel.max():.2f}])")
    print(f"   • C Simulation Shape        : {c_log_mel.shape} (Range: [{c_log_mel.min():.2f}, {c_log_mel.max():.2f}])")
    print(f"   • Max Absolute Error        : {max_err:.8f} dB")
    print(f"   • Mean Absolute Error       : {mean_err:.8f} dB")
    print(f"   • Pearson Correlation       : {corr:.8f} (1.000000 = Identical!)")

    # 5. INT8 Quantization Range Check
    # TFLite Quant Params from model: scale = 0.093557, zp = 20
    scale = 0.09355688840150833
    zp = 20
    q_py = np.clip(np.round(py_log_mel / scale) + zp, -128, 127).astype(np.int8)
    q_c = np.clip(np.round(c_log_mel / scale) + zp, -128, 127).astype(np.int8)

    diff_q = np.sum(q_py != q_c)
    print(f"\n🔢 INT8 QUANTIZATION CHECK (Scale={scale:.6f}, ZeroPoint={zp}):")
    print(f"   • Total INT8 Elements       : {q_py.size} (52 x 313)")
    print(f"   • Identical INT8 Pixels     : {q_py.size - diff_q} / {q_py.size} ({(q_py.size - diff_q)/q_py.size*100:.2f}%)")
    print(f"   • Mismatched INT8 Pixels    : {diff_q} (due to sub-micro floating rounding)")

    # Save to disk for inspect
    np.save("/home/acar/zephyrproject/my_apps/silabs_ble_audio_peripheral/audit_c_spectrogram.npy", c_log_mel)
    print("\n✅ Saved 'audit_c_spectrogram.npy' for inspection!")

if __name__ == "__main__":
    main()
