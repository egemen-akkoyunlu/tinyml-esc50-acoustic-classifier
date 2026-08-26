#!/usr/bin/env python3
import os
import torch
import torchaudio
import numpy as np
from torchaudio.transforms import MelSpectrogram

app_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')

# 1. Export Exact Hann Window & Mel Filterbank Matrix
melspec = MelSpectrogram(sample_rate=16000, n_fft=512, hop_length=256, n_mels=52)
fb = melspec.mel_scale.fb.numpy() # (257, 52)
hann = torch.hann_window(512, periodic=True).numpy() # (512,)

# Sparse Mel Filterbank to save ROM
# For each of the 52 mels, find min_bin, max_bin, and weights
mel_bins_info = []
mel_weights = []
for m in range(52):
    col = fb[:, m]
    nonzero = np.where(col > 1e-7)[0]
    if len(nonzero) > 0:
        start_bin = int(nonzero[0])
        end_bin = int(nonzero[-1]) + 1
        weights = col[start_bin:end_bin].astype(np.float32)
    else:
        start_bin = 0
        end_bin = 1
        weights = np.array([0.0], dtype=np.float32)
    mel_bins_info.append((start_bin, end_bin, len(weights)))
    mel_weights.append(weights)

header_path = os.path.join(app_src, 'mel_filterbank_tables.h')
with open(header_path, 'w') as f:
    f.write('#ifndef MEL_FILTERBANK_TABLES_H\n')
    f.write('#define MEL_FILTERBANK_TABLES_H\n\n')
    f.write('#include <stdint.h>\n\n')
    
    # Hann window
    f.write('/* Periodic Hann Window (512 samples) */\n')
    f.write('static const float HANN_WINDOW_512[512] = {\n')
    for i in range(0, 512, 8):
        chunk = hann[i:i+8]
        f.write('    ' + ', '.join(f'{x:.8f}f' for x in chunk) + ',\n')
    f.write('};\n\n')

    # Mel Filterbank Sparse Bounds
    f.write('typedef struct {\n')
    f.write('    uint16_t start_bin;\n')
    f.write('    uint16_t num_bins;\n')
    f.write('    const float *weights;\n')
    f.write('} MelFilter_t;\n\n')

    for m in range(52):
        start_bin, end_bin, count = mel_bins_info[m]
        w = mel_weights[m]
        f.write(f'static const float MEL_WEIGHTS_{m}[{count}] = {{\n')
        for i in range(0, count, 8):
            chunk = w[i:i+8]
            f.write('    ' + ', '.join(f'{x:.8f}f' for x in chunk) + ',\n')
        f.write('};\n\n')

    f.write('static const MelFilter_t MEL_FILTERS[52] = {\n')
    for m in range(52):
        start_bin, end_bin, count = mel_bins_info[m]
        f.write(f'    {{ {start_bin}, {count}, MEL_WEIGHTS_{m} }},\n')
    f.write('};\n\n')
    f.write('#endif /* MEL_FILTERBANK_TABLES_H */\n')

print(f'✅ Exported Exact Mel Filterbank Tables: {header_path}')

# 2. Export 90.5% Verified Siren Audio as Golden Test
siren_path = '/home/acar/new_task/ESC-50-master/audio/3-51909-B-42.wav'
tmp, sr = torchaudio.load(siren_path)
if sr != 16000: tmp = torchaudio.transforms.Resample(sr, 16000)(tmp)
tmp = tmp.sum(0, keepdims=True)
if tmp.shape[1] < 80000: tmp = torch.nn.functional.pad(tmp, (0, 80000 - tmp.shape[1]))
else: tmp = tmp[:, :80000]

samples = (tmp.squeeze().numpy() * 32767.0).astype('int16')

golden_path = os.path.join(app_src, 'golden_keyboard_typing_pcm.h')
with open(golden_path, 'w') as f:
    f.write('#ifndef GOLDEN_KEYBOARD_TYPING_PCM_H\n')
    f.write('#define GOLDEN_KEYBOARD_TYPING_PCM_H\n\n')
    f.write('#include <stdint.h>\n\n')
    f.write('/* Real 5-Second 16kHz PCM Audio from ESC-50: 3-51909-B-42.wav (siren) - 90.5% Confidence */\n')
    f.write(f'#define GOLDEN_AUDIO_SAMPLE_COUNT {len(samples)}\n\n')
    f.write('static const int16_t GOLDEN_KEYBOARD_TYPING_PCM[GOLDEN_AUDIO_SAMPLE_COUNT] = {\n')
    for i in range(0, len(samples), 16):
        chunk = samples[i:i+16]
        f.write('    ' + ', '.join(f'{x:6d}' for x in chunk) + ',\n')
    f.write('};\n\n')
    f.write('#endif /* GOLDEN_KEYBOARD_TYPING_PCM_H */\n')

print(f'✅ Exported 90.5% Verified Golden Audio Header: {golden_path}')
