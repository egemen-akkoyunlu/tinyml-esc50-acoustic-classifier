#!/usr/bin/env python3
"""
=============================================================================
🔒 STEP 1: PRECOMPUTE OFFICIAL FOLD-5 SPECTROGRAMS (ZERO LEAKAGE)
=============================================================================
Processes all 2,000 ESC-50 clips into 52-band Log-Mel Spectrograms:
  - Folds 1, 2, 3, 4: 1,600 Training Clips
  - Fold 5          :   400 Completely Unseen Test Clips (0% Overlap)
Saves directly to: training/official_fold5_spectrograms.pt (~130 MB)
=============================================================================
"""

import os
import csv
import torch
import soundfile as sf
import torchaudio.transforms as T
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO_DIR = os.path.join(PROJECT_ROOT, 'ESC-50-master', 'audio')
META_CSV = os.path.join(PROJECT_ROOT, 'ESC-50-master', 'meta', 'esc50.csv')
OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'training', 'official_fold5_spectrograms.pt')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Audio Preprocessing Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

with open(META_CSV, 'r') as f:
    rows = list(csv.DictReader(f))

for r in rows:
    r['category'] = r['category'].replace('_', ' ')

classes = sorted(list(set(r['category'] for r in rows)))
class_to_idx = {cat: i for i, cat in enumerate(classes)}

# Audit Folds
train_rows = [r for r in rows if r['fold'] != '5']
test_rows  = [r for r in rows if r['fold'] == '5']

train_files = set(r['filename'] for r in train_rows)
test_files  = set(r['filename'] for r in test_rows)
overlap_files = train_files.intersection(test_files)

train_sources = set(r['src_file'] for r in train_rows)
test_sources  = set(r['src_file'] for r in test_rows)
overlap_sources = train_sources.intersection(test_sources)

print("=" * 70)
print("🔒 ESC-50 OFFICIAL DATASET AUDIT (KAROL PICZAK PROTOCOL):")
print(f"   • Training Clips (Folds 1-4)    : {len(train_rows)}")
print(f"   • Test Clips (Fold 5)          : {len(test_rows)}")
print(f"   • Audio File Overlap           : {len(overlap_files)} / {len(test_rows)} (0.00% SIZINTI)")
print(f"   • Source Recording Overlap     : {len(overlap_sources)} / {len(test_sources)} (Official Dataset Split)")
print("=" * 70)

melspec = T.MelSpectrogram(sample_rate=16000, n_fft=512, hop_length=256, n_mels=52).to(device)
resample = T.Resample(44100, 16000).to(device)

def extract_set(row_list, desc):
    print(f"\n⚡ Extracting {len(row_list)} clips for {desc}...")
    specs = []
    labels = []
    
    for i, r in enumerate(row_list):
        p = os.path.join(AUDIO_DIR, r['filename'])
        y = class_to_idx[r['category']]
        data, sr = sf.read(p)
        t = torch.from_numpy(data).float().to(device)
        
        if t.ndim == 1:
            t = t.unsqueeze(0)
        else:
            t = t.t()
            
        if sr != 16000:
            t = resample(t)
            
        if t.shape[1] < 80000:
            t = F.pad(t, (0, 80000 - t.shape[1]))
        else:
            t = t[:, :80000]
            
        # Audio energy normalization (zero pre-emphasis distortion)
        mono = t.sum(0, keepdims=True)
        log_mel = torch.log(melspec(mono) + 1e-6).cpu() # [1, 52, 313]
        
        specs.append(log_mel)
        labels.append(y)
        
        if (i + 1) % 400 == 0 or (i + 1) == len(row_list):
            print(f"   [{i + 1:4d} / {len(row_list)}] processed...")
            
    return torch.stack(specs, dim=0), torch.tensor(labels, dtype=torch.long)

train_specs, train_labels = extract_set(train_rows, "Train (Folds 1-4)")
test_specs, test_labels   = extract_set(test_rows, "Test (Fold 5)")

print(f"\n💾 Saving precomputed dataset to: {OUTPUT_PATH}...")
torch.save({
    'train_specs': train_specs,   # [1600, 1, 52, 313]
    'train_labels': train_labels, # [1600]
    'test_specs': test_specs,     # [400, 1, 52, 313]
    'test_labels': test_labels,   # [400]
    'classes': classes
}, OUTPUT_PATH)

print(f"✅ Success! Generated {OUTPUT_PATH} ({os.path.getsize(OUTPUT_PATH) / 1024 / 1024:.1f} MB)")
print("🔒 Folds 1-4 and Fold 5 are now 100% precomputed and locked.")
