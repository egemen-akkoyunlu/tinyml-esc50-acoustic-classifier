#!/usr/bin/env python3
import os
import sys
import csv
import torch
import soundfile as sf
import torchaudio.transforms as T
import torch.nn.functional as F
import torch.ao.quantization as quantization
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from training.train_ultimate_int8_csr import AudioPhiNetCRNNClassifierQAT

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

cache_file = os.path.join(PROJECT_ROOT, 'training', 'cached_features_fold5_rtx4050.pt')

meta_csv = os.path.join(PROJECT_ROOT, 'ESC-50-master', 'meta', 'esc50.csv')
with open(meta_csv, 'r') as f:
    rows = list(csv.DictReader(f))

for r in rows:
    r['category'] = r['category'].replace('_', ' ')

classes = sorted(list(set(r['category'] for r in rows)))
class_to_idx = {cat: i for i, cat in enumerate(classes)}
targets = [class_to_idx[r['category']] for r in rows]
audio_paths = [os.path.join(PROJECT_ROOT, 'ESC-50-master', 'audio', r['filename']) for r in rows]

# Official Piczak Fold Split (Folds 1-4 Train, Fold 5 Test)
train_idx = [i for i, r in enumerate(rows) if r['fold'] != '5']
val_idx   = [i for i, r in enumerate(rows) if r['fold'] == '5']

# Check source overlap
train_src = set(rows[i]['src_file'] for i in train_idx)
val_src   = set(rows[i]['src_file'] for i in val_idx)
overlap   = train_src.intersection(val_src)

print("=" * 70)
print("🔒 ESC-50 OFFICIAL ZERO-LEAKAGE FOLD-5 AUDIT:")
print(f"   • Train Clips (Folds 1-4)    : {len(train_idx)}")
print(f"   • Test Clips (Fold 5)        : {len(val_idx)}")
print(f"   • File Overlap               : {len(set(train_idx).intersection(set(val_idx)))} / {len(val_idx)} (0.00%)")
print(f"   • Unique Train Sound Sources : {len(train_src)}")
print(f"   • Unique Test Sound Sources  : {len(val_src)}")
print(f"   • Source Recording Overlap   : {len(overlap)} (Dataset Freesound source overlap)")
print("=" * 70)

# Load Master Teacher
print("\n🎓 Loading 91.50% Master Teacher Model...")
teacher = AudioPhiNetCRNNClassifierQAT(num_classes=50).to(device)
teacher.eval()
teacher.qconfig = quantization.get_default_qat_qconfig('fbgemm')
torch.ao.quantization.fuse_modules(teacher, [['stem.0', 'stem.1']], inplace=True)
torch.ao.quantization.fuse_modules(teacher.phi_blocks[0], [['conv.0', 'conv.1'], ['conv.3', 'conv.4']], inplace=True)
torch.ao.quantization.fuse_modules(teacher.phi_blocks[2], [['conv.0', 'conv.1'], ['conv.3', 'conv.4']], inplace=True)
teacher.gru.qconfig = None
teacher.bottleneck.qconfig = None
teacher.fc.qconfig = None
teacher.train()
quantization.prepare_qat(teacher, inplace=True)

t_ckpt = os.path.join(PROJECT_ROOT, 'models', 'best_distilled_qat_model.pth')
t_sd = torch.load(t_ckpt, map_location=device, weights_only=False)
teacher.load_state_dict(t_sd, strict=True)
teacher.eval()

melspec = T.MelSpectrogram(sample_rate=16000, n_fft=512, hop_length=256, n_mels=52).to(device)
resample = T.Resample(44100, 16000).to(device)

def compute_features(indices):
    features, labels, t_logits = [], [], []
    with torch.no_grad():
        for idx in indices:
            p = audio_paths[idx]
            y = targets[idx]
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

            log_mel = torch.log(melspec(t.sum(0, keepdims=True)) + 1e-6).unsqueeze(0)
            feat = teacher.extract_features(log_mel).cpu()
            t_out = teacher(log_mel).cpu()

            features.append(feat)
            labels.append(y)
            t_logits.append(t_out)

    return torch.cat(features, dim=0), torch.tensor(labels, dtype=torch.long), torch.cat(t_logits, dim=0)

print("⚡ Extracting and caching features for Official Fold 5...")
train_feat, train_labels, teacher_train_logits = compute_features(train_idx)
val_feat, val_labels, _ = compute_features(val_idx)

torch.save({
    'train_feat': train_feat,
    'train_labels': train_labels,
    'teacher_train_logits': teacher_train_logits,
    'val_feat': val_feat,
    'val_labels': val_labels
}, cache_file)
print(f"✅ Saved precomputed features to {cache_file}!")
