#!/usr/bin/env python3
import os
import torch
import torchaudio
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
audio_path = os.path.join(PROJECT_ROOT, "ESC-50-master", "audio", "1-94231-B-32.wav")
output_header = os.path.join(PROJECT_ROOT, "firmware", "efr32mg24", "src", "golden_keyboard_typing_pcm.h")

waveform, sr = torchaudio.load(audio_path)
if sr != 16000:
    waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

if waveform.shape[1] < 80000:
    waveform = F.pad(waveform, (0, 80000 - waveform.shape[1]))
else:
    waveform = waveform[:, :80000]

pcm_int16 = (waveform.squeeze(0).clamp(-1.0, 1.0) * 32767.0).to(torch.int16).numpy()

content = []
content.append("#ifndef GOLDEN_KEYBOARD_TYPING_PCM_H\n#define GOLDEN_KEYBOARD_TYPING_PCM_H\n\n#include <stdint.h>\n\n")
content.append("/* Real 5-Second 16kHz PCM Audio from Held-Out Validation: 1-94231-B-32.wav (keyboard_typing) */\n")
content.append(f"#define GOLDEN_AUDIO_SAMPLE_COUNT {len(pcm_int16)}\n\n")
content.append(f"static const int16_t GOLDEN_KEYBOARD_TYPING_PCM[GOLDEN_AUDIO_SAMPLE_COUNT] = {{\n")

for i in range(0, len(pcm_int16), 16):
    chunk = ", ".join(f"{v:6d}" for v in pcm_int16[i:i+16])
    if i + 16 < len(pcm_int16):
        content.append(f"  {chunk},\n")
    else:
        content.append(f"  {chunk}\n")

content.append("};\n\n#endif // GOLDEN_KEYBOARD_TYPING_PCM_H\n")

with open(output_header, "w") as f:
    f.writelines(content)

print(f"✅ Generated {output_header} from true held-out validation clip: {os.path.basename(audio_path)} ({len(pcm_int16)} samples)!")
