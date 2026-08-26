#!/usr/bin/env python3
import os
import torch
import torchaudio
import torch.nn.functional as F

audio_path = "/home/acar/new_task/ESC-50-master/audio/1-94231-B-32.wav"
output_header = "/home/acar/zephyrproject/my_apps/silabs_ble_audio_peripheral/src/golden_keyboard_typing_pcm.h"

waveform, sr = torchaudio.load(audio_path)
if sr != 16000:
    waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

if waveform.shape[1] < 80000:
    waveform = F.pad(waveform, (0, 80000 - waveform.shape[1]))
else:
    waveform = waveform[:, :80000]

# Convert float [-1.0, 1.0] to int16 [-32768, 32767]
pcm_int16 = (waveform.squeeze(0).clamp(-1.0, 1.0) * 32767.0).to(torch.int16).numpy()

content = []
content.append("#ifndef GOLDEN_KEYBOARD_TYPING_PCM_H\n#define GOLDEN_KEYBOARD_TYPING_PCM_H\n\n#include <stdint.h>\n\n")
content.append("/* Real 5-Second 16kHz PCM Audio from ESC-50: 1-137-A-32.wav (keyboard_typing) */\n")
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

print(f"✅ Generated {output_header} from {audio_path} ({len(pcm_int16)} samples)!")
