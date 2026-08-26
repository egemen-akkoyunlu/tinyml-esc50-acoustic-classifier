#!/usr/bin/env python3
import os
import torchaudio
import torch.nn.functional as F

wav_path = '/home/acar/new_task/ESC-50-master/audio/1-76831-D-42.wav'
output_header = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src', 'golden_keyboard_typing_pcm.h')

waveform, sr = torchaudio.load(wav_path)
if sr != 16000:
    waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

waveform = waveform.sum(0, keepdims=True)
if waveform.shape[1] < 80000:
    waveform = F.pad(waveform, (0, 80000 - waveform.shape[1]))
else:
    waveform = waveform[:, :80000]

samples = waveform.squeeze().numpy()
samples_int16 = (samples * 32767.0).astype('int16')

with open(output_header, 'w') as f:
    f.write('#ifndef GOLDEN_KEYBOARD_TYPING_PCM_H\n')
    f.write('#define GOLDEN_KEYBOARD_TYPING_PCM_H\n\n')
    f.write('#include <stdint.h>\n\n')
    f.write('/* Real 5-Second 16kHz PCM Audio from ESC-50: 1-137-A-32.wav (keyboard_typing) */\n')
    f.write(f'#define GOLDEN_AUDIO_SAMPLE_COUNT {len(samples_int16)}\n\n')
    f.write('static const int16_t GOLDEN_KEYBOARD_TYPING_PCM[GOLDEN_AUDIO_SAMPLE_COUNT] = {\n')
    
    for i in range(0, len(samples_int16), 16):
        chunk = samples_int16[i:i+16]
        line = '    ' + ', '.join(f'{x:6d}' for x in chunk) + ',\n'
        f.write(line)
        
    f.write('};\n\n')
    f.write('#endif /* GOLDEN_KEYBOARD_TYPING_PCM_H */\n')

print(f'✅ Successfully exported {len(samples_int16)} samples to {output_header} ({os.path.getsize(output_header)/1024:.1f} KB)')
