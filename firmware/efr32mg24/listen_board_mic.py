#!/usr/bin/env python3
"""
Listen to Silicon Labs EFR32MG24 Microphone over UART (/dev/ttyACM0)
Saves recorded audio to WAV, plays it back, and evaluates PyTorch AI model.
"""

import sys
import os
import time
import subprocess
import serial
import numpy as np
from scipy.io import wavfile

PORT = "/dev/ttyACM0"
BAUD = 115200
SAMPLE_RATE = 16000
EXPECTED_SAMPLES = 48000 # 3 seconds @ 16 kHz
OUTPUT_WAV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "board_mic_test.wav")

def main():
    print("=" * 70)
    print(" 🎙️ SILICON LABS EFR32MG24 3-SECOND MICROPHONE STREAMER & PLAYER")
    print(f" Port: {PORT} @ {BAUD} baud | Target: {EXPECTED_SAMPLES} samples ({SAMPLE_RATE} Hz - 3 sec)")
    print("=" * 70)

    try:
        ser = serial.Serial(PORT, BAUD, timeout=10)
    except Exception as e:
        print(f"❌ Error opening serial port {PORT}: {e}")
        print("Make sure no other minicom/screen terminal is using /dev/ttyACM0!")
        sys.exit(1)

    ser.reset_input_buffer()
    print("⚡ Connected to board! Waiting for '=== AUDIO_DUMP_START ==='...")
    print("👉 Speak, clap, whistle, or type near the microcontroller now!")

    while True:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if not line:
            continue
        
        if "=== AUDIO_DUMP_START ===" in line:
            print("\n🎙️ [CAPTURE ACTIVE] Receiving 80,000 audio samples from microcontroller...")
            samples = []
            start_time = time.time()

            while len(samples) < EXPECTED_SAMPLES:
                sample_line = ser.readline().decode('utf-8', errors='ignore').strip()
                if not sample_line:
                    continue
                if "=== AUDIO_DUMP_END ===" in sample_line:
                    break
                try:
                    val = int(sample_line)
                    samples.append(val)
                    if len(samples) % 10000 == 0:
                        pct = (len(samples) / EXPECTED_SAMPLES) * 100
                        print(f"   📥 Progress: {len(samples)}/{EXPECTED_SAMPLES} ({pct:.0f}%)")
                except ValueError:
                    pass

            elapsed = time.time() - start_time
            print(f"✅ Received {len(samples)} samples in {elapsed:.2f} seconds!")

            audio_arr = np.array(samples, dtype=np.int16)

            # Audio Analysis & Quality Diagnostics
            min_val = int(np.min(audio_arr))
            max_val = int(np.max(audio_arr))
            mean_val = float(np.mean(audio_arr))
            rms_val = float(np.sqrt(np.mean(audio_arr.astype(np.float64)**2)))

            print("\n" + "-" * 50)
            print(" 📊 MICROPHONE AUDIO SIGNAL ANALYSIS:")
            print(f"   • Min Sample   : {min_val}")
            print(f"   • Max Sample   : {max_val}")
            print(f"   • DC Offset    : {mean_val:.2f}")
            print(f"   • RMS Energy   : {rms_val:.2f}")
            print("-" * 50)

            # Save to WAV file
            wavfile.write(OUTPUT_WAV, SAMPLE_RATE, audio_arr)
            print(f"💾 Saved Audio to: {OUTPUT_WAV}")

            # Playback audio on PC speakers
            print("🔊 Playing back recorded audio through PC speakers...")
            try:
                subprocess.run(["aplay", OUTPUT_WAV], check=False)
            except Exception as e:
                print(f"⚠️ Could not execute aplay: {e}")

            # Run PyTorch Model on Recorded Audio
            try:
                print("\n🧠 Evaluating PyTorch Model on this recorded audio...")
                subprocess.run([
                    "/home/acar/kws_env/bin/python", 
                    "-c", 
                    f"""
import sys
sys.path.append('/home/acar/new_task')
import torch, torchaudio, numpy as np
from scipy.io import wavfile
from qat_training import HybridPhiNetKWS, compute_melspectrogram_db

ESC50_CLASSES = [
    "airplane", "breathing", "brushing_teeth", "can_opening", "car_horn",
    "cat", "chainsaw", "chirping_birds", "church_bells", "clapping",
    "clock_alarm", "clock_tick", "coughing", "cow", "crackling_fire",
    "crickets", "crow", "crying_baby", "dog", "door_wood_creaks",
    "door_wood_knock", "drinking_sipping", "engine", "fireworks", "footsteps",
    "frog", "glass_breaking", "groan", "gunshot", "hand_saw",
    "helicopter", "hen", "honking", "insects", "keyboard_typing",
    "laughing", "mouse_click", "pig", "pouring_water", "rain",
    "rooster", "sea_waves", "siren", "sneezing", "snoring",
    "sound_of_waterfalls", "train", "toilet_flush", "vacuum_cleaner", "washer_dryer"
]

sr, data = wavfile.read('{OUTPUT_WAV}')
pcm = data.astype(np.float32)
if np.max(np.abs(pcm)) > 0:
    pcm /= 32768.0

mel = compute_melspectrogram_db(pcm)
x = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0)

model = HybridPhiNetKWS(num_classes=50)
ckpt = torch.load('/home/acar/new_task/best_qat_model.pth', map_location='cpu')
model.load_state_dict(ckpt, strict=False)
model.eval()

with torch.no_grad():
    logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze().numpy()
    top5_idx = np.argsort(probs)[::-1][:5]

print("🏆 PYTORCH TOP-5 PREDICTIONS ON RECORDED AUDIO:")
for rank, idx in enumerate(top5_idx, 1):
    print(f"   #{rank}: {{ESC50_CLASSES[idx]:<22}} (Confidence: {{probs[idx]*100:.2f}}%)")
"""
                ])
            except Exception as e:
                print(f"⚠️ PyTorch evaluation error: {e}")

            print("\n" + "=" * 70)
            print("👉 Ready for next recording! Press Ctrl+C to exit.")
            print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
