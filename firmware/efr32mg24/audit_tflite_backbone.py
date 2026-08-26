#!/usr/bin/env python3
"""
================================================================================
🔬 AUDIT SCRIPT 2: STAGE 1 INT8 TFLITE MODEL EXECUTION & FEATURE FORENSICS
================================================================================
Runs the compiled INT8 TFLite model on the spectrogram.
Dumps the 39 time-step x 32-channel feature map and compares with Stage 2 input.
================================================================================
"""

import os
import sys
import numpy as np
import ai_edge_litert.interpreter as litert

MODEL_PATH = "/home/acar/new_task/tflite_models/cnn_backbone_out/cnn_backbone_full_integer_quant.tflite"
SPEC_PATH = "/home/acar/zephyrproject/my_apps/silabs_ble_audio_peripheral/audit_c_spectrogram.npy"

def main():
    print("=" * 80)
    print("🔬 AUDIT 2: TFLITE INT8 BACKBONE EXECUTION & FEATURE AUDIT")
    print("=" * 80)

    if not os.path.exists(SPEC_PATH):
        print(f"❌ Error: Please run audit_dsp_spectrogram.py first to generate {SPEC_PATH}!")
        sys.exit(1)

    # 1. Load Spectrogram
    c_log_mel = np.load(SPEC_PATH) # (52, 313)

    # 2. Instantiate TFLite Interpreter
    interp = litert.Interpreter(model_path=MODEL_PATH)
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]

    in_scale, in_zp = in_det['quantization']
    out_scale, out_zp = out_det['quantization']

    print(f"📦 Model: {os.path.basename(MODEL_PATH)}")
    print(f"   • Input Tensor  : Shape={in_det['shape']}, Dtype={in_det['dtype']}, Scale={in_scale:.6f}, ZP={in_zp}")
    print(f"   • Output Tensor : Shape={out_det['shape']}, Dtype={out_det['dtype']}, Scale={out_scale:.6f}, ZP={out_zp}")

    # 3. Quantize Spectrogram into [1, 52, 313, 1] INT8
    spec_int8 = np.clip(np.round(c_log_mel / in_scale) + in_zp, -128, 127).astype(np.int8)
    spec_nhwc = np.expand_dims(np.expand_dims(spec_int8, 0), -1)

    interp.set_tensor(in_det['index'], spec_nhwc)
    interp.invoke()

    out_tensor = interp.get_tensor(out_det['index']) # Shape: (1, 1, 40, 32)
    print(f"\n📊 TFLite Stage 1 Output Shape: {out_tensor.shape}")
    print(f"   • INT8 Min/Max Values: [{out_tensor.min()}, {out_tensor.max()}]")

    # Dequantize first 39 time steps: (39, 32)
    features_39x32 = (out_tensor[0, 0, :39, :].astype(np.float32) - out_zp) * out_scale
    print(f"   • Dequantized Feature Matrix (39 x 32): Range [{features_39x32.min():.4f}, {features_39x32.max():.4f}]")

    # Save to disk for Stage 2 audit
    np.save("/home/acar/zephyrproject/my_apps/silabs_ble_audio_peripheral/audit_tflite_features.npy", features_39x32)
    print("✅ Saved 'audit_tflite_features.npy' for Stage 2 analysis!")

if __name__ == "__main__":
    main()
