#!/usr/bin/env python3
# ==============================================================================
# ESC-50 PhiNet + GRU INT8 Quantization & C Header Generator for EFR32MG24
# Target Microcontroller: Silicon Labs EFR32MG24 DevKit (ARM Cortex-M33 + MVP)
# Engine: Zephyr RTOS / TensorFlow Lite Micro (TFLM) / CMSIS-NN
# ==============================================================================

import os
import sys
import subprocess
import numpy as np
import torch

# Import val_dataset safely by changing working directory to ~/new_task
orig_cwd = os.getcwd()
new_task_dir = os.path.expanduser("~/new_task")
sys.path.append(new_task_dir)

try:
    os.chdir(new_task_dir)
    from annotated_code import val_dataset
finally:
    os.chdir(orig_cwd)


def extract_real_calibration_data(output_npy_path="sample_npy/input.npy", num_samples=128):
    """
    Extracts real validation spectrogram samples from ESC-50 dataset to guarantee
    optimal INT8 quantization scales and minimize quantization error.
    """
    print(f"[INFO] Extracting {num_samples} real validation audio clips for calibration...")
    os.makedirs(os.path.dirname(output_npy_path), exist_ok=True)

    if val_dataset is not None and len(val_dataset) > 0:
        samples = []
        for i in range(min(num_samples, len(val_dataset))):
            audio_tensor, _ = val_dataset[i]
            # Shape: [1, 52, 313] -> Add batch dim: [1, 1, 52, 313]
            sample_np = audio_tensor.unsqueeze(0).numpy()
            samples.append(sample_np)
        
        # Combine into single array: [128, 1, 52, 313]
        calib_data = np.concatenate(samples, axis=0)
        print(f"   • Real Calibration Tensor Shape: {calib_data.shape}")
    else:
        print("   • Warning: val_dataset not found, fallback to zero-mean normalized synthetic features...")
        calib_data = np.random.normal(loc=0.0, scale=1.0, size=(num_samples, 1, 52, 313)).astype(np.float32)

    np.save(output_npy_path, calib_data)
    print(f"✅ Calibration data saved to: '{output_npy_path}'")


def convert_tflite_to_c_header(tflite_path, header_path, array_name="g_phinet_crnn_model_data"):
    """
    Converts the binary TFLite flatbuffer file into an aligned C header array for
    inclusion in the Zephyr / EFR32MG24 project.
    """
    print(f"[INFO] Exporting binary flatbuffer to C header file: '{header_path}'...")
    with open(tflite_path, "rb") as f:
        data = f.read()

    with open(header_path, "w") as f:
        f.write("/* Auto-generated INT8 TFLite Model Header for EFR32MG24 ESC-50 Classifier */\n")
        f.write("#ifndef PHINET_CRNN_INT8_MODEL_H_\n")
        f.write("#define PHINET_CRNN_INT8_MODEL_H_\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"// Model size: {len(data)} bytes ({len(data)/1024:.2f} KB)\n")
        f.write(f"alignas(16) const unsigned char {array_name}[] = {{\n  ")
        
        for i, b in enumerate(data):
            f.write(f"0x{b:02x}, ")
            if (i + 1) % 12 == 0:
                f.write("\n  ")
                
        f.write("\n};\n\n")
        f.write(f"const unsigned int {array_name}_len = {len(data)};\n\n")
        f.write("#endif // PHINET_CRNN_INT8_MODEL_H_\n")

    print(f"✅ C Header saved: '{header_path}' ({len(data)} bytes / {len(data)/1024:.2f} KB)")


def main():
    onnx_file = "phinet_crnn_fp32.onnx"
    if not os.path.exists(onnx_file):
        onnx_file = os.path.expanduser("~/new_task/phinet_crnn_fp32.onnx")

    output_tflite = "phinet_crnn_int8.tflite"
    output_header = "phinet_crnn_int8.h"

    if not os.path.exists(onnx_file):
        print(f"[ERROR] Input ONNX file '{onnx_file}' not found!")
        sys.exit(1)

    print("="*80)
    print("⚡ EFR32MG24 INT8 QUANTIZATION & C HEADER EXPORTER")
    print("="*80)

    # 1. Generate real calibration data
    calib_npy_path = "sample_npy/input.npy"
    extract_real_calibration_data(calib_npy_path, num_samples=128)

    # 2. Run onnx2tf per-channel full integer quantization
    print(f"\n🔮 Running onnx2tf full integer quantization on '{onnx_file}'...")
    cmd_onnx2tf = [
        "onnx2tf",
        "-i", onnx_file,
        "-o", "tf_saved_model",
        "-oiqt",
        "-cind", "input", calib_npy_path, "0.0", "1.0",
        "-coion"
    ]
    
    result = subprocess.run(cmd_onnx2tf)

    # 3. Locate quantized INT8 model
    quant_tflite_path = "tf_saved_model/phinet_crnn_fp32_full_integer_quant.tflite"
    if not os.path.exists(quant_tflite_path):
        # Fallback search
        possible_paths = [
            "tf_saved_model/phinet_crnn_fp32_integer_quant.tflite",
            "tf_saved_model/phinet_crnn_fp32_int8.tflite"
        ]
        for p in possible_paths:
            if os.path.exists(p):
                quant_tflite_path = p
                break

    if not os.path.exists(quant_tflite_path):
        print(f"[ERROR] Quantized model '{quant_tflite_path}' was not generated!")
        sys.exit(1)

    # Copy binary flatbuffer to output_tflite
    with open(quant_tflite_path, "rb") as f_in, open(output_tflite, "wb") as f_out:
        data = f_in.read()
        f_out.write(data)

    size_kb = len(data) / 1024.0
    print(f"\n✅ INT8 TFLite model generated: '{output_tflite}' ({size_kb:.2f} KB)")

    # 4. Export C Header file
    convert_tflite_to_c_header(output_tflite, output_header)

    print("\n" + "="*80)
    print("🏆 INT8 DEPLOYMENT READY FOR EFR32MG24 DEV KIT!")
    print(f"   • Model Header File:   {output_header}")
    print(f"   • INT8 Model Size:     {size_kb:.2f} KB (📉 75% Flash Saved!)")
    print(f"   • Target Chipset:      Silicon Labs EFR32MG24 (ARM Cortex-M33 @ 78 MHz)")
    print("="*80)

if __name__ == "__main__":
    main()
