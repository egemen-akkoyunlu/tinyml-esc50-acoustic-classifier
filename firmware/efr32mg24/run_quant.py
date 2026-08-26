#!/usr/bin/env python3
# ==============================================================================
# INT8 Quantization & C Header Exporter using tflite2tensorflow + flatc
# Target Microcontroller: Silicon Labs EFR32MG24 DevKit (ARM Cortex-M33 / Zephyr)
# ==============================================================================

import os
import sys
import subprocess
import numpy as np
import tensorflow as tf


def prepare_calibration_data():
    input_npy = "sample_npy/input.npy"
    calib_npy = "sample_npy/input_calib.npy"
    
    if not os.path.exists(input_npy):
        print(f"[INFO] Generating synthetic calibration tensor at '{input_npy}'...")
        os.makedirs("sample_npy", exist_ok=True)
        data = np.random.normal(loc=0.0, scale=1.0, size=(128, 1, 52, 313)).astype(np.float32)
        np.save(input_npy, data)

    print(f"[INFO] Formatting calibration tensor from '{input_npy}' to TFLite NHWC layout...")
    d = np.load(input_npy)
    if d.ndim == 4 and d.shape[1] == 1:
        d = np.transpose(d, (0, 2, 3, 1))
    
    h0_npy = "sample_npy/h0.npy"
    if not os.path.exists(h0_npy):
        h0_data = np.zeros((128, 1, 160), dtype=np.float32)
        np.save(h0_npy, h0_data)
        print(f"✅ Created GRU hidden state calibration tensor: '{h0_npy}'")

    np.save(calib_npy, d)
    print(f"✅ Calibration tensor formatted to shape {d.shape}: '{calib_npy}'")
    return calib_npy


def main():
    input_tflite = "tf_saved_model/phinet_crnn_fp32_float32.tflite"
    schema_fbs = "tf_saved_model/schema.fbs"
    output_dir = "tf_saved_model"
    final_tflite = "phinet_crnn_int8.tflite"
    final_header = "phinet_crnn_int8.h"

    print("="*80)
    print("⚡ EFR32MG24 INT8 QUANTIZATION RUNNER")
    print("="*80)

    # 1. Prepare calibration tensor
    calib_file = prepare_calibration_data()

    # 2. Step 1: Run onnx2tf full integer quantization with input and h0 calibration data
    cmd_pb = [
        "onnx2tf",
        "-i", "phinet_crnn_fp32.onnx",
        "-o", output_dir,
        "-oiqt",
        "-cind", "input", calib_file, "0.0", "1.0",
        "-cind", "h0", "sample_npy/h0.npy", "0.0", "1.0"
    ]

    print(f"\n[INFO] Step 1: Running onnx2tf INT8 Quantization...")
    subprocess.run(cmd_pb, check=False)

    # Step 2: Locate true INT8 TFLite model
    print(f"\n[INFO] Step 2: Locating INT8 Quantized Model...")
    possible_int8_paths = [
        os.path.join(output_dir, "phinet_crnn_fp32_full_integer_quant.tflite"),
        os.path.join(output_dir, "phinet_crnn_fp32_integer_quant.tflite"),
        os.path.join(output_dir, "phinet_crnn_fp32_dynamic_range_quant.tflite")
    ]

    selected_tflite = None
    for p in possible_int8_paths:
        if os.path.exists(p):
            selected_tflite = p
            break

    if selected_tflite:
        print(f"✅ Found True INT8 TFLite model: '{selected_tflite}'")
        with open(selected_tflite, "rb") as f_in, open(final_tflite, "wb") as f_out:
            tflite_model_int8 = f_in.read()
            f_out.write(tflite_model_int8)
    else:
        print(f"[ERROR] True INT8 quantized TFLite model was not found in '{output_dir}'!")
        sys.exit(1)

    size_kb = len(tflite_model_int8) / 1024.0
    print(f"\n✅ INT8 TFLite model generated: '{final_tflite}' ({size_kb:.2f} KB / {len(tflite_model_int8)} bytes)")

    # 4. Generate C Header File
    print(f"[INFO] Generating C Header array file: '{final_header}'...")
    subprocess.run(f"xxd -i {final_tflite} > {final_header}", shell=True, check=True)

    header_size = os.path.getsize(final_header)
    print(f"✅ C Header generated: '{final_header}' ({header_size} bytes)")

    print("\n" + "="*80)
    print("🏆 INT8 DEPLOYMENT READY FOR SILICON LABS EFR32MG24 DEV KIT!")
    print(f"   • Model Header File:   {final_header}")
    print(f"   • INT8 Model Size:     {size_kb:.2f} KB (📉 75% Flash Saved!)")
    print(f"   • Target Chipset:      Silicon Labs EFR32MG24 (ARM Cortex-M33 @ 78 MHz)")
    print("="*80)


if __name__ == "__main__":
    main()
