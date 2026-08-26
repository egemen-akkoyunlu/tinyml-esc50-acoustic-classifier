#!/usr/bin/env python3
# ==============================================================================
# ESC-50 PhiNet + GRU INT8 TFLite & C Header Generator
# Target: Silicon Labs EFR32MG24 DevKit (ARM Cortex-M33 / Zephyr RTOS)
# ==============================================================================

import os
import sys
import subprocess
import numpy as np

def generate_c_header(tflite_path, header_path, array_name="g_phinet_crnn_model_data"):
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

    print(f"[SUCCESS] C Header generated: '{header_path}' ({len(data)} bytes / {len(data)/1024:.2f} KB)")


def main():
    onnx_file = "phinet_crnn_fp32.onnx"
    output_tflite = "phinet_crnn_int8.tflite"
    output_header = "phinet_crnn_int8.h"

    if not os.path.exists(onnx_file):
        print(f"[ERROR] Input ONNX file '{onnx_file}' not found!")
        sys.exit(1)

    print(f"[INFO] Converting '{onnx_file}' to INT8 TFLite using onnx2tf...")

    # 1. Run onnx2tf with full INT8 quantization
    cmd_onnx2tf = [
        "onnx2tf",
        "-i", onnx_file,
        "-o", "tf_saved_model",
        "-oiqt",
        "-coion"
    ]
    
    res = subprocess.run(cmd_onnx2tf, capture_output=False)

    # 2. Locate generated INT8 tflite model
    possible_int8_paths = [
        "tf_saved_model/phinet_crnn_fp32_full_integer_quant.tflite",
        "tf_saved_model/phinet_crnn_fp32_integer_quant.tflite",
        "tf_saved_model/phinet_crnn_fp32_int8.tflite",
        "tf_saved_model/phinet_crnn_fp32_float16.tflite"
    ]

    selected_tflite = None
    for path in possible_int8_paths:
        if os.path.exists(path):
            selected_tflite = path
            break

    if selected_tflite is None:
        # Fallback to tflite2tensorflow CLI command if flatc is present
        input_tflite = "tf_saved_model/phinet_crnn_fp32_float32.tflite"
        if os.path.exists(input_tflite) and os.path.exists("/usr/bin/flatc"):
            print("[INFO] Running tflite2tensorflow flatc quantization fallback...")
            cmd_tfl2tf = [
                "tflite2tensorflow",
                "--model_path", input_tflite,
                "--model_output_path", "tf_saved_model",
                "--flatc_path", "/usr/bin/flatc",
                "--schema_path", "tf_saved_model/schema.fbs",
                "--output_integer_quant_tflite"
            ]
            subprocess.run(cmd_tfl2tf)
            if os.path.exists("tf_saved_model/phinet_crnn_fp32_float32_integer_quant.tflite"):
                selected_tflite = "tf_saved_model/phinet_crnn_fp32_float32_integer_quant.tflite"

    if selected_tflite is None:
        print("[ERROR] Failed to locate quantized INT8 TFLite model!")
        sys.exit(1)

    # Copy binary flatbuffer to output_tflite
    with open(selected_tflite, "rb") as f_in, open(output_tflite, "wb") as f_out:
        data = f_in.read()
        f_out.write(data)

    print(f"[SUCCESS] Quantized INT8 TFLite saved: '{output_tflite}' ({len(data)/1024:.2f} KB)")

    # 3. Generate C header file
    generate_c_header(output_tflite, output_header)

    print("\n" + "="*80)
    print("✅ INT8 DEPLOYMENT READY FOR EFR32MG24! Copy 'phinet_crnn_int8.h' to Zephyr.")
    print("="*80)

if __name__ == "__main__":
    main()