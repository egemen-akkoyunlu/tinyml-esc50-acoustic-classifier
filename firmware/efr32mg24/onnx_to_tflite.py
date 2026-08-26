#!/usr/bin/env python3
# ==============================================================================
# ESC-50 PhiNet + GRU FP32 ONNX to INT8 TFLite Quantization Script
# Target Microcontroller: Silicon Labs EFR32MG24 (ARM Cortex-M33 + MVP)
# Engine: TensorFlow Lite Micro (TFLM) / CMSIS-NN
# ==============================================================================

import os
import sys
import numpy as np

# Ensure dependencies are available
try:
    import onnx
except ImportError:
    print("[ERROR] ONNX module not found. Please install via: pip install onnx")
    sys.exit(1)

try:
    import tensorflow as tf
except ImportError:
    print("[ERROR] TensorFlow module not found. Please install via: pip install tensorflow")
    sys.exit(1)


def get_onnx_input_details(onnx_path):
    """
    Inspects the input schema of the ONNX graph to determine exact dynamic/static
    dimensions required for representative dataset calibration.
    """
    print(f"[INFO] Inspecting ONNX model file: {onnx_path}")
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    
    input_info = []
    for inp in model.graph.input:
        shape = []
        for i, dim in enumerate(inp.type.tensor_type.shape.dim):
            if dim.dim_value > 0:
                shape.append(dim.dim_value)
            else:
                # Default dynamic dimensions: batch=1, time=313, frequency=52
                if i == 0:
                    shape.append(1)
                elif i == 2:
                    shape.append(52)
                elif i == 3:
                    shape.append(313)
                else:
                    shape.append(1)
        
        # Override to target ESC-50 Log-Mel Spectrogram shape if 4D tensor
        if len(shape) == 4 and (shape[2] == 1 or shape[2] is None):
            shape = [1, 1, 52, 313]

        input_info.append({
            'name': inp.name,
            'shape': shape,
            'elem_type': inp.type.tensor_type.elem_type
        })
        print(f"  -> Detected Input: name='{inp.name}', shape={shape}")

    for out in model.graph.output:
        shape = [dim.dim_value if dim.dim_value > 0 else 1 for dim in out.type.tensor_type.shape.dim]
        print(f"  -> Detected Output: name='{out.name}', shape={shape}")

    return input_info


def build_representative_dataset(input_shapes, num_samples=100):
    def _generator():
        for _ in range(num_samples):
            sample_tensors = []
            for shape in input_shapes:
                sample = np.random.normal(loc=0.0, scale=1.0, size=shape).astype(np.float32)
                sample_tensors.append(sample)
            yield sample_tensors
    return _generator


def convert_onnx_to_tflite(onnx_path, output_tflite_path):
    # 1. Parse ONNX inputs
    input_details = get_onnx_input_details(onnx_path)
    input_shapes = [inp['shape'] for inp in input_details]

    # 2. Convert ONNX model to TensorFlow SavedModel using onnx2tf
    saved_model_dir = "tf_saved_model"
    print(f"[INFO] Converting ONNX graph to TF SavedModel format in '{saved_model_dir}'...")
    
    try:
        import onnx2tf
        onnx2tf.convert(
            input_onnx_file_path=onnx_path,
            output_folder_path=saved_model_dir,
            copy_onnx_input_output_names_to_tflite=True,
            non_verbose=True
        )
    except Exception as e:
        print(f"[INFO] onnx2tf conversion finished with notice: {e}")

    # Check if SavedModel directory was successfully created
    if not os.path.exists(saved_model_dir):
        print(f"[ERROR] SavedModel directory '{saved_model_dir}' was not created!")
        sys.exit(1)

    # 3. Create TFLite Converter from SavedModel
    print("[INFO] Creating TFLite Converter from SavedModel...")
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

    # 4. Configure INT8 Post-Training Quantization (PTQ)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = build_representative_dataset(input_shapes, num_samples=100)
    
    # Enforce full INT8 quantization (int8 input, int8 output, int8 internal ops)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTIN_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    print("[INFO] Running INT8 quantization calibration...")
    tflite_model_int8 = converter.convert()

    # 5. Save quantized flatbuffer model
    with open(output_tflite_path, "wb") as f:
        f.write(tflite_model_int8)
    
    size_kb = len(tflite_model_int8) / 1024.0
    print(f"[SUCCESS] INT8 TFLite model generated successfully: '{output_tflite_path}' ({size_kb:.2f} KB)")
    return tflite_model_int8


def convert_tflite_to_c_header(tflite_path, header_path, array_name="g_phinet_crnn_model_data"):
    """
    Converts the binary TFLite flatbuffer file into a C header array ready for
    inclusion into the Zephyr / EFR32MG24 C project.
    """
    print(f"[INFO] Exporting binary flatbuffer to C header file: '{header_path}'")
    with open(tflite_path, "rb") as f:
        data = f.read()

    with open(header_path, "w") as f:
        f.write("/* Auto-generated TFLite INT8 Model Array for EFR32MG24 ESC-50 Classifier */\n")
        f.write("#ifndef PHINET_CRNN_INT8_MODEL_H_\n")
        f.write("#define PHINET_CRNN_INT8_MODEL_H_\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"// Model size: {len(data)} bytes\n")
        f.write(f"alignas(16) const unsigned char {array_name}[] = {{\n  ")
        
        for i, b in enumerate(data):
            f.write(f"0x{b:02x}, ")
            if (i + 1) % 12 == 0:
                f.write("\n  ")
                
        f.write("\n};\n\n")
        f.write(f"const unsigned int {array_name}_len = {len(data)};\n\n")
        f.write("#endif // PHINET_CRNN_INT8_MODEL_H_\n")

    print(f"[SUCCESS] C Header generated: '{header_path}' ({len(data)} bytes)")


if __name__ == "__main__":
    onnx_file = "phinet_crnn_fp32.onnx"
    tflite_file = "phinet_crnn_int8.tflite"
    c_header_file = "phinet_crnn_int8.h"

    if not os.path.exists(onnx_file):
        print(f"[ERROR] Specified ONNX file '{onnx_file}' does not exist.")
        sys.exit(1)

    # Execute end-to-end quantization pipeline
    convert_onnx_to_tflite(onnx_file, tflite_file)
    convert_tflite_to_c_header(tflite_file, c_header_file)
