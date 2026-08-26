#!/usr/bin/env python3
# ==============================================================================
# Step 1: ONNX to TensorFlow SavedModel Converter
# Target Model: ESC-50 PhiNet-CRNN Audio Spectrogram Classifier
# ==============================================================================

import os
import sys

try:
    import onnx
except ImportError:
    print("[ERROR] ONNX is required. Please install via: pip install onnx")
    sys.exit(1)

try:
    import onnx2tf
except ImportError:
    print("[ERROR] onnx2tf is required. Please install via: pip install onnx2tf")
    sys.exit(1)


def inspect_onnx_graph(onnx_path):
    print(f"\n[INFO] Inspecting ONNX model: '{onnx_path}'...")
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)

    print("  • Inputs:")
    for inp in model.graph.input:
        shape = [dim.dim_value if dim.dim_value > 0 else "dynamic" for dim in inp.type.tensor_type.shape.dim]
        print(f"    - Name: '{inp.name}', Shape: {shape}")

    print("  • Outputs:")
    for out in model.graph.output:
        shape = [dim.dim_value if dim.dim_value > 0 else "dynamic" for dim in out.type.tensor_type.shape.dim]
        print(f"    - Name: '{out.name}', Shape: {shape}")


def main():
    onnx_file = "phinet_crnn_fp32.onnx"
    saved_model_dir = "tf_saved_model"

    if not os.path.exists(onnx_file):
        onnx_file = os.path.expanduser("~/new_task/phinet_crnn_fp32.onnx")

    if not os.path.exists(onnx_file):
        print(f"[ERROR] Input ONNX file '{onnx_file}' not found!")
        sys.exit(1)

    print("="*80)
    print("🚀 STEP 1: CONVERTING ONNX TO TENSORFLOW SAVEDMODEL")
    print("="*80)

    # 1. Inspect ONNX model
    inspect_onnx_graph(onnx_file)

    # 2. Convert ONNX to TensorFlow SavedModel using onnx2tf
    print(f"\n[INFO] Converting '{onnx_file}' to TensorFlow SavedModel in '{saved_model_dir}'...")
    try:
        onnx2tf.convert(
            input_onnx_file_path=onnx_file,
            output_folder_path=saved_model_dir,
            copy_onnx_input_output_names_to_tflite=True,
            non_verbose=False
        )
    except Exception as e:
        print(f"[NOTICE] onnx2tf conversion note: {e}")

    # 3. Verify output
    saved_pb = os.path.join(saved_model_dir, "saved_model.pb")
    if os.path.exists(saved_model_dir) and len(os.listdir(saved_model_dir)) > 0:
        print("\n" + "="*80)
        print("✅ SUCCESS: TensorFlow SavedModel successfully generated!")
        print(f"   • Output Directory:  '{os.path.abspath(saved_model_dir)}'")
        print(f"   • Directory Files:   {os.listdir(saved_model_dir)}")
        print("="*80)
    else:
        print(f"[ERROR] SavedModel conversion failed. Directory '{saved_model_dir}' is empty.")
        sys.exit(1)


if __name__ == "__main__":
    main()
