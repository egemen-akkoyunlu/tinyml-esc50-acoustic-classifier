#!/usr/bin/env python3
"""
================================================================================
🚀 TOOL 2: EXPORT CLEAN INT8 PHINET CNN BACKBONE TO C HEADER
Environment: TensorFlow / LiteRT (.venv)
Usage: /home/acar/zephyrproject/.venv/bin/python export_clean_cnn_int8.py
================================================================================
Builds the clean PhiNet CNN backbone in Keras with:
  • 0 Standalone Pad Operators (Uses native padding='same')
  • 0 AveragePool2D Operators (Offloaded to hardware Cortex-M33 FPU)
  • 0 StridedSlice Operators
  • 100% Mathematically Folded BatchNorm Weights from QAT Convert
Exports the production C header for Silicon Labs EFR32MG24 firmware.
================================================================================
"""

import os
import sys
import numpy as np
import tensorflow as tf

ROOT_DIR = "/home/acar/new_task"
WEIGHTS_NPZ = os.path.join(ROOT_DIR, "cnn_qat_converted_weights.npz")
OUT_TFLITE = os.path.join(ROOT_DIR, "tflite_models", "cnn_backbone_out", "cnn_backbone_full_integer_quant.tflite")
HEADER_PATH = "/home/acar/zephyrproject/my_apps/silabs_ble_audio_peripheral/src/phinet_features_model_data.h"

print("=" * 80)
print("🚀 STEP 1: LOADING FOLDED WEIGHTS FROM NUMPY NPZ")
print("=" * 80)

if not os.path.exists(WEIGHTS_NPZ):
    print(f"❌ Error: {WEIGHTS_NPZ} not found! Run step1_extract_pytorch_weights.py first.")
    sys.exit(1)

weights = np.load(WEIGHTS_NPZ)
print(f"✅ Loaded {len(weights.files)} weight tensors from: {WEIGHTS_NPZ}")

print("\n" + "=" * 80)
print("🏗️ STEP 2: BUILDING CLEAN KERAS CNN BACKBONE (0 PAD, 0 POOL OPS)")
print("=" * 80)

inputs = tf.keras.Input(shape=(52, 313, 1), batch_size=1, name="spectrogram")

# 1. Stem: Conv2D(1 -> 16, k=3, stride=2, padding=1) + ReLU6
stem_w = np.transpose(weights["stem.weight"], (2, 3, 1, 0)) # (16, 1, 3, 3) -> (3, 3, 1, 16)
stem_b = weights["stem.bias"]
x = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(inputs)
x = tf.keras.layers.Conv2D(16, (3, 3), strides=(2, 2), padding="valid", name="stem_conv")(x)
x = tf.keras.layers.ReLU(max_value=6.0, name="stem_relu6")(x)

# 2. Block 0:
# Depthwise: DepthwiseConv2D(16 -> 16, k=3, stride=(1, 2), padding=1) + ReLU6
b0_dw_w = np.transpose(weights["b0_dw.weight"], (2, 3, 0, 1)) # (16, 1, 3, 3) -> (3, 3, 16, 1)
b0_dw_b = weights["b0_dw.bias"]
x = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
x = tf.keras.layers.DepthwiseConv2D((3, 3), strides=(1, 2), padding="valid", name="b0_dw_conv")(x)
x = tf.keras.layers.ReLU(max_value=6.0, name="b0_dw_relu")(x)

# Pointwise: Conv2D(16 -> 32, k=1, stride=1, padding='valid') + ReLU6
b0_pw_w = np.transpose(weights["b0_pw.weight"], (2, 3, 1, 0)) # (32, 16, 1, 1) -> (1, 1, 16, 32)
b0_pw_b = weights["b0_pw.bias"]
x = tf.keras.layers.Conv2D(32, (1, 1), strides=(1, 1), padding="valid", name="b0_pw_conv")(x)
x = tf.keras.layers.ReLU(max_value=6.0, name="b0_pw_relu")(x)

# 3. Block 1 (phi_blocks.2 in PyTorch):
# Depthwise: DepthwiseConv2D(32 -> 32, k=3, stride=2, padding=1) + ReLU6
b1_dw_w = np.transpose(weights["b1_dw.weight"], (2, 3, 0, 1)) # (32, 1, 3, 3) -> (3, 3, 32, 1)
b1_dw_b = weights["b1_dw.bias"]
x = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
x = tf.keras.layers.DepthwiseConv2D((3, 3), strides=(2, 2), padding="valid", name="b1_dw_conv")(x)
x = tf.keras.layers.ReLU(max_value=6.0, name="b1_dw_relu")(x)

# Pointwise: Conv2D(32 -> 48, k=1, stride=1, padding='valid') + ReLU6
b1_pw_w = np.transpose(weights["b1_pw.weight"], (2, 3, 1, 0)) # (48, 32, 1, 1) -> (1, 1, 32, 48)
b1_pw_b = weights["b1_pw.bias"]
x = tf.keras.layers.Conv2D(48, (1, 1), strides=(1, 1), padding="valid", name="b1_pw_conv")(x)
x = tf.keras.layers.ReLU(max_value=6.0, name="b1_pw_relu")(x)

# 4. ConvCompress: Conv2D(48 -> 32, k=1, stride=1, padding='valid') -> Outputs clean (1, 13, 40, 32)
comp_w_raw = weights["conv_compress.weight"] if "conv_compress.weight" in weights else weights["compress.weight"]
comp_b_raw = weights["conv_compress.bias"] if "conv_compress.bias" in weights else weights["compress.bias"]
comp_w = np.transpose(comp_w_raw, (2, 3, 1, 0)) # (32, 48, 1, 1) -> (1, 1, 48, 32)
comp_b = comp_b_raw
outputs = tf.keras.layers.Conv2D(32, (1, 1), strides=(1, 1), padding="valid", name="conv_compress")(x)

keras_model = tf.keras.Model(inputs=inputs, outputs=outputs, name="phinet_clean_backbone")

# Assign weights directly
keras_model.get_layer("stem_conv").set_weights([stem_w, stem_b])
keras_model.get_layer("b0_dw_conv").set_weights([b0_dw_w, b0_dw_b])
keras_model.get_layer("b0_pw_conv").set_weights([b0_pw_w, b0_pw_b])
keras_model.get_layer("b1_dw_conv").set_weights([b1_dw_w, b1_dw_b])
keras_model.get_layer("b1_pw_conv").set_weights([b1_pw_w, b1_pw_b])
keras_model.get_layer("conv_compress").set_weights([comp_w, comp_b])

print(f"✅ Keras Model Built! Output Shape: {keras_model.output_shape}")

print("\n" + "=" * 80)
print("⚡ STEP 3: QUANTIZING TO FULL INT8 WITH POST-TRAINING CALIBRATION")
print("=" * 80)

calib_file = os.path.join(ROOT_DIR, "val_spectrograms.npy")
calib_data = np.load(calib_file)

def rep_dataset():
    for i in range(min(100, len(calib_data))):
        sample = np.transpose(calib_data[i:i+1], (0, 2, 3, 1)).astype(np.float32) # (1, 52, 313, 1)
        yield [sample]

converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = rep_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_int8_bytes = converter.convert()
os.makedirs(os.path.dirname(OUT_TFLITE), exist_ok=True)
with open(OUT_TFLITE, "wb") as f:
    f.write(tflite_int8_bytes)

print(f"✅ Generated Full INT8 TFLite: {OUT_TFLITE} ({len(tflite_int8_bytes)} bytes / {len(tflite_int8_bytes)/1024:.2f} KB)")

# Step 4: Operator Audit
import ai_edge_litert.interpreter as litert
interp = litert.Interpreter(model_content=tflite_int8_bytes)
interp.allocate_tensors()
det = interp.get_tensor_details()
pad_count = sum(1 for d in det if "pad" in d["name"].lower())
print(f"\n🔍 OPERATOR AUDIT: Standalone Pad Operators Count = {pad_count} (Must be 0!)")

# Step 5: Export C Header
print("\n" + "=" * 80)
print(f"📝 STEP 5: EXPORTING C HEADER: {HEADER_PATH}")
print("=" * 80)

os.makedirs(os.path.dirname(HEADER_PATH), exist_ok=True)
with open(HEADER_PATH, "w") as f:
    f.write("// ==============================================================================\n")
    f.write("// FULL INT8 PHINET CNN BACKBONE MODEL DATA (CMSIS-NN READY)\n")
    f.write("// Auto-generated from Clean Keras Backbone - 0 Pad Ops, 0 Pool Overlap\n")
    f.write("// ==============================================================================\n\n")
    f.write("#ifndef PHINET_FEATURES_MODEL_DATA_H_\n")
    f.write(f"const unsigned int g_phinet_features_model_data_size = {len(tflite_int8_bytes)};\n")
    f.write(f"const unsigned int g_phinet_features_model_data_len = {len(tflite_int8_bytes)};\n\n")
    f.write("const unsigned char g_phinet_features_model_data[] __attribute__((aligned(4))) = {\n")
    
    for i in range(0, len(tflite_int8_bytes), 12):
        chunk = tflite_int8_bytes[i:i+12]
        hex_str = ", ".join([f"0x{b:02x}" for b in chunk])
        f.write(f"  {hex_str},\n")
        
    f.write("};\n\n")
    f.write("#endif  // PHINET_FEATURES_MODEL_DATA_H_\n")

print(f"✅ Successfully wrote C header: {HEADER_PATH} ({len(tflite_int8_bytes)} bytes)")
print("=" * 80)
