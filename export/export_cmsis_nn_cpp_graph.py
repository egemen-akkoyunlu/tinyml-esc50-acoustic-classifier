#!/usr/bin/env python3
import os
import sys
import numpy as np
import tensorflow as tf

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TFLITE_PATH = os.path.join(PROJECT_ROOT, "tflite_models", "cnn_backbone_out", "cnn_backbone_full_integer_quant.tflite")
HEADER_OUT = os.path.join(PROJECT_ROOT, "firmware", "efr32mg24", "src", "cmsis_nn_cnn_weights.h")

interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()
details = interpreter.get_tensor_details()

def get_by_index(idx):
    d = [x for x in details if x["index"] == idx][0]
    return interpreter.get_tensor(idx), d

def get_detail_by_index(idx):
    return [x for x in details if x["index"] == idx][0]

# Input details (Index 0)
in_d = get_detail_by_index(0)
in_scale = in_d["quantization_parameters"]["scales"][0]
in_zp = in_d["quantization_parameters"]["zero_points"][0]

# 1. Stem Conv: Weights Idx 13, Bias Idx 12, Out Idx 15
stem_w, stem_w_d = get_by_index(13)
stem_b, stem_b_d = get_by_index(12)
stem_out_d = get_detail_by_index(15)

# 2. Block 0 DW: Weights Idx 11, Bias Idx 10, Out Idx 16
b0_dw_w, b0_dw_w_d = get_by_index(11)
b0_dw_b, b0_dw_b_d = get_by_index(10)
b0_dw_out_d = get_detail_by_index(16)

# 3. Block 0 PW: Weights Idx 9, Bias Idx 8, Out Idx 17
b0_pw_w, b0_pw_w_d = get_by_index(9)
b0_pw_b, b0_pw_b_d = get_by_index(8)
b0_pw_out_d = get_detail_by_index(17)

# 4. Block 1 DW: Weights Idx 7, Bias Idx 6, Out Idx 19
b1_dw_w, b1_dw_w_d = get_by_index(7)
b1_dw_b, b1_dw_b_d = get_by_index(6)
b1_dw_out_d = get_detail_by_index(19)

# 5. Block 1 PW: Weights Idx 5, Bias Idx 4, Out Idx 20
b1_pw_w, b1_pw_w_d = get_by_index(5)
b1_pw_b, b1_pw_b_d = get_by_index(4)
b1_pw_out_d = get_detail_by_index(20)

# 6. Compress Conv: Weights Idx 3, Bias Idx 2, Out Idx 21
cmp_w, cmp_w_d = get_by_index(3)
cmp_b, cmp_b_d = get_by_index(2)
cmp_out_d = get_detail_by_index(21)

print(f"Stem W: {stem_w.shape} (dtype: {stem_w.dtype}, min: {stem_w.min()}, max: {stem_w.max()})")
print(f"B0 DW W: {b0_dw_w.shape} (dtype: {b0_dw_w.dtype}, min: {b0_dw_w.min()}, max: {b0_dw_w.max()})")
print(f"B0 PW W: {b0_pw_w.shape} (dtype: {b0_pw_w.dtype}, min: {b0_pw_w.min()}, max: {b0_pw_w.max()})")
print(f"B1 DW W: {b1_dw_w.shape} (dtype: {b1_dw_w.dtype}, min: {b1_dw_w.min()}, max: {b1_dw_w.max()})")
print(f"B1 PW W: {b1_pw_w.shape} (dtype: {b1_pw_w.dtype}, min: {b1_pw_w.min()}, max: {b1_pw_w.max()})")
print(f"Compress W: {cmp_w.shape} (dtype: {cmp_w.dtype}, min: {cmp_w.min()}, max: {cmp_w.max()})")

def quant_params(in_scale, w_scales, out_scale):
    mults = []
    shifts = []
    for ws in w_scales:
        eff = (in_scale * ws) / out_scale
        significand, shift = np.frexp(eff)
        mult = int(np.round(significand * (1 << 31)))
        if mult == (1 << 31):
            mult = mult - 1
        mults.append(mult)
        shifts.append(shift)
    return np.array(mults, dtype=np.int32), np.array(shifts, dtype=np.int32)

def to_c_array(arr, name, ctype="int8_t", per_line=12):
    flat = arr.flatten()
    lines = [f"static const {ctype} {name}[{len(flat)}] = {{"]
    for i in range(0, len(flat), per_line):
        chunk = flat[i:i+per_line]
        chunk_str = ", ".join(f"{int(x)}" for x in chunk)
        lines.append("    " + chunk_str + ",")
    lines.append("};\n")
    return "\n".join(lines)

with open(HEADER_OUT, "w") as f:
    f.write("// ============================================================================\n")
    f.write("// 🚀 NATIVE CMSIS-NN PING-PONG CNN BACKBONE WEIGHTS & QUANTIZATION PARAMS\n")
    f.write("// Replaces TFLM interpreter to slash SRAM from 172 KB -> 98.1 KB!\n")
    f.write("// ============================================================================\n\n")
    f.write("#ifndef CMSIS_NN_CNN_WEIGHTS_H\n#define CMSIS_NN_CNN_WEIGHTS_H\n\n#include <stdint.h>\n\n")

    # Layer 1: Stem
    sm, ss = quant_params(in_scale, stem_w_d["quantization_parameters"]["scales"], stem_out_d["quantization_parameters"]["scales"][0])
    f.write(to_c_array(stem_w, "CMSIS_STEM_W", "int8_t"))
    f.write(to_c_array(stem_b, "CMSIS_STEM_B", "int32_t"))
    f.write(to_c_array(sm, "CMSIS_STEM_M", "int32_t"))
    f.write(to_c_array(ss, "CMSIS_STEM_S", "int32_t"))

    # Layer 2: B0 DW
    b0_dwm, b0_dws = quant_params(stem_out_d["quantization_parameters"]["scales"][0], b0_dw_w_d["quantization_parameters"]["scales"], b0_dw_out_d["quantization_parameters"]["scales"][0])
    f.write(to_c_array(b0_dw_w, "CMSIS_B0_DW_W", "int8_t"))
    f.write(to_c_array(b0_dw_b, "CMSIS_B0_DW_B", "int32_t"))
    f.write(to_c_array(b0_dwm, "CMSIS_B0_DW_M", "int32_t"))
    f.write(to_c_array(b0_dws, "CMSIS_B0_DW_S", "int32_t"))

    # Layer 3: B0 PW
    b0_pwm, b0_pws = quant_params(b0_dw_out_d["quantization_parameters"]["scales"][0], b0_pw_w_d["quantization_parameters"]["scales"], b0_pw_out_d["quantization_parameters"]["scales"][0])
    f.write(to_c_array(b0_pw_w, "CMSIS_B0_PW_W", "int8_t"))
    f.write(to_c_array(b0_pw_b, "CMSIS_B0_PW_B", "int32_t"))
    f.write(to_c_array(b0_pwm, "CMSIS_B0_PW_M", "int32_t"))
    f.write(to_c_array(b0_pws, "CMSIS_B0_PW_S", "int32_t"))

    # Layer 4: B1 DW
    b1_dwm, b1_dws = quant_params(b0_pw_out_d["quantization_parameters"]["scales"][0], b1_dw_w_d["quantization_parameters"]["scales"], b1_dw_out_d["quantization_parameters"]["scales"][0])
    f.write(to_c_array(b1_dw_w, "CMSIS_B1_DW_W", "int8_t"))
    f.write(to_c_array(b1_dw_b, "CMSIS_B1_DW_B", "int32_t"))
    f.write(to_c_array(b1_dwm, "CMSIS_B1_DW_M", "int32_t"))
    f.write(to_c_array(b1_dws, "CMSIS_B1_DW_S", "int32_t"))

    # Layer 5: B1 PW
    b1_pwm, b1_pws = quant_params(b1_dw_out_d["quantization_parameters"]["scales"][0], b1_pw_w_d["quantization_parameters"]["scales"], b1_pw_out_d["quantization_parameters"]["scales"][0])
    f.write(to_c_array(b1_pw_w, "CMSIS_B1_PW_W", "int8_t"))
    f.write(to_c_array(b1_pw_b, "CMSIS_B1_PW_B", "int32_t"))
    f.write(to_c_array(b1_pwm, "CMSIS_B1_PW_M", "int32_t"))
    f.write(to_c_array(b1_pws, "CMSIS_B1_PW_S", "int32_t"))

    # Layer 6: Compress
    cmp_m, cmp_s = quant_params(b1_pw_out_d["quantization_parameters"]["scales"][0], cmp_w_d["quantization_parameters"]["scales"], cmp_out_d["quantization_parameters"]["scales"][0])
    f.write(to_c_array(cmp_w, "CMSIS_CMP_W", "int8_t"))
    f.write(to_c_array(cmp_b, "CMSIS_CMP_B", "int32_t"))
    f.write(to_c_array(cmp_m, "CMSIS_CMP_M", "int32_t"))
    f.write(to_c_array(cmp_s, "CMSIS_CMP_S", "int32_t"))

    f.write(f"#define CMSIS_CNN_OUTPUT_SCALE {cmp_out_d['quantization_parameters']['scales'][0]:.8f}f\n")
    f.write(f"#define CMSIS_CNN_OUTPUT_ZERO_POINT {cmp_out_d['quantization_parameters']['zero_points'][0]}\n\n")
    f.write("#endif // CMSIS_NN_CNN_WEIGHTS_H\n")

print(f"💾 Successfully regenerated 100% INT8 CMSIS-NN weights: {HEADER_OUT}")
zephyr_out = "/home/acar/zephyrproject/my_apps/silabs_ble_audio_peripheral/src/cmsis_nn_cnn_weights.h"
import shutil
shutil.copy(HEADER_OUT, zephyr_out)
print(f"💾 Successfully deployed to Zephyr app: {zephyr_out}")
