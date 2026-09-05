#!/usr/bin/env python3
import os
import sys
import torch
import numpy as np

PROJECT_ROOT = "/home/acar/new_task"
CKPT_PATH = os.path.join(PROJECT_ROOT, "best_distilled_qat_model.pth")
OUT_NEW_TASK = os.path.join(PROJECT_ROOT, "firmware", "efr32mg24", "src", "gru_classifier_weights.h")
OUT_ZEPHYR = "/home/acar/zephyrproject/my_apps/silabs_ble_audio_peripheral/src/gru_classifier_weights.h"

print(f"📦 Loading Fold-5 checkpoint: {CKPT_PATH}")
sd = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)

# 1. Pre-GRU BatchNorm
pre_w = sd["pre_gru_bn.weight"].numpy()
pre_b = sd["pre_gru_bn.bias"].numpy()
pre_m = sd["pre_gru_bn.running_mean"].numpy()
pre_v = sd["pre_gru_bn.running_var"].numpy()
pre_scale = pre_w / np.sqrt(pre_v + 1e-5)
pre_bias = pre_b - pre_m * pre_scale

# 2. GRU Weights & Biases
w_ih = sd["gru.weight_ih_l0"].numpy() # [480, 32]
w_hh = sd["gru.weight_hh_l0"].numpy() # [480, 160]
b_ih = sd["gru.bias_ih_l0"].numpy()   # [480]
b_hh = sd["gru.bias_hh_l0"].numpy()   # [480]

# 3. Post-GRU BatchNorm
post_w = sd["post_gru_bn.weight"].numpy()
post_b = sd["post_gru_bn.bias"].numpy()
post_m = sd["post_gru_bn.running_mean"].numpy()
post_v = sd["post_gru_bn.running_var"].numpy()
post_scale = post_w / np.sqrt(post_v + 1e-5)
post_bias = post_b - post_m * post_scale

# 4. Bottleneck
btn_w = sd["bottleneck.weight"].numpy() # [128, 160]
btn_b = sd["bottleneck.bias"].numpy()   # [128]

# 5. FC Classifier Head
fc_w = sd["fc.weight"].numpy() # [50, 128]
fc_b = sd["fc.bias"].numpy()   # [50]

def dump_f32_arr(name, arr, per_line=8):
    flat = arr.flatten()
    lines = [f"static const float {name}[{len(flat)}] = {{\n  "]
    for idx, val in enumerate(flat):
        lines.append(f"{val:.8e}f, ")
        if (idx + 1) % per_line == 0 and (idx + 1) < len(flat):
            lines.append("\n  ")
    lines.append("\n};\n\n")
    return "".join(lines)

header = [
    "// ==============================================================================\n",
    "// STAGE 2: NATIVE C++ RECURRENT GRU + ATTENTION + CLASSIFIER WEIGHTS (FP32 BASELINE)\n",
    "// Target: ARM Cortex-M33 Hardware FPU (Direct Flash RODATA Execution)\n",
    "// Model: Official Karol Piczak Fold-5 Clean Distilled PhiNet-CRNN (FP32 Reference)\n",
    "// ==============================================================================\n\n",
    "#ifndef GRU_CLASSIFIER_WEIGHTS_H_\n",
    "#define GRU_CLASSIFIER_WEIGHTS_H_\n\n",
    "#include <stdint.h>\n\n",
    "#define GRU_INPUT_DIM      32\n",
    "#define GRU_HIDDEN_DIM     160\n",
    "#define GRU_TIME_STEPS     39\n",
    "#define BOTTLENECK_DIM     128\n",
    "#define NUM_OUTPUT_CLASSES 50\n\n",
    dump_f32_arr("PRE_GRU_BN_SCALE", pre_scale, per_line=6),
    dump_f32_arr("PRE_GRU_BN_BIAS", pre_bias, per_line=6),
    dump_f32_arr("GRU_W_IH", w_ih, per_line=8),
    dump_f32_arr("GRU_W_HH", w_hh, per_line=8),
    dump_f32_arr("GRU_B_IH", b_ih, per_line=6),
    dump_f32_arr("GRU_B_HH", b_hh, per_line=6),
    dump_f32_arr("POST_GRU_BN_SCALE", post_scale, per_line=6),
    dump_f32_arr("POST_GRU_BN_BIAS", post_bias, per_line=6),
    dump_f32_arr("BOTTLENECK_W", btn_w, per_line=8),
    dump_f32_arr("BOTTLENECK_B", btn_b, per_line=8),
    dump_f32_arr("FC_W", fc_w, per_line=8),
    dump_f32_arr("FC_B", fc_b, per_line=8),
    "#endif /* GRU_CLASSIFIER_WEIGHTS_H_ */\n"
]

content = "".join(header)

with open(OUT_NEW_TASK, "w") as f:
    f.write(content)
print(f"💾 Generated: {OUT_NEW_TASK}")

with open(OUT_ZEPHYR, "w") as f:
    f.write(content)
print(f"💾 Generated: {OUT_ZEPHYR}")
print("🎉 Fold-5 FP32 Baseline GRU Header successfully exported!")
