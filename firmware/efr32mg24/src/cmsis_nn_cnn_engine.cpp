#include "cmsis_nn_cnn_engine.h"
#include "cmsis_nn_cnn_weights.h"
#include <arm_nnfunctions.h>
#include <stdio.h>
#include <string.h>

// CMSIS-NN Scratchpad Buffer for temporary im2col computations
alignas(16) static int8_t s_scratchpad[4096];

void cmsis_nn_cnn_init(void) {
    printf("-> [OK] Native CMSIS-NN Ping-Pong Engine Initialized (SRAM: 98.5 KB vs TFLM: 172 KB).\n");
}

int cmsis_nn_cnn_run(const int8_t *input_spec_int8, int8_t *ping_pong_A, int8_t *ping_pong_B, int8_t *out_features_int8) {
    cmsis_nn_context ctx;
    ctx.buf = s_scratchpad;
    ctx.size = sizeof(s_scratchpad);

    // =========================================================================
    // LAYER 1: Stem Conv2D (1 -> 16, k=3x3, stride=2x2, pad=1x1) + ReLU6
    // In:  [1, 52, 313, 1]  (scale=0.0931, zp=20)
    // Out: [1, 26, 157, 16] -> ping_pong_A (65,312 bytes, zp=-128)
    // =========================================================================
    cmsis_nn_dims in_dims = {1, 52, 313, 1};
    cmsis_nn_dims out_dims = {1, 26, 157, 16};
    cmsis_nn_dims filter_dims = {16, 3, 3, 1};
    cmsis_nn_dims bias_dims = {1, 1, 1, 16};

    cmsis_nn_conv_params conv_params;
    conv_params.input_offset = -20;
    conv_params.output_offset = -128; // zp = -128
    conv_params.stride.h = 2;
    conv_params.stride.w = 2;
    conv_params.padding.h = 1;
    conv_params.padding.w = 1;
    conv_params.dilation.h = 1;
    conv_params.dilation.w = 1;
    conv_params.activation.min = -128;
    conv_params.activation.max = 127;

    cmsis_nn_per_channel_quant_params quant_params;
    quant_params.multiplier = (int32_t*)CMSIS_STEM_M;
    quant_params.shift = (int32_t*)CMSIS_STEM_S;

    arm_cmsis_nn_status status;
    status = arm_convolve_wrapper_s8(&ctx, &conv_params, &quant_params, &in_dims, input_spec_int8,
                                     &filter_dims, CMSIS_STEM_W, &bias_dims, CMSIS_STEM_B,
                                     &out_dims, ping_pong_A);
    if (status != ARM_CMSIS_NN_SUCCESS) return -1;

    // =========================================================================
    // LAYER 2: Block 0 Depthwise Conv2D (16 -> 16, k=3x3, stride=1x2, pad=1x1) + ReLU6
    // In:  [1, 26, 157, 16] -> ping_pong_A (zp=-128)
    // Out: [1, 26, 79, 16]  -> ping_pong_B (32,864 bytes, zp=-128)
    // =========================================================================
    in_dims = {1, 26, 157, 16};
    out_dims = {1, 26, 79, 16};
    filter_dims = {1, 3, 3, 16};
    bias_dims = {1, 1, 1, 16};

    cmsis_nn_dw_conv_params dw_params;
    dw_params.input_offset = 128;
    dw_params.output_offset = -128;
    dw_params.ch_mult = 1;
    dw_params.stride.h = 1;
    dw_params.stride.w = 2;
    dw_params.padding.h = 1;
    dw_params.padding.w = 1;
    dw_params.dilation.h = 1;
    dw_params.dilation.w = 1;
    dw_params.activation.min = -128;
    dw_params.activation.max = 127;

    quant_params.multiplier = (int32_t*)CMSIS_B0_DW_M;
    quant_params.shift = (int32_t*)CMSIS_B0_DW_S;

    status = arm_depthwise_conv_wrapper_s8(&ctx, &dw_params, &quant_params, &in_dims, ping_pong_A,
                                           &filter_dims, CMSIS_B0_DW_W, &bias_dims, CMSIS_B0_DW_B,
                                           &out_dims, ping_pong_B);
    if (status != ARM_CMSIS_NN_SUCCESS) return -2;

    // =========================================================================
    // LAYER 3: Block 0 Pointwise Conv2D (16 -> 32, k=1x1) + ReLU6
    // In:  [1, 26, 79, 16] -> ping_pong_B (zp=-128)
    // Out: [1, 26, 79, 32] -> ping_pong_A (65,728 bytes, zp=-128)
    // =========================================================================
    in_dims = {1, 26, 79, 16};
    out_dims = {1, 26, 79, 32};
    filter_dims = {32, 1, 1, 16};
    bias_dims = {1, 1, 1, 32};

    conv_params.input_offset = 128;
    conv_params.output_offset = -128;
    conv_params.stride.h = 1;
    conv_params.stride.w = 1;
    conv_params.padding.h = 0;
    conv_params.padding.w = 0;

    quant_params.multiplier = (int32_t*)CMSIS_B0_PW_M;
    quant_params.shift = (int32_t*)CMSIS_B0_PW_S;

    status = arm_convolve_wrapper_s8(&ctx, &conv_params, &quant_params, &in_dims, ping_pong_B,
                                     &filter_dims, CMSIS_B0_PW_W, &bias_dims, CMSIS_B0_PW_B,
                                     &out_dims, ping_pong_A);
    if (status != ARM_CMSIS_NN_SUCCESS) return -3;

    // =========================================================================
    // LAYER 4: Block 1 Depthwise Conv2D (32 -> 32, k=3x3, stride=2x2, pad=1x1) + ReLU6
    // In:  [1, 26, 79, 32] -> ping_pong_A (zp=-128)
    // Out: [1, 13, 40, 32] -> ping_pong_B (16,640 bytes, zp=-128)
    // =========================================================================
    in_dims = {1, 26, 79, 32};
    out_dims = {1, 13, 40, 32};
    filter_dims = {1, 3, 3, 32};
    bias_dims = {1, 1, 1, 32};

    dw_params.input_offset = 128;
    dw_params.output_offset = -128;
    dw_params.ch_mult = 1;
    dw_params.stride.h = 2;
    dw_params.stride.w = 2;
    dw_params.padding.h = 1;
    dw_params.padding.w = 1;

    quant_params.multiplier = (int32_t*)CMSIS_B1_DW_M;
    quant_params.shift = (int32_t*)CMSIS_B1_DW_S;

    status = arm_depthwise_conv_wrapper_s8(&ctx, &dw_params, &quant_params, &in_dims, ping_pong_A,
                                           &filter_dims, CMSIS_B1_DW_W, &bias_dims, CMSIS_B1_DW_B,
                                           &out_dims, ping_pong_B);
    if (status != ARM_CMSIS_NN_SUCCESS) return -4;

    // =========================================================================
    // LAYER 5: Block 1 Pointwise Conv2D (32 -> 48, k=1x1) + ReLU6
    // In:  [1, 13, 40, 32] -> ping_pong_B (zp=-128)
    // Out: [1, 13, 40, 48] -> ping_pong_A (24,960 bytes, zp=-128)
    // =========================================================================
    in_dims = {1, 13, 40, 32};
    out_dims = {1, 13, 40, 48};
    filter_dims = {48, 1, 1, 32};
    bias_dims = {1, 1, 1, 48};

    conv_params.input_offset = 128;
    conv_params.output_offset = -128;
    conv_params.stride.h = 1;
    conv_params.stride.w = 1;
    conv_params.padding.h = 0;
    conv_params.padding.w = 0;

    quant_params.multiplier = (int32_t*)CMSIS_B1_PW_M;
    quant_params.shift = (int32_t*)CMSIS_B1_PW_S;

    status = arm_convolve_wrapper_s8(&ctx, &conv_params, &quant_params, &in_dims, ping_pong_B,
                                     &filter_dims, CMSIS_B1_PW_W, &bias_dims, CMSIS_B1_PW_B,
                                     &out_dims, ping_pong_A);
    if (status != ARM_CMSIS_NN_SUCCESS) return -5;

    // =========================================================================
    // LAYER 6: Compress Conv2D (48 -> 32, k=1x1)
    // In:  [1, 13, 40, 48] -> ping_pong_A (zp=-128)
    // Out: [1, 13, 40, 32] -> out_features_int8 (16,640 bytes) (scale=0.155061, zp=5)
    // =========================================================================
    in_dims = {1, 13, 40, 48};
    out_dims = {1, 13, 40, 32};
    filter_dims = {32, 1, 1, 48};
    bias_dims = {1, 1, 1, 32};

    conv_params.input_offset = 128;
    conv_params.output_offset = 5; // output zp = 5, offset = 5
    conv_params.stride.h = 1;
    conv_params.stride.w = 1;
    conv_params.padding.h = 0;
    conv_params.padding.w = 0;
    conv_params.activation.min = -128;
    conv_params.activation.max = 127;

    quant_params.multiplier = (int32_t*)CMSIS_CMP_M;
    quant_params.shift = (int32_t*)CMSIS_CMP_S;

    status = arm_convolve_wrapper_s8(&ctx, &conv_params, &quant_params, &in_dims, ping_pong_A,
                                     &filter_dims, CMSIS_CMP_W, &bias_dims, CMSIS_CMP_B,
                                     &out_dims, out_features_int8);
    if (status != ARM_CMSIS_NN_SUCCESS) return -6;

    return 0;
}
