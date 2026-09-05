// ==============================================================================
// ⚡ 1D DILATED TC-RESNET + FREQUENCY-FOLDING NATIVE C++ INFERENCE ENGINE
// Silicon Labs EFR32MG24 (ARM Cortex-M33 @ 78 MHz)
// Register-Tile Cached INT8 2D CNN + INT8 Weights Flash Storage (<50 KB) + SIMD FPU
// ==============================================================================

#include "tcn_inference_engine.h"
#include "config.h"
#include <zephyr/kernel.h>
#include <arm_math.h>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <algorithm>

#if (ACTIVE_MODEL_PROFILE == PROFILE_SLIM_TCN_81)
    #include "tcn_slim_classifier_weights_int8_81.h"
    #define IN_QUANT_SCALE       TCN_SLIM_QUANT_SCALE[0]
    #define IN_QUANT_ZP          TCN_SLIM_QUANT_ZERO_POINT

    #define STEM_W               TCN_SLIM_STEM_CONV_WEIGHT_W
    #define STEM_W_SCALE         TCN_SLIM_STEM_CONV_WEIGHT_SCALE
    #define STEM_BIAS            TCN_SLIM_STEM_CONV_BIAS
    #define STEM_OUT_SCALE       TCN_SLIM_STEM_CONV_SCALE[0]

    #define PB0_DW_W             TCN_SLIM_PHI_BLOCKS_0_CONV1_WEIGHT_W
    #define PB0_DW_SCALE         TCN_SLIM_PHI_BLOCKS_0_CONV1_WEIGHT_SCALE
    #define PB0_DW_BIAS          TCN_SLIM_PHI_BLOCKS_0_CONV1_BIAS
    #define PB0_DW_OUT_SCALE     TCN_SLIM_PHI_BLOCKS_0_CONV1_SCALE[0]

    #define PB0_PW_W             TCN_SLIM_PHI_BLOCKS_0_CONV2_WEIGHT_W
    #define PB0_PW_SCALE         TCN_SLIM_PHI_BLOCKS_0_CONV2_WEIGHT_SCALE
    #define PB0_PW_BIAS          TCN_SLIM_PHI_BLOCKS_0_CONV2_BIAS
    #define PB0_PW_OUT_SCALE     TCN_SLIM_PHI_BLOCKS_0_CONV2_SCALE[0]

    #define PB2_DW_W             TCN_SLIM_PHI_BLOCKS_2_CONV1_WEIGHT_W
    #define PB2_DW_SCALE         TCN_SLIM_PHI_BLOCKS_2_CONV1_WEIGHT_SCALE
    #define PB2_DW_BIAS          TCN_SLIM_PHI_BLOCKS_2_CONV1_BIAS
    #define PB2_DW_OUT_SCALE     TCN_SLIM_PHI_BLOCKS_2_CONV1_SCALE[0]

    #define PB2_PW_W             TCN_SLIM_PHI_BLOCKS_2_CONV2_WEIGHT_W
    #define PB2_PW_SCALE         TCN_SLIM_PHI_BLOCKS_2_CONV2_WEIGHT_SCALE
    #define PB2_PW_BIAS          TCN_SLIM_PHI_BLOCKS_2_CONV2_BIAS
    #define PB2_PW_OUT_SCALE     TCN_SLIM_PHI_BLOCKS_2_CONV2_SCALE[0]

    #define PHI_MID_CH           24
    #define PHI_FINAL_CH         24

    #define TCN_STAGE0_IN_CH     96
    #define TCN_STAGE0_OUT_CH    64
    #define TCN_MID_CH           64
    #define TCN_FINAL_CH         96
    #define TCN_BOTTLENECK_DIM   48

    #define W_TCN_0_DW_W         TCN_SLIM_TCN_0_DW_CONV_WEIGHT_W
    #define W_TCN_0_DW_BIAS      TCN_SLIM_TCN_0_DW_CONV_BIAS
    #define W_TCN_0_DW_SCALE     TCN_SLIM_TCN_0_DW_CONV_WEIGHT_SCALE
    #define W_TCN_0_PW_W         TCN_SLIM_TCN_0_PW_CONV_WEIGHT_W
    #define W_TCN_0_PW_BIAS      TCN_SLIM_TCN_0_PW_CONV_BIAS
    #define W_TCN_0_PW_SCALE     TCN_SLIM_TCN_0_PW_CONV_WEIGHT_SCALE
    #define W_TCN_0_SC_W         TCN_SLIM_TCN_0_SHORTCUT_CONV_WEIGHT_W
    #define W_TCN_0_SC_BIAS      TCN_SLIM_TCN_0_SHORTCUT_CONV_BIAS
    #define W_TCN_0_SC_SCALE     TCN_SLIM_TCN_0_SHORTCUT_CONV_WEIGHT_SCALE

    #define W_TCN_1_DW_W         TCN_SLIM_TCN_1_DW_CONV_WEIGHT_W
    #define W_TCN_1_DW_BIAS      TCN_SLIM_TCN_1_DW_CONV_BIAS
    #define W_TCN_1_DW_SCALE     TCN_SLIM_TCN_1_DW_CONV_WEIGHT_SCALE
    #define W_TCN_1_PW_W         TCN_SLIM_TCN_1_PW_CONV_WEIGHT_W
    #define W_TCN_1_PW_BIAS      TCN_SLIM_TCN_1_PW_CONV_BIAS
    #define W_TCN_1_PW_SCALE     TCN_SLIM_TCN_1_PW_CONV_WEIGHT_SCALE

    #define W_TCN_2_DW_W         TCN_SLIM_TCN_2_DW_CONV_WEIGHT_W
    #define W_TCN_2_DW_BIAS      TCN_SLIM_TCN_2_DW_CONV_BIAS
    #define W_TCN_2_DW_SCALE     TCN_SLIM_TCN_2_DW_CONV_WEIGHT_SCALE
    #define W_TCN_2_PW_W         TCN_SLIM_TCN_2_PW_CONV_WEIGHT_W
    #define W_TCN_2_PW_BIAS      TCN_SLIM_TCN_2_PW_CONV_BIAS
    #define W_TCN_2_PW_SCALE     TCN_SLIM_TCN_2_PW_CONV_WEIGHT_SCALE

    #define W_TCN_3_DW_W         TCN_SLIM_TCN_3_DW_CONV_WEIGHT_W
    #define W_TCN_3_DW_BIAS      TCN_SLIM_TCN_3_DW_CONV_BIAS
    #define W_TCN_3_DW_SCALE     TCN_SLIM_TCN_3_DW_CONV_WEIGHT_SCALE
    #define W_TCN_3_PW_W         TCN_SLIM_TCN_3_PW_CONV_WEIGHT_W
    #define W_TCN_3_PW_BIAS      TCN_SLIM_TCN_3_PW_CONV_BIAS
    #define W_TCN_3_PW_SCALE     TCN_SLIM_TCN_3_PW_CONV_WEIGHT_SCALE

    #define W_TCN_4_DW_W         TCN_SLIM_TCN_4_DW_CONV_WEIGHT_W
    #define W_TCN_4_DW_BIAS      TCN_SLIM_TCN_4_DW_CONV_BIAS
    #define W_TCN_4_DW_SCALE     TCN_SLIM_TCN_4_DW_CONV_WEIGHT_SCALE
    #define W_TCN_4_PW_W         TCN_SLIM_TCN_4_PW_CONV_WEIGHT_W
    #define W_TCN_4_PW_BIAS      TCN_SLIM_TCN_4_PW_CONV_BIAS
    #define W_TCN_4_PW_SCALE     TCN_SLIM_TCN_4_PW_CONV_WEIGHT_SCALE
    #define W_TCN_4_SC_W         TCN_SLIM_TCN_4_SHORTCUT_CONV_WEIGHT_W
    #define W_TCN_4_SC_BIAS      TCN_SLIM_TCN_4_SHORTCUT_CONV_BIAS
    #define W_TCN_4_SC_SCALE     TCN_SLIM_TCN_4_SHORTCUT_CONV_WEIGHT_SCALE

    #define W_ATTN_W             TCN_SLIM_ATTENTION_ATTN_CONV_WEIGHT
    #define W_ATTN_BIAS          TCN_SLIM_ATTENTION_ATTN_CONV_BIAS
    #define W_POST_BN_W          TCN_SLIM_POST_TCN_BN_WEIGHT
    #define W_POST_BN_BIAS       TCN_SLIM_POST_TCN_BN_BIAS
    #define W_POST_BN_MEAN       TCN_SLIM_POST_TCN_BN_RUNNING_MEAN
    #define W_POST_BN_VAR        TCN_SLIM_POST_TCN_BN_RUNNING_VAR

    #define W_BOTTLENECK_W       TCN_SLIM_BOTTLENECK_WEIGHT
    #define W_BOTTLENECK_BIAS    TCN_SLIM_BOTTLENECK_BIAS
    #define W_FC_W               TCN_SLIM_FC_WEIGHT
    #define W_FC_BIAS            TCN_SLIM_FC_BIAS
#else
    #include "tcn_classifier_weights_int8_85.h"
    #define IN_QUANT_SCALE       TCN_QUANT_SCALE[0]
    #define IN_QUANT_ZP          TCN_QUANT_ZERO_POINT

    #define STEM_W               TCN_STEM_CONV_WEIGHT_W
    #define STEM_W_SCALE         TCN_STEM_CONV_WEIGHT_SCALE
    #define STEM_BIAS            TCN_STEM_CONV_BIAS
    #define STEM_OUT_SCALE       TCN_STEM_CONV_SCALE[0]

    #define PB0_DW_W             TCN_PHI_BLOCKS_0_CONV1_WEIGHT_W
    #define PB0_DW_SCALE         TCN_PHI_BLOCKS_0_CONV1_WEIGHT_SCALE
    #define PB0_DW_BIAS          TCN_PHI_BLOCKS_0_CONV1_BIAS
    #define PB0_DW_OUT_SCALE     TCN_PHI_BLOCKS_0_CONV1_SCALE[0]

    #define PB0_PW_W             TCN_PHI_BLOCKS_0_CONV2_WEIGHT_W
    #define PB0_PW_SCALE         TCN_PHI_BLOCKS_0_CONV2_WEIGHT_SCALE
    #define PB0_PW_BIAS          TCN_PHI_BLOCKS_0_CONV2_BIAS
    #define PB0_PW_OUT_SCALE     TCN_PHI_BLOCKS_0_CONV2_SCALE[0]

    #define PB2_DW_W             TCN_PHI_BLOCKS_2_CONV1_WEIGHT_W
    #define PB2_DW_SCALE         TCN_PHI_BLOCKS_2_CONV1_WEIGHT_SCALE
    #define PB2_DW_BIAS          TCN_PHI_BLOCKS_2_CONV1_BIAS
    #define PB2_DW_OUT_SCALE     TCN_PHI_BLOCKS_2_CONV1_SCALE[0]

    #define PB2_PW_W             TCN_PHI_BLOCKS_2_CONV2_WEIGHT_W
    #define PB2_PW_SCALE         TCN_PHI_BLOCKS_2_CONV2_WEIGHT_SCALE
    #define PB2_PW_BIAS          TCN_PHI_BLOCKS_2_CONV2_BIAS
    #define PB2_PW_OUT_SCALE     TCN_PHI_BLOCKS_2_CONV2_SCALE[0]

    #define PHI_MID_CH           32
    #define PHI_FINAL_CH         32

    #define TCN_STAGE0_IN_CH     128
    #define TCN_STAGE0_OUT_CH    96
    #define TCN_MID_CH           96
    #define TCN_FINAL_CH         128
    #define TCN_BOTTLENECK_DIM   64

    #define W_TCN_0_DW_W         TCN_TCN_0_DW_CONV_WEIGHT_W
    #define W_TCN_0_DW_BIAS      TCN_TCN_0_DW_CONV_BIAS
    #define W_TCN_0_DW_SCALE     TCN_TCN_0_DW_CONV_WEIGHT_SCALE
    #define W_TCN_0_PW_W         TCN_TCN_0_PW_CONV_WEIGHT_W
    #define W_TCN_0_PW_BIAS      TCN_TCN_0_PW_CONV_BIAS
    #define W_TCN_0_PW_SCALE     TCN_TCN_0_PW_CONV_WEIGHT_SCALE
    #define W_TCN_0_SC_W         TCN_TCN_0_SHORTCUT_CONV_WEIGHT_W
    #define W_TCN_0_SC_BIAS      TCN_TCN_0_SHORTCUT_CONV_BIAS
    #define W_TCN_0_SC_SCALE     TCN_TCN_0_SHORTCUT_CONV_WEIGHT_SCALE

    #define W_TCN_1_DW_W         TCN_TCN_1_DW_CONV_WEIGHT_W
    #define W_TCN_1_DW_BIAS      TCN_TCN_1_DW_CONV_BIAS
    #define W_TCN_1_DW_SCALE     TCN_TCN_1_DW_CONV_WEIGHT_SCALE
    #define W_TCN_1_PW_W         TCN_TCN_1_PW_CONV_WEIGHT_W
    #define W_TCN_1_PW_BIAS      TCN_TCN_1_PW_CONV_BIAS
    #define W_TCN_1_PW_SCALE     TCN_TCN_1_PW_CONV_WEIGHT_SCALE

    #define W_TCN_2_DW_W         TCN_TCN_2_DW_CONV_WEIGHT_W
    #define W_TCN_2_DW_BIAS      TCN_TCN_2_DW_CONV_BIAS
    #define W_TCN_2_DW_SCALE     TCN_TCN_2_DW_CONV_WEIGHT_SCALE
    #define W_TCN_2_PW_W         TCN_TCN_2_PW_CONV_WEIGHT_W
    #define W_TCN_2_PW_BIAS      TCN_TCN_2_PW_CONV_BIAS
    #define W_TCN_2_PW_SCALE     TCN_TCN_2_PW_CONV_WEIGHT_SCALE

    #define W_TCN_3_DW_W         TCN_TCN_3_DW_CONV_WEIGHT_W
    #define W_TCN_3_DW_BIAS      TCN_TCN_3_DW_CONV_BIAS
    #define W_TCN_3_DW_SCALE     TCN_TCN_3_DW_CONV_WEIGHT_SCALE
    #define W_TCN_3_PW_W         TCN_TCN_3_PW_CONV_WEIGHT_W
    #define W_TCN_3_PW_BIAS      TCN_TCN_3_PW_CONV_BIAS
    #define W_TCN_3_PW_SCALE     TCN_TCN_3_PW_CONV_WEIGHT_SCALE

    #define W_TCN_4_DW_W         TCN_TCN_4_DW_CONV_WEIGHT_W
    #define W_TCN_4_DW_BIAS      TCN_TCN_4_DW_CONV_BIAS
    #define W_TCN_4_DW_SCALE     TCN_TCN_4_DW_CONV_WEIGHT_SCALE
    #define W_TCN_4_PW_W         TCN_TCN_4_PW_CONV_WEIGHT_W
    #define W_TCN_4_PW_BIAS      TCN_TCN_4_PW_CONV_BIAS
    #define W_TCN_4_PW_SCALE     TCN_TCN_4_PW_CONV_WEIGHT_SCALE
    #define W_TCN_4_SC_W         TCN_TCN_4_SHORTCUT_CONV_WEIGHT_W
    #define W_TCN_4_SC_BIAS      TCN_TCN_4_SHORTCUT_CONV_BIAS
    #define W_TCN_4_SC_SCALE     TCN_TCN_4_SHORTCUT_CONV_WEIGHT_SCALE

    #define W_ATTN_W             TCN_ATTENTION_ATTN_CONV_WEIGHT
    #define W_ATTN_BIAS          TCN_ATTENTION_ATTN_CONV_BIAS
    #define W_POST_BN_W          TCN_POST_TCN_BN_WEIGHT
    #define W_POST_BN_BIAS       TCN_POST_TCN_BN_BIAS
    #define W_POST_BN_MEAN       TCN_POST_TCN_BN_RUNNING_MEAN
    #define W_POST_BN_VAR        TCN_POST_TCN_BN_RUNNING_VAR

    #define W_BOTTLENECK_W       TCN_BOTTLENECK_WEIGHT
    #define W_BOTTLENECK_BIAS    TCN_BOTTLENECK_BIAS
    #define W_FC_W               TCN_FC_WEIGHT
    #define W_FC_BIAS            TCN_FC_BIAS
#endif

int tcn_inference_init(void) {
    return 0;
}

void tcn_inference_get_quant_params(float *out_scale, int32_t *out_zero_point) {
    if (out_scale) *out_scale = IN_QUANT_SCALE;
    if (out_zero_point) *out_zero_point = IN_QUANT_ZP;
}

// ⚡ Fast Cache-Aligned 1D Dilated Depthwise Conv + Pointwise Residual Block
static void run_dilated_residual_block_fast(
    const float *input, float *output, float *scratch_dw,
    int in_ch, int out_ch, int dilation,
    const int8_t *dw_w, const float *dw_bias, float dw_scale,
    const int8_t *pw_w, const float *pw_bias, float pw_scale,
    const int8_t *sc_w, const float *sc_bias, float sc_scale)
{
    const int T = 40;

    // 1. Dilated Depthwise Conv1D [T, in_ch]
    for (int t = 0; t < T; t++) {
        for (int c = 0; c < in_ch; c++) {
            const int8_t *kw = &dw_w[c * 3];
            float dot = 0.0f;

            for (int k = 0; k < 3; k++) {
                int in_t = t + (k - 1) * dilation;
                if (in_t >= 0 && in_t < T) {
                    dot += input[in_t * in_ch + c] * (float)kw[k];
                }
            }
            float sum = (dw_bias ? dw_bias[c] : 0.0f) + (dot * dw_scale);
            scratch_dw[t * in_ch + c] = (sum > 0.0f) ? sum : 0.0f;
        }
    }

    // 2. Pointwise Conv1D (in_ch -> out_ch) [T, out_ch]
    for (int t = 0; t < T; t++) {
        const float *in_row = &scratch_dw[t * in_ch];
        float *out_row = &output[t * out_ch];

        for (int oc = 0; oc < out_ch; oc++) {
            const int8_t *w_row = &pw_w[oc * in_ch];
            float dot = 0.0f;

            #pragma GCC unroll 8
            for (int ic = 0; ic < in_ch; ic++) {
                dot += in_row[ic] * (float)w_row[ic];
            }
            float sum = (pw_bias ? pw_bias[oc] : 0.0f) + (dot * pw_scale);
            out_row[oc] = (sum > 0.0f) ? sum : 0.0f;
        }
    }

    // 3. Skip Connection
    if (in_ch == out_ch) {
        #pragma GCC unroll 8
        for (int i = 0; i < T * out_ch; i++) {
            output[i] += input[i];
        }
    } else if (sc_w != nullptr) {
        for (int t = 0; t < T; t++) {
            const float *in_row = &input[t * in_ch];
            float *out_row = &output[t * out_ch];

            for (int oc = 0; oc < out_ch; oc++) {
                const int8_t *w_row = &sc_w[oc * in_ch];
                float dot = 0.0f;

                #pragma GCC unroll 8
                for (int ic = 0; ic < in_ch; ic++) {
                    dot += in_row[ic] * (float)w_row[ic];
                }
                out_row[oc] += (sc_bias ? sc_bias[oc] : 0.0f) + (dot * sc_scale);
            }
        }
    }
}

// 🛡️ Safe Fixed-Point INT8 Clamping Helper
static inline int8_t clamp_int8(int32_t val) {
    if (val > 127) return 127;
    if (val < -128) return -128;
    return (int8_t)val;
}

int tcn_inference_run(const int8_t *spectrogram_int8, int8_t *ping_pong_A, int8_t *ping_pong_B, int *out_class_id, float *out_confidence) {
    if (!spectrogram_int8 || !ping_pong_A || !ping_pong_B || !out_class_id || !out_confidence) {
        return -1;
    }

    uint32_t t_cnn_start = k_cycle_get_32();

    // =========================================================================
    // STAGE 1: ULTRA-FAST INT8 2D PHINET STEM & INVERTED BOTTLENECKS
    // =========================================================================
    int8_t *buf_A = ping_pong_A;
    int8_t *buf_B = ping_pong_B;

    const float in_scale = IN_QUANT_SCALE;
    const int32_t in_zp = IN_QUANT_ZP;

    // Pre-calculate bias offsets (Saves 500,000 divisions!)
    float stem_bias_scaled[16];
    const float stem_mult = (in_scale * STEM_W_SCALE) / STEM_OUT_SCALE;
    for (int oc = 0; oc < 16; oc++) {
        stem_bias_scaled[oc] = STEM_BIAS[oc] / STEM_OUT_SCALE;
    }

    // ⚡ Layer 1: Stem Conv2D with 3x3 Register-Tile Caching (1 -> 16, k=3x3, s=2x2)
    // Out: [26, 157, 16] -> buf_A (65,312 bytes)
    int32_t l1_min = 127, l1_max = -128;
    for (int h = 0; h < 26; h++) {
        int in_h0 = h * 2 - 1;
        int in_h1 = h * 2;
        int in_h2 = h * 2 + 1;

        for (int w = 0; w < 157; w++) {
            int in_w0 = w * 2 - 1;
            int in_w1 = w * 2;
            int in_w2 = w * 2 + 1;

            // Load 3x3 input patch ONCE into CPU registers (reused across all 16 channels!)
            int32_t p0 = (in_h0 >= 0 && in_w0 >= 0) ? (spectrogram_int8[in_h0 * 313 + in_w0] - in_zp) : 0;
            int32_t p1 = (in_h0 >= 0 && in_w1 < 313) ? (spectrogram_int8[in_h0 * 313 + in_w1] - in_zp) : 0;
            int32_t p2 = (in_h0 >= 0 && in_w2 < 313) ? (spectrogram_int8[in_h0 * 313 + in_w2] - in_zp) : 0;

            int32_t p3 = (in_h1 < 52 && in_w0 >= 0) ? (spectrogram_int8[in_h1 * 313 + in_w0] - in_zp) : 0;
            int32_t p4 = (in_h1 < 52 && in_w1 < 313) ? (spectrogram_int8[in_h1 * 313 + in_w1] - in_zp) : 0;
            int32_t p5 = (in_h1 < 52 && in_w2 < 313) ? (spectrogram_int8[in_h1 * 313 + in_w2] - in_zp) : 0;

            int32_t p6 = (in_h2 < 52 && in_w0 >= 0) ? (spectrogram_int8[in_h2 * 313 + in_w0] - in_zp) : 0;
            int32_t p7 = (in_h2 < 52 && in_w1 < 313) ? (spectrogram_int8[in_h2 * 313 + in_w1] - in_zp) : 0;
            int32_t p8 = (in_h2 < 52 && in_w2 < 313) ? (spectrogram_int8[in_h2 * 313 + in_w2] - in_zp) : 0;

            int8_t *out_row = &buf_A[(h * 157 + w) * 16];

            #pragma GCC unroll 16
            for (int oc = 0; oc < 16; oc++) {
                const int8_t *kw = &STEM_W[oc * 9];
                int32_t dot = p0 * (int32_t)kw[0] + p1 * (int32_t)kw[1] + p2 * (int32_t)kw[2] +
                              p3 * (int32_t)kw[3] + p4 * (int32_t)kw[4] + p5 * (int32_t)kw[5] +
                              p6 * (int32_t)kw[6] + p7 * (int32_t)kw[7] + p8 * (int32_t)kw[8];

                float val = stem_bias_scaled[oc] + ((float)dot * stem_mult);
                int32_t q_val = (val > 0.0f) ? (int32_t)(val + 0.5f) : 0;
                int8_t clamped = clamp_int8(q_val);
                out_row[oc] = clamped;
                if (clamped < l1_min) l1_min = clamped;
                if (clamped > l1_max) l1_max = clamped;
            }
        }
    }

    // ⚡ Layer 2: PB0 DW Conv2D with Row Stencils (16 -> 16, k=3x3, s=1x2)
    // In: buf_A [26, 157, 16] -> Out: buf_B [26, 79, 16] (32,864 bytes)
    float pb0_dw_bias_scaled[16];
    const float pb0_dw_mult = (STEM_OUT_SCALE * PB0_DW_SCALE) / PB0_DW_OUT_SCALE;
    for (int c = 0; c < 16; c++) {
        pb0_dw_bias_scaled[c] = PB0_DW_BIAS[c] / PB0_DW_OUT_SCALE;
    }

    for (int h = 0; h < 26; h++) {
        int in_h0 = h - 1;
        int in_h1 = h;
        int in_h2 = h + 1;

        const int8_t *row0 = (in_h0 >= 0) ? &buf_A[in_h0 * 157 * 16] : nullptr;
        const int8_t *row1 = &buf_A[in_h1 * 157 * 16];
        const int8_t *row2 = (in_h2 < 26) ? &buf_A[in_h2 * 157 * 16] : nullptr;

        for (int w = 0; w < 79; w++) {
            int in_w0 = w * 2 - 1;
            int in_w1 = w * 2;
            int in_w2 = w * 2 + 1;

            int8_t *out_row = &buf_B[(h * 79 + w) * 16];

            #pragma GCC unroll 16
            for (int c = 0; c < 16; c++) {
                const int8_t *kw = &PB0_DW_W[c * 9];
                int32_t dot = 0;

                if (row0) {
                    if (in_w0 >= 0)  dot += (int32_t)row0[in_w0 * 16 + c] * (int32_t)kw[0];
                    if (in_w1 < 157) dot += (int32_t)row0[in_w1 * 16 + c] * (int32_t)kw[1];
                    if (in_w2 < 157) dot += (int32_t)row0[in_w2 * 16 + c] * (int32_t)kw[2];
                }
                if (in_w0 >= 0)  dot += (int32_t)row1[in_w0 * 16 + c] * (int32_t)kw[3];
                if (in_w1 < 157) dot += (int32_t)row1[in_w1 * 16 + c] * (int32_t)kw[4];
                if (in_w2 < 157) dot += (int32_t)row1[in_w2 * 16 + c] * (int32_t)kw[5];

                if (row2) {
                    if (in_w0 >= 0)  dot += (int32_t)row2[in_w0 * 16 + c] * (int32_t)kw[6];
                    if (in_w1 < 157) dot += (int32_t)row2[in_w1 * 16 + c] * (int32_t)kw[7];
                    if (in_w2 < 157) dot += (int32_t)row2[in_w2 * 16 + c] * (int32_t)kw[8];
                }

                float val = pb0_dw_bias_scaled[c] + ((float)dot * pb0_dw_mult);
                int32_t q_val = (val > 0.0f) ? (int32_t)(val + 0.5f) : 0;
                out_row[c] = clamp_int8(q_val);
            }
        }
    }

    // ⚡ Layer 3: PB0 PW Conv2D with __SMLAD (16 -> PHI_MID_CH, k=1x1)
    // In: buf_B [26, 79, 16] -> Out: buf_A [26, 79, PHI_MID_CH] (49,296 bytes)
    float pb0_pw_bias_scaled[PHI_MID_CH];
    const float pb0_pw_mult = (PB0_DW_OUT_SCALE * PB0_PW_SCALE) / PB0_PW_OUT_SCALE;
    for (int oc = 0; oc < PHI_MID_CH; oc++) {
        pb0_pw_bias_scaled[oc] = PB0_PW_BIAS[oc] / PB0_PW_OUT_SCALE;
    }

    for (int i = 0; i < 26 * 79; i++) {
        const int8_t *in_c = &buf_B[i * 16];
        int8_t *out_c = &buf_A[i * PHI_MID_CH];

        for (int oc = 0; oc < PHI_MID_CH; oc++) {
            const int8_t *kw = &PB0_PW_W[oc * 16];
            int32_t dot = 0;
            #pragma GCC unroll 8
            for (int ic = 0; ic < 16; ic += 2) {
                uint32_t in_p = (uint32_t)(uint16_t)((int16_t)in_c[ic]) | ((uint32_t)(uint16_t)((int16_t)in_c[ic+1]) << 16);
                uint32_t w_p  = (uint32_t)(uint16_t)((int16_t)kw[ic])   | ((uint32_t)(uint16_t)((int16_t)kw[ic+1]) << 16);
                dot = __SMLAD(in_p, w_p, dot);
            }
            float val = pb0_pw_bias_scaled[oc] + ((float)dot * pb0_pw_mult);
            int32_t q_val = (val > 0.0f) ? (int32_t)(val + 0.5f) : 0;
            out_c[oc] = clamp_int8(q_val);
        }
    }

    // ⚡ Layer 4: PB2 DW Conv2D with Row Stencils (PHI_MID_CH -> PHI_MID_CH, k=3x3, s=2x2)
    // In: buf_A [26, 79, PHI_MID_CH] -> Out: buf_B [13, 40, PHI_MID_CH] (12,480 bytes)
    float pb2_dw_bias_scaled[PHI_MID_CH];
    const float pb2_dw_mult = (PB0_PW_OUT_SCALE * PB2_DW_SCALE) / PB2_DW_OUT_SCALE;
    for (int c = 0; c < PHI_MID_CH; c++) {
        pb2_dw_bias_scaled[c] = PB2_DW_BIAS[c] / PB2_DW_OUT_SCALE;
    }

    for (int h = 0; h < 13; h++) {
        int in_h0 = h * 2 - 1;
        int in_h1 = h * 2;
        int in_h2 = h * 2 + 1;

        const int8_t *row0 = (in_h0 >= 0) ? &buf_A[in_h0 * 79 * PHI_MID_CH] : nullptr;
        const int8_t *row1 = &buf_A[in_h1 * 79 * PHI_MID_CH];
        const int8_t *row2 = (in_h2 < 26) ? &buf_A[in_h2 * 79 * PHI_MID_CH] : nullptr;

        for (int w = 0; w < 40; w++) {
            int in_w0 = w * 2 - 1;
            int in_w1 = w * 2;
            int in_w2 = w * 2 + 1;

            int8_t *out_row = &buf_B[(h * 40 + w) * PHI_MID_CH];

            for (int c = 0; c < PHI_MID_CH; c++) {
                const int8_t *kw = &PB2_DW_W[c * 9];
                int32_t dot = 0;

                if (row0) {
                    if (in_w0 >= 0)  dot += (int32_t)row0[in_w0 * PHI_MID_CH + c] * (int32_t)kw[0];
                    if (in_w1 < 79)  dot += (int32_t)row0[in_w1 * PHI_MID_CH + c] * (int32_t)kw[1];
                    if (in_w2 < 79)  dot += (int32_t)row0[in_w2 * PHI_MID_CH + c] * (int32_t)kw[2];
                }
                if (in_w0 >= 0)  dot += (int32_t)row1[in_w0 * PHI_MID_CH + c] * (int32_t)kw[3];
                if (in_w1 < 79)  dot += (int32_t)row1[in_w1 * PHI_MID_CH + c] * (int32_t)kw[4];
                if (in_w2 < 79)  dot += (int32_t)row1[in_w2 * PHI_MID_CH + c] * (int32_t)kw[5];

                if (row2) {
                    if (in_w0 >= 0)  dot += (int32_t)row2[in_w0 * PHI_MID_CH + c] * (int32_t)kw[6];
                    if (in_w1 < 79)  dot += (int32_t)row2[in_w1 * PHI_MID_CH + c] * (int32_t)kw[7];
                    if (in_w2 < 79)  dot += (int32_t)row2[in_w2 * PHI_MID_CH + c] * (int32_t)kw[8];
                }

                float val = pb2_dw_bias_scaled[c] + ((float)dot * pb2_dw_mult);
                int32_t q_val = (val > 0.0f) ? (int32_t)(val + 0.5f) : 0;
                out_row[c] = clamp_int8(q_val);
            }
        }
    }

    // ⚡ Layer 5: PB2 PW Conv2D with __SMLAD (PHI_MID_CH -> PHI_FINAL_CH, k=1x1)
    // In: buf_B [13, 40, PHI_MID_CH] -> Out: buf_A [13, 40, PHI_FINAL_CH] (12,480 bytes)
    float pb2_pw_bias_scaled[PHI_FINAL_CH];
    const float pb2_pw_mult = (PB2_DW_OUT_SCALE * PB2_PW_SCALE) / PB2_PW_OUT_SCALE;
    for (int oc = 0; oc < PHI_FINAL_CH; oc++) {
        pb2_pw_bias_scaled[oc] = PB2_PW_BIAS[oc] / PB2_PW_OUT_SCALE;
    }

    int32_t l5_min = 127, l5_max = -128;
    for (int i = 0; i < 13 * 40; i++) {
        const int8_t *in_c = &buf_B[i * PHI_MID_CH];
        int8_t *out_c = &buf_A[i * PHI_FINAL_CH];

        for (int oc = 0; oc < PHI_FINAL_CH; oc++) {
            const int8_t *kw = &PB2_PW_W[oc * PHI_MID_CH];
            int32_t dot = 0;
            #pragma GCC unroll 8
            for (int ic = 0; ic < PHI_MID_CH; ic += 2) {
                uint32_t in_p = (uint32_t)(uint16_t)((int16_t)in_c[ic]) | ((uint32_t)(uint16_t)((int16_t)in_c[ic+1]) << 16);
                uint32_t w_p  = (uint32_t)(uint16_t)((int16_t)kw[ic])   | ((uint32_t)(uint16_t)((int16_t)kw[ic+1]) << 16);
                dot = __SMLAD(in_p, w_p, dot);
            }
            float val = pb2_pw_bias_scaled[oc] + ((float)dot * pb2_pw_mult);
            int32_t q_val = (val > 0.0f) ? (int32_t)(val + 0.5f) : 0;
            int8_t clamped = clamp_int8(q_val);
            out_c[oc] = clamped;
            if (clamped < l5_min) l5_min = clamped;
            if (clamped > l5_max) l5_max = clamped;
        }
    }

    uint32_t t_cnn_end = k_cycle_get_32();

    // =========================================================================
    // FREQUENCY-TO-CHANNEL FOLDING (13 Bins -> 4 Sub-Bands x PHI_FINAL_CH -> TCN_IN_CH)
    // ⚠️ Channel-Major: folded_ch = (c * 4) + sb
    // Out: s_tcn_buf_a [40 Time, TCN_STAGE0_IN_CH] in buf_B (15.3 KB float)
    // =========================================================================
    uint32_t t_tcn_start = k_cycle_get_32();

    float *s_tcn_buf_a = (float*)ping_pong_B; // 40 * 96 * 4 = 15,360 bytes
    float *s_tcn_buf_b = (float*)ping_pong_A; // 40 * 96 * 4 = 15,360 bytes
    float *s_tcn_residual = (float*)(ping_pong_A + 16384);

    float tcn_in_min = 1e9f, tcn_in_max = -1e9f, tcn_in_energy = 0.0f;
    for (int t = 0; t < 40; t++) {
        for (int c = 0; c < PHI_FINAL_CH; c++) {
            for (int sb = 0; sb < 4; sb++) {
                int f_start = (sb == 0) ? 0 : (sb == 1) ? 3 : (sb == 2) ? 6 : 9;
                int f_end   = (sb == 0) ? 3 : (sb == 1) ? 6 : (sb == 2) ? 9 : 13;
                int f_count = f_end - f_start;

                float sum = 0.0f;
                for (int f = f_start; f < f_end; f++) {
                    sum += (float)buf_A[(f * 40 + t) * PHI_FINAL_CH + c];
                }
                float dequant_val = (sum / (float)f_count) * PB2_PW_OUT_SCALE;
                int folded_ch = c * 4 + sb;
                s_tcn_buf_a[t * TCN_STAGE0_IN_CH + folded_ch] = dequant_val;

                if (dequant_val < tcn_in_min) tcn_in_min = dequant_val;
                if (dequant_val > tcn_in_max) tcn_in_max = dequant_val;
                tcn_in_energy += dequant_val * dequant_val;
            }
        }
    }

    // =========================================================================
    // STAGE 2: 5-STAGE 1D DILATED TC-RESNET (FAST FPU & INT8 WEIGHTS)
    // =========================================================================
    // Stage 0 (d=1)
    run_dilated_residual_block_fast(s_tcn_buf_a, s_tcn_buf_b, s_tcn_residual, TCN_STAGE0_IN_CH, TCN_STAGE0_OUT_CH, 1,
                                   W_TCN_0_DW_W, W_TCN_0_DW_BIAS, W_TCN_0_DW_SCALE,
                                   W_TCN_0_PW_W, W_TCN_0_PW_BIAS, W_TCN_0_PW_SCALE,
                                   W_TCN_0_SC_W, W_TCN_0_SC_BIAS, W_TCN_0_SC_SCALE);

    // Stage 1 (d=2)
    run_dilated_residual_block_fast(s_tcn_buf_b, s_tcn_buf_a, s_tcn_residual, TCN_MID_CH, TCN_MID_CH, 2,
                                   W_TCN_1_DW_W, W_TCN_1_DW_BIAS, W_TCN_1_DW_SCALE,
                                   W_TCN_1_PW_W, W_TCN_1_PW_BIAS, W_TCN_1_PW_SCALE,
                                   nullptr, nullptr, 0.0f);

    // Stage 2 (d=4)
    run_dilated_residual_block_fast(s_tcn_buf_a, s_tcn_buf_b, s_tcn_residual, TCN_MID_CH, TCN_MID_CH, 4,
                                   W_TCN_2_DW_W, W_TCN_2_DW_BIAS, W_TCN_2_DW_SCALE,
                                   W_TCN_2_PW_W, W_TCN_2_PW_BIAS, W_TCN_2_PW_SCALE,
                                   nullptr, nullptr, 0.0f);

    // Stage 3 (d=8)
    run_dilated_residual_block_fast(s_tcn_buf_b, s_tcn_buf_a, s_tcn_residual, TCN_MID_CH, TCN_MID_CH, 8,
                                   W_TCN_3_DW_W, W_TCN_3_DW_BIAS, W_TCN_3_DW_SCALE,
                                   W_TCN_3_PW_W, W_TCN_3_PW_BIAS, W_TCN_3_PW_SCALE,
                                   nullptr, nullptr, 0.0f);

    // Stage 4 (d=16)
    run_dilated_residual_block_fast(s_tcn_buf_a, s_tcn_buf_b, s_tcn_residual, TCN_MID_CH, TCN_FINAL_CH, 16,
                                   W_TCN_4_DW_W, W_TCN_4_DW_BIAS, W_TCN_4_DW_SCALE,
                                   W_TCN_4_PW_W, W_TCN_4_PW_BIAS, W_TCN_4_PW_SCALE,
                                   W_TCN_4_SC_W, W_TCN_4_SC_BIAS, W_TCN_4_SC_SCALE);

    // =========================================================================
    // STAGE 3: TEMPORAL SOFTMAX ATTENTION & CLASSIFIER HEAD
    // =========================================================================
    const int T = 40;
    const int C = TCN_FINAL_CH;
    float scores[40];
    float max_score = -1e9f;

    for (int t = 0; t < T; t++) {
        const float *t_feat = &s_tcn_buf_b[t * C];
        float sum = W_ATTN_BIAS[0];
        #pragma GCC unroll 8
        for (int c = 0; c < C; c++) {
            sum += t_feat[c] * W_ATTN_W[c];
        }
        scores[t] = sum;
        if (sum > max_score) max_score = sum;
    }

    float sum_exp = 0.0f;
    for (int t = 0; t < T; t++) {
        float diff = scores[t] - max_score;
        if (diff < -60.0f) diff = -60.0f;
        scores[t] = expf(diff);
        sum_exp += scores[t];
    }
    if (sum_exp < 1e-6f) sum_exp = 1e-6f;
    int peak_attn_t = 0;
    float max_attn_w = 0.0f;
    for (int t = 0; t < T; t++) {
        scores[t] /= sum_exp;
        if (scores[t] > max_attn_w) {
            max_attn_w = scores[t];
            peak_attn_t = t;
        }
    }

    // Context Vector [C]
    float context[TCN_FINAL_CH];
    for (int c = 0; c < C; c++) {
        float sum = 0.0f;
        for (int t = 0; t < T; t++) {
            sum += s_tcn_buf_b[t * C + c] * scores[t];
        }
        // Post-TCN BatchNorm
        float std_dev = sqrtf(W_POST_BN_VAR[c] + 1e-5f);
        float norm = (sum - W_POST_BN_MEAN[c]) / std_dev;
        context[c] = norm * W_POST_BN_W[c] + W_POST_BN_BIAS[c];
    }

    // Bottleneck Linear (C -> TCN_BOTTLENECK_DIM) + ReLU
    float h_bottleneck[TCN_BOTTLENECK_DIM];
    for (int j = 0; j < TCN_BOTTLENECK_DIM; j++) {
        const float *w_row = &W_BOTTLENECK_W[j * C];
        float sum = W_BOTTLENECK_BIAS[j];
        #pragma GCC unroll 8
        for (int c = 0; c < C; c++) {
            sum += context[c] * w_row[c];
        }
        h_bottleneck[j] = (sum > 0.0f) ? sum : 0.0f;
    }

    // FC Linear (TCN_BOTTLENECK_DIM -> 50 Logits)
    float raw_logits[NUM_ESC50_CLASSES];
    int top_class = 0;
    float max_logit = -1e9f;

    for (int i = 0; i < NUM_ESC50_CLASSES; i++) {
        const float *w_row = &W_FC_W[i * TCN_BOTTLENECK_DIM];
        float sum = W_FC_BIAS[i];
        #pragma GCC unroll 8
        for (int j = 0; j < TCN_BOTTLENECK_DIM; j++) {
            sum += h_bottleneck[j] * w_row[j];
        }
        raw_logits[i] = sum;
        if (sum > max_logit) {
            max_logit = sum;
            top_class = i;
        }
    }

    // Numerically Stable Softmax Confidence
    float sum_l = 0.0f;
    for (int i = 0; i < NUM_ESC50_CLASSES; i++) {
        float diff_l = raw_logits[i] - max_logit;
        if (diff_l < -60.0f) diff_l = -60.0f;
        sum_l += expf(diff_l);
    }
    if (sum_l < 1e-6f) sum_l = 1e-6f;
    float confidence = 1.0f / sum_l;

    uint32_t t_tcn_end = k_cycle_get_32();

    // =========================================================================
    // 📊 ON-CHIP FORENSICS & INTERMEDIATE LAYER HEALTH AUDIT
    // =========================================================================
    uint32_t cnn_time_us = k_cyc_to_us_near32(t_cnn_end - t_cnn_start);
    uint32_t tcn_time_us = k_cyc_to_us_near32(t_tcn_end - t_tcn_start);
    float cnn_time_ms = (float)cnn_time_us / 1000.0f;
    float tcn_time_ms = (float)tcn_time_us / 1000.0f;

    printf("\n  🔍 ON-CHIP FORENSICS (1D Dilated TC-ResNet):\n");
    printf("    • Model Profile : %s\n", (ACTIVE_MODEL_PROFILE == PROFILE_SLIM_TCN_81) ? "Slim 48k Channel-Pruned (69.5% INT8 Fold-5)" : "Standard 93k (67.75% INT8 Fold-5)");
    printf("    • Working Arena : 96.3 KB RAM (Shared Ping-Pong)\n");
    printf("    • Layer 1 Stem  : INT8 range [%d .. %d]\n", l1_min, l1_max);
    printf("    • Layer 5 PhiOut: INT8 range [%d .. %d]\n", l5_min, l5_max);
    printf("    • TCN In Energy : %.2f (range [%.2f .. %.2f])\n", (double)tcn_in_energy, (double)tcn_in_min, (double)tcn_in_max);
    printf("    • Attn Peak Step: Step #%d (Weight: %.1f%%)\n", peak_attn_t, (double)(max_attn_w * 100.0f));
    printf("    • Top-1 Class   : Class %2d (%s)\n", top_class, ESC50_CLASS_NAMES[top_class]);

    printf("\n  📊 Classifier Top 5 Logits:\n");
    int picked[5] = {-1, -1, -1, -1, -1};
    for (int rank = 0; rank < 5; rank++) {
        int best_k = -1;
        float best_l = -1e9f;
        for (int k = 0; k < NUM_ESC50_CLASSES; k++) {
            bool used = false;
            for (int r = 0; r < rank; r++) if (picked[r] == k) used = true;
            if (!used && raw_logits[k] > best_l) {
                best_l = raw_logits[k];
                best_k = k;
            }
        }
        if (best_k >= 0) {
            picked[rank] = best_k;
            float diff_top = raw_logits[best_k] - max_logit;
            if (diff_top < -60.0f) diff_top = -60.0f;
            float prob = expf(diff_top) / sum_l;
            printf("    #%d: Class %2d (%-18s) -> %5.1f%% (logit: %.2f)\n",
                   rank + 1, best_k, ESC50_CLASS_NAMES[best_k], (double)(prob * 100.0f), (double)raw_logits[best_k]);
        }
    }

    printf("\n  ⏱️ INFERENCE LATENCY: Stage 1 2D CNN = %.2f ms | Stage 2 1D TCN = %.2f ms | Total ML = %.2f ms\n",
           (double)cnn_time_ms, (double)tcn_time_ms, (double)(cnn_time_ms + tcn_time_ms));

    *out_class_id = top_class;
    *out_confidence = confidence;

    return 0;
}
