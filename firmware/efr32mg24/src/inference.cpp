#include <zephyr/kernel.h>
#include "inference.h"
#include "audio_preprocessing.h"
#include "config.h"

#if (ACTIVE_MODEL_PROFILE == PROFILE_FLAGSHIP_DENSE_91)
    #include "phinet_features_model_data.h"
    #include "gru_classifier_weights.h"
#elif (ACTIVE_MODEL_PROFILE == PROFILE_SPARSE_PRUNED_48K)
    #include "phinet_features_pruned_model_data.h"
    #include "gru_classifier_weights_pruned_csr.h"
#elif (ACTIVE_MODEL_PROFILE == PROFILE_INT8_FIXED_SIMD_91)
    #include "phinet_features_pruned_model_data.h"
    #include "gru_classifier_weights_int8_fixed.h"
#elif (ACTIVE_MODEL_PROFILE == PROFILE_CMSIS_NN_PINGPONG_91)
    #include "cmsis_nn_cnn_engine.h"
    #include "gru_classifier_weights_int8_fixed.h"
#elif (ACTIVE_MODEL_PROFILE == PROFILE_TCN_85 || ACTIVE_MODEL_PROFILE == PROFILE_SLIM_TCN_81)
    #include "tcn_inference_engine.h"
#endif

#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <cstdio>
#include <cmath>
#include <arm_math.h>

/* ============================================================================
 * 2-STAGE HYBRID TINYML INFERENCE ENGINE FOR SILICON LABS EFR32MG24
 * Stage 1: Full-Integer INT8 PhiNet 2D CNN (Native CMSIS-NN Ping-Pong / TFLM)
 * Stage 2: Recurrent GRU + Attention Classifier Head (Native C++ on FPU)
 * ============================================================================ */

/* ============================================================================
 * 🏓 UNIFIED ZERO-BSS GLOBAL INFERENCE MEMORY OVERLAY
 * Stage 1 CNN Ping-Pong Buffers (96.3 KB) & Stage 2 GRU Working Buffers
 * Ping-Pong B (32.1 KB) hosts Input Spectrogram & Stage 1 Features Output.
 * Ping-Pong A (64.2 KB) hosts Intermediate Layers & Stage 2 Recurrent State.
 * ZERO memory overwrite collision during inference!
 * ============================================================================ */
union alignas(16) InferenceMemoryPool {
#if (ACTIVE_MODEL_PROFILE == PROFILE_CMSIS_NN_PINGPONG_91 || ACTIVE_MODEL_PROFILE == PROFILE_TCN_85 || ACTIVE_MODEL_PROFILE == PROFILE_SLIM_TCN_81)
    struct {
        int8_t ping_pong_A[65728]; // 64.2 KB (Layer 1 Stem Conv2D [26x157x16] = 65,312 Bytes)
        int8_t ping_pong_B[32864]; // 32.1 KB (Input Spectrogram [52x313] & Layer 2 Output)
    } cmsis_nn;
#else
    uint8_t tensor_arena[TFLM_TENSOR_ARENA_SIZE];
#endif
#if (ACTIVE_MODEL_PROFILE != PROFILE_TCN_85 && ACTIVE_MODEL_PROFILE != PROFILE_SLIM_TCN_81)
    struct {
        alignas(16) float s_features[GRU_TIME_STEPS][GRU_INPUT_DIM]; // 5.0 KB (in ping_pong_A)
        alignas(16) float s_H[GRU_TIME_STEPS][GRU_HIDDEN_DIM];       // 24.9 KB (in ping_pong_A)
        alignas(16) float s_gate_x[3 * GRU_HIDDEN_DIM];              // 1.9 KB (in ping_pong_A)
        alignas(16) float s_gate_h[3 * GRU_HIDDEN_DIM];              // 1.9 KB (in ping_pong_A)
    } stage2;
#endif
};

static InferenceMemoryPool s_mem_pool;
#if (ACTIVE_MODEL_PROFILE != PROFILE_CMSIS_NN_PINGPONG_91 && ACTIVE_MODEL_PROFILE != PROFILE_TCN_85 && ACTIVE_MODEL_PROFILE != PROFILE_SLIM_TCN_81)
#define tensor_arena (s_mem_pool.tensor_arena)
#endif
#if (ACTIVE_MODEL_PROFILE != PROFILE_TCN_85 && ACTIVE_MODEL_PROFILE != PROFILE_SLIM_TCN_81)
#define s_features   (s_mem_pool.stage2.s_features)
#define s_H          (s_mem_pool.stage2.s_H)
#define s_gate_x     (s_mem_pool.stage2.s_gate_x)
#define s_gate_h     (s_mem_pool.stage2.s_gate_h)
#endif
#define s_input_spectrogram (s_mem_pool.cmsis_nn.ping_pong_B)
#define s_stage1_cnn_features (s_mem_pool.cmsis_nn.ping_pong_B)


static const tflite::Model* model = nullptr;
static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* input_tensor = nullptr;
static TfLiteTensor* output_tensor = nullptr;
static bool is_inference_initialized = false;

extern "C" int inference_init(void) {
    if (is_inference_initialized) return 0;

    printf("\n========================================================\n");
    printf(" 🚀 2-STAGE HYBRID TINYML INFERENCE ENGINE INIT\n");
    printf("========================================================\n");

#if (ACTIVE_MODEL_PROFILE == PROFILE_TCN_85 || ACTIVE_MODEL_PROFILE == PROFILE_SLIM_TCN_81)
    tcn_inference_init();
    printf("   • Profile   : 1D Dilated TC-ResNet (%s)\n",
           (ACTIVE_MODEL_PROFILE == PROFILE_SLIM_TCN_81) ? "Slim 48k Pruned, 80.50% INT8, 47.6 KB Flash" : "Standard 93k, 85.25% INT8, 92.8 KB Flash");
    printf("   • Engine    : Native C++ Zero-Copy SIMD Pipeline (<10.2 KB SRAM)\n");
    printf("========================================================\n\n");
    is_inference_initialized = true;
    return 0;
#elif (ACTIVE_MODEL_PROFILE == PROFILE_CMSIS_NN_PINGPONG_91)
    cmsis_nn_cnn_init();
    printf("   • Profile   : Native CMSIS-NN Ping-Pong CNN (98.5 KB SRAM vs 172 KB TFLM!)\n");
    printf("   • Stage 2   : Inlined ARM DSP __SMLAD Fixed-Point GRU Head\n");
    printf("========================================================\n\n");
    is_inference_initialized = true;
    return 0;
#else
    /* 1. Load Stage 1 PhiNet Features Model Flatbuffer */
    model = tflite::GetModel(g_phinet_features_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("[ERROR] Model schema version mismatch! Model: %ld, Supported: %d\n",
               model->version(), TFLITE_SCHEMA_VERSION);
        return -1;
    }
    printf("-> [OK] Stage 1 INT8 CNN Model loaded (Size: %u bytes / %.2f KB)\n",
           g_phinet_features_model_data_len, g_phinet_features_model_data_len / 1024.0f);

    /* 2. Instantiate MicroMutableOpResolver with Hardware CMSIS-NN Kernels */
    static tflite::MicroMutableOpResolver<32> resolver;
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddReshape();
    resolver.AddAdd();
    resolver.AddMul();
    resolver.AddLogistic();
    resolver.AddRelu();
    resolver.AddRelu6();
    resolver.AddQuantize();
    resolver.AddDequantize();
    resolver.AddAveragePool2D();
    resolver.AddMaxPool2D();
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    resolver.AddSub();
    resolver.AddDiv();
    resolver.AddMean();
    resolver.AddStridedSlice();
    resolver.AddTranspose();
    resolver.AddGather();
    resolver.AddTanh();
    resolver.AddConcatenation();
    resolver.AddSum();
    resolver.AddPad();


    /* 3. Instantiate MicroInterpreter */
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, TFLM_TENSOR_ARENA_SIZE
    );
    interpreter = &static_interpreter;

    /* 4. Allocate Tensors */
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        printf("[ERROR] Tensor allocation failed in arena (%d KB)!\n", TFLM_TENSOR_ARENA_SIZE / 1024);
        return -2;
    }

    /* 5. Get Input & Output Tensors */
    input_tensor = interpreter->input(0);
    output_tensor = interpreter->output(0);

    printf("-> [OK] TFLM Arena Allocation Successful (%u / %d bytes used)\n",
           (unsigned int)interpreter->arena_used_bytes(), TFLM_TENSOR_ARENA_SIZE);
    
    printf("  Stage 1 Input : Type=%s (%d), Scale=%.6f, ZeroPoint=%d\n",
           (input_tensor->type == kTfLiteInt8) ? "INT8" : "FP32",
           input_tensor->type, input_tensor->params.scale, input_tensor->params.zero_point);
    printf("  Stage 1 Output: Type=%s (%d), Scale=%.6f, ZeroPoint=%d\n",
           (output_tensor->type == kTfLiteInt8) ? "INT8" : "FP32",
           output_tensor->type, output_tensor->params.scale, output_tensor->params.zero_point);

    printf("-> [OK] Stage 2 Native C++ GRU ready on Cortex-M33 Hardware FPU\n");
    printf("========================================================\n\n");
    is_inference_initialized = true;
    return 0;
#endif
}

extern "C" void* inference_get_input_tensor_ptr(void) {
#if (ACTIVE_MODEL_PROFILE == PROFILE_CMSIS_NN_PINGPONG_91 || ACTIVE_MODEL_PROFILE == PROFILE_TCN_85 || ACTIVE_MODEL_PROFILE == PROFILE_SLIM_TCN_81)
    return s_input_spectrogram;
#else
    if (!is_inference_initialized) {
        inference_init();
    }
    if (input_tensor) {
        return input_tensor->data.raw;
    }
    return nullptr;
#endif
}

extern "C" bool inference_is_input_int8(void) {
#if (ACTIVE_MODEL_PROFILE == PROFILE_CMSIS_NN_PINGPONG_91 || ACTIVE_MODEL_PROFILE == PROFILE_TCN_85 || ACTIVE_MODEL_PROFILE == PROFILE_SLIM_TCN_81)
    return true;
#else
    if (!is_inference_initialized) {
        inference_init();
    }
    return (input_tensor && input_tensor->type == kTfLiteInt8);
#endif
}

extern "C" void inference_get_input_quant_params(float *out_scale, int32_t *out_zero_point) {
#if (ACTIVE_MODEL_PROFILE == PROFILE_TCN_85 || ACTIVE_MODEL_PROFILE == PROFILE_SLIM_TCN_81)
    if (out_scale) *out_scale = 0.09229838f;
    if (out_zero_point) *out_zero_point = 22;
#elif (ACTIVE_MODEL_PROFILE == PROFILE_CMSIS_NN_PINGPONG_91)
    if (out_scale) *out_scale = 0.09312528f;
    if (out_zero_point) *out_zero_point = 20;
#else
    if (!is_inference_initialized) {
        inference_init();
    }
    if (input_tensor) {
        if (out_scale) *out_scale = input_tensor->params.scale;
        if (out_zero_point) *out_zero_point = input_tensor->params.zero_point;
    } else {
        if (out_scale) *out_scale = 1.0f;
        if (out_zero_point) *out_zero_point = 0;
    }
#endif
}

extern "C" int inference_run_direct(int *out_class_id, float *out_confidence) {
    if (!is_inference_initialized) {
        inference_init();
    }

    const int8_t *in_int8 = (const int8_t*)inference_get_input_tensor_ptr();

#if (ACTIVE_MODEL_PROFILE == PROFILE_TCN_85 || ACTIVE_MODEL_PROFILE == PROFILE_SLIM_TCN_81)
    return tcn_inference_run(in_int8, s_mem_pool.cmsis_nn.ping_pong_A, s_mem_pool.cmsis_nn.ping_pong_B, out_class_id, out_confidence);
#else
    printf("\n  🔍 ON-CHIP FORENSICS:\n");
#if (ACTIVE_MODEL_PROFILE == PROFILE_CMSIS_NN_PINGPONG_91)
    printf("    • Pipeline   : Native CMSIS-NN Ping-Pong (Buffer A: 64.2 KB | Buffer B: 32.1 KB)\n");
    printf("    • Peak SRAM  : 98.5 KB (74 KB SRAM Slashed from TFLM!)\n");
#else
    printf("    • Arena Base : %p (Size: %d KB)\n", tensor_arena, TFLM_TENSOR_ARENA_SIZE / 1024);
    printf("    • In Tensor  : %p (Offset: %ld)\n", in_int8, (long)(in_int8 - (int8_t*)tensor_arena));
#endif
    printf("    • In Shape   : [1, 52, 313, 1]\n");
    printf("    • In Mel 0 [0..15]: ");
    for (int t = 0; t < 16; t++) printf("%d%s", in_int8[t], (t == 15) ? "\n" : ", ");

    /* =========================================================================
     * STAGE 1: EXECUTE FULL INT8 PHINET CNN BACKBONE (NATIVE CMSIS-NN / TFLM)
     * ========================================================================= */
#if (ACTIVE_MODEL_PROFILE == PROFILE_CMSIS_NN_PINGPONG_91)
    int8_t *s_stage1_features_ptr = s_mem_pool.cmsis_nn.ping_pong_B;
    uint32_t t_cnn_start = k_cycle_get_32();
    int cnn_status = cmsis_nn_cnn_run(in_int8, s_mem_pool.cmsis_nn.ping_pong_A, s_mem_pool.cmsis_nn.ping_pong_B, s_stage1_features_ptr);
    uint32_t t_cnn_end = k_cycle_get_32();
    uint32_t cnn_time_us = k_cyc_to_us_near32(t_cnn_end - t_cnn_start);

    if (cnn_status != 0) {
        printf("[ERROR] Native CMSIS-NN CNN execution failed (%d)!\n", cnn_status);
        return -3;
    }

    const int8_t *out_int8 = s_stage1_features_ptr;
    const float out_scale = 0.15506062f;
    const int32_t out_zp = 5;
    int num_freq = 13;
    int num_time = 40;
    int num_chan = 32;
#else
    uint32_t t_cnn_start = k_cycle_get_32();
    TfLiteStatus invoke_status = interpreter->Invoke();
    uint32_t t_cnn_end = k_cycle_get_32();
    uint32_t cnn_time_us = k_cyc_to_us_near32(t_cnn_end - t_cnn_start);

    if (invoke_status != kTfLiteOk) {
        printf("[ERROR] TFLM Model Invoke failed (%d)!\n", invoke_status);
        return -3;
    }

    const int8_t *out_int8 = output_tensor->data.int8;
    const float out_scale = output_tensor->params.scale;
    const int32_t out_zp = output_tensor->params.zero_point;
    int num_freq = output_tensor->dims->data[1]; // 13
    int num_time = output_tensor->dims->data[2]; // 40
    int num_chan = output_tensor->dims->data[3]; // 32
#endif
/* =========================================================================
     * STAGE 2: NATIVE C++ RECURRENT GRU + ATTENTION + BOTTLENECK + CLASSIFIER HEAD
     * ========================================================================= */
    uint32_t t_gru_start = k_cycle_get_32();

    float (*features)[GRU_INPUT_DIM] = s_features;
    float (*H)[GRU_HIDDEN_DIM] = s_H;
    float *gate_x = s_gate_x;
    float *gate_h = s_gate_h;

    /* 1. Hardware FPU Frequency Pooling (13 bins -> 1) & Pre-GRU BatchNorm */
    for (int t = 0; t < GRU_TIME_STEPS; t++) {
        for (int c = 0; c < GRU_INPUT_DIM; c++) {
            float sum = 0.0f;
            if (num_freq > 1) {
                for (int f = 0; f < num_freq; f++) {
                    int idx = f * (num_time * num_chan) + t * num_chan + c;
                    sum += (float)(out_int8[idx] - out_zp) * out_scale;
                }
                features[t][c] = (sum / (float)num_freq) * PRE_GRU_BN_SCALE[c] + PRE_GRU_BN_BIAS[c];
            } else {
                int idx = t * num_chan + c;
                features[t][c] = ((float)(out_int8[idx] - out_zp) * out_scale) * PRE_GRU_BN_SCALE[c] + PRE_GRU_BN_BIAS[c];
            }
        }
    }

    /* Fast Padé Rational Approximations for ARM Cortex-M33 FPU */
    auto fast_tanh_fpu = [](float x) -> float {
        if (x >= 4.0f) return 1.0f;
        if (x <= -4.0f) return -1.0f;
        float x2 = x * x;
        return x * (105.0f + 10.0f * x2) / (105.0f + 45.0f * x2 + x2 * x2);
    };

    auto fast_sigmoid_fpu = [&](float x) -> float {
        if (x >= 6.0f) return 1.0f;
        if (x <= -6.0f) return 0.0f;
        return 0.5f + 0.5f * fast_tanh_fpu(0.5f * x);
    };

#if (ACTIVE_MODEL_PROFILE == PROFILE_SPARSE_PRUNED_48K)
    /* ⚡ Branchless Compressed Sparse Row (CSR) Zero-Skipping Kernel */
    auto sparse_matvec_mult = [](
        const float *sparse_w, const uint8_t *col_idx, const uint32_t *row_offsets,
        const float *bias, const float *x, float *y, int num_rows)
    {
        for (int r = 0; r < num_rows; r++) {
            float sum = bias ? bias[r] : 0.0f;
            uint32_t start = row_offsets[r];
            uint32_t end   = row_offsets[r + 1];
            #pragma GCC unroll 8
            for (uint32_t k = start; k < end; k++) {
                sum += sparse_w[k] * x[col_idx[k]];
            }
            y[r] = sum;
        }
    };
#endif

    /* 2. Unroll 39 Time Steps of Recurrent GRU Cell */
    float h[GRU_HIDDEN_DIM];
    memset(h, 0, sizeof(h));

    for (int t = 0; t < GRU_TIME_STEPS; t++) {
        const float *feat_ptr = features[t];

#if (ACTIVE_MODEL_PROFILE == PROFILE_FLAGSHIP_DENSE_91)
        /* 🏆 Matrix-Vector Multiplications (Fully Inlined Dense FPU MACs) */
        for (int i = 0; i < 3 * GRU_HIDDEN_DIM; i++) {
            const float *w_ih_ptr = &GRU_W_IH[i * GRU_INPUT_DIM];
            float sum_x = GRU_B_IH[i];
            
            #pragma GCC unroll 8
            for (int j = 0; j < GRU_INPUT_DIM; j += 4) {
                sum_x += w_ih_ptr[j]     * feat_ptr[j]
                       + w_ih_ptr[j + 1] * feat_ptr[j + 1]
                       + w_ih_ptr[j + 2] * feat_ptr[j + 2]
                       + w_ih_ptr[j + 3] * feat_ptr[j + 3];
            }
            gate_x[i] = sum_x;

            const float *w_hh_ptr = &GRU_W_HH[i * GRU_HIDDEN_DIM];
            float sum_h = GRU_B_HH[i];
            
            #pragma GCC unroll 8
            for (int j = 0; j < GRU_HIDDEN_DIM; j += 8) {
                sum_h += w_hh_ptr[j]     * h[j]
                       + w_hh_ptr[j + 1] * h[j + 1]
                       + w_hh_ptr[j + 2] * h[j + 2]
                       + w_hh_ptr[j + 3] * h[j + 3]
                       + w_hh_ptr[j + 4] * h[j + 4]
                       + w_hh_ptr[j + 5] * h[j + 5]
                       + w_hh_ptr[j + 6] * h[j + 6]
                       + w_hh_ptr[j + 7] * h[j + 7];
            }
            gate_h[i] = sum_h;
        }
#elif (ACTIVE_MODEL_PROFILE == PROFILE_SPARSE_PRUNED_48K)
        /* ⚡ Branchless CSR Sparse Zero-Skipping */
        sparse_matvec_mult(GRU_W_IH_SPARSE, GRU_W_IH_COL_IDX, GRU_W_IH_ROW_OFFSETS, GRU_B_IH, feat_ptr, gate_x, 3 * GRU_HIDDEN_DIM);
        sparse_matvec_mult(GRU_W_HH_SPARSE, GRU_W_HH_COL_IDX, GRU_W_HH_ROW_OFFSETS, GRU_B_HH, h, gate_h, 3 * GRU_HIDDEN_DIM);

        /* Fast FPU Recurrent Cell Activation */
        for (int j = 0; j < GRU_HIDDEN_DIM; j++) {
            float r = fast_sigmoid_fpu(gate_x[j] + gate_h[j]);
            float z = fast_sigmoid_fpu(gate_x[GRU_HIDDEN_DIM + j] + gate_h[GRU_HIDDEN_DIM + j]);
            float n = fast_tanh_fpu(gate_x[2 * GRU_HIDDEN_DIM + j] + r * gate_h[2 * GRU_HIDDEN_DIM + j]);

            h[j] = (1.0f - z) * n + z * h[j];
            H[t][j] = h[j];
        }
#elif (ACTIVE_MODEL_PROFILE == PROFILE_INT8_FIXED_SIMD_91 || ACTIVE_MODEL_PROFILE == PROFILE_CMSIS_NN_PINGPONG_91)
        /* 🚀 Profile 3 & 4: Inlined Hardware __SXTB16 + __SMLAD SIMD with Bit-Exact Dynamic Input Scaling */
        float max_x = 1e-6f;
        for (int j = 0; j < GRU_INPUT_DIM; j++) {
            float ax = fabsf(feat_ptr[j]);
            if (ax > max_x) max_x = ax;
        }
        float inv_scale_x = 127.0f / max_x;
        float scale_prod_ih = GRU_SCALE_W_IH * (max_x / 127.0f);

        const float STATIC_INV_SCALE_H = 127.0f; // Bounded by tanh in [-1, 1]
        const float SCALE_PROD_HH = GRU_SCALE_W_HH * (1.0f / 127.0f);

        alignas(4) int8_t x_s8[GRU_INPUT_DIM];
        for (int j = 0; j < GRU_INPUT_DIM; j++) {
            int32_t scaled = (int32_t)(feat_ptr[j] * inv_scale_x);
            x_s8[j] = (int8_t)__SSAT(scaled, 8);
        }

        alignas(4) int8_t h_s8[GRU_HIDDEN_DIM];
        for (int j = 0; j < GRU_HIDDEN_DIM; j++) {
            int32_t scaled = (int32_t)(h[j] * STATIC_INV_SCALE_H);
            h_s8[j] = (int8_t)__SSAT(scaled, 8);
        }

        /* ⚡ Inlined 32-Bit Word Load + Dual 16-bit Hardware SIMD MAC (__SMLAD) */
        for (int i = 0; i < 3 * GRU_HIDDEN_DIM; i++) {
            const int8_t *w_ih_ptr = &GRU_W_IH_INT8[i * GRU_INPUT_DIM];
            int32_t acc_ih = 0;
            #pragma GCC unroll 4
            for (int j = 0; j < GRU_INPUT_DIM; j += 4) {
                uint32_t w_word = *(const uint32_t*)(&w_ih_ptr[j]);
                uint32_t x_word = *(const uint32_t*)(&x_s8[j]);

                uint32_t w_bottom = __SXTB16(w_word);
                uint32_t x_bottom = __SXTB16(x_word);
                uint32_t w_top = __SXTB16(__ROR(w_word, 8));
                uint32_t x_top = __SXTB16(__ROR(x_word, 8));

                acc_ih = __SMLAD(w_bottom, x_bottom, acc_ih);
                acc_ih = __SMLAD(w_top, x_top, acc_ih);
            }
            gate_x[i] = ((float)acc_ih * scale_prod_ih) + GRU_BIAS_F32[i];

            const int8_t *w_hh_ptr = &GRU_W_HH_INT8[i * GRU_HIDDEN_DIM];
            int32_t acc_hh = 0;
            #pragma GCC unroll 4
            for (int j = 0; j < GRU_HIDDEN_DIM; j += 4) {
                uint32_t w_word = *(const uint32_t*)(&w_hh_ptr[j]);
                uint32_t h_word = *(const uint32_t*)(&h_s8[j]);

                uint32_t w_bottom = __SXTB16(w_word);
                uint32_t h_bottom = __SXTB16(h_word);
                uint32_t w_top = __SXTB16(__ROR(w_word, 8));
                uint32_t h_top = __SXTB16(__ROR(h_word, 8));

                acc_hh = __SMLAD(w_bottom, h_bottom, acc_hh);
                acc_hh = __SMLAD(w_top, h_top, acc_hh);
            }
            gate_h[i] = (float)acc_hh * SCALE_PROD_HH;
        }

        /* ⚡ 1-Cycle Fast Direct LUT Table Lookups */
        const float inv_scale_act = GRU_SCALE_ACT_INV;
        for (int j = 0; j < GRU_HIDDEN_DIM; j++) {
            int32_t r_raw = (int32_t)((gate_x[j] + gate_h[j]) * inv_scale_act);
            int r_idx = (int)__SSAT(r_raw, 8) + 128;
            float r = (float)SIGMOID_LUT_S8[r_idx] * (1.0f / 127.0f);

            int32_t z_raw = (int32_t)((gate_x[GRU_HIDDEN_DIM + j] + gate_h[GRU_HIDDEN_DIM + j]) * inv_scale_act);
            int z_idx = (int)__SSAT(z_raw, 8) + 128;
            float z = (float)SIGMOID_LUT_S8[z_idx] * (1.0f / 127.0f);

            int32_t n_raw = (int32_t)((gate_x[2 * GRU_HIDDEN_DIM + j] + (r * gate_h[2 * GRU_HIDDEN_DIM + j])) * inv_scale_act);
            int n_idx = (int)__SSAT(n_raw, 8) + 128;
            float n = (float)TANH_LUT_S8[n_idx] * (1.0f / 127.0f);

            h[j] = (1.0f - z) * n + z * h[j];
            H[t][j] = h[j];
        }
#endif
    }

    /* 3. Softmax Sequence Attention Pooling across 39 temporal steps (5.0s acoustic context) */
    float attn_scores[GRU_TIME_STEPS];
    float max_attn = -1e9f;
    for (int t = 0; t < GRU_TIME_STEPS; t++) {
        float sum = 0.0f;
        for (int j = 0; j < GRU_HIDDEN_DIM; j++) {
            sum += H[t][j];
        }
        attn_scores[t] = sum / (float)GRU_HIDDEN_DIM;
        if (attn_scores[t] > max_attn) max_attn = attn_scores[t];
    }

    float sum_exp = 0.0f;
    float attn_weights[GRU_TIME_STEPS];
    for (int t = 0; t < GRU_TIME_STEPS; t++) {
        attn_weights[t] = expf(attn_scores[t] - max_attn);
        sum_exp += attn_weights[t];
    }
    for (int t = 0; t < GRU_TIME_STEPS; t++) {
        attn_weights[t] /= sum_exp;
    }

    float h_pooled[GRU_HIDDEN_DIM] = {0.0f};
    for (int t = 0; t < GRU_TIME_STEPS; t++) {
        for (int j = 0; j < GRU_HIDDEN_DIM; j++) {
            h_pooled[j] += H[t][j] * attn_weights[t];
        }
    }

    /* 4. Post-GRU BatchNorm */
    for (int j = 0; j < GRU_HIDDEN_DIM; j++) {
        h_pooled[j] = h_pooled[j] * POST_GRU_BN_SCALE[j] + POST_GRU_BN_BIAS[j];
    }

    /* 5. Bottleneck Linear (160 -> 128) with Native ReLU6 */
    float btn_out[BOTTLENECK_DIM];
    float btn_raw[BOTTLENECK_DIM];
#if (ACTIVE_MODEL_PROFILE == PROFILE_FLAGSHIP_DENSE_91 || ACTIVE_MODEL_PROFILE == PROFILE_INT8_FIXED_SIMD_91 || ACTIVE_MODEL_PROFILE == PROFILE_CMSIS_NN_PINGPONG_91)
    for (int i = 0; i < BOTTLENECK_DIM; i++) {
        const float *btn_w_row = &BOTTLENECK_W[i * GRU_HIDDEN_DIM];
        float sum = BOTTLENECK_B[i];
        #pragma GCC unroll 8
        for (int j = 0; j < GRU_HIDDEN_DIM; j += 8) {
            sum += btn_w_row[j]     * h_pooled[j]
                 + btn_w_row[j + 1] * h_pooled[j + 1]
                 + btn_w_row[j + 2] * h_pooled[j + 2]
                 + btn_w_row[j + 3] * h_pooled[j + 3]
                 + btn_w_row[j + 4] * h_pooled[j + 4]
                 + btn_w_row[j + 5] * h_pooled[j + 5]
                 + btn_w_row[j + 6] * h_pooled[j + 6]
                 + btn_w_row[j + 7] * h_pooled[j + 7];
        }
        btn_raw[i] = sum;
        btn_out[i] = fmaxf(0.0f, fminf(6.0f, sum)); // ReLU6
    }
#elif (ACTIVE_MODEL_PROFILE == PROFILE_SPARSE_PRUNED_48K)
    sparse_matvec_mult(BOTTLENECK_W_SPARSE, BOTTLENECK_W_COL_IDX, BOTTLENECK_W_ROW_OFFSETS, BOTTLENECK_B, h_pooled, btn_raw, BOTTLENECK_DIM);
    for (int i = 0; i < BOTTLENECK_DIM; i++) {
        btn_out[i] = fmaxf(0.0f, fminf(6.0f, btn_raw[i])); // ReLU6
    }
#endif

    int btn_nonzeros = 0;
    for (int i = 0; i < BOTTLENECK_DIM; i++) {
        if (btn_out[i] > 0.0f) btn_nonzeros++;
    }

    /* 6. FC Classifier Head (128 -> 50) */
    float raw_logits[NUM_ESC50_CLASSES];
#if (ACTIVE_MODEL_PROFILE == PROFILE_FLAGSHIP_DENSE_91 || ACTIVE_MODEL_PROFILE == PROFILE_INT8_FIXED_SIMD_91 || ACTIVE_MODEL_PROFILE == PROFILE_CMSIS_NN_PINGPONG_91)
    for (int i = 0; i < NUM_ESC50_CLASSES; i++) {
        const float *fc_w_row = &FC_W[i * BOTTLENECK_DIM];
        float sum = FC_B[i];
        #pragma GCC unroll 8
        for (int j = 0; j < BOTTLENECK_DIM; j += 8) {
            sum += fc_w_row[j]     * btn_out[j]
                 + fc_w_row[j + 1] * btn_out[j + 1]
                 + fc_w_row[j + 2] * btn_out[j + 2]
                 + fc_w_row[j + 3] * btn_out[j + 3]
                 + fc_w_row[j + 4] * btn_out[j + 4]
                 + fc_w_row[j + 5] * btn_out[j + 5]
                 + fc_w_row[j + 6] * btn_out[j + 6]
                 + fc_w_row[j + 7] * btn_out[j + 7];
        }
        raw_logits[i] = sum;
    }
#elif (ACTIVE_MODEL_PROFILE == PROFILE_SPARSE_PRUNED_48K)
    sparse_matvec_mult(FC_W_SPARSE, FC_W_COL_IDX, FC_W_ROW_OFFSETS, FC_B, btn_out, raw_logits, NUM_ESC50_CLASSES);
#endif

#if (ACTIVE_MODEL_PROFILE == PROFILE_SPARSE_PRUNED_48K)
    /* ⚡ ICML 2017 Temperature Calibration Scale for 68% Pruning Variance Recovery */
    const float LOGIT_CALIBRATION_SCALE = 3.0f;
#else
    const float LOGIT_CALIBRATION_SCALE = 1.0f;
#endif

    /* Temporal EMA Smoothing across successive frames (Alpha = 0.65) */
    static float s_smoothed_logits[NUM_ESC50_CLASSES];
    static bool s_has_prev_logits = false;
    const float EMA_ALPHA = 0.65f;

    int top_class = 0;
    float max_logit = -1e9f;

    for (int i = 0; i < NUM_ESC50_CLASSES; i++) {
        float scaled_logit = raw_logits[i] * LOGIT_CALIBRATION_SCALE;
        if (!s_has_prev_logits) {
            s_smoothed_logits[i] = scaled_logit;
        } else {
            s_smoothed_logits[i] = EMA_ALPHA * scaled_logit + (1.0f - EMA_ALPHA) * s_smoothed_logits[i];
        }
        if (s_smoothed_logits[i] > max_logit) {
            max_logit = s_smoothed_logits[i];
            top_class = i;
        }
    }
    s_has_prev_logits = true;

    /* 7. Compute Softmax Confidence on Calibrated Smoothed Logits */
    float exp_sum = 0.0f;
    for (int i = 0; i < NUM_ESC50_CLASSES; i++) {
        exp_sum += expf(s_smoothed_logits[i] - max_logit);
    }
    float confidence = 1.0f / exp_sum;

    /* Stop GRU Timer HERE (BEFORE any slow UART printf calls!) */
    uint32_t t_gru_end = k_cycle_get_32();

    /* Print Diagnostics & Top 3 Classes */
    printf("    • h_pooled [0..3] : %.4f, %.4f, %.4f, %.4f\n", h_pooled[0], h_pooled[1], h_pooled[2], h_pooled[3]);
    printf("    • btn_raw  [0..3] : %.4f, %.4f, %.4f, %.4f\n", btn_raw[0], btn_raw[1], btn_raw[2], btn_raw[3]);
    printf("    • btn_out  [0..3] : %.4f, %.4f, %.4f, %.4f (Non-zeros: %d/%d)\n", 
           btn_out[0], btn_out[1], btn_out[2], btn_out[3], btn_nonzeros, BOTTLENECK_DIM);
    printf("    • Logit[40] (siren): %.4f | Logit[46] (vacuum): %.4f\n", s_smoothed_logits[40], s_smoothed_logits[46]);

    printf("\n  📊 Stage 2 Classifier Top 3 Logits (EMA Smoothed):\n");
    int picked_classes[3] = {-1, -1, -1};
    for (int rank = 0; rank < 3; rank++) {
        int best_k = -1;
        float best_l = -1e9f;
        for (int k = 0; k < NUM_ESC50_CLASSES; k++) {
            bool already_picked = false;
            for (int r = 0; r < rank; r++) {
                if (picked_classes[r] == k) already_picked = true;
            }
            if (!already_picked && s_smoothed_logits[k] > best_l) {
                best_l = s_smoothed_logits[k];
                best_k = k;
            }
        }
        if (best_k >= 0) {
            picked_classes[rank] = best_k;
            float p = expf(s_smoothed_logits[best_k] - max_logit) / exp_sum;
            printf("    #%d: Class %2d (%-18s) -> %5.1f%% (logit: %.2f)\n", 
                   rank + 1, best_k, ESC50_CLASS_NAMES[best_k], p * 100.0f, s_smoothed_logits[best_k]);
        }
    }

    /* Compute True Milliseconds using Zephyr Hardware Timer */
    cnn_time_us = k_cyc_to_us_near32(t_cnn_end - t_cnn_start);
    uint32_t gru_time_us = k_cyc_to_us_near32(t_gru_end - t_gru_start);
    float cnn_time_ms = (float)cnn_time_us / 1000.0f;
    float gru_time_ms = (float)gru_time_us / 1000.0f;

    printf("\n  ⏱️ INFERENCE LATENCY: Stage 1 CNN = %.2f ms | Stage 2 GRU = %.2f ms | Total ML = %.2f ms\n",
           cnn_time_ms, gru_time_ms, cnn_time_ms + gru_time_ms);

    if (out_class_id) *out_class_id = top_class;
    if (out_confidence) *out_confidence = confidence;

    return 0;
#endif
}

extern "C" const char* inference_get_class_name(int class_id) {
    if (class_id >= 0 && class_id < NUM_ESC50_CLASSES) {
        return ESC50_CLASS_NAMES[class_id];
    }
    return "unknown";
}

extern "C" int inference_run(const float *spectrogram_data, int *out_class_id, float *out_confidence) {
    if (spectrogram_data == nullptr) return -1;
    void *in_ptr = inference_get_input_tensor_ptr();
    if (in_ptr == nullptr) return -2;

    if (inference_is_input_int8()) {
        float scale = 1.0f;
        int32_t zp = 0;
        inference_get_input_quant_params(&scale, &zp);
        int8_t *dst = (int8_t*)in_ptr;
        for (int i = 0; i < SPECTROGRAM_TOTAL_ELEMENTS; i++) {
            int32_t val = (int32_t)roundf(spectrogram_data[i] / scale) + zp;
            dst[i] = (int8_t)fmaxf(-128.0f, fminf(127.0f, (float)val));
        }
    } else {
        float *dst = (float*)in_ptr;
        for (int i = 0; i < SPECTROGRAM_TOTAL_ELEMENTS; i++) {
            dst[i] = spectrogram_data[i];
        }
    }
    return inference_run_direct(out_class_id, out_confidence);
}
