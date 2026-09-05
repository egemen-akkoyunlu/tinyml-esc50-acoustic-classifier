// ==============================================================================
// ⚡ 1D DILATED TC-RESNET + FREQUENCY-FOLDING NATIVE C++ INFERENCE ENGINE
// Silicon Labs EFR32MG24 (ARM Cortex-M33) & Espressif ESP32-S3 (Xtensa PIE SIMD)
// 100% Zero-Heap, Zero-BSS Ping-Pong Buffer Memory Execution
// ==============================================================================

#ifndef TCN_INFERENCE_ENGINE_H_
#define TCN_INFERENCE_ENGINE_H_

#include <stdint.h>
#include <stdbool.h>
#include "config.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initializes the TCN engine and sets up internal quant parameters.
 * @return 0 on success.
 */
int tcn_inference_init(void);

/**
 * @brief Runs full end-to-end INT8 inference on a pre-computed Log-Mel Spectrogram using shared ping-pong RAM.
 * @param spectrogram_int8 Pointer to input [52 Freq x 313 Time] INT8 array.
 * @param ping_pong_A Pointer to shared scratchpad memory buffer A (64 KB).
 * @param ping_pong_B Pointer to shared scratchpad memory buffer B (32 KB).
 * @param out_class_id Pointer to store top-1 predicted class index (0..49).
 * @param out_confidence Pointer to store top-1 softmax probability (0.0f..1.0f).
 * @return 0 on success, negative error code on failure.
 */
int tcn_inference_run(const int8_t *spectrogram_int8, int8_t *ping_pong_A, int8_t *ping_pong_B, int *out_class_id, float *out_confidence);

/**
 * @brief Retrieves input quantization parameters (scale and zero-point) for current active TCN profile.
 */
void tcn_inference_get_quant_params(float *out_scale, int32_t *out_zero_point);

#ifdef __cplusplus
}
#endif

#endif // TCN_INFERENCE_ENGINE_H_
