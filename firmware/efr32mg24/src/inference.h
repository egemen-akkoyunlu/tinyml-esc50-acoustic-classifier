#ifndef INFERENCE_H
#define INFERENCE_H

#include <stdint.h>
#include <stdbool.h>
#include "config.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initializes TensorFlow Lite Micro interpreter, memory arena, and model flatbuffer.
 * @return 0 on success, negative error code on failure.
 */
int inference_init(void);

/**
 * @brief Gets pointer to the model input tensor buffer directly (0 extra RAM overhead).
 * @return Pointer to input tensor data array.
 */
void* inference_get_input_tensor_ptr(void);

/**
 * @brief Checks whether the loaded TFLM model expects INT8 input tensor (true) or FP32 (false).
 * @return true if input tensor is kTfLiteInt8, false if kTfLiteFloat32.
 */
bool inference_is_input_int8(void);

/**
 * @brief Retrieves model input tensor scale factor and zero-point for direct INT8 quantization.
 * @param out_scale Pointer to store scale factor.
 * @param out_zero_point Pointer to store zero-point offset.
 */
void inference_get_input_quant_params(float *out_scale, int32_t *out_zero_point);

/**
 * @brief Runs model inference after input tensor has been populated.
 * @param out_class_id Pointer to store top predicted ESC-50 class ID (0 to 49).
 * @param out_confidence Pointer to store top prediction confidence score.
 * @return 0 on success, negative error code on failure.
 */
int inference_run_direct(int *out_class_id, float *out_confidence);

/**
 * @brief Legacy wrapper running model inference on Log-Mel Spectrogram input.
 */
int inference_run(const float *spectrogram_data, int *out_class_id, float *out_confidence);

/**
 * @brief Returns human-readable class name string for a given ESC-50 class ID.
 * @param class_id Class index (0 to 49).
 * @return Pointer to class name string.
 */
const char* inference_get_class_name(int class_id);

#ifdef __cplusplus
}
#endif

#endif /* INFERENCE_H */
