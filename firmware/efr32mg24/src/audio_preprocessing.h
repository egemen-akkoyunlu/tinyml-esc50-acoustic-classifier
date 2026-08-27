#ifndef AUDIO_PREPROCESSING_H
#define AUDIO_PREPROCESSING_H

#include <stdint.h>
#include <stdbool.h>
#include "config.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initializes audio preprocessing module (Mel filterbank weights, window functions).
 */
void audio_preprocessing_init(void);

/**
 * @brief Computes DC offset average from raw 16-bit PCM audio samples.
 * @param raw_pcm Pointer to raw int16 PCM audio samples.
 * @param num_samples Number of samples in raw_pcm.
 * @return Calculated DC offset value.
 */
int32_t audio_preprocess_compute_dc(const int16_t *raw_pcm, int num_samples);

/**
 * @brief Performs Log-Mel extraction and direct quantization into the model input tensor (0 extra RAM).
 * @param raw_pcm Pointer to raw int16 PCM audio samples.
 * @param num_samples Number of samples.
 * @param dc_offset Pre-computed DC offset.
 * @param scale TFLM input tensor scale factor.
 * @param zero_point TFLM input tensor zero point.
 * @param is_int8 True if model input is INT8, false if FP32.
 * @param out_tensor_ptr Pointer directly to TFLM input tensor buffer.
 */
void audio_preprocess_extract_melspectrogram_direct(
    const int16_t *raw_pcm, 
    int num_samples, 
    int32_t dc_offset, 
    float scale, 
    int32_t zero_point, 
    bool is_int8, 
    void *out_tensor_ptr
);

/**
 * @brief Full end-to-end preprocessing pipeline directly writing to TFLM input tensor (0 extra RAM).
 * @param raw_pcm Pointer to raw int16 PCM audio buffer.
 * @param num_samples Number of samples in raw_pcm.
 * @param scale TFLM input tensor scale factor.
 * @param zero_point TFLM input tensor zero point.
 * @param is_int8 True if model input is INT8, false if FP32.
 * @param out_tensor_ptr Pointer directly to TFLM input tensor buffer.
 */
void audio_preprocess_run_direct(
    const int16_t *raw_pcm, 
    int num_samples, 
    float scale, 
    int32_t zero_point, 
    bool is_int8, 
    void *out_tensor_ptr
);

/**
 * @brief Shifts existing spectrogram frames in TFLM input tensor left and appends newly computed Mel frames (0 extra RAM).
 */
void audio_preprocess_append_chunk_direct(
    const int16_t *new_pcm_chunk, 
    int num_samples, 
    float scale, 
    int32_t zero_point, 
    bool is_int8, 
    void *out_tensor_ptr
);

/**
 * @brief Computes Mel frames for an active audio chunk and tiles it across the full 5.0s (313 frames) TFLM input tensor.
 * Eliminates temporal dilution during real-time streaming (0 extra RAM).
 */
void audio_preprocess_tile_chunk_direct(
    const int16_t *new_pcm_chunk, 
    int num_samples, 
    float scale, 
    int32_t zero_point, 
    bool is_int8, 
    void *out_tensor_ptr
);

/**
 * @brief Quantizes floating-point Log-Mel Spectrogram features into model INT8 tensor format.
 */
void audio_preprocess_quantize_int8(const float *in_spectrogram, int8_t *out_int8_tensor, float scale, int32_t zero_point);

#ifdef __cplusplus
}
#endif

#endif /* AUDIO_PREPROCESSING_H */
