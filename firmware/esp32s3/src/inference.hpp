/*
 * Copyright (c) 2026 Seeed Studio / Senior ML Engineer
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef INFERENCE_HPP
#define INFERENCE_HPP

#include "dl_model_base.hpp"
#include "dl_tensor_base.hpp"
#include "dl_fbank.hpp"
#include <zephyr/kernel.h>
#include <cstdint>

class KWSInference {
public:
    KWSInference();
    ~KWSInference();

    /**
     * @brief Initialize neural network model, Mel filterbank, and PSRAM sliding window buffer.
     * @return true on success, false on failure.
     */
    bool init();

    /**
     * @brief Process an audio frame (256 samples), extract Fbank features, update sliding window,
     * and run KWS inference every INFERENCE_STRIDE frames.
     * @param audio_window Array of 256 audio samples (int16_t).
     * @param frame_counter Current frame counter.
     * @param log_details Whether detailed frame logs should be printed.
     */
    void process_audio_window(const int16_t *audio_window, int frame_counter, bool log_details);

    /**
     * @brief Execute neural network inference on Core 1 in background thread.
     */
    void run_inference_now();

    struct k_sem inf_sem;

private:
    void compute_softmax(const float *logits, float *probs, int len);

    dl::Model *model;
    dl::audio::Fbank *fbank;
    dl::TensorBase *input_tensor;
    dl::TensorBase *output_tensor;
    float *sliding_window_buf;
    float *normalized_buf;
    int cooldown_counter;
    int last_detected_class;
    int consecutive_count;
    int dropout_count;

    // Timing diagnostic variables
    uint32_t last_ap_start;
    uint32_t last_ap_end;
    double last_ap_time_ms;

    uint32_t last_inf_start;
    uint32_t last_inf_end;
    double last_inf_time_ms;
};

#endif // INFERENCE_HPP
