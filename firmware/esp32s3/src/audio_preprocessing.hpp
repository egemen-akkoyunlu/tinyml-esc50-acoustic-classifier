/*
 * Copyright (c) 2026 Seeed Studio / Senior ML Engineer
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef AUDIO_PREPROCESSING_HPP
#define AUDIO_PREPROCESSING_HPP

#include <cstdint>

class AudioProcessor {
public:
    AudioProcessor();

    /**
     * @brief Process a raw stereo sample pair from PDM mic: applies DC-blocking HPF,
     * Soft Limiter, updates diagnostics, dynamically selects active channel,
     * and returns the filtered sample of the active channel.
     * @param raw_l Raw left channel sample
     * @param raw_r Raw right channel sample
     * @return Filtered active channel sample
     */
    int16_t process_sample_pair(int16_t raw_l, int16_t raw_r);

    /**
     * @brief Check if diagnostic reporting threshold is reached and output diagnosis report if needed.
     */
    void check_and_report_diagnostics();

    /**
     * @brief Check whether the right channel (CH1) is currently active.
     */
    bool is_using_right_channel() const { return use_right_channel; }

private:
    void reset_diagnostics();

    bool use_right_channel;
    bool is_crossfading;
    int crossfade_count;
    bool target_right_channel;

    // Filter states
    float hpf_x1_l, hpf_y1_l;
    float hpf_x1_r, hpf_y1_r;
    float lpf1_x1_l, lpf1_x2_l, lpf1_y1_l, lpf1_y2_l;
    float lpf1_x1_r, lpf1_x2_r, lpf1_y1_r, lpf1_y2_r;
    float lpf2_x1_l, lpf2_x2_l, lpf2_y1_l, lpf2_y2_l;
    float lpf2_x1_r, lpf2_x2_r, lpf2_y1_r, lpf2_y2_r;

    // Diagnostics accumulators
    uint64_t diag_sum_sq_l, diag_sum_sq_r;
    uint64_t raw_sum_sq_l, raw_sum_sq_r;
    int64_t diag_sum_l, diag_sum_r;
    int16_t diag_min_l, diag_max_l;
    int16_t diag_min_r, diag_max_r;
    int16_t raw_min_l, raw_max_l;
    int16_t raw_min_r, raw_max_r;
    int diag_zcr_l, diag_zcr_r;
    int diag_clip_l, diag_clip_r;
    int16_t prev_sample_l, prev_sample_r;
    int diag_sample_count;
};

#endif // AUDIO_PREPROCESSING_HPP
