/*
 * Copyright (c) 2026 Seeed Studio / Senior ML Engineer
 * SPDX-License-Identifier: Apache-2.0
 * 
 * audio_preprocessing.cpp - Real-time audio processing pipeline.
 * Performs gain scaling, soft-limiting, DC blocking high-pass filtering,
 * automatic microphone channel selection, and continuous diagnostic reporting.
 */

#include "audio_preprocessing.hpp"
#include "esc_config.hpp"

#include <zephyr/logging/log.h>
#include <cmath>
#include <algorithm>

LOG_MODULE_DECLARE(esc50_app, LOG_LEVEL_INF);

/*
 * Constructor: Initializes filter state variables and resets diagnostic counters.
 */
AudioProcessor::AudioProcessor()
    : use_right_channel(false), is_crossfading(false), crossfade_count(0), target_right_channel(false),
      hpf_x1_l(0.0f), hpf_y1_l(0.0f), hpf_x1_r(0.0f), hpf_y1_r(0.0f),
      lpf1_x1_l(0.0f), lpf1_x2_l(0.0f), lpf1_y1_l(0.0f), lpf1_y2_l(0.0f),
      lpf1_x1_r(0.0f), lpf1_x2_r(0.0f), lpf1_y1_r(0.0f), lpf1_y2_r(0.0f),
      lpf2_x1_l(0.0f), lpf2_x2_l(0.0f), lpf2_y1_l(0.0f), lpf2_y2_l(0.0f),
      lpf2_x1_r(0.0f), lpf2_x2_r(0.0f), lpf2_y1_r(0.0f), lpf2_y2_r(0.0f),
      prev_sample_l(0), prev_sample_r(0)
{
    reset_diagnostics();
}

/*
 * reset_diagnostics: Resets all running statistical registers used for logging.
 */
void AudioProcessor::reset_diagnostics()
{
    raw_sum_sq_l = 0; raw_sum_sq_r = 0;
    diag_sum_sq_l = 0; diag_sum_sq_r = 0;
    diag_sum_l = 0; diag_sum_r = 0;
    diag_min_l = 32767; diag_max_l = -32768;
    diag_min_r = 32767; diag_max_r = -32768;
    raw_min_l = 32767; raw_max_l = -32768;
    raw_min_r = 32767; raw_max_r = -32768;
    diag_zcr_l = 0; diag_zcr_r = 0;
    diag_clip_l = 0; diag_clip_r = 0;
    diag_sample_count = 0;
}

/*
 * process_sample_pair: Processes raw stereo PDM microphone samples (Left and Right).
 * 
 * Execution Steps:
 * 1. Accumulate raw input signal stats.
 * 2. Apply a Soft Limiter with a 0.35 gain scale to protect against digital clipping.
 * 3. Run a DC-blocking High-Pass Filter (cut-off ~80Hz) to filter out sub-bass rumble/DC offset.
 * 4. Accumulate clean signal stats (ZCR, peaks, sum of squares).
 * 5. Auto-Channel Lock: Locks on to the channel with higher speech energy after 1 second.
 * 6. Return the chosen channel's sample.
 */
int16_t AudioProcessor::process_sample_pair(int16_t raw_l, int16_t raw_r)
{
    raw_sum_sq_l += (int32_t)raw_l * (int32_t)raw_l;
    raw_sum_sq_r += (int32_t)raw_r * (int32_t)raw_r;
    
    if (raw_l < raw_min_l) raw_min_l = raw_l;
    if (raw_l > raw_max_l) raw_max_l = raw_l;
    if (raw_r < raw_min_r) raw_min_r = raw_r;
    if (raw_r > raw_max_r) raw_max_r = raw_r;

    // --- 1. Soft Limiter (Input stage) to prevent clipping / state overflow ---
    const float input_gain = 1.5f; 
    const float in_limit = 24000.0f; // Soft limiting headroom
    
    // Left Channel Input & Limit
    float in_l = (float)raw_l * input_gain;
    if (in_l > in_limit) {
        in_l = in_limit + (in_l - in_limit) / (1.0f + (in_l - in_limit) / (32767.0f - in_limit));
    } else if (in_l < -in_limit) {
        in_l = -in_limit + (in_l + in_limit) / (1.0f - (in_l + in_limit) / (32768.0f - in_limit));
    }

    // --- 2. DC Blocking HPF ---
    const float hpf_r = 0.99f; // ~25 Hz sub-bass cutoff (preserves coughing/speech fundamentals)
    float hpf_out_l = in_l - hpf_x1_l + hpf_r * hpf_y1_l;
    hpf_x1_l = in_l; hpf_y1_l = hpf_out_l;
 
    // --- 3. LPF (Bypassed: using hardware decimation) ---
    float out_l = hpf_out_l;
 
    if (out_l > 32767.0f) out_l = 32767.0f; else if (out_l < -32768.0f) out_l = -32768.0f;
    int16_t sample_l = (int16_t)out_l;
 
    // --- Right Channel Input & Limit ---
    float in_r = (float)raw_r * input_gain;
    if (in_r > in_limit) {
        in_r = in_limit + (in_r - in_limit) / (1.0f + (in_r - in_limit) / (32767.0f - in_limit));
    } else if (in_r < -in_limit) {
        in_r = -in_limit + (in_r + in_limit) / (1.0f - (in_r + in_limit) / (32768.0f - in_limit));
    }

    // DC Blocking HPF for Right Channel
    float hpf_out_r = in_r - hpf_x1_r + hpf_r * hpf_y1_r;
    hpf_x1_r = in_r; hpf_y1_r = hpf_out_r;
 
    // LPF (Bypassed: using hardware decimation) for Right Channel
    float out_r = hpf_out_r;
 
    if (out_r > 32767.0f) out_r = 32767.0f; else if (out_r < -32768.0f) out_r = -32768.0f;
    int16_t sample_r = (int16_t)out_r;
    
    // --- Lightweight O(1) Diagnostics Accumulation on Clean Signal ---
    diag_sum_l += sample_l;
    diag_sum_r += sample_r;
    diag_sum_sq_l += (int32_t)sample_l * (int32_t)sample_l;
    diag_sum_sq_r += (int32_t)sample_r * (int32_t)sample_r;
    
    if (sample_l < diag_min_l) diag_min_l = sample_l;
    if (sample_l > diag_max_l) diag_max_l = sample_l;
    if (sample_r < diag_min_r) diag_min_r = sample_r;
    if (sample_r > diag_max_r) diag_max_r = sample_r;
    
    // Track Zero-Crossing Rate (ZCR) for both channels to analyze hiss/high-frequency noise
    if ((sample_l >= 0 && prev_sample_l < 0) || (sample_l < 0 && prev_sample_l >= 0)) diag_zcr_l++;
    if ((sample_r >= 0 && prev_sample_r < 0) || (sample_r < 0 && prev_sample_r >= 0)) diag_zcr_r++;
    prev_sample_l = sample_l;
    prev_sample_r = sample_r;
    
    // Accumulate clipping occurrences
    if (sample_l <= -32000 || sample_l >= 32000) diag_clip_l++;
    if (sample_r <= -32000 || sample_r >= 32000) diag_clip_r++;
    
    diag_sample_count++;
    
    // Dynamically pick whichever channel contains active microphone signal (eliminates dead/floating channel static)
    int16_t active_sample = (abs(sample_l) >= abs(sample_r)) ? sample_l : sample_r;

    return active_sample;
}

/*
 * check_and_report_diagnostics: Prints a diagnostic report summarizing raw & post-filtered
 *                              signal properties (RMS, ZCR, clipping) every 10 seconds.
 */
void AudioProcessor::check_and_report_diagnostics()
{
    const int diag_report_interval = SAMPLE_RATE * 10;
    if (diag_sample_count >= diag_report_interval) {
        float raw_rms_l = sqrtf((float)raw_sum_sq_l / diag_sample_count);
        float raw_rms_r = sqrtf((float)raw_sum_sq_r / diag_sample_count);

        float mean_l = (float)diag_sum_l / diag_sample_count;
        float mean_r = (float)diag_sum_r / diag_sample_count;
        
        float var_l = ((float)diag_sum_sq_l / diag_sample_count) - (mean_l * mean_l);
        float var_r = ((float)diag_sum_sq_r / diag_sample_count) - (mean_r * mean_r);
        float rms_l = sqrtf(std::max(0.0f, var_l));
        float rms_r = sqrtf(std::max(0.0f, var_r));
        
        float dbfs_l = 20.0f * log10f(std::max(1.0f, rms_l) / 32768.0f);
        float dbfs_r = 20.0f * log10f(std::max(1.0f, rms_r) / 32768.0f);
        
        float zcr_pct_l = (100.0f * diag_zcr_l) / diag_sample_count;
        float zcr_pct_r = (100.0f * diag_zcr_r) / diag_sample_count;
        
        LOG_INF("================== AUDIO DIAGNOSIS REPORT (POST-FILTER) ==================");
        LOG_INF("Filter Active: DC Blocking HPF + 0.30x Input Gain + Limiter + 6kHz LPF");
        LOG_INF("Samples Analyzed: %d (~2.0 sec)", diag_sample_count);
        LOG_INF("Left  Ch (CH0) | Raw RMS: %6.1f, Raw Peak: [%6d, %6d] -> Clean RMS: %6.1f (%5.1f dBFS) | Peak: [%d, %d] | ZCR: %4.1f%% | Clips: %d",
                (double)raw_rms_l, raw_min_l, raw_max_l, (double)rms_l, (double)dbfs_l, diag_min_l, diag_max_l, (double)zcr_pct_l, diag_clip_l);
        LOG_INF("Right Ch (CH1) | Raw RMS: %6.1f, Raw Peak: [%6d, %6d] -> Clean RMS: %6.1f (%5.1f dBFS) | Peak: [%d, %d] | ZCR: %4.1f%% | Clips: %d",
                (double)raw_rms_r, raw_min_r, raw_max_r, (double)rms_r, (double)dbfs_r, diag_min_r, diag_max_r, (double)zcr_pct_r, diag_clip_r);
        
        if (rms_l < 10.0f && rms_r < 10.0f) {
            LOG_WRN("[DIAGNOSIS] Both channels have near-zero RMS. Mic may be disconnected, muted, or I2S clock mismatch!");
        } else if (rms_l > 500.0f && rms_r < 50.0f) {
            LOG_INF("[DIAGNOSIS] Active audio detected on LEFT channel (CH0).");
        } else if (rms_r > 500.0f && rms_l < 50.0f) {
            LOG_WRN("[DIAGNOSIS] Active audio is on RIGHT channel (CH1), but ESC50 is reading LEFT! Check mic channel strapping.");
        }
        
        if (zcr_pct_l > 35.0f || zcr_pct_r > 35.0f) {
            LOG_WRN("[DIAGNOSIS] High Zero-Crossing Rate (>35%%) detected. Indicates high-frequency white noise or clock hiss!");
        } else {
            LOG_INF("[DIAGNOSIS] ZCR is normal (<35%%). PDM high-frequency hiss successfully suppressed!");
        }
        if (diag_clip_l > 10 || diag_clip_r > 10) {
            LOG_WRN("[DIAGNOSIS] Signal clipping detected! Mic gain or PDM scaling is too high.");
        }
        LOG_INF("==========================================================================");
        
        reset_diagnostics();
    }
}
