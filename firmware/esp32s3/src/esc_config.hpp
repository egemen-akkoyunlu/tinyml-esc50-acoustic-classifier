/*
 * Copyright (c) 2026 Seeed Studio / Senior ML Engineer
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ESC_CONFIG_HPP
#define ESC_CONFIG_HPP

#ifdef __cplusplus
#include <cstddef>
#include <cstdint>
#else
#include <stddef.h>
#include <stdint.h>
#endif

#define SAMPLE_RATE 16000
#define FRAME_LEN_SAMPLES 512
#define FRAME_SHIFT_SAMPLES 256
#define NUM_MEL_BINS 52
#define TIME_FRAMES 313

#define I2S_BLOCK_SIZE 1024
#define SAMPLES_PER_BLOCK 128

#define NUM_CLASSES 50
#define CONFIDENCE_THRESHOLD 0.30f          // ESC-50 confidence threshold (40% for 50-class split)
#define CONFIDENCE_MARGIN 0.05f             // Minimum margin over second best class (10%)
#define COOLDOWN_FRAMES 0                  // Cooldown frames after detection
#define INFERENCE_STRIDE 10
#define VOICE_ACTIVITY_RMS_THRESHOLD 100.0f   // Energy squelch: skip model run if RMS < threshold

// ============================================================================
// MULTI-DEVICE CONFIGURATION FOR LATE FUSION
// Set DEVICE_ID to 1 for Device 1 (XIAO_SENSE_BLE_1)
// Set DEVICE_ID to 2 for Device 2 (XIAO_SENSE_BLE_2)
// ============================================================================
#ifndef DEVICE_ID
#define DEVICE_ID 1
#endif

#if DEVICE_ID == 2
#define CONFIG_CUSTOM_DEVICE_NAME "XIAO_SENSE_BLE_2"
#else
#define CONFIG_CUSTOM_DEVICE_NAME "XIAO_SENSE_BLE_1"
#endif

// ============================================================================
// AUDIO STREAMING TEST MODE (For Audacity Analysis via record_to_wav.py)
// ============================================================================
// 0: Normal Inference Mode (ASCII serial logs enabled)
// 1: Stream PRE-FILTER Raw PCM over Serial (Before DC-Block HPF & Butterworth LPF)
// 2: Stream POST-FILTER Clean PCM over Serial (After DC-Block HPF & Butterworth LPF)
#define STREAM_RAW_PCM_MODE 0

#endif // ESC_CONFIG_HPP
