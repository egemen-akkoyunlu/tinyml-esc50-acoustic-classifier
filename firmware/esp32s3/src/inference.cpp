/*
 * Copyright (c) 2026 Seeed Studio / Senior ML Engineer
 * SPDX-License-Identifier: Apache-2.0
 */

#include "inference.hpp"
#include "esc_config.hpp"
#include "compat/esp_heap_caps.h"

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <zephyr/drivers/gpio.h>
#include <cstring>
#include <cmath>
#include <algorithm>

LOG_MODULE_DECLARE(esc50_app, LOG_LEVEL_INF);

/* Otii Sync GPIO Specifications */
#define SYNC_AP_NODE DT_ALIAS(sync_ap)
#define SYNC_INF_NODE DT_ALIAS(sync_inf)

static const struct gpio_dt_spec sync_ap = GPIO_DT_SPEC_GET(SYNC_AP_NODE, gpios);
static const struct gpio_dt_spec sync_inf = GPIO_DT_SPEC_GET(SYNC_INF_NODE, gpios);

static const char *CLASS_LABELS[50] = {
    "airplane", "breathing", "brushing_teeth", "can_opening", "car_horn",
    "cat", "chainsaw", "chirping_birds", "church_bells", "clapping",
    "clock_alarm", "clock_tick", "coughing", "cow", "crackling_fire",
    "crickets", "crow", "crying_baby", "dog", "door_wood_creaks",
    "door_wood_knock", "drinking_sipping", "engine", "fireworks", "footsteps",
    "frog", "glass_breaking", "hand_saw", "helicopter", "hen",
    "insects", "keyboard_typing", "laughing", "mouse_click", "pig",
    "pouring_water", "rain", "rooster", "sea_waves", "sheep",
    "siren", "sneezing", "snoring", "thunderstorm", "toilet_flush",
    "train", "vacuum_cleaner", "washing_machine", "water_drops", "wind"
};

static const float CLASS_THRESHOLDS[4] = {
    0.97f, // go (voiced vowel, high confidence)
    0.94f, // stop (sibilant start, lower confidence on mic)
    0.94f, // left (sibilant/fricative end, lower confidence on mic)
    0.96f  // right (voiced diphthong, high confidence)
};

static const int CLASS_DEBOUNCE_THRESHOLDS[4] = {
    2, // go (requires 4 consecutive frames to prevent false triggers)
    2, // stop (requires 3 consecutive frames)
    2, // left (requires 4 consecutive frames)
    2  // right (requires 4 consecutive frames)
};

static const unsigned char model_espdl_bin[] __attribute__((aligned(16))) = {
#include "model_espdl.inc"
};

KWSInference::KWSInference()
    : model(nullptr), fbank(nullptr), input_tensor(nullptr), output_tensor(nullptr),
      sliding_window_buf(nullptr), normalized_buf(nullptr), cooldown_counter(0),
      last_detected_class(-1), consecutive_count(0), dropout_count(0),
      last_ap_start(0), last_ap_end(0), last_ap_time_ms(0.0),
      last_inf_start(0), last_inf_end(0), last_inf_time_ms(0.0)
{
}

KWSInference::~KWSInference()
{
    if (model) delete model;
    if (fbank) delete fbank;
    if (sliding_window_buf) heap_caps_free(sliding_window_buf);
    if (normalized_buf) heap_caps_free(normalized_buf);
}

void KWSInference::compute_softmax(const float *logits, float *probs, int len)
{
    float max_logit = logits[0];
    for (int i = 1; i < len; i++) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
        }
    }

    float sum_exp = 0.0f;
    for (int i = 0; i < len; i++) {
        probs[i] = expf(logits[i] - max_logit);
        sum_exp += probs[i];
    }
    for (int i = 0; i < len; i++) {
        probs[i] /= sum_exp;
    }
}

bool KWSInference::init()
{
    LOG_INF("[FLAG: STAGE_0_START] Loading Model from Flash...");
    LOG_INF("Model binary at: %p (size: %zu bytes)", 
            (void *)model_espdl_bin, sizeof(model_espdl_bin));
    
    /*
     * Initialize the Flatbuffer Model object.
     * 
     * IMPORTANT OPTIMIZATION: 
     * We pass "false" as the 6th argument (param_copy) to run model weights
     * directly from Flash RODATA. This saves ~80 KB of internal SRAM,
     * preventing allocation failures on the system heap.
     */
    model = new dl::Model((const char *)model_espdl_bin, 
                          fbs::MODEL_LOCATION_IN_FLASH_RODATA,
                          0,
                          dl::MEMORY_MANAGER_GREEDY,
                          nullptr,
                          false);

    if (model == nullptr) {
        LOG_ERR("[FLAG: ERROR] Failed to initialize dl::Model!");
        return false;
    }

    LOG_INF("[FLAG: SUCCESS] Model initialized successfully!");

    auto inputs = model->get_inputs();
    auto outputs = model->get_outputs();
    
    if (inputs.empty() || outputs.empty()) {
        LOG_ERR("[FLAG: ERROR] Model has no inputs or outputs!");
        return false;
    }
    
    LOG_INF("--- Model Inputs ---");
    for (auto const& [name, tensor] : inputs) {
        std::string shape_str = "";
        for (int d : tensor->shape) shape_str += std::to_string(d) + " ";
        LOG_INF("  Input '%s': shape=[ %s] (bytes: %d)", name.c_str(), shape_str.c_str(), tensor->get_bytes());
    }
    
    // Default to first input, which should be the spectrogram
    input_tensor = inputs.begin()->second;

    LOG_INF("--- Model Outputs ---");
    output_tensor = nullptr;
    for (auto const& [name, tensor] : outputs) {
        std::string shape_str = "";
        for (int d : tensor->shape) shape_str += std::to_string(d) + " ";
        LOG_INF("  Output '%s': shape=[ %s] (bytes: %d)", name.c_str(), shape_str.c_str(), tensor->get_bytes());
        
        // Calculate total elements in this output tensor
        int elements = 1;
        for (int d : tensor->shape) elements *= d;
        
        // The logits tensor must have exactly NUM_CLASSES elements (50)
        if (elements == NUM_CLASSES) {
            output_tensor = tensor;
            LOG_INF("  -> Selected '%s' as the active logits output tensor", name.c_str());
        }
    }

    if (output_tensor == nullptr) {
        LOG_WRN("[FLAG: WARNING] No output tensor with %d elements found! Defaulting to first output.", NUM_CLASSES);
        output_tensor = outputs.begin()->second;
    }

    // Verify input/output dimension sizes (Batch=1, Freq=52, Time=313)
    if (input_tensor->shape[0] != 1 || input_tensor->shape[1] != NUM_MEL_BINS || input_tensor->shape[2] != TIME_FRAMES) {
        LOG_ERR("[FLAG: ERROR] Input tensor shape mismatch! Expected [1, %d, %d], got [%d, %d, %d]", 
                NUM_MEL_BINS, TIME_FRAMES, input_tensor->shape[0], input_tensor->shape[1], input_tensor->shape[2]);
        return false;
    }

    if (output_tensor->shape[0] != 1 || output_tensor->shape[1] != NUM_CLASSES) {
        LOG_ERR("[FLAG: ERROR] Output tensor shape mismatch! Expected [1, %d], got [%d, %d]", 
                NUM_CLASSES, output_tensor->shape[0], output_tensor->shape[1]);
        return false;
    }

    LOG_INF("[FLAG: STAGE_0_START] Initializing Mel Filterbank...");
    dl::audio::SpeechFeatureConfig fbank_cfg;
    fbank_cfg.sample_rate = SAMPLE_RATE;       // 16000
    fbank_cfg.frame_length = 32;               // 32 ms window (512 samples at 16kHz)
    fbank_cfg.frame_shift = 16;                // 16 ms step (256 samples at 16kHz)
    fbank_cfg.num_mel_bins = NUM_MEL_BINS;     // 52 mel bins
    fbank_cfg.low_freq = 0.0f;                 // 0.0 Hz (matches PyTorch's f_min = 0.0)
    fbank_cfg.use_log_fbank = 2;               // logf(x + epsilon)
    fbank_cfg.log_epsilon = 1e-10f;
    fbank_cfg.preemphasis = 0.0f;              // No preemphasis
    fbank_cfg.remove_dc_offset = true;         // Remove DC
    fbank_cfg.window_type = dl::audio::WinType::HANN;

    fbank = new dl::audio::Fbank(fbank_cfg);
    if (fbank == nullptr) {
        LOG_ERR("[FLAG: ERROR] Failed to initialize dl::audio::Fbank!");
        return false;
    }
    LOG_INF("[FLAG: SUCCESS] Fbank initialized successfully!");

    sliding_window_buf = (float *)heap_caps_malloc(
        NUM_MEL_BINS * TIME_FRAMES * sizeof(float), MALLOC_CAP_SPIRAM);
    
    if (!sliding_window_buf) {
        LOG_ERR("[FLAG: ERROR] Failed to allocate sliding window buffer in PSRAM!");
        return false;
    }
    memset(sliding_window_buf, 0, NUM_MEL_BINS * TIME_FRAMES * sizeof(float));
    LOG_INF("[FLAG: SUCCESS] Sliding window buffer allocated and zero-initialized in PSRAM");

    normalized_buf = (float *)heap_caps_malloc(
        NUM_MEL_BINS * TIME_FRAMES * sizeof(float), MALLOC_CAP_SPIRAM);
    if (!normalized_buf) {
        LOG_ERR("[FLAG: ERROR] Failed to allocate normalized buffer in PSRAM!");
        heap_caps_free(sliding_window_buf);
        sliding_window_buf = nullptr;
        return false;
    }
    memset(normalized_buf, 0, NUM_MEL_BINS * TIME_FRAMES * sizeof(float));
    LOG_INF("[FLAG: SUCCESS] Normalized buffer allocated and zero-initialized in PSRAM");
    LOG_INF("[FLAG: SUCCESS] All buffers allocated successfully!");
    
    /* Initialize Otii Sync GPIOs as output, inactive by default */
    if (device_is_ready(sync_ap.port)) {
        gpio_pin_configure_dt(&sync_ap, GPIO_OUTPUT_INACTIVE);
        LOG_INF("[FLAG: SUCCESS] Audio Processing Sync GPIO initialized on Pin D1 (GPIO2)");
    } else {
        LOG_WRN("[FLAG: WARNING] Audio Processing Sync GPIO device not ready");
    }

    if (device_is_ready(sync_inf.port)) {
        gpio_pin_configure_dt(&sync_inf, GPIO_OUTPUT_INACTIVE);
        LOG_INF("[FLAG: SUCCESS] Inference Sync GPIO initialized on Pin D2 (GPIO3)");
    } else {
        LOG_WRN("[FLAG: WARNING] Inference Sync GPIO device not ready");
    }

    k_sem_init(&inf_sem, 0, 1);

    return true;
}

void KWSInference::process_audio_window(const int16_t *audio_window, int frame_counter, bool log_details)
{
    gpio_pin_set_dt(&sync_ap, 1); /* Set AP pin High */
    last_ap_start = k_cycle_get_32();
    float mel_frame_out[NUM_MEL_BINS] = {0};
    fbank->process_frame(audio_window, FRAME_LEN_SAMPLES, mel_frame_out);

    // Pass natural log ln(x) from esp-dl Fbank directly to match PyTorch's torch.log()

    // Shift each Mel channel's time-series left by 1 frame and append the new Mel coefficient.
    // Memory layout is Freq-major [NUM_MEL_BINS, TIME_FRAMES] ([52, 313])
    for (int m = 0; m < NUM_MEL_BINS; m++) {
        float *row = sliding_window_buf + m * TIME_FRAMES;
        memmove(row, row + 1, (TIME_FRAMES - 1) * sizeof(float));
        row[TIME_FRAMES - 1] = mel_frame_out[m];
    }
    last_ap_end = k_cycle_get_32();
    gpio_pin_set_dt(&sync_ap, 0); /* Set AP pin Low */
    last_ap_time_ms = (double)(last_ap_end - last_ap_start) / (sys_clock_hw_cycles_per_sec() / 1000.0);

    if (frame_counter % INFERENCE_STRIDE == 0) {
        // Calculate audio frame RMS over 256 samples to check for sound activity
        int64_t win_sum_sq = 0;
        for (int i = 0; i < FRAME_LEN_SAMPLES; i++) {
            win_sum_sq += (int32_t)audio_window[i] * (int32_t)audio_window[i];
        }
        float win_rms = sqrtf((float)win_sum_sq / FRAME_LEN_SAMPLES);

        if (win_rms >= VOICE_ACTIVITY_RMS_THRESHOLD) {
            k_sem_give(&inf_sem);
        }
    }
}

void KWSInference::run_inference_now()
{
    gpio_pin_set_dt(&sync_inf, 1); /* Set INF pin High */
    last_inf_start = k_cycle_get_32();

    // Feed raw Log-Mel Spectrogram directly (matching PyTorch & Silicon Labs pipeline 1-to-1)
    dl::TensorBase float_sliding_tensor({1, NUM_MEL_BINS, TIME_FRAMES, 1}, 
                                        sliding_window_buf, 
                                        0, 
                                        dl::DATA_TYPE_FLOAT);

    input_tensor->assign(&float_sliding_tensor);
    model->reset(); // Reset GRU hidden states to zero to match PyTorch evaluation
    model->run();   // Run the neural network
    last_inf_end = k_cycle_get_32();
    gpio_pin_set_dt(&sync_inf, 0); /* Set INF pin Low */
    last_inf_time_ms = (double)(last_inf_end - last_inf_start) / (sys_clock_hw_cycles_per_sec() / 1000.0);

    int8_t *int8_logits = (int8_t *)output_tensor->data;
    float scale = DL_SCALE(output_tensor->exponent);

    float fp32_logits[NUM_CLASSES];
    for (int c = 0; c < NUM_CLASSES; c++) {
        fp32_logits[c] = int8_logits[c] * scale;
    }

    float probs[NUM_CLASSES] = {0.0f};
    compute_softmax(fp32_logits, probs, NUM_CLASSES);

    // EMA Temporal Probability Smoothing (alpha = 0.60: 60% current frame, 40% memory)
    static float smoothed_probs[NUM_CLASSES] = {0.0f};
    static bool is_first_run = true;
    const float alpha = 0.60f;

    if (is_first_run) {
        for (int c = 0; c < NUM_CLASSES; c++) {
            smoothed_probs[c] = probs[c];
        }
        is_first_run = false;
    } else {
        for (int c = 0; c < NUM_CLASSES; c++) {
            smoothed_probs[c] = alpha * probs[c] + (1.0f - alpha) * smoothed_probs[c];
        }
    }

    // Compute Top 3 predicted ESC-50 classes from smoothed probabilities
    int top_idx[3] = {0, 0, 0};
    float top_p[3] = {-1.0f, -1.0f, -1.0f};

    for (int c = 0; c < NUM_CLASSES; c++) {
        if (smoothed_probs[c] > top_p[0]) {
            top_p[2] = top_p[1]; top_idx[2] = top_idx[1];
            top_p[1] = top_p[0]; top_idx[1] = top_idx[0];
            top_p[0] = smoothed_probs[c]; top_idx[0] = c;
        } else if (smoothed_probs[c] > top_p[1]) {
            top_p[2] = top_p[1]; top_idx[2] = top_idx[1];
            top_p[1] = smoothed_probs[c]; top_idx[1] = c;
        } else if (smoothed_probs[c] > top_p[2]) {
            top_p[2] = smoothed_probs[c]; top_idx[2] = c;
        }
    }

    LOG_INF("ESC-50: #1 %s (%.1f%%) | #2 %s (%.1f%%) | #3 %s (%.1f%%) [Inf: %.2f ms | DSP: %.2f ms]",
            CLASS_LABELS[top_idx[0]], (double)(top_p[0] * 100.0f),
            CLASS_LABELS[top_idx[1]], (double)(top_p[1] * 100.0f),
            CLASS_LABELS[top_idx[2]], (double)(top_p[2] * 100.0f),
            last_inf_time_ms, last_ap_time_ms);
}
