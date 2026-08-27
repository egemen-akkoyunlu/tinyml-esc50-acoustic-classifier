#ifndef CONFIG_H
#define CONFIG_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * AUDIO HARDWARE CONFIGURATION
 * ============================================================================ */
#define SAMPLE_RATE             16000
#define RECORD_SECONDS          5
#define TOTAL_SAMPLES           (SAMPLE_RATE * RECORD_SECONDS) /* 80000 samples */
#define AUDIO_RING_BUFFER_SAMPLES (SAMPLE_RATE * 1)            /* 16000 samples (1.0s) = 32 KB RAM (Full Acoustic Fidelity) */
#define DUMP_RECORD_SECONDS     3                              /* 3 seconds quick dump = 48000 samples */
#define DUMP_TOTAL_SAMPLES      (SAMPLE_RATE * DUMP_RECORD_SECONDS)
#define AUDIO_SOFTWARE_GAIN_MULTIPLIER 48                      /* 48x Software Digital Gain (Full Acoustic Dynamic Range) */

/* ============================================================================
 * FIRMWARE OPERATING MODES
 * MODE_REALTIME_INFERENCE    : Continuous Real-Time Environmental Sound Classification
 * MODE_AUDIO_DUMP_TO_PC      : Stream 3-Sec Raw Microphone Audio to PC over UART (WAV)
 * MODE_INJECT_GOLDEN_SAMPLE  : Benchmark Golden Pre-Recorded Audio (No Mic)
 * ============================================================================ */
#define MODE_REALTIME_INFERENCE    0
#define MODE_AUDIO_DUMP_TO_PC      1
#define MODE_INJECT_GOLDEN_SAMPLE  2

#define FIRMWARE_OPERATION_MODE    MODE_REALTIME_INFERENCE /* Benchmark Golden Pre-Recorded Audio (No Mic) */

#define ENABLE_STREAMING_INFERENCE         true
#define ENABLE_BLE_COMMUNICATION           false /* Set to false for standalone microphone audio inference over UART */

/* Sliding window stride in seconds for continuous mode (e.g. 1 sec hop) */
#define INFERENCE_STREAMING_STRIDE_SEC     1
#define INFERENCE_STREAMING_STRIDE_SAMPLES (SAMPLE_RATE * INFERENCE_STREAMING_STRIDE_SEC)

/* ============================================================================
 * AUDIO PREPROCESSING & MEL-SPECTROGRAM CONFIGURATION
 * ============================================================================ */
#define FFT_SIZE                512
#define HOP_LENGTH              256
#define N_MELS                  52
#define NUM_FREQ_BINS           (FFT_SIZE / 2 + 1) /* 257 bins */

/* Model expects 313 time frames (5 seconds @ 16kHz with 256 hop, center=True). */
#define SPECTROGRAM_TIME_STEPS  313
#define SPECTROGRAM_TOTAL_ELEMENTS (N_MELS * SPECTROGRAM_TIME_STEPS) /* 52 * 313 = 16276 */

/* ============================================================================
 * TENSORFLOW LITE MICRO CONFIGURATION
 * Model requires ~68 KB SRAM arena with CMSIS-NN scratch. Set to 70 KB.
 * ============================================================================ */
#define TFLM_TENSOR_ARENA_SIZE  (172 * 1024) /* 175 KB RAM Arena for 91% Symmetric Model */
#define NUM_ESC50_CLASSES       50

/* ============================================================================
 * ESC-50 CLASS NAMES (Exact Alphabetical Match with Trained Model)
 * ============================================================================ */
static const char* const ESC50_CLASS_NAMES[NUM_ESC50_CLASSES] = {
    "airplane",            "breathing",           "brushing teeth",      "can opening",         "car horn",
    "cat",                 "chainsaw",            "chirping birds",      "church bells",        "clapping",
    "clock alarm",         "clock tick",          "coughing",            "cow",                 "crackling fire",
    "crickets",            "crow",                "crying baby",         "dog",                 "door wood creaks",
    "door wood knock",     "drinking sipping",    "engine",              "fireworks",           "footsteps",
    "frog",                "glass breaking",      "hand saw",            "helicopter",          "hen",
    "insects",             "keyboard typing",     "laughing",            "mouse click",         "pig",
    "pouring water",       "rain",                "rooster",             "sea waves",           "sheep",
    "siren",               "sneezing",            "snoring",             "thunderstorm",        "toilet flush",
    "train",               "vacuum cleaner",      "washing machine",     "water drops",         "wind"
};

#ifdef __cplusplus
}
#endif

#endif /* CONFIG_H */
