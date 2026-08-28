#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>
#include <stdio.h>
#include <math.h>
#include "em_cmu.h"
#include "em_device.h"
#include "SEGGER_RTT.h"
#include "config.h"
#include "audio.h"
#include "audio_preprocessing.h"
#include "inference.h"
#include "ble_server.h"
#if (FIRMWARE_OPERATION_MODE == MODE_INJECT_GOLDEN_SAMPLE)
#include "golden_keyboard_typing_pcm.h"
#endif

int main(void) {
    /* 0. BOOST CPU CLOCK TO 80 MHz MAXIMUM PERFORMANCE */
    CMU_HFRCODPLLBandSet(cmuHFRCODPLLFreq_80M0Hz);
    CMU_ClockSelectSet(cmuClock_SYSCLK, cmuSelect_HFRCODPLL);
    SystemCoreClockUpdate();

    /* EARLY HEARTBEAT PRINTK (Guaranteed direct UART output to /dev/ttyACM0) */
    printk("\n\n");
    printk("************************************************************\n");
    printk(" *** EFR32MG24 BOOT TEST: UART CONSOLE LOGGING IS ALIVE! ***\n");
    printk("************************************************************\n\n");

    SEGGER_RTT_Init();
    
    uint32_t cpu_freq_hz = CMU_ClockFreqGet(cmuClock_SYSCLK);
    printf("\n============================================================\n");
    printf(" EFR32MG24 TINYML ESC-50 SOUND CLASSIFICATION PIPELINE\n");
    printf(" Target: Silicon Labs EFR32MG24 DevKit (ARM Cortex-M33)\n");
    printf(" Clock : %lu MHz High-Performance Core Clock\n", cpu_freq_hz / 1000000);
    printf(" Engine: Zephyr RTOS + TensorFlow Lite Micro (CMSIS-NN)\n");
    printf(" Mode  : %s\n", ENABLE_STREAMING_INFERENCE ? 
           "CONTINUOUS SLIDING-WINDOW STREAMING" : "ONE-SHOT CAPTURE");
    printf("============================================================\n\n");

    /* 1. Hardware & Module Initialization */
    printk("-> Step 1: Initializing hardware & TinyML modules...\n");
    audio_init();
    audio_preprocessing_init();
    
    if (inference_init() != 0) {
        printk("[FATAL] TinyML inference initialization failed!\n");
        return -1;
    }

#if ENABLE_BLE_COMMUNICATION
    ble_server_init();

    /* 2. Wait for BLE Client Connection */
    printk("\n-> Step 2: Waiting for BLE Client Connection...\n");
    while (!ble_is_connected_and_ready()) {
        printk(".");
        k_sleep(K_MSEC(500));
    }
    printk("\n-> BLE Connected!\n");
#else
    printk("\n-> [STANDALONE MODE] BLE is disabled. Starting real-time audio inference directly via UART!\n");
#endif

    void *input_tensor_ptr = inference_get_input_tensor_ptr();
    bool is_input_int8 = inference_is_input_int8();
    float scale = 1.0f;
    int32_t zero_point = 0;
    inference_get_input_quant_params(&scale, &zero_point);

#if (FIRMWARE_OPERATION_MODE == MODE_INJECT_GOLDEN_SAMPLE)
    /* ========================================================================
     * 🧪 MODE 2: DIRECT AUDIO INJECTION (GOLDEN ARRAY BENCHMARK - NO MIC)
     * ======================================================================== */
    printk("\n============================================================\n");
    printk(" 🧪 FIRMWARE MODE 2: DIRECT AUDIO INJECTION BENCHMARK\n");
    printk(" Target: Silicon Labs EFR32MG24 DevKit (ARM Cortex-M33)\n");
    printk(" Input : Golden Validation 'keyboard_typing' ESC-50 Array (1-94231-B-32.wav)\n");
    printk(" Bypassing Physical Microphone to Test DSP & Neural Network!\n");
    printk("============================================================\n\n");

    printk("-> Step 1: Running Bit-Exact On-Chip DSP Mel-Spectrogram Extraction...\n");
    audio_preprocess_run_direct(GOLDEN_KEYBOARD_TYPING_PCM, GOLDEN_AUDIO_SAMPLE_COUNT, scale, zero_point, is_input_int8, input_tensor_ptr);

    printk("-> Step 2: Executing 2-Stage Hybrid TinyML Model Inference...\n");
    int predicted_class_id = -1;
    float confidence = 0.0f;

    if (inference_run_direct(&predicted_class_id, &confidence) == 0) {
        const char *class_name = inference_get_class_name(predicted_class_id);
        printk("\n============================================================\n");
        printk(" 🏆 GOLDEN AUDIO INJECTION INFERENCE RESULT:\n");
        printk("   • Injected Sound       : 'keyboard_typing' (1-94231-B-32.wav - 100% UNSEEN VAL)\n");
        printk("   • Predicted Class ID   : %d\n", predicted_class_id);
        printk("   • Predicted Sound Name : '%s'\n", class_name);
        printk("   • Prediction Confidence: %d%%\n", (int)(confidence * 100.0f));
        printk("============================================================\n\n");
    } else {
        printk("[ERROR] Inference execution failed!\n");
    }

    while (1) {
        k_sleep(K_SECONDS(2));
    }

#elif (FIRMWARE_OPERATION_MODE == MODE_AUDIO_DUMP_TO_PC)
    /* ========================================================================
     * 🎙️ DIAGNOSTIC MODE: STREAM 3-SECOND MICROPHONE AUDIO TO PC (WAV DUMP)
     * ======================================================================== */
    printk("\n============================================================\n");
    printk(" 🎙️ FIRMWARE MODE: 3-SECOND MICROPHONE AUDIO DUMP TO PC\n");
    printk(" Target: Silicon Labs EFR32MG24 DevKit\n");
    printk(" Rate  : 16000 Hz, 16-bit PCM Mono, Total: 48000 samples (3 sec)\n");
    printk(" Speak, type, or play sounds near the board now!\n");
    printk("============================================================\n\n");

    int dump_iteration = 0;
    while (1) {
        printk("\n-> [READY #%d] 🎙️ Recording 3 seconds of live audio from mic...\n", ++dump_iteration);
        printk("=== AUDIO_DUMP_START ===\n");

        for (int sec = 0; sec < DUMP_RECORD_SECONDS; sec++) {
            audio_record_to_ram();
            int16_t *raw_pcm = get_audio_buffer();

            for (int i = 0; i < AUDIO_RING_BUFFER_SAMPLES; i++) {
                printk("%d\n", (int)raw_pcm[i]);
            }
        }

        printk("=== AUDIO_DUMP_END ===\n");
        printk("-> [OK] 3-second capture complete! Waiting 3 seconds before next clip...\n");
        k_sleep(K_SECONDS(3));
    }

#elif (FIRMWARE_OPERATION_MODE == MODE_REALTIME_INFERENCE)
    /* ========================================================================
     * 🧠 MODE 0: CONTINUOUS SLIDING-WINDOW STREAMING INFERENCE
     * ======================================================================== */
    printk("\n-> [CONTINUOUS STREAMING MODE] Starting Real-Time Acoustic Inference Loop...\n");
    
    /* Zero-initialize the TFLM input tensor buffer to true silence (-128) */
    if (is_input_int8) {
        memset(input_tensor_ptr, -128, SPECTROGRAM_TOTAL_ELEMENTS);
    } else {
        memset(input_tensor_ptr, 0, sizeof(float) * SPECTROGRAM_TOTAL_ELEMENTS);
    }

    const int SILENCE_AC_RMS_THRESHOLD = 3500;      /* Squelch threshold: Ignores ambient noise (<3500), triggers on active sounds */
    const float SOUND_CONFIDENCE_THRESHOLD = 0.35f; /* 35% minimum confidence */
    int inference_counter = 0;

    while (1) {
        /* 1. Record 1.0-second audio chunk directly into audio_ring_buffer */
        audio_record_to_ram();
        int16_t *raw_pcm = get_audio_buffer();

        /* 2. Calculate DC Mean & True AC RMS */
        int64_t sum_samples = 0;
        for (int i = 0; i < AUDIO_RING_BUFFER_SAMPLES; i++) {
            sum_samples += raw_pcm[i];
        }
        int16_t dc_mean = (int16_t)(sum_samples / AUDIO_RING_BUFFER_SAMPLES);

        int64_t sum_ac_sq = 0;
        for (int i = 0; i < AUDIO_RING_BUFFER_SAMPLES; i++) {
            int16_t ac_s = raw_pcm[i] - dc_mean;
            sum_ac_sq += ((int32_t)ac_s * ac_s);
        }
        int ac_rms = (int)sqrtf((float)(sum_ac_sq / AUDIO_RING_BUFFER_SAMPLES));

        /* 3. Compute and tile active Mel chunk across full 5.0s (313 frames) to eliminate temporal dilution */
        uint32_t t_dsp_start = k_cycle_get_32();
        audio_preprocess_tile_chunk_direct(raw_pcm, AUDIO_RING_BUFFER_SAMPLES, scale, zero_point, is_input_int8, input_tensor_ptr);
        uint32_t t_dsp_end = k_cycle_get_32();
        uint32_t dsp_time_us = k_cyc_to_us_near32(t_dsp_end - t_dsp_start);

        /* 4. Evaluate Dual-Stage Silence & Confidence Thresholds */
        uint32_t uptime_ms = k_uptime_get_32();
        if (ac_rms < SILENCE_AC_RMS_THRESHOLD) {
            printk("[%02d:%02d.%03d] [#%04d] 🎙️ [AC-RMS: %4d | DSP: %.2f ms] -> [Quiet / Ambient Silence]\n", 
                   (uptime_ms / 60000) % 60, (uptime_ms / 1000) % 60, uptime_ms % 1000,
                   ++inference_counter, ac_rms, (float)dsp_time_us / 1000.0f);
        } else {
            /* Run Hybrid 2-Stage Model Inference */
            int predicted_class_id = -1;
            float confidence = 0.0f;

            if (inference_run_direct(&predicted_class_id, &confidence) == 0) {
                if (confidence >= SOUND_CONFIDENCE_THRESHOLD) {
                    const char *class_name = inference_get_class_name(predicted_class_id);
                    printk("[%02d:%02d.%03d] [#%04d] 🎙️ [AC-RMS: %5d | DSP: %.2f ms] -> 🎯 CONFIDENT SOUND: %-20s (Conf: %2d%%) 🌟\n",
                           (uptime_ms / 60000) % 60, (uptime_ms / 1000) % 60, uptime_ms % 1000,
                           ++inference_counter, ac_rms, (float)dsp_time_us / 1000.0f, class_name, (int)(confidence * 100.0f));
                }
            }
        }
    }
#endif

    return 0;
}
