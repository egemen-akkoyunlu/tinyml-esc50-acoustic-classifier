#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <zephyr/drivers/gpio.h>
#include <cstring>
#include <cstdio>

#include "esc_config.hpp"
#include "audio.hpp"
#include "audio_preprocessing.hpp"
#include "inference.hpp"

LOG_MODULE_REGISTER(esc50_app, LOG_LEVEL_INF);

/* Built-in LED: GPIO21, active-low (LED_ON = low, LED_OFF = high) */
#define LED0_NODE DT_ALIAS(led0)
static const struct gpio_dt_spec led = GPIO_DT_SPEC_GET(LED0_NODE, gpios);

K_THREAD_STACK_DEFINE(inf_stack, 16384);
static struct k_thread inf_thread_data;

static void inf_thread_entry(void *p1, void *p2, void *p3)
{
    KWSInference *kws = (KWSInference *)p1;
    while (1) {
        k_sem_take(&kws->inf_sem, K_FOREVER);
        kws->run_inference_now();
    }
}

int main(void)
{
    LOG_INF("==================================================");
    LOG_INF("  XIAO ESP32-S3 Sense: ESC-50 50-Class Classifier (Zephyr)");
    LOG_INF("==================================================");

    /* --------------------------------------------------------
     * STARTUP BLINK: 3 fast blinks to confirm power-up and
     * firmware boot. Visible without a serial console.
     * -------------------------------------------------------- */
    if (device_is_ready(led.port)) {
        gpio_pin_configure_dt(&led, GPIO_OUTPUT_INACTIVE);
        for (int i = 0; i < 3; i++) {
            gpio_pin_set_dt(&led, 1);   /* LED ON  */
            k_sleep(K_MSEC(200));
            gpio_pin_set_dt(&led, 0);   /* LED OFF */
            k_sleep(K_MSEC(200));
        }
    }

    KWSInference kws_engine;
    if (!kws_engine.init()) {
        LOG_ERR("[FLAG: ERROR] ESC-50 Inference engine initialization failed!");
        return -1;
    }

    if (!i2s_mic_init()) {
        LOG_ERR("[FLAG: ERROR] I2S PDM Microphone initialization failed!");
        return -1;
    }

    /* Spawn background inference thread on Core 1 after I2S hardware is initialized */
    k_thread_create(&inf_thread_data, inf_stack, K_THREAD_STACK_SIZEOF(inf_stack),
                    inf_thread_entry, &kws_engine, NULL, NULL,
                    7, 0, K_NO_WAIT);

    LOG_INF("==================================================");
    LOG_INF("[FLAG: LIVE] Using real PDM microphone!");
    LOG_INF("Ready for real-time INT8 ESC-50 50-class inference!");
    LOG_INF("==================================================");

    AudioProcessor audio_processor;

    int frame_counter = 0;
    int audio_buffer_index = 0;
    int16_t audio_pool[FRAME_SHIFT_SAMPLES] = {0};
    int16_t audio_window[FRAME_LEN_SAMPLES] = {0};

    while (1) {
        void *mem_block = nullptr;
        size_t block_size = 0;
        int ret = i2s_mic_read_block(&mem_block, &block_size);

        if (ret == 0 && mem_block != nullptr && block_size == I2S_BLOCK_SIZE) {
            uint8_t *byte_ptr = (uint8_t *)mem_block;
            
            for (int i = 0; i < SAMPLES_PER_BLOCK; i++) {
                int16_t raw_l = (int16_t)(byte_ptr[i * 8 + 2] | (byte_ptr[i * 8 + 3] << 8));
                int16_t raw_r = (int16_t)(byte_ptr[i * 8 + 6] | (byte_ptr[i * 8 + 7] << 8));
                
                int16_t clean_sample = audio_processor.process_sample_pair(raw_l, raw_r);

#if STREAM_RAW_PCM_MODE > 0
                int16_t stream_val = (STREAM_RAW_PCM_MODE == 1) ? raw_l : clean_sample;
                putchar(stream_val & 0xFF);
                putchar((stream_val >> 8) & 0xFF);
#else
                audio_pool[audio_buffer_index++] = clean_sample;

                if (audio_buffer_index >= FRAME_SHIFT_SAMPLES) {
                    audio_buffer_index = 0;

                    // Shift window
                    memmove(audio_window, audio_window + FRAME_SHIFT_SAMPLES, 
                            (FRAME_LEN_SAMPLES - FRAME_SHIFT_SAMPLES) * sizeof(int16_t));
                    
                    memcpy(audio_window + (FRAME_LEN_SAMPLES - FRAME_SHIFT_SAMPLES), 
                           audio_pool, FRAME_SHIFT_SAMPLES * sizeof(int16_t));

                    bool log_details = false;

                    kws_engine.process_audio_window(audio_window, frame_counter, log_details);

                    frame_counter++;
                    if (frame_counter > 1000000) frame_counter = 0;
                }
#endif
            }
#if STREAM_RAW_PCM_MODE == 0
            // audio_processor.check_and_report_diagnostics();
#endif
            i2s_mic_free_block(mem_block);
        } else {
            if (mem_block != nullptr) {
                i2s_mic_free_block(mem_block);
            }
            k_sleep(K_MSEC(1));
        }
    }
    return 0;
}
