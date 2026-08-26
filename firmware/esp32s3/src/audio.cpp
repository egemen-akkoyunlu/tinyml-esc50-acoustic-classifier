/*
 * Copyright (c) 2026 Seeed Studio / Senior ML Engineer
 * SPDX-License-Identifier: Apache-2.0
 * 
 * audio_i2s.cpp - I2S driver wrapper for the PDM microphone on the Xiao ESP32-S3.
 * Configures the ESP32-S3 I2S peripheral in PDM RX mode, manages the DMA
 * memory slabs, and provides automatic recovery from audio stream underflows.
 */

#include "audio.hpp"
#include "esc_config.hpp"

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <zephyr/drivers/i2s.h>
#include <zephyr/drivers/gpio.h>

/* ESP32-S3 direct hardware register access definitions */
#include <soc/i2s_struct.h>
#include <soc/i2s_reg.h>

LOG_MODULE_DECLARE(esc50_app, LOG_LEVEL_INF);

/* 
 * Define static memory slab pool for I2S DMA.
 * Allocates 128 blocks of I2S_BLOCK_SIZE bytes, aligned on 8-byte boundaries.
 */
K_MEM_SLAB_DEFINE_STATIC(rx_mem_slab, I2S_BLOCK_SIZE, 128, 8);

static const struct device *i2s_dev = nullptr;

/*
 * apply_pdm_regs: Directly configures the ESP32-S3 I2S hardware registers
 *                 to enable PDM-to-PCM demodulation and SINC decimation filter.
 */
static void apply_pdm_regs(void)
{
    I2S0.rx_conf.rx_pdm_en = 1;              // Enable PDM RX channel
    I2S0.rx_conf.rx_pdm2pcm_en = 1;          // Enable hardware PDM-to-PCM demodulator
    I2S0.rx_conf.rx_pdm_sinc_dsr_16_en = 1;  // Set oversampling rate to 16
    I2S0.rx_conf.rx_tdm_en = 0;              // Disable TDM mode
    I2S0.rx_conf.rx_mono = 0;                 // Disable Mono mode (read stereo)
    I2S0.rx_conf.rx_update = 1;              // Apply changes immediately
}

/*
 * full_i2s_reset: Performs a full hardware reset on the I2S peripheral,
 *                 clears the DMA queue, and restarts the audio capture.
 */
static void full_i2s_reset(void)
{
    LOG_WRN("Full I2S reset...");
    
    /* Stop DMA capture */
    i2s_trigger(i2s_dev, I2S_DIR_RX, I2S_TRIGGER_DROP);
    k_sleep(K_MSEC(10));
    
    /* Reset RX unit and FIFO */
    I2S0.rx_conf.rx_reset = 1;
    I2S0.rx_conf.rx_fifo_reset = 1;
    k_sleep(K_MSEC(10));
    I2S0.rx_conf.rx_reset = 0;
    I2S0.rx_conf.rx_fifo_reset = 0;
    
    /* Flush the DMA queue and free all allocated blocks */
    void *dummy;
    size_t dummy_size;
    while (i2s_read(i2s_dev, &dummy, &dummy_size) == 0 && dummy != NULL) {
        k_mem_slab_free(&rx_mem_slab, dummy);
    }
    
    apply_pdm_regs();
    
    /* Restart DMA capture */
    if (i2s_trigger(i2s_dev, I2S_DIR_RX, I2S_TRIGGER_START) != 0) {
        LOG_ERR("Failed to start I2S after reset!");
        return;
    }
    
    LOG_INF("I2S reset complete");
}

/*
 * i2s_mic_init: Gets the I2S device from the device tree, configures 
 *               the sampling rate, registers PDM settings, and starts capturing.
 */
bool i2s_mic_init(void)
{
    LOG_INF("[FLAG: STAGE_0_START] Initializing I2S PDM Microphone...");

    /* Get I2S device node from DTS */
    i2s_dev = DEVICE_DT_GET(DT_NODELABEL(i2s0));
    if (!device_is_ready(i2s_dev)) {
        LOG_ERR("[FLAG: ERROR] I2S device not ready!");
        return false;
    }
    LOG_INF("I2S device found: %s", i2s_dev->name);

    /* Setup I2S configuration parameters */
    struct i2s_config i2s_cfg = {
        .word_size = 16,
        .channels = 2,
        .format = I2S_FMT_DATA_FORMAT_I2S,
        .options = I2S_OPT_BIT_CLK_MASTER | I2S_OPT_FRAME_CLK_MASTER,
        .frame_clk_freq = SAMPLE_RATE * 8, // Set decimation clock frequency
        .mem_slab = &rx_mem_slab,
        .block_size = I2S_BLOCK_SIZE,
        .timeout = 1000,
    };

    if (i2s_configure(i2s_dev, I2S_DIR_RX, &i2s_cfg) != 0) {
        LOG_ERR("[FLAG: ERROR] Failed to configure I2S!");
        return false;
    }

    apply_pdm_regs();

    /* Trigger the I2S capture to start filling memory blocks via DMA */
    if (i2s_trigger(i2s_dev, I2S_DIR_RX, I2S_TRIGGER_START) != 0) {
        LOG_ERR("[FLAG: ERROR] Failed to start I2S!");
        return false;
    }

    k_sleep(K_MSEC(100));
    LOG_INF("[FLAG: SUCCESS] I2S started with PDM RX enabled!");
    return true;
}

/*
 * i2s_mic_read_block: Reads a processed PCM audio block from the DMA queue.
 *                     If reading fails (e.g. queue overflow or buffer underflow),
 *                     it triggers an automatic hardware reset to recover.
 */
int i2s_mic_read_block(void **mem_block, size_t *block_size)
{
    if (!i2s_dev) {
        return -1;
    }

    /* Force PDM register settings prior to reading to avoid hardware drift */
    apply_pdm_regs();

    int ret = i2s_read(i2s_dev, mem_block, block_size);
    
    /* Handle read failure or missing buffer pointers */
    if (ret != 0 || mem_block == NULL) {
        static int error_count = 0;
        error_count++;
        LOG_WRN("I2S read issue: ret=%d, mem_block=%p (count: %d)", ret, mem_block, error_count);
        
        /* Stop and reset the I2S capture pipeline */
        i2s_trigger(i2s_dev, I2S_DIR_RX, I2S_TRIGGER_DROP);
        k_sleep(K_MSEC(10));
        
        I2S0.rx_conf.rx_reset = 1;
        I2S0.rx_conf.rx_fifo_reset = 1;
        k_sleep(K_MSEC(10));
        I2S0.rx_conf.rx_reset = 0;
        I2S0.rx_conf.rx_fifo_reset = 0;
        
        apply_pdm_regs();
        
        if (i2s_trigger(i2s_dev, I2S_DIR_RX, I2S_TRIGGER_START) != 0) {
            LOG_ERR("Failed to start I2S after reset!");
            return -1;
        }
        
        /* Retry the read after resetting the pipeline */
        ret = i2s_read(i2s_dev, mem_block, block_size);
        if (ret != 0 || mem_block == NULL) {
            LOG_ERR("I2S still failing after reset! ret=%d", ret);
            return ret;
        }
        
        LOG_INF("I2S recovered successfully!");
    }
    
    return 0;
}

/*
 * i2s_mic_free_block: Frees the processed memory block back to the mem slab pool.
 */
void i2s_mic_free_block(void *mem_block)
{
    if (mem_block) {
        k_mem_slab_free(&rx_mem_slab, mem_block);
    }
}
