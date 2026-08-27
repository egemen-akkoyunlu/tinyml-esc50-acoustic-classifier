#include "audio.h"
#include "em_ldma.h"
#include "em_device.h"
#include "em_cmu.h"
#include "em_usart.h"
#include "em_gpio.h" 
#include <zephyr/kernel.h>
#include <zephyr/irq.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DMA_BUF_WORDS 1024

/* ============================================================================
 * AUDIO RING BUFFER CONFIGURATION
 * To eliminate linker RAM overflow and save >125 KB of microcontroller SRAM,
 * we maintain a 1-second streaming ring buffer (16,000 samples = 32 KB RAM)
 * instead of a monolithic 5-second 160 KB array.
 * ============================================================================ */
#ifndef AUDIO_RING_BUFFER_SAMPLES
#define AUDIO_RING_BUFFER_SAMPLES (SAMPLE_RATE * 1) /* 16,000 samples = 32 KB RAM */
#endif

static int16_t audio_ring_buffer[AUDIO_RING_BUFFER_SAMPLES];
static uint16_t dma_buf[DMA_BUF_WORDS] __attribute__((aligned(4)));
static LDMA_Descriptor_t dma_desc;

static void check_usart_errors(const char* stage) {
    uint32_t if_flags = USART0->IF;
    uint32_t status = USART0->STATUS;
    
    printf("\n--- [USART DONANIM HATA VE DURUM ANALIZI: %s] ---\n", stage);
    printf("  USART0->IF     = 0x%08X\n", if_flags);
    printf("  USART0->STATUS = 0x%08X\n", status);
    
    bool err = false;
    if (if_flags & (1 << 4))  { printf("  [!] RXOF : Alici FIFO Tasmasi (RX Overflow)!\n"); err = true; }
    if (if_flags & (1 << 5))  { printf("  [!] RXUF : Alici FIFO Bos Okuma (RX Underflow)!\n"); err = true; }
    if (if_flags & (1 << 6))  { printf("  [!] TXOF : Verici FIFO Tasmasi (TX Overflow)!\n"); err = true; }
    if (if_flags & (1 << 7))  { printf("  [!] TXUF : Verici Underflow (TX Underflow)!\n"); err = true; }
    if (if_flags & (1 << 8))  { printf("  [!] PERR : Parite Hatasi!\n"); err = true; }
    if (if_flags & (1 << 9))  { printf("  [!] FERR : Cerceveleme Hatasi!\n"); err = true; }
    if (if_flags & (1 << 12)) { printf("  [!] CCF  : Cakisma Hatasi!\n"); err = true; }
    
    if (!err) {
        printf("  [OK] Hicbir donanim hata bayragi aktif degil.\n");
    }
    
    printf("  Anlik Durum: RX_Aktif=%d, TX_Aktif=%d, RXDATAV=%d, RXFULL=%d\n",
           (status & (1 << 1)) ? 1 : 0,
           (status & (1 << 0)) ? 1 : 0,
           (status & (1 << 7)) ? 1 : 0,
           (status & (1 << 8)) ? 1 : 0);
    printf("-------------------------------------------------------------\n");
}

void audio_init(void) {
    printf("-> Initializing hardware-level I2S and power...\n");

    /* 1. Enable CMU Clocks */
    CMU_ClockEnable(cmuClock_USART0, true);
    CMU_ClockEnable(cmuClock_GPIO, true); 

    /* 2. Microphone Power Supply (PC08 and PC09) */
    GPIO_PinModeSet(gpioPortC, 8, gpioModePushPull, 1);
    GPIO_PinModeSet(gpioPortC, 9, gpioModePushPull, 1);

    k_sleep(K_MSEC(50)); 

    /* 3. Configure Physical GPIO Pins */
    GPIO_PinModeSet(gpioPortD, 3, gpioModePushPull, 0); /* CLK */
    GPIO_PinModeSet(gpioPortD, 5, gpioModePushPull, 0); /* WS */
    GPIO_PinModeSet(gpioPortD, 4, gpioModeInput, 0);    /* RX */

    GPIO->USARTROUTE[0].CLKROUTE = (gpioPortD << _GPIO_USART_CLKROUTE_PORT_SHIFT) | (3 << _GPIO_USART_CLKROUTE_PIN_SHIFT);
    GPIO->USARTROUTE[0].RXROUTE  = (gpioPortD << _GPIO_USART_RXROUTE_PORT_SHIFT)  | (4 << _GPIO_USART_RXROUTE_PIN_SHIFT);
    GPIO->USARTROUTE[0].CSROUTE  = (gpioPortD << _GPIO_USART_CSROUTE_PORT_SHIFT)  | (5 << _GPIO_USART_CSROUTE_PIN_SHIFT);

    /* 4. USART Route Matrix */
    GPIO->USARTROUTE[0].ROUTEEN = GPIO_USART_ROUTEEN_CLKPEN | 
                                  GPIO_USART_ROUTEEN_RXPEN  | 
                                  GPIO_USART_ROUTEEN_CSPEN;

    /* 5. I2S Configuration: W32D16 Stereo */
    USART_InitI2s_TypeDef init = USART_INITI2S_DEFAULT;
    init.sync.baudrate = 1024000;
    init.sync.enable = usartEnable; 
    init.sync.master = true;
    init.format = usartI2sFormatW32D16;
    init.mono = false;
    init.sync.autoTx = true;
    init.sync.autoCsEnable = true;
    USART_InitI2s(USART0, &init);

    /* Force 16-bit databits */
    USART0->FRAME = (USART0->FRAME & ~_USART_FRAME_DATABITS_MASK) | USART_FRAME_DATABITS_SIXTEEN;

    USART_Tx(USART0, 0x00000000);

    /* Enable Clock Auto-TX */
    USART0->CTRL |= USART_CTRL_AUTOTX;
    USART_Enable(USART0, usartEnable); 
    USART0->TXDATA = 0x0000; 

    printf("-> [OK] I2S Initialized: W32D16 Stereo mode active.\n");
    
    /* 6. Initialize LDMA Controller */
    CMU_ClockEnable(cmuClock_LDMA, true);
    LDMA_Init_t ldma_init = LDMA_INIT_DEFAULT;
    LDMA_Init(&ldma_init);
    NVIC_DisableIRQ(LDMA_IRQn);
}

static int internal_record_to_buffer(int16_t *buffer, int num_samples) {
    memset(dma_buf, 0, sizeof(dma_buf));
    USART0->IF_CLR = 0xFFFFFFFF; 
    
    k_sleep(K_MSEC(10));

    USART0->CMD = USART_CMD_CLEARRX;
    
    LDMA_TransferCfg_t transfer_cfg = LDMA_TRANSFER_CFG_PERIPHERAL(ldmaPeripheralSignal_USART0_RXDATAV);

    dma_desc = (LDMA_Descriptor_t)LDMA_DESCRIPTOR_LINKREL_P2M_WORD(
        (void*)&(USART0->RXDOUBLE), 
        dma_buf,                  
        DMA_BUF_WORDS,
        0
    );
    dma_desc.xfer.size = ldmaCtrlSizeHalf;     /* 16-BIT TRANSFER */
    dma_desc.xfer.dstInc = ldmaCtrlSrcIncOne;  

    /* Synchronize to Left Channel (WS=0) start edge */
    while (GPIO_PinInGet(gpioPortD, 5) == 0);
    while (GPIO_PinInGet(gpioPortD, 5) == 1);

    USART0->CMD = USART_CMD_CLEARRX;
    LDMA_StartTransfer(0, &transfer_cfg, &dma_desc);
    
    int read_idx = 0;
    int interrupt_key = irq_lock();
    int catchup_count = 0;

    for (int i = 0; i < num_samples; ) {
        uint32_t current_dst = LDMA->CH[0].DST;
        int write_idx = ((current_dst - (uint32_t)dma_buf) / 2) & (DMA_BUF_WORDS - 1);
        int available = (write_idx - read_idx + DMA_BUF_WORDS) & (DMA_BUF_WORDS - 1);

        if (available > DMA_BUF_WORDS - 8) {
            catchup_count++;
            read_idx = (write_idx - 8 + DMA_BUF_WORDS) & (DMA_BUF_WORDS - 1);
            read_idx &= ~1; 
            available = 8;
        }

        if (available >= 2) {
            int16_t raw_l = (int16_t)dma_buf[read_idx];
            int16_t raw_r = (int16_t)dma_buf[read_idx + 1];
            read_idx = (read_idx + 2) & (DMA_BUF_WORDS - 1);
            
            /* Automatically select active microphone channel (Left vs Right) */
            int16_t raw_sample = (abs((int)raw_l) >= abs((int)raw_r)) ? raw_l : raw_r;
            
            /* Software Digital Gain with Saturation Protection */
            int32_t amplified = (int32_t)raw_sample * AUDIO_SOFTWARE_GAIN_MULTIPLIER;
            if (amplified > 32767) amplified = 32767;
            if (amplified < -32768) amplified = -32768;
            buffer[i] = (int16_t)amplified;
            i++;
        } else {
            irq_unlock(interrupt_key);
            k_busy_wait(20); 
            interrupt_key = irq_lock();
        }
    }

    irq_unlock(interrupt_key);
    LDMA_StopTransfer(0);
    return num_samples;
}

int audio_record_to_ram(void) {
    memset(audio_ring_buffer, 0, sizeof(audio_ring_buffer));
    return internal_record_to_buffer(audio_ring_buffer, AUDIO_RING_BUFFER_SAMPLES);
}

int audio_record_chunk(int16_t *out_chunk, int num_samples) {
    return internal_record_to_buffer(out_chunk, num_samples);
}

void audio_update_ring_buffer(const int16_t *new_chunk, int num_samples) {
    if (num_samples >= AUDIO_RING_BUFFER_SAMPLES) {
        memcpy(audio_ring_buffer, new_chunk + (num_samples - AUDIO_RING_BUFFER_SAMPLES), sizeof(int16_t) * AUDIO_RING_BUFFER_SAMPLES);
        return;
    }
    
    int keep_samples = AUDIO_RING_BUFFER_SAMPLES - num_samples;
    /* Shift old audio left */
    memmove(audio_ring_buffer, audio_ring_buffer + num_samples, sizeof(int16_t) * keep_samples);
    /* Append new audio chunk at tail */
    memcpy(audio_ring_buffer + keep_samples, new_chunk, sizeof(int16_t) * num_samples);
}

int16_t* get_audio_buffer(void) {
    return audio_ring_buffer;
}