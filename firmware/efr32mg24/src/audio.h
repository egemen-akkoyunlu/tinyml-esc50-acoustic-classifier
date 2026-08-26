#ifndef AUDIO_H
#define AUDIO_H

#include <stdint.h>
#include "config.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initializes I2S hardware, clocks, DMA (LDMA), and microphone power pins.
 */
void audio_init(void);

/**
 * @brief Records raw full PCM audio samples (TOTAL_SAMPLES) from hardware microphone into RAM.
 * @return Number of captured 16-bit PCM audio samples.
 */
int audio_record_to_ram(void);

/**
 * @brief Records a continuous chunk of PCM audio samples (e.g. 1 second stride) from hardware I2S.
 * @param out_chunk Buffer to receive new PCM samples.
 * @param num_samples Number of samples to capture.
 * @return Number of captured samples.
 */
int audio_record_chunk(int16_t *out_chunk, int num_samples);

/**
 * @brief Shifts the 5-second trailing audio buffer left and appends new incoming samples.
 * @param new_chunk Pointer to new audio samples.
 * @param num_samples Number of new audio samples.
 */
void audio_update_ring_buffer(const int16_t *new_chunk, int num_samples);

/**
 * @brief Gets pointer to the current trailing 5-second PCM audio buffer.
 * @return Pointer to raw audio sample buffer (80,000 samples).
 */
int16_t* get_audio_buffer(void);

#ifdef __cplusplus
}
#endif

#endif /* AUDIO_H */