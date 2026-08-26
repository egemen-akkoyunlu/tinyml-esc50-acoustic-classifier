/*
 * Copyright (c) 2026 Seeed Studio / Senior ML Engineer
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef AUDIO_HPP
#define AUDIO_HPP

#include <cstddef>

/**
 * @brief Initialize the I2S PDM microphone device and memory slabs.
 * @return true if initialization succeeded, false otherwise.
 */
bool i2s_mic_init();

/**
 * @brief Read an audio block from the I2S microphone.
 * @param mem_block Output pointer to received memory block.
 * @param block_size Output pointer to size of received block.
 * @return 0 on success, negative error code on failure.
 */
int i2s_mic_read_block(void **mem_block, size_t *block_size);

/**
 * @brief Free a previously read I2S memory block.
 * @param mem_block Memory block to free.
 */
void i2s_mic_free_block(void *mem_block);

#endif // AUDIO_HPP
