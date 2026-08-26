/*
 * Copyright (c) 2026 Seeed Studio / Senior ML Engineer
 * SPDX-License-Identifier: Apache-2.0
 *
 * @file esp_partition.h
 * @brief Zephyr RTOS compatibility wrapper for ESP-IDF's esp_partition.h.
 */

#pragma once

#include <zephyr/kernel.h>
#include <stddef.h>
#include <stdint.h>
#include "spi_flash_mmap.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef uint32_t esp_partition_mmap_handle_t;

typedef struct {
    uint32_t size;      /**< Partition size in bytes */
    const char *label;  /**< Partition text label */
} esp_partition_t;

typedef enum {
    ESP_PARTITION_TYPE_DATA = 0
} esp_partition_type_t;

typedef enum {
    ESP_PARTITION_SUBTYPE_ANY = 0
} esp_partition_subtype_t;

typedef spi_flash_mmap_memory_t esp_partition_mmap_memory_t;
#define ESP_PARTITION_MMAP_DATA SPI_FLASH_MMAP_DATA

static inline const esp_partition_t *esp_partition_find_first(esp_partition_type_t type,
                                                              esp_partition_subtype_t subtype,
                                                              const char *label)
{
    return NULL;
}

static inline int esp_partition_mmap(const esp_partition_t *partition, uint32_t offset,
                                     uint32_t size, esp_partition_mmap_memory_t memory,
                                     const void **out_ptr, esp_partition_mmap_handle_t *out_handle)
{
    *out_ptr = (const void *)offset;
    *out_handle = 1;
    return 0;
}

static inline void esp_partition_munmap(esp_partition_mmap_handle_t handle)
{
    (void)handle;
}

#ifdef __cplusplus
}
#endif
