/*
 * Copyright (c) 2026 Seeed Studio / Senior ML Engineer
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    SPI_FLASH_MMAP_DATA = 0
} spi_flash_mmap_memory_t;

static inline int spi_flash_mmap_get_free_pages(spi_flash_mmap_memory_t memory)
{
    (void)memory;
    return 1024;
}

#ifdef __cplusplus
}
#endif
