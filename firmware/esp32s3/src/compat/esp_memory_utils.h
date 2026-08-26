/*
 * Copyright (c) 2026 Seeed Studio / Senior ML Engineer
 * SPDX-License-Identifier: Apache-2.0
 *
 * Zephyr compatibility wrapper for <esp_memory_utils.h>.
 * Ensures C linkage and pulls in esp_heap_caps.h.
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#if __has_include_next(<esp_memory_utils.h>)
#include_next <esp_memory_utils.h>
#elif __has_include_next("esp_memory_utils.h")
#include_next "esp_memory_utils.h"
#endif

#include "esp_heap_caps.h"

#ifdef __cplusplus
}
#endif
