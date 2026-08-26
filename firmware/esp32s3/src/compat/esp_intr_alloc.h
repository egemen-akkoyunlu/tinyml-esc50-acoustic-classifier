/*
 * Copyright (c) 2026 Seeed Studio / Senior ML Engineer
 * SPDX-License-Identifier: Apache-2.0
 *
 * Zephyr compatibility wrapper for <esp_intr_alloc.h>.
 * Ensures C linkage when included from C++ files.
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#if __has_include_next(<esp_intr_alloc.h>)
#include_next <esp_intr_alloc.h>
#elif __has_include_next("esp_intr_alloc.h")
#include_next "esp_intr_alloc.h"
#endif

#ifdef __cplusplus
}
#endif
