/*
 * Copyright (c) 2026 Seeed Studio / Senior ML Engineer
 * SPDX-License-Identifier: Apache-2.0
 *
 * Zephyr compatibility wrapper for <esp_cpu.h>.
 * Wraps esp_cpu.h and underlying xtensa/irq headers in extern "C" when included by C++ files
 * to prevent conflicting C vs C++ linkage declarations for arch_irq_lock and IRQ primitives.
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#if __has_include_next(<esp_cpu.h>)
#include_next <esp_cpu.h>
#elif __has_include_next("esp_cpu.h")
#include_next "esp_cpu.h"
#endif

#ifdef __cplusplus
}
#endif
