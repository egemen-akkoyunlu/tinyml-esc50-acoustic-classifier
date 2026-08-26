/*
 * Copyright (c) 2026 Seeed Studio / Senior ML Engineer
 * SPDX-License-Identifier: Apache-2.0
 *
 * Zephyr compatibility wrapper for <soc/soc_caps.h>.
 * Ensures <cstdint> is loaded for uint32_t definitions and pulls in esp_heap_caps.h.
 */

#pragma once

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#ifdef __cplusplus
#include <cstdint>
#include <cstddef>
#include <cstdlib>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifdef __cplusplus
extern "C" {
#endif

#if __has_include_next(<soc/soc_caps.h>)
#include_next <soc/soc_caps.h>
#elif __has_include_next("soc/soc_caps.h")
#include_next "soc/soc_caps.h"
#endif

#include "esp_heap_caps.h"

#ifdef __cplusplus
}
#endif
