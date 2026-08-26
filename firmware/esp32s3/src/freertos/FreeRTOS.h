/*
 * Copyright (c) 2026 Seeed Studio / Senior ML Engineer
 * SPDX-License-Identifier: Apache-2.0
 *
 * Zephyr RTOS compatibility wrapper for ESP-IDF / FreeRTOS headers used by esp-dl.
 * Maps FreeRTOS types and task calls directly to Zephyr primitives.
 */

#pragma once

#include <zephyr/kernel.h>
#include <zephyr/types.h>

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int BaseType_t;
typedef unsigned int UBaseType_t;
typedef void * TaskHandle_t;
typedef void * SemaphoreHandle_t;

#ifndef portMAX_DELAY
#define portMAX_DELAY (0xFFFFFFFFUL)
#endif

#ifndef pdPASS
#define pdPASS (1)
#endif

#ifndef pdFAIL
#define pdFAIL (0)
#endif

// Map FreeRTOS Core/Task functions to dummy or Zephyr equivalents for single-threaded inference
static inline BaseType_t xPortGetCoreID(void) { return 0; }
static inline UBaseType_t uxTaskPriorityGet(TaskHandle_t xTask) { return 1; }
static inline TaskHandle_t xTaskGetCurrentTaskHandle(void) { return NULL; }
static inline void vTaskSuspend(TaskHandle_t xTaskHandle) { k_sleep(K_FOREVER); }
static inline void vTaskDelete(TaskHandle_t xTaskHandle) { }

// Semaphore compatibility stubs mapped to Zephyr integers/no-ops
static inline SemaphoreHandle_t xSemaphoreCreateCounting(UBaseType_t maxCount, UBaseType_t initialCount) { return (SemaphoreHandle_t)1; }
static inline void vSemaphoreDelete(SemaphoreHandle_t xSemaphore) { }
static inline BaseType_t xSemaphoreGive(SemaphoreHandle_t xSemaphore) { return pdPASS; }
static inline BaseType_t xSemaphoreTake(SemaphoreHandle_t xSemaphore, uint32_t xTicksToWait) { return pdPASS; }

// When dual-core task splitting is invoked in esp-dl, execute synchronously on the current Zephyr thread
static inline void xTaskCreatePinnedToCore(void (*pxTaskCode)(void*), const char * const pcName,
                                           const uint32_t usStackDepth, void * const pvParameters,
                                           UBaseType_t uxPriority, TaskHandle_t * const pxCreatedTask,
                                           const BaseType_t xCoreID)
{
    if (pxTaskCode) {
        pxTaskCode(pvParameters);
    }
}

#ifdef __cplusplus
}
#endif
