#pragma once

#include <zephyr/kernel.h>
#include <stdlib.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Define the macros that ESP-DL and Espressif HAL expect
#define MALLOC_CAP_EXEC             (1<<0)
#define MALLOC_CAP_32BIT            (1<<1)
#define MALLOC_CAP_8BIT             (1<<2)
#define MALLOC_CAP_DMA              (1<<3)
#define MALLOC_CAP_PID2             (1<<4)
#define MALLOC_CAP_PID3             (1<<5)
#define MALLOC_CAP_PID4             (1<<6)
#define MALLOC_CAP_PID5             (1<<7)
#define MALLOC_CAP_PID6             (1<<8)
#define MALLOC_CAP_PID7             (1<<9)
#define MALLOC_CAP_SPIRAM           (1<<10)
#define MALLOC_CAP_INTERNAL         (1<<11)
#define MALLOC_CAP_DEFAULT          (1<<12)
#define MALLOC_CAP_IRAM_8BIT        (1<<13)
#define MALLOC_CAP_RETENTION        (1<<14)
#define MALLOC_CAP_RTCRAM           (1<<15)
#define MALLOC_CAP_SPM              (1<<16)
#define MALLOC_CAP_DMA_DESC_AHB     (1<<17)
#define MALLOC_CAP_DMA_DESC_AXI     (1<<18)
#define MALLOC_CAP_CACHE_ALIGNED    (1<<19)
#define MALLOC_CAP_SIMD             (1<<20)
#define MALLOC_CAP_SPIRAM_NO_ENC    (1<<21)
#define MALLOC_CAP_INVALID          (1<<31)

#ifndef HEAP_IRAM_ATTR
#define HEAP_IRAM_ATTR
#endif

typedef void (*esp_alloc_failed_hook_t)(size_t size, uint32_t caps, const char * function_name);

typedef struct {
    size_t total_free_bytes;
    size_t total_allocated_bytes;
    size_t largest_free_block;
    size_t minimum_free_bytes;
    size_t allocated_blocks;
    size_t free_blocks;
    size_t total_blocks;
} multi_heap_info_t;

// Function declarations
extern int heap_caps_register_failed_alloc_callback(esp_alloc_failed_hook_t callback);
extern void* heap_caps_malloc(size_t size, uint32_t caps);
extern void heap_caps_malloc_extmem_enable(size_t limit);
extern void* heap_caps_malloc_default(size_t size);
extern void* heap_caps_malloc_prefer(size_t size, size_t num, ...);
extern void heap_caps_free(void* ptr);
extern void* heap_caps_realloc(void* ptr, size_t size, uint32_t caps);
extern void* heap_caps_realloc_default(void* ptr, size_t size);
extern void* heap_caps_realloc_prefer(void* ptr, size_t size, size_t num, ...);
extern void* heap_caps_calloc(size_t n, size_t size, uint32_t caps);
extern void* heap_caps_calloc_prefer(size_t n, size_t size, size_t num, ...);
extern void* heap_caps_aligned_alloc(size_t alignment, size_t size, uint32_t caps);
extern void heap_caps_aligned_free(void* ptr);
extern void* heap_caps_aligned_calloc(size_t alignment, size_t n, size_t size, uint32_t caps);
extern size_t heap_caps_get_largest_free_block(uint32_t caps);
extern size_t heap_caps_get_free_size(uint32_t caps);
extern size_t heap_caps_get_total_size(uint32_t caps);
extern size_t heap_caps_get_minimum_free_size(uint32_t caps);
extern void heap_caps_get_info(multi_heap_info_t *info, uint32_t caps);
extern void heap_caps_print_heap_info(uint32_t caps);
extern bool heap_caps_check_integrity(uint32_t caps, bool print_errors);
extern bool heap_caps_check_integrity_all(bool print_errors);
extern bool heap_caps_check_integrity_addr(intptr_t addr, bool print_errors);
extern void heap_caps_dump(uint32_t caps);
extern void heap_caps_dump_all(void);
extern size_t heap_caps_get_allocated_size(void *ptr);

#ifdef __cplusplus
}
#endif
