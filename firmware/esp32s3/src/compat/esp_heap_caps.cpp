#include "esp_heap_caps.h"
#include <zephyr/kernel.h>
#include <zephyr/multi_heap/shared_multi_heap.h>
#include <esp_memory_utils.h>
#include <string.h>
#include <mutex>
#include <stdarg.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations for wrap functions
void* __wrap_heap_caps_malloc(size_t size, uint32_t caps);
void __wrap_heap_caps_free(void* ptr);
void* __wrap_heap_caps_realloc(void* ptr, size_t size, uint32_t caps);
void* __wrap_heap_caps_calloc(size_t n, size_t size, uint32_t caps);
void* __wrap_heap_caps_aligned_alloc(size_t alignment, size_t size, uint32_t caps);
void __wrap_heap_caps_aligned_free(void* ptr);
void* __wrap_heap_caps_aligned_calloc(size_t alignment, size_t n, size_t size, uint32_t caps);
size_t __wrap_heap_caps_get_free_size(uint32_t caps);
size_t __wrap_heap_caps_get_largest_free_block(uint32_t caps);

#ifndef CONFIG_HEAP_MEM_POOL_SIZE
#define CONFIG_HEAP_MEM_POOL_SIZE 163840
#endif

#ifndef CONFIG_ESP_SPIRAM_HEAP_SIZE
#define CONFIG_ESP_SPIRAM_HEAP_SIZE 4194304
#endif

// Fixed-size heap allocation tracker to avoid early-boot C++ static initialization issues and recursive allocations
#define MAX_TRACKED_ALLOCATIONS 128

struct TrackerEntry {
    void* ptr;
    size_t size;
};

static TrackerEntry g_entries[MAX_TRACKED_ALLOCATIONS];
static int g_num_entries = 0;
static size_t g_allocated_internal = 0;
static size_t g_allocated_external = 0;

static void track_allocation(void* ptr, size_t size)
{
    if (ptr) {
        unsigned int key = irq_lock();
        
        // Update if it already exists (unlikely, but safe)
        for (int i = 0; i < g_num_entries; i++) {
            if (g_entries[i].ptr == ptr) {
                g_entries[i].size = size;
                irq_unlock(key);
                return;
            }
        }
        
        if (g_num_entries < MAX_TRACKED_ALLOCATIONS) {
            g_entries[g_num_entries++] = {ptr, size};
        }
        
        if (esp_ptr_external_ram(ptr)) {
            g_allocated_external += size;
        } else {
            g_allocated_internal += size;
        }
        irq_unlock(key);
    }
}

static void track_deallocation(void* ptr)
{
    if (ptr) {
        unsigned int key = irq_lock();
        for (int i = 0; i < g_num_entries; i++) {
            if (g_entries[i].ptr == ptr) {
                size_t size = g_entries[i].size;
                if (esp_ptr_external_ram(ptr)) {
                    if (g_allocated_external >= size) {
                        g_allocated_external -= size;
                    } else {
                        g_allocated_external = 0;
                    }
                } else {
                    if (g_allocated_internal >= size) {
                        g_allocated_internal -= size;
                    } else {
                        g_allocated_internal = 0;
                    }
                }
                
                // Swap with the last entry to remove in O(1)
                g_entries[i] = g_entries[g_num_entries - 1];
                g_num_entries--;
                irq_unlock(key);
                return;
            }
        }
        irq_unlock(key);
    }
}

static size_t get_allocation_size(void* ptr)
{
    if (!ptr) return 0;
    unsigned int key = irq_lock();
    for (int i = 0; i < g_num_entries; i++) {
        if (g_entries[i].ptr == ptr) {
            size_t size = g_entries[i].size;
            irq_unlock(key);
            return size;
        }
    }
    irq_unlock(key);
    return 0;
}

void* __wrap_heap_caps_malloc(size_t size, uint32_t caps)
{
    void* ptr = NULL;
    if (caps & MALLOC_CAP_SPIRAM) {
        ptr = shared_multi_heap_alloc(SMH_REG_ATTR_EXTERNAL, size);
    } else {
        ptr = k_malloc(size);
    }
    if (ptr) {
        track_allocation(ptr, size);
    } else if (size > 0) {
        printk("[HEAP ERROR] heap_caps_malloc(%zu, 0x%x) failed! Internal allocations: %zu, External allocations: %zu\n",
               size, caps, g_allocated_internal, g_allocated_external);
    }
    return ptr;
}

void __wrap_heap_caps_free(void* ptr)
{
    if (ptr == NULL) {
        return;
    }
    track_deallocation(ptr);
    if (esp_ptr_external_ram(ptr)) {
        shared_multi_heap_free(ptr);
    } else {
        k_free(ptr);
    }
}

void* __wrap_heap_caps_realloc(void* ptr, size_t size, uint32_t caps)
{
    if (ptr == NULL) {
        return __wrap_heap_caps_malloc(size, caps);
    }
    if (size == 0) {
        __wrap_heap_caps_free(ptr);
        return NULL;
    }

    bool is_src_psram = esp_ptr_external_ram(ptr);
    bool is_dst_psram = (caps & MALLOC_CAP_SPIRAM) != 0;

    void* new_ptr = NULL;
    if (is_src_psram == is_dst_psram) {
        size_t old_size = get_allocation_size(ptr);
        track_deallocation(ptr);
        if (is_src_psram) {
            new_ptr = shared_multi_heap_realloc(SMH_REG_ATTR_EXTERNAL, ptr, size);
        } else {
            new_ptr = k_realloc(ptr, size);
        }
        if (new_ptr) {
            track_allocation(new_ptr, size);
        } else {
            // Restore tracking for the original pointer if realloc failed
            track_allocation(ptr, old_size);
        }
    } else {
        new_ptr = __wrap_heap_caps_malloc(size, caps);
        if (new_ptr) {
            size_t old_size = get_allocation_size(ptr);
            size_t copy_size = (size < old_size) ? size : old_size;
            if (copy_size > 0) {
                memcpy(new_ptr, ptr, copy_size);
            }
            __wrap_heap_caps_free(ptr);
        }
    }
    return new_ptr;
}

void* __wrap_heap_caps_calloc(size_t n, size_t size, uint32_t caps)
{
    size_t size_bytes;
    if (__builtin_mul_overflow(n, size, &size_bytes)) {
        return NULL;
    }
    void* ptr = __wrap_heap_caps_malloc(size_bytes, caps);
    if (ptr) {
        memset(ptr, 0, size_bytes);
    }
    return ptr;
}

void* __wrap_heap_caps_aligned_alloc(size_t alignment, size_t size, uint32_t caps)
{
    void* ptr = NULL;
    if (caps & MALLOC_CAP_SPIRAM) {
        ptr = shared_multi_heap_aligned_alloc(SMH_REG_ATTR_EXTERNAL, alignment, size);
    } else {
        ptr = k_aligned_alloc(alignment, size);
        if (!ptr) {
            ptr = shared_multi_heap_aligned_alloc(SMH_REG_ATTR_EXTERNAL, alignment, size);
        }
    }
    if (ptr) {
        track_allocation(ptr, size);
    } else if (size > 0) {
        printk("[HEAP ERROR] heap_caps_aligned_alloc(align=%zu, size=%zu, 0x%x) failed! Internal allocations: %zu, External allocations: %zu\n",
               alignment, size, caps, g_allocated_internal, g_allocated_external);
    }
    return ptr;
}

void __wrap_heap_caps_aligned_free(void* ptr)
{
    __wrap_heap_caps_free(ptr);
}

void* __wrap_heap_caps_aligned_calloc(size_t alignment, size_t n, size_t size, uint32_t caps)
{
    size_t size_bytes;
    if (__builtin_mul_overflow(n, size, &size_bytes)) {
        return NULL;
    }
    void* ptr = __wrap_heap_caps_aligned_alloc(alignment, size_bytes, caps);
    if (ptr) {
        memset(ptr, 0, size_bytes);
    }
    return ptr;
}

size_t __wrap_heap_caps_get_largest_free_block(uint32_t caps)
{
    return __wrap_heap_caps_get_free_size(caps);
}

size_t __wrap_heap_caps_get_free_size(uint32_t caps)
{
    if (caps & MALLOC_CAP_SPIRAM) {
        if (CONFIG_ESP_SPIRAM_HEAP_SIZE > g_allocated_external) {
            return CONFIG_ESP_SPIRAM_HEAP_SIZE - g_allocated_external;
        }
        return 0;
    }
    if (CONFIG_HEAP_MEM_POOL_SIZE > g_allocated_internal) {
        return CONFIG_HEAP_MEM_POOL_SIZE - g_allocated_internal;
    }
    return 0;
}

size_t __wrap_heap_caps_get_allocated_size(void *ptr)
{
    return get_allocation_size(ptr);
}

size_t __wrap_heap_caps_get_total_size(uint32_t caps)
{
    if (caps & MALLOC_CAP_SPIRAM) {
        return CONFIG_ESP_SPIRAM_HEAP_SIZE;
    }
    return CONFIG_HEAP_MEM_POOL_SIZE;
}

size_t __wrap_heap_caps_get_minimum_free_size(uint32_t caps)
{
    return __wrap_heap_caps_get_free_size(caps);
}

void __wrap_heap_caps_get_info(multi_heap_info_t *info, uint32_t caps)
{
    if (info) {
        size_t free_sz = __wrap_heap_caps_get_free_size(caps);
        info->total_free_bytes = free_sz;
        info->total_allocated_bytes = (caps & MALLOC_CAP_SPIRAM) ? g_allocated_external : g_allocated_internal;
        info->largest_free_block = free_sz;
        info->minimum_free_bytes = free_sz;
        info->allocated_blocks = g_num_entries;
        info->free_blocks = 0;
        info->total_blocks = info->allocated_blocks;
    }
}

void* __wrap_heap_caps_malloc_default(size_t size)
{
    return __wrap_heap_caps_malloc(size, MALLOC_CAP_DEFAULT);
}

void* __wrap_heap_caps_realloc_default(void* ptr, size_t size)
{
    return __wrap_heap_caps_realloc(ptr, size, MALLOC_CAP_DEFAULT);
}

void* __wrap_heap_caps_malloc_prefer(size_t size, size_t num, ...)
{
    uint32_t caps = MALLOC_CAP_DEFAULT;
    if (num > 0) {
        va_list args;
        va_start(args, num);
        caps = va_arg(args, uint32_t);
        va_end(args);
    }
    return __wrap_heap_caps_malloc(size, caps);
}

void* __wrap_heap_caps_realloc_prefer(void* ptr, size_t size, size_t num, ...)
{
    uint32_t caps = MALLOC_CAP_DEFAULT;
    if (num > 0) {
        va_list args;
        va_start(args, num);
        caps = va_arg(args, uint32_t);
        va_end(args);
    }
    return __wrap_heap_caps_realloc(ptr, size, caps);
}

void* __wrap_heap_caps_calloc_prefer(size_t n, size_t size, size_t num, ...)
{
    uint32_t caps = MALLOC_CAP_DEFAULT;
    if (num > 0) {
        va_list args;
        va_start(args, num);
        caps = va_arg(args, uint32_t);
        va_end(args);
    }
    return __wrap_heap_caps_calloc(n, size, caps);
}

int __wrap_heap_caps_register_failed_alloc_callback(esp_alloc_failed_hook_t callback)
{
    return 0;
}

void __wrap_heap_caps_malloc_extmem_enable(size_t limit)
{
}

void __wrap_heap_caps_print_heap_info(uint32_t caps)
{
}

bool __wrap_heap_caps_check_integrity(uint32_t caps, bool print_errors)
{
    return true;
}

bool __wrap_heap_caps_check_integrity_all(bool print_errors)
{
    return true;
}

bool __wrap_heap_caps_check_integrity_addr(intptr_t addr, bool print_errors)
{
    return true;
}

/* ====================================================================
 * __wrap_malloc: Intercepts standard malloc calls. Redirects the allocation
 *                from the libc heap to Zephyr's larger system heap (k_malloc)
 *                and updates our tracking index.
 * ==================================================================== */
void* __wrap_malloc(size_t size)
{
    void* ptr = k_malloc(size);
    if (ptr) {
        track_allocation(ptr, size);
    }
    return ptr;
}

/* ====================================================================
 * __wrap_free: Intercepts standard free calls. Checks the address of the
 *              pointer to determine if it belongs to internal SRAM or
 *              external PSRAM, and calls the corresponding free routine.
 * ==================================================================== */
void __wrap_free(void* ptr)
{
    if (ptr == NULL) {
        return;
    }
    track_deallocation(ptr);
    if (esp_ptr_external_ram(ptr)) {
        shared_multi_heap_free(ptr);
    } else {
        k_free(ptr);
    }
}

/* ====================================================================
 * __wrap_calloc: Intercepts standard calloc calls. Allocates memory via
 *                __wrap_malloc and zero-initializes it.
 * ==================================================================== */
void* __wrap_calloc(size_t nmemb, size_t size)
{
    size_t size_bytes;
    if (__builtin_mul_overflow(nmemb, size, &size_bytes)) {
        return NULL;
    }
    void* ptr = __wrap_malloc(size_bytes);
    if (ptr) {
        memset(ptr, 0, size_bytes);
    }
    return ptr;
}

/* ====================================================================
 * __wrap_realloc: Intercepts standard realloc calls. Safely relocates
 *                 data blocks, maintaining their memory space (internal
 *                 SRAM vs external PSRAM) using our allocation tracking.
 * ==================================================================== */
void* __wrap_realloc(void* ptr, size_t size)
{
    if (ptr == NULL) {
        return __wrap_malloc(size);
    }
    if (size == 0) {
        __wrap_free(ptr);
        return NULL;
    }
    size_t old_size = get_allocation_size(ptr);
    void* new_ptr = NULL;
    if (esp_ptr_external_ram(ptr)) {
        new_ptr = shared_multi_heap_alloc(SMH_REG_ATTR_EXTERNAL, size);
    } else {
        new_ptr = k_malloc(size);
    }
    if (new_ptr) {
        track_allocation(new_ptr, size);
        size_t copy_size = (size < old_size) ? size : old_size;
        if (copy_size > 0) {
            memcpy(new_ptr, ptr, copy_size);
        }
        __wrap_free(ptr);
    }
    return new_ptr;
}

void __wrap_heap_caps_dump(uint32_t caps)
{
}

void __wrap_heap_caps_dump_all(void)
{
}


#ifdef __cplusplus
}
#endif
