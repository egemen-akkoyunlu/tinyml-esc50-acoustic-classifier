#pragma once
#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>

#define ESP_LOGE(tag, format, ...) printk("[ERROR] [%s] " format "\n", tag, ##__VA_ARGS__)
#define ESP_LOGI(tag, format, ...) printk("[INFO] [%s] " format "\n", tag, ##__VA_ARGS__)
#define ESP_LOGW(tag, format, ...) printk("[WARN] [%s] " format "\n", tag, ##__VA_ARGS__)
#define ESP_LOGD(tag, format, ...) printk("[DEBUG] [%s] " format "\n", tag, ##__VA_ARGS__)
#define ESP_LOGV(tag, format, ...) printk("[VERBOSE] [%s] " format "\n", tag, ##__VA_ARGS__)

#ifdef __cplusplus
extern "C" {
#endif

// ESP logging functions needed by FBS library
static inline uint32_t esp_log_timestamp(void) { 
    return k_uptime_get_32(); 
}

static inline void esp_log_write(int level, const char* tag, const char* format, ...) {
    // Zephyr already handles this via printk in the macros
}

#ifdef __cplusplus
}
#endif
