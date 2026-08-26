# Copyright (c) 2026 Seeed Studio / Senior ML Engineer
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20.0)

# Find Zephyr package
find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(esc50_zephyr)

# Include esp-dl path (located in parent directory)
set(ESPDL_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../streamable-kws/esp-dl/esp-dl)
set(DL_FFT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../streamable-kws/esp-dl/tools/dl_fft)

# ============================================================
# ESP-IDF HAL Paths (for PDM support)
# ============================================================
set(ESP_HAL_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../zephyrproject/modules/hal/espressif)

target_include_directories(app PRIVATE
    src/compat
    ${ESPDL_PATH}/dl
    ${ESPDL_PATH}/dl/tool/include
    ${ESPDL_PATH}/dl/tensor/include
    ${ESPDL_PATH}/dl/base
    ${ESPDL_PATH}/dl/base/isa
    ${ESPDL_PATH}/dl/base/isa/tie728
    ${ESPDL_PATH}/dl/base/isa/xtensa
    ${ESPDL_PATH}/dl/math/include
    ${ESPDL_PATH}/dl/model/include
    ${ESPDL_PATH}/dl/module/include
    ${ESPDL_PATH}/fbs_loader/include
    ${ESPDL_PATH}/audio/common
    ${ESPDL_PATH}/audio/speech_features
    ${DL_FFT_PATH}
    ${DL_FFT_PATH}/base
    ${DL_FFT_PATH}/base/isa
    ${CMAKE_CURRENT_BINARY_DIR}
    src
    # ESP-IDF HAL includes - CORRECTED PATHS
    ${ESP_HAL_PATH}/components/esp_hal_i2s/include/hal
    ${ESP_HAL_PATH}/components/esp_common/include
    ${ESP_HAL_PATH}/components/driver/include
    ${ESP_HAL_PATH}/components/esp_hw_support/include
    ${ESP_HAL_PATH}/components/esp_hw_support/include/soc
    ${ESP_HAL_PATH}/components/soc/esp32s3/include
    ${ESP_HAL_PATH}/components/soc/include
    ${ESP_HAL_PATH}/components/esp_rom/include
    ${ESP_HAL_PATH}/components/esp_rom/include/esp32s3
    ${ESP_HAL_PATH}/components/esp_private/include
    ${ESP_HAL_PATH}/components/esp_hal_i2s/include
)

target_compile_definitions(app PRIVATE
    _USE_MATH_DEFINES
    _GNU_SOURCE
    M_PI=3.14159265358979323846
    # ESP-IDF definitions for HAL
    ESP_PLATFORM=1
    SOC_ESP32S3=1
    CONFIG_IDF_TARGET_ESP32S3=1
    CONFIG_SPIRAM=1
)

file(GLOB ESPDL_SRCS
    ${ESPDL_PATH}/dl/tool/src/*.cpp
    ${ESPDL_PATH}/dl/tool/isa/tie728/*.S
    ${ESPDL_PATH}/dl/tool/isa/xtensa/*.S
    ${ESPDL_PATH}/dl/tensor/src/*.cpp
    ${ESPDL_PATH}/dl/base/*.cpp
    ${ESPDL_PATH}/dl/base/isa/tie728/*.S
    ${ESPDL_PATH}/dl/base/isa/xtensa/*.S
    ${ESPDL_PATH}/dl/math/src/*.cpp
    ${ESPDL_PATH}/dl/model/src/*.cpp
    ${ESPDL_PATH}/dl/module/src/*.cpp
    ${ESPDL_PATH}/fbs_loader/src/*.cpp
    ${ESPDL_PATH}/audio/common/*.cpp
    ${ESPDL_PATH}/audio/speech_features/*.cpp
    ${DL_FFT_PATH}/*.c
    ${DL_FFT_PATH}/base/*.c
    ${DL_FFT_PATH}/base/isa/esp32s3/*.S
)

target_sources(app PRIVATE 
    src/main.cpp 
    src/audio.cpp
    src/audio_preprocessing.cpp
    src/inference.cpp
    src/compat/posix_stubs.c
    src/compat/esp_heap_caps.cpp 
    ${ESPDL_SRCS}
)

# Link ESP-IDF HAL library if available
target_link_libraries(app PRIVATE 
    ${ESPDL_PATH}/fbs_loader/lib/esp32s3/libfbs_model.a
    "-Wl,--wrap=heap_caps_malloc"
    "-Wl,--wrap=heap_caps_free"
    "-Wl,--wrap=heap_caps_realloc"
    "-Wl,--wrap=heap_caps_calloc"
    "-Wl,--wrap=heap_caps_aligned_alloc"
    "-Wl,--wrap=heap_caps_aligned_free"
    "-Wl,--wrap=heap_caps_aligned_calloc"
    "-Wl,--wrap=heap_caps_get_free_size"
    "-Wl,--wrap=heap_caps_get_largest_free_block"
    "-Wl,--wrap=heap_caps_get_allocated_size"
    "-Wl,--wrap=heap_caps_get_total_size"
    "-Wl,--wrap=heap_caps_get_minimum_free_size"
    "-Wl,--wrap=heap_caps_get_info"
    "-Wl,--wrap=heap_caps_malloc_default"
    "-Wl,--wrap=heap_caps_realloc_default"
    "-Wl,--wrap=heap_caps_malloc_prefer"
    "-Wl,--wrap=heap_caps_realloc_prefer"
    "-Wl,--wrap=heap_caps_calloc_prefer"
    "-Wl,--wrap=heap_caps_register_failed_alloc_callback"
    "-Wl,--wrap=heap_caps_malloc_extmem_enable"
    "-Wl,--wrap=heap_caps_print_heap_info"
    "-Wl,--wrap=heap_caps_check_integrity"
    "-Wl,--wrap=heap_caps_check_integrity_all"
    "-Wl,--wrap=heap_caps_check_integrity_addr"
    "-Wl,--wrap=heap_caps_dump"
    "-Wl,--wrap=heap_caps_dump_all"
    "-Wl,--wrap=malloc"
    "-Wl,--wrap=free"
    "-Wl,--wrap=calloc"
    "-Wl,--wrap=realloc"
)

# Try to link ESP-IDF HAL libraries if they exist
if(EXISTS ${ESP_HAL_PATH}/components/esp_common/lib/esp32s3/libesp_common.a)
    target_link_libraries(app PRIVATE 
        ${ESP_HAL_PATH}/components/esp_common/lib/esp32s3/libesp_common.a
    )
endif()

if(EXISTS ${ESP_HAL_PATH}/components/driver/lib/esp32s3/libdriver.a)
    target_link_libraries(app PRIVATE 
        ${ESP_HAL_PATH}/components/driver/lib/esp32s3/libdriver.a
    )
endif()

generate_inc_file_for_target(app src/model.espdl ${CMAKE_CURRENT_BINARY_DIR}/model_espdl.inc)
