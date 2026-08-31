cmake_minimum_required(VERSION 3.20.0)
find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(silabs_ble_audio_peripheral)

# Module source files for audio, preprocessing, TFLM inference, and BLE
target_sources(app PRIVATE 
    src/main.cpp 
    src/audio.c 
    src/audio_preprocessing.c
    src/inference.cpp
    src/cmsis_nn_cnn_engine.cpp
    src/tcn_inference_engine.cpp
    src/ble_server.c
)

# EMLIB hardware USART and LDMA drivers
target_sources(app PRIVATE /home/acar/zephyrproject/modules/hal/silabs/gecko/emlib/src/em_usart.c)
target_sources(app PRIVATE /home/acar/zephyrproject/modules/hal/silabs/gecko/emlib/src/em_ldma.c)
