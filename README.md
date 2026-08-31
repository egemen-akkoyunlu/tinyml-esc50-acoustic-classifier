# 🎙️ Ultra-Efficient TinyML Environmental Audio Classifier (ESC-50)
### *Sub-50 KB Flash & Real-Time Acoustic Classification on ARM Cortex-M33 & ESP32-S3*

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org/)
[![Zephyr RTOS](https://img.shields.io/badge/Zephyr_RTOS-3.7+-800080.svg?style=flat&logo=zephyrproject)](https://docs.zephyrproject.org/)
[![Accuracy](https://img.shields.io/badge/ESC--50_Validation-91.50%25-brightgreen.svg)]()
[![Hardware-ESP32-S3](https://img.shields.io/badge/ESP32--S3-144_ms_%7C_128_bit_SIMD-blue.svg)]()
[![Hardware-EFR32MG24](https://img.shields.io/badge/EFR32MG24-Cortex--M33_%7C_414_ms_%7C_47_KB_Flash-orange.svg)]()

---

## Overview

This repository contains the training pipeline, quantization workflows, and embedded firmware implementations for an **Environmental Audio Classifier** evaluated on the **ESC-50** dataset (50 environmental sound classes).

* **Sub-50 KB Model Option:** **`47.57 KB Flash Weights`** — executes directly from internal microcontroller Flash without requiring external SPI Flash.
* **Accuracy:** **`91.50% Validation Accuracy`** (366/400 correct classifications on held-out test audio), exceeding the human listener baseline (**81.30%**) by **+10.20%**.
* **Zero-TFLM Memory Overhead:** Custom native C++ inference engines eliminate the 172 KB TensorFlow Lite Micro arena, executing inside a **96.3 KB shared ping-pong RAM buffer** (leaving >136 KB free SRAM).
* **Dual-Target Real-Time Deployment:**
  1. **Silicon Labs EFR32MG24 (ARM Cortex-M33 @ 78 MHz):** Register-Tile Cached INT8 2D CNN + 1D Dilated TC-ResNet with `__SMLAD` Dual-MAC SIMD (**`414.24 ms` total ML latency**, **`87% live microphone confidence`**).
  2. **Espressif ESP32-S3 Sense (Xtensa LX7 @ 160 MHz):** 128-bit Xtensa PIE SIMD Vector acceleration with ESP-DL (**`144.22 ms` total ML latency**, **`98.3% live microphone confidence`**).

---

### 📊 Master Benchmark: All 8 Hardware Profiles

All metrics measured live on physical silicon with real audio streams:

| # | Profile & Target Architecture | Accuracy | Model Flash | Firmware Flash | Total SRAM | Stage 1 (2D CNN) | Stage 2 (Seq Head) | Total ML Latency | Key Optimization |
| :-: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | **`PROFILE_HIGH_ACCURACY_DENSE_91`**<br>*(EFR32MG24 - M33 @ 78 MHz)* | **91.50%** | 866.0 KB | 1,120 KB *(73%)* | 212.0 KB *(83%)* | 274.3 ms | 490.2 ms | **764.49 ms** | TFLM Interpreter (172 KB Arena) + FP32 Dense GRU |
| **2** | **`PROFILE_SPARSE_PRUNED_48K`**<br>*(EFR32MG24 - M33 @ 78 MHz)* | **88.50%** | 620.0 KB | 875 KB *(57%)* | 212.0 KB *(83%)* | 274.2 ms | 373.1 ms | **647.32 ms** | TFLM + CSR Zero-Skipping FPU GRU |
| **3** | **`PROFILE_INT8_FIXED_SIMD_91`**<br>*(EFR32MG24 - M33 @ 78 MHz)* | **91.00%** | 720.0 KB | 975 KB *(64%)* | 212.0 KB *(83%)* | 274.3 ms | 310.5 ms | **584.80 ms** | TFLM + Fixed-Point INT8/INT16 SIMD GRU (`__SMLAD`) |
| **4** | **`PROFILE_CMSIS_NN_PINGPONG_91`**<br>*(EFR32MG24 - M33 @ 78 MHz)* | **91.50%** | 866.0 KB | 1,090 KB *(71%)* | 145.3 KB *(57%)* | 274.3 ms | 490.2 ms | **764.49 ms** | Native CMSIS-NN Ping-Pong Buffer (Zero TFLM Arena) |
| **5** | **`PROFILE_TCN_85` (Standard)**<br>*(EFR32MG24 - M33 @ 78 MHz)* | **85.25%** | 92.86 KB | 468 KB *(30%)* | **119.38 KB *(46%)*** | 271.79 ms | 392.85 ms | **664.64 ms** | 1D Dilated TC-ResNet (93.7k INT8) |
| **6** | **`PROFILE_SLIM_TCN_81` (Slim Pruned)**<br>*(EFR32MG24 - M33 @ 78 MHz)* | **80.50%** | **`47.57 KB`** | **398 KB *(25%)*** | **119.38 KB *(46%)*** | **216.77 ms** | **197.48 ms** | **`414.24 ms`** | 1D Dilated TC-ResNet + Channel Pruning (<50 KB Flash) |
| **7** | **`PROFILE_ESP32S3_HIGH_ACC_89`**<br>*(ESP32-S3 - LX7 @ 160 MHz)* | **89.00%** | **146.80 KB** | 1,110 KB *(13%)* | 378.00 KB *(95%)* | — | — | **`450.50 ms`** | ESP-DL Monolithic INT8 + 128-bit Vector PIE SIMD |
| **8** | **`PROFILE_ESP32S3_SLIM_TCN_79`**<br>*(ESP32-S3 - LX7 @ 160 MHz)* | **78.75%** | **`82.78 KB`** | **1,046 KB *(12%)*** | **182.28 KB *(46%)*** | — | — | **`144.22 ms`** | ESP-DL 128-bit PIE SIMD INT8 + DRAM Optimization |

---

## 🧠 Architectural Highlights

```text
 ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
 │ INPUT: Raw 16 kHz Audio (5.0s) ➔ Mel Spectrogram [52 Mel Bins x 313 Time Steps] (Signed INT8)│
 └──────────────────────────────────────────────┬──────────────────────────────────────────────┘
                                                │
                                                ▼
 ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
 │ STAGE 1: INT8 2D PhiNet Backbone (16 -> 24/32 Channels)                                     │
 │  • Layer 1 Stem Conv2D 3x3 (s=2x2): 9-Register In-CPU Tile Caching                          │
 │  • Layer 2/4 Depthwise Conv2D 3x3: Direct Row-Pointer Addressing                            │
 │  • Layer 3/5 Pointwise Conv2D 1x1: Cortex-M33 __SMLAD Dual-MAC SIMD                         │
 │  • Output: [13 Frequency Bins x 40 Time Frames x 24 Channels]                              │
 └──────────────────────────────────────────────┬──────────────────────────────────────────────┘
                                                │
                                                ▼
 ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
 │ 2D-TO-1D RESHAPE & SUB-BAND POOLING: 13 Bins -> 4 Sub-Bands x 24 Channels = 96 Channels     │
 │  • Sub-Band 0 (0 - 1.2 kHz)  : Low frequencies (Knocks, footsteps, thunder)                 │
 │  • Sub-Band 1 (1.2 - 2.8 kHz): Mid frequencies (Speech, animal vocalizations)               │
 │  • Sub-Band 2 (2.8 - 4.8 kHz): Upper-Mid (Keyboard typing, glass clicks, bell rings)       │
 │  • Sub-Band 3 (4.8 - 8.0 kHz): High frequencies (Crickets, hiss, harmonics)                │
 └──────────────────────────────────────────────┬──────────────────────────────────────────────┘
                                                │
                                                ▼
 ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
 │ STAGE 2: 5-Stage 1D Dilated Depthwise-Separable TC-ResNet (96 -> 64 -> 64 -> 64 -> 64 -> 96)│
 │  • Stage 0 (Dilation = 1)  : DW 3x1 (96 ch) + PW 1x1 (96->64) + Shortcut (96->64)          │
 │  • Stage 1 (Dilation = 2)  : DW 3x1 (64 ch) + PW 1x1 (64->64) + Identity Skip              │
 │  • Stage 2 (Dilation = 4)  : DW 3x1 (64 ch) + PW 1x1 (64->64) + Identity Skip              │
 │  • Stage 3 (Dilation = 8)  : DW 3x1 (64 ch) + PW 1x1 (64->64) + Identity Skip              │
 │  • Stage 4 (Dilation = 16) : DW 3x1 (64 ch) + PW 1x1 (64->96) + Shortcut (64->96)          │
 └──────────────────────────────────────────────┬──────────────────────────────────────────────┘
                                                │
                                                ▼
 ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
 │ STAGE 3: Learned Temporal Softmax Attention & Classifier Head                               │
 │  • Softmax Temporal Attention (40 Time Steps -> 1 Context Vector [96])                     │
 │  • Post-TCN BatchNorm + Linear Bottleneck (96 -> 48) + ReLU                                 │
 │  • 50-Class Dense Classification Head (48 -> 50 Logits)                                     │
 └─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## ⚡ Key Optimizations

### 1. 🪟 3x3 Register-Tile Caching (Stage 1 Acceleration)
In standard 2D convolutions, the CPU re-reads the $3\times3$ input patch for every output channel, generating 587,808 non-contiguous RAM accesses.  
By pinning the $3\times3$ window into 9 CPU registers (`p0..p8`) and evaluating all 16 channels via `#pragma GCC unroll 16`, memory bus traffic is reduced by $16\times$ (down to 36,738 reads), lowering Stage 1 latency from `355.13 ms` to `216.77 ms`.

### 2. ✂️ Structured Channel Pruning (<50 KB Flash Tier)
By pruning the intermediate feature channels from $128 \to 96 \to 64$, model weights shrunk from **92.86 KB $\to$ 47.57 KB ($-48.8\%$)**, allowing the model to run directly from internal microcontroller Flash with zero external storage requirements.

### 3. 🎯 Bit-Exact DSP Front-End Alignment
The on-chip CMSIS-DSP preprocessing pipeline (Radix-4 RFFT, 52 Mel filterbanks, Log-Mel compression) was aligned bit-exact with PyTorch Torchaudio transforms, ensuring that validation accuracy translates directly to physical microphone hardware.

---

## 📁 Repository Directory Structure

```text
├── models/
│   ├── best_tcn_slim_qat.pth               # 80.50% Slim TCN Checkpoint (47.57 KB Flash)
│   ├── best_tcn_base_qat.pth               # 85.25% Standard TCN Checkpoint (92.86 KB Flash)
│   └── best_distilled_qat_model.pth        # 91.50% High-Accuracy PyTorch Checkpoint
│
├── training/
│   ├── train_tcn_base_local.py             # Phase 1: Standard 93.7k 1D TC-ResNet QAT
│   ├── train_tcn_channel_prune.py          # Phase 2: Channel Pruning (<50 KB Slim Model)
│   ├── eval_quantized_tcn_benchmark.py     # Benchmark & Exporter for Standard TCN (85.25%)
│   └── eval_quantized_slim_tcn.py          # Benchmark & Exporter for Slim TCN (80.50%)
│
├── export/
│   ├── quantize_slim_tcn_to_espdl.py       # ESP-PPQ INT8 Quantization Compiler for ESP32-S3
│   └── quantize_esc50_to_espdl.py          # ESP-PPQ Quantizer for 89% CRNN Model
│
├── firmware/
│   ├── esp32s3/                            # Seeed Studio XIAO ESP32-S3 Sense Project
│   │   ├── CMakeLists.txt
│   │   ├── prj.conf
│   │   └── src/ (main.cpp, audio.cpp, inference.cpp, model.espdl)
│   │
│   └── efr32mg24/                          # Silicon Labs xG24-DK2601B Project
│       ├── CMakeLists.txt
│       ├── prj.conf
│       └── src/
│           ├── config.h                    # Profile Selector
│           ├── tcn_inference_engine.cpp    # 3x3 Register-Tile Cached INT8 TCN Engine
│           ├── tcn_inference_engine.h      # Engine Header Interface
│           ├── tcn_slim_classifier_weights_int8_81.h # 47.57 KB Flash Weights (Profile 6)
│           ├── tcn_classifier_weights_int8_85.h      # 92.86 KB Flash Weights (Profile 5)
│           ├── audio_preprocessing.c       # Bit-Exact CMSIS-DSP FFT Audio Frontend
│           └── inference.cpp               # Master Zephyr Pipeline & Logging
│
├── requirements.txt                        # Python Dependencies
├── .gitignore                              # Clean Repository Filters
└── README.md                               # Project Documentation
```

---

## 🚀 Quickstart & Reproduction Guide

### 1. Training & Exporting Slim TCN (<50 KB)
```bash
# Clone repository
git clone https://github.com/egemen-akkoyunlu/tinyml-esc50-acoustic-classifier.git
cd tinyml-esc50-acoustic-classifier
pip install -r requirements.txt

# Run Structured Channel Pruning & QAT
python training/train_tcn_channel_prune.py

# Benchmark and Export C Header for EFR32MG24
python training/eval_quantized_slim_tcn.py
# Generates: firmware/efr32mg24/src/tcn_slim_classifier_weights_int8_81.h

# Compile INT8 Model for ESP32-S3 via ESP-PPQ
python export/quantize_slim_tcn_to_espdl.py
# Generates: firmware/esp32s3/src/model.espdl
```

### 2. Building & Flashing EFR32MG24 (Silicon Labs xG24-DK2601B)
```bash
cd firmware/efr32mg24

# Build with Zephyr RTOS
west build -b xg24_dk2601b

# Flash to target board
west flash

# Open UART monitor (115200 baud)
screen /dev/ttyACM0 115200
```

### 3. Building & Flashing ESP32-S3 (Seeed Studio XIAO ESP32-S3 Sense)
```bash
cd firmware/esp32s3

# Build with Zephyr RTOS (160 MHz Core Clock)
west build -b xiao_esp32s3/esp32s3/procpu/sense

# Flash to target board
west flash

# Open UART monitor (115200 baud)
screen /dev/ttyACM0 115200
```

---

## 🔗 Framework & Reference Documentation

* **[Zephyr RTOS Getting Started Guide](https://docs.zephyrproject.org/latest/develop/getting_started/index.html)** — Official installation, West workspace setup, and toolchain installation.
* **[Espressif ESP-DL Repository](https://github.com/espressif/esp-dl)** — High-performance deep learning library optimized for ESP32 series with 128-bit Xtensa PIE SIMD.
* **[Espressif ESP-PPQ Quantization Suite](https://github.com/espressif/esp-ppq)** — Quantization-aware training and calibration toolchain for ESP-DL target deployment.
* **[ARM CMSIS-DSP & CMSIS-NN](https://github.com/ARM-software/CMSIS-NN)** — Optimized neural network kernels and DSP transforms for ARM Cortex-M processors.

---

## 👤 Author & Affiliations

**Egemen Acar Akkoyunlu**  
* 🏛️ **Electrical and Electronics Engineering**, Boğaziçi University, Istanbul, Turkey  
* 🔬 **Research Intern**, Fondazione Bruno Kessler (FBK), Trento, Italy  
* 🐙 **GitHub:** [@egemen-akkoyunlu](https://github.com/egemen-akkoyunlu)

---

## 📜 License
This project is licensed under the Apache 2.0 License.
