# 🎙️ Ultra-Efficient TinyML Environmental Audio Classifier (ESC-50)
### *Sub-150k Parameter Class Leader via ResNet-34 Knowledge Distillation & Real-Time Dual-Edge Deployment*

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org/)
[![Zephyr RTOS](https://img.shields.io/badge/Zephyr_RTOS-3.7+-800080.svg?style=flat&logo=zephyrproject)](https://zephyrproject.org/)
[![Accuracy](https://img.shields.io/badge/ESC--50_Validation-91.00%25-brightgreen.svg)]()
[![Hardware-ESP32-S3](https://img.shields.io/badge/ESP32--S3-161_mW_%7C_450_ms-blue.svg)]()
[![Hardware-EFR32MG24](https://img.shields.io/badge/EFR32MG24-Cortex--M33_%7C_35_KB_RAM-orange.svg)]()

---

## 🌟 Executive Summary

This repository contains the complete research, training pipeline, and embedded firmware implementation for an **Ultra-Efficient TinyML Acoustic Classifier** on the **ESC-50** dataset (50 environmental sound classes).

* **Model Footprint:** Only **124,866 parameters (~124.9k)** — `#1 in parameter efficiency for sub-150k parameter edge models`.
* **Accuracy:** **`91.00% Validation Accuracy`** (PyTorch QAT) and **`89.00% End-to-End INT8 ESP-DL`**, exceeding the human ear baseline (**81.30%**) by nearly **+10.0%**.
* **Dual-Target Real-Time Deployment:**
  1. **Espressif ESP32-S3 Sense:** Monolithic 100% INT8 ESP-DL with 128-bit Xtensa PIE SIMD Vector acceleration (`450 ms` latency, `43.5 mA / 161 mW` sustained power, **`92.0% live mic confidence`**).
  2. **Silicon Labs EFR32MG24:** 2-Stage Hybrid INT8 CNN + Float32 FPU GRU (`731 ms` latency, `~35 KB` active SRAM, **`81.0% live mic confidence`**).

---

## 📊 Dual-Platform Hardware Benchmarks

Measured on physical hardware using continuous microphone audio streams and **Otii Arc Pro** high-precision power analyzer:

| Metric | Espressif ESP32-S3 Sense | Silicon Labs EFR32MG24 (xG24-DK2601B) |
| :--- | :--- | :--- |
| **Processor Architecture** | Xtensa LX7 Dual-Core @ 160 MHz | ARM Cortex-M33 @ 78 MHz |
| **Hardware Acceleration** | Xtensa PIE (128-bit SIMD Vector Engine) | CMSIS-NN / MVP + Single-Cycle FPU |
| **Quantization Scheme** | 100% Monolithic Full-Integer INT8 | 2-Stage Hybrid (INT8 CNN + FPU GRU) |
| **Total ML Inference Time** | **`450.50 ms`** | **`731.90 ms`** |
| **DSP Feature Extraction** | **`4.38 ms`** (ESP-DL Fbank) | **`54.11 ms`** (CMSIS-DSP) |
| **Sustained Active Current** | **`43.5 mA`** (@ 3.3V) | **`~10 - 12 mA`** (@ 3.3V) |
| **Sustained Active Power** | **`161 mW`** (Otii Arc Pro) | **`~35 - 40 mW`** |
| **Energy per Inference** | **`72.4 mJ`** | **`~25.0 mJ`** |
| **Active Working SRAM** | `378 KB DRAM / 5 MB PSRAM` | **`172 KB Arena + 35 KB FPU`** |
| **Model Flash Binary** | **`146.8 KB`** (`model.espdl`) | **`14.6 KB CNN + 480 KB Weights`** |
| **Live Mic Peak Confidence** | **`92.0%`** (`keyboard typing`) | **`81.0%`** (`keyboard typing`) |

---

## 🧠 Model Architecture & Distillation

```text
 ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
 │ INPUT: Raw 16 kHz Audio (5.0 Seconds) ➔ Log-Mel Spectrogram [1, 52, 313, 1]                │
 └──────────────────────────────────────────────┬──────────────────────────────────────────────┘
                                                │
                                                ▼
 ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
 │ STAGE 1: INT8 PhiNet 2D CNN Backbone (4,160 Parameters)                                     │
 │  • Stem: Conv2D 3x3 (stride 2x2) ➔ [1, 26, 157, 16]                                         │
 │  • Block 0: Inverted Bottleneck (stride 1x2, exp 1.5) ➔ [1, 26, 79, 32]                     │
 │  • Block 1: Inverted Bottleneck (stride 2x2, exp 1.5) ➔ [1, 13, 40, 48]                     │
 │  • Pointwise 1x1 Conv Compression ➔ [1, 13, 40, 32]                                         │
 └──────────────────────────────────────────────┬──────────────────────────────────────────────┘
                                                │
                                                ▼
 ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
 │ STAGE 2: Recurrent GRU & Attention Classifier (120,706 Parameters)                          │
 │  • FPU Frequency Average Pooling (13 -> 1) ➔ [39, 32] Sequence                              │
 │  • Folded Pre-GRU Layer Normalization                                                       │
 │  • Unrolled Recurrent GRU Cell (32 In -> 160 Hidden) ➔ [39, 160] Temporal Representation    │
 │  • Softmax Self-Attention Pooling (39 Time Steps -> 1 Context Vector)                       │
 │  • Post-GRU Layer Normalization + Linear Bottleneck (160 -> 128) + ReLU                      │
 │  • 50-Class Dense Classification Head (128 -> 50 Logits)                                    │
 └─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📈 Confusion Matrix & Validation Results

Evaluated on all 400 test clips across the 50 ESC-50 classes:

![Confusion Matrix](confusion_matrix_esc50_91.png)

* **Overall Accuracy:** **91.00%** (364 / 400 correct classifications).
* **26 Classes:** **100.0% Perfect Accuracy** (8/8 clips correct).
* **Top Performance:** `keyboard typing (100%)`, `chainsaw (100%)`, `crackling fire (100%)`, `rain (100%)`, `siren (100%)`, `sea waves (100%)`.

---

## 📁 Repository Directory Structure

```text
├── models/
│   ├── best_distilled_qat_model.pth        # 91.00% PyTorch Checkpoint (~500 KB)
│   └── model.espdl                         # Monolithic INT8 ESP-DL Binary (~146.9 KB)
│
├── export/
│   ├── quantize_esc50_to_espdl.py          # ESP-PPQ Quantizer for ESP32-S3
│   ├── export_clean_cnn_int8.py            # TFLM INT8 Header Generator for EFR32
│   └── export_golden_keyboard_pcm.py       # Bit-Exact Validation Audio Exporter
│
├── training/
│   ├── train_distill_91_colab.py           # ResNet-34 Knowledge Distillation Script
│   └── qat_training.py                     # Quantization-Aware Fine-Tuning
│
├── zephyr_esc/                             # ESP32-S3 Firmware (ESP-DL + Zephyr RTOS)
│   ├── CMakeLists.txt
│   ├── prj.conf
│   ├── app.overlay
│   └── src/
│       ├── main.cpp
│       ├── inference.cpp / inference.hpp
│       ├── audio_preprocessing.cpp
│       └── model.espdl
│
├── silabs_ble_audio_peripheral/            # EFR32MG24 Firmware (TFLM + CMSIS-NN)
│   ├── CMakeLists.txt
│   ├── prj.conf
│   └── src/
│       ├── main.cpp
│       ├── inference.cpp
│       ├── audio_preprocessing.c
│       └── gru_classifier_weights.h
│
├── requirements.txt                        # Python Dependencies
├── .gitignore                              # Clean Repository Filters
└── README.md                               # Project Documentation
```

---

## 🚀 Quick Start Guide

### 1. Python Environment Setup
```bash
git clone https://github.com/<your-username>/tinyml-esc50-acoustic-classifier.git
cd tinyml-esc50-acoustic-classifier
pip install -r requirements.txt
```

### 2. Export / Quantize Model for ESP32-S3
```bash
python export/quantize_esc50_to_espdl.py
```

### 3. Build & Flash ESP32-S3 Firmware (Seeed XIAO Sense)
```bash
cd zephyr_esc
west build -b xiao_esp32s3 -p auto
west flash
minicom -D /dev/ttyACM0 -b 115200
```

### 4. Build & Flash EFR32MG24 Firmware (Silicon Labs xG24)
```bash
cd ../silabs_ble_audio_peripheral
west build -b xg24_dk2601b -p auto
west flash
minicom -D /dev/ttyACM0 -b 115200
```

---

## 📜 License
This project is licensed under the Apache 2.0 License.
