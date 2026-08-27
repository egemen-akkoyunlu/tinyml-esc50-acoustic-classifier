# 🎙️ Ultra-Efficient TinyML Environmental Audio Classifier (ESC-50)
### *Sub-150k Parameter Architecture via ResNet-34 Knowledge Distillation & Real-Time Dual-Edge Deployment*

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org/)
[![Zephyr RTOS](https://img.shields.io/badge/Zephyr_RTOS-3.7+-800080.svg?style=flat&logo=zephyrproject)](https://zephyrproject.org/)
[![Accuracy](https://img.shields.io/badge/ESC--50_Validation-90.50%25-brightgreen.svg)]()
[![Hardware-ESP32-S3](https://img.shields.io/badge/ESP32--S3-161_mW_%7C_450_ms-blue.svg)]()
[![Hardware-EFR32MG24](https://img.shields.io/badge/EFR32MG24-Cortex--M33_%7C_35_KB_RAM-orange.svg)]()
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

---

## 🌟 Executive Summary

This repository contains the complete research, training pipeline, and embedded firmware implementation for an **Ultra-Efficient TinyML Acoustic Classifier** on the **ESC-50** dataset (50 environmental sound classes).

* **Model Footprint:** Only **124,866 parameters (~124.9k)** — designed specifically for resource-constrained microcontrollers (fitting in <210 KB active working SRAM).
* **Accuracy:** **`90.50% Validation Accuracy`** (PyTorch QAT: 362/400) and **`89.00% On-Chip INT8 ESP-DL`** (356/400), exceeding the human ear baseline (**81.30%**) by **+9.20%** (QAT) and **+7.70%** (INT8).
* **Dual-Target Real-Time Deployment:**
  1. **Espressif ESP32-S3 Sense:** Monolithic INT8 ESP-DL with 128-bit Xtensa PIE SIMD Vector acceleration (`450 ms` latency, `43.5 mA / 161 mW` sustained power, **`92.0% live mic confidence`**).
  2. **Silicon Labs EFR32MG24:** 2-Stage Hybrid INT8 CNN + Float32 FPU GRU (`731 ms` latency, `172 KB Arena + ~35 KB FPU RAM`, **`81.0% live mic peak confidence`**).

---

## 📊 Multi-Target Hardware & Profile Benchmarks

Measured directly on physical edge hardware across live audio streams and held-out validation sets:

| Metric | ⚡ Espressif ESP32-S3 Sense | 🏆 Silicon Labs EFR32MG24 (Flagship Dense 91) | 🚀 Silicon Labs EFR32MG24 (Sparse Pruned CSR) |
| :--- | :--- | :--- | :--- |
| **Processor Architecture** | Xtensa LX7 Dual-Core @ 160 MHz | ARM Cortex-M33 @ 78 MHz | ARM Cortex-M33 @ 78 MHz |
| **Hardware Acceleration** | Xtensa PIE (128-bit SIMD Vector Engine) | CMSIS-NN / MVP + Single-Cycle FPU | CMSIS-NN / MVP + Branchless CSR Zero-Skipping FPU |
| **Active Model Parameters** | 124,866 (100% Dense INT8) | 124,866 (100% Dense Hybrid) | **`48,874 Active Non-Zeros`** (61.0% Sparsity) |
| **Model Quantization Scheme** | Full-Integer INT8 Layers + FP32 Softmax | 2-Stage Hybrid (INT8 CNN + Dense FPU GRU) | 2-Stage Hybrid (INT8 CNN + Sparse CSR FPU GRU) |
| **Pre-Quant PyTorch QAT Accuracy** | **`90.50%`** (362 / 400 test clips) | **`90.50%`** (362 / 400 test clips) | **`88.50%`** (354 / 400 test clips) 🌟 |
| **Post-Quantization On-Chip Accuracy** | **`89.00%`** (356 / 400 test clips) | **`90.25%`** (361 / 400 test clips) | **`88.50%`** (354 / 400 test clips) |
| **Stage 2 GRU Latency (Cortex-M33)** | *(Monolithic Single-Stage)* | `490.23 ms` | **`373.08 ms`** ⚡ *(**-117.15 ms Faster!**)* |
| **Total ML Inference Time** | **`450.50 ms`** | **`764.49 ms`** | **`647.28 ms`** ⚡ *(**-117.21 ms Speedup!**)* |
| **DSP Feature Extraction** | **`4.38 ms`** (ESP-DL Fbank) | **`59.45 ms`** (CMSIS-DSP + Pre-Emphasis) | **`59.45 ms`** (CMSIS-DSP + Pre-Emphasis) |
| **Sustained Active Current / Power** | **`43.5 mA / 161 mW`** (Otii Arc Pro @ 3.3V) | *(Low-power wireless profile)* | *(Low-power wireless profile)* |
| **Firmware Flash Memory Usage** | **`146.8 KB`** (`model.espdl` in Flash) | `866 KB (55.0% of Flash)` | **`607 KB (38.6% of Flash)`** 💾 *(**-258.4 KB Reclaimed!**)* |
| **Active Working SRAM** | **`~48 KB`** *(Internal SRAM)* | **`172 KB Arena + 33.8 KB Union Pool`** <sup>†</sup> | **`172 KB Arena + 33.8 KB Union Pool`** <sup>†</sup> |
| **Live Mic Peak Confidence** | **`92.0%`** (`keyboard typing` sustained) | **`80.4%`** (`keyboard typing` peak) | **`69.9% (~70%)`** (`keyboard typing` calibrated) |

> <sup>†</sup> **Zero-BSS Memory Overlay:** By sharing the 172 KB TFLM Tensor Arena with Stage 2 working buffers via an `InferenceMemoryPool` union overlay, SRAM consumption is reduced by **33.8 KB**, holding total RAM usage at 86.7% with zero dynamic heap allocation.

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
 │ STAGE 2: Recurrent GRU & Softmax Temporal Self-Attention (120,706 Parameters)               │
 │  • FPU Frequency Average Pooling (13 -> 1) ➔ [39, 32] Sequence                              │
 │  • Folded Pre-GRU Layer Normalization                                                       │
 │  • Unrolled Recurrent GRU Cell (32 In -> 160 Hidden) ➔ [39, 160] Temporal Representation    │
 │  • Softmax Temporal Self-Attention (39 Time Steps -> 1 Context Vector)                      │
 │  • Post-GRU Layer Normalization + Linear Bottleneck (160 -> 128) + ReLU                      │
 │  • 50-Class Dense Classification Head (128 -> 50 Logits)                                    │
 └─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🎛️ Dual Deployment Profiles: High-Precision vs. Ultra-Compressed CSR

To support different hardware memory tiers, this repository features a compile-time profile selector in [`config.h`](firmware/efr32mg24/src/config.h):

| Metric | 🏆 Profile 1: **Flagship Dense Model** | ⚡ Profile 2: **Sparse Pruned CSR Model** |
| :--- | :--- | :--- |
| **Model Type** | 124.9k Dense Distilled Architecture | **`48.9k Active Non-Zeros`** (61.0% Sparsity / 77k Zeros) |
| **Execution Kernel** | Dense SIMD FPU Loops (`#pragma GCC unroll 8`) | **Branchless CSR Zero-Skipping (`sparse_matvec_mult`)** |
| **Held-Out Validation Accuracy** | **`90.50%`** (362 / 400 test clips) | **`88.50%`** (354 / 400 test clips) 🌟 |
| **Flash Memory Usage** | `866 KB (55% of Flash)` | **`607 KB (38% of Flash)`** 💾 *(**-258.4 KB Reclaimed!**)* |
| **Active Working SRAM** | `86.7%` *(with 33.8 KB Union Pool)* | `86.7%` *(with 33.8 KB Union Pool)* |
| **Stage 2 GRU Latency (Cortex-M33)** | `490.45 ms` | **`344.45 ms`** ⚡ *(**-146.0 ms Faster!**)* |
| **Total ML Inference Time** | `764.68 ms` | **`618.13 ms`** |
| **Live Microphone Confidence** | **`80.4%`** (`keyboard typing` peak) | **`75.2%`** (`keyboard typing` sustained) |
| **Target Hardware** | High-performance MCUs (1MB+ Flash) | Memory & Power Constrained Nodes (<512KB Flash) |

---

## ⚡ Compressed Sparse Row (CSR) Hardware Zero-Skipping

On the Silicon Labs EFR32MG24 (ARM Cortex-M33 @ 78 MHz), Profile 2 uses custom **Compressed Sparse Row (CSR)** encoding (`SPARSE_WEIGHTS`, `COL_INDICES`, `ROW_OFFSETS`). 

Instead of multiplying 77,448 zero-weights, the single-cycle hardware FPU skips directly to non-zero indices:

```cpp
/* Branchless CSR Matrix-Vector Multiply (inference.cpp) */
auto sparse_matvec_mult = [](
    const float *sparse_w, const uint8_t *col_idx, const uint32_t *row_offsets,
    const float *bias, const float *x, float *y, int num_rows)
{
    for (int r = 0; r < num_rows; r++) {
        float sum = bias ? bias[r] : 0.0f;
        uint32_t start = row_offsets[r];
        uint32_t end   = row_offsets[r + 1];
        #pragma GCC unroll 8
        for (uint32_t k = start; k < end; k++) {
            sum += sparse_w[k] * x[col_idx[k]]; /* ZERO-SKIPPING! */
        }
        y[r] = sum;
    }
};
```

---

## 📈 Confusion Matrix & Validation Results

### 1. 🏆 Flagship 90.50% Dense Model
![Flagship Confusion Matrix](models/confusion_matrix_esc50_91.png)

### 2. ⚡ Ultra-Compressed 88.50% Sparse Pruned Model (48.9k Parameters)
![Pruned Confusion Matrix](models/confusion_matrix_pruned_88_5.png)

* **Overall Accuracy:** **`88.50%`** (354 / 400 correct classifications).
* **30+ Classes with >87.5% - 100% Perfect Accuracy:**  
  `breathing`, `chainsaw`, `chirping birds`, `church bells`, `clock alarm`, `clock tick`, `coughing`, `cow`, `crickets`, `crow`, `crying baby`, `dog`, `door wood knock`, `fireworks`, `frog`, `glass breaking`, `hand saw`, `helicopter`, `insects`, `pig`, `pouring water`, `rain`, `rooster`, `sea waves`, `siren`, `water drops`.

---

## 📁 Repository Directory Structure

```text
├── models/
│   ├── best_distilled_qat_model.pth        # 90.50% Flagship PyTorch Checkpoint (~500 KB)
│   ├── confusion_matrix_esc50_91.png       # 50x50 Evaluation Heatmap (90.50% Flagship)
│   └── confusion_matrix_pruned_88_5.png    # 50x50 Evaluation Heatmap (88.50% Pruned CSR)
│
├── training/
│   ├── qat_training.py                     # Step 1: Base QAT Training Pipeline
│   ├── train_distill_91_colab.py           # Step 2: ResNet-34 Distillation (Flagship 90.50%)
│   ├── train_distill_91_colab.ipynb        # Step 2: Interactive Colab Notebook
│   ├── train_prune_csr_colab.py            # Step 3: L1 Pruning + Distillation (88.50% CSR)
│   └── generate_confusion_matrix.py        # Validation & Heatmap Generator
│
├── export/
│   ├── quantize_esc50_to_espdl.py          # ESP-PPQ Quantizer for ESP32-S3
│   ├── export_clean_cnn_int8.py            # TFLM INT8 Header Generator for EFR32
│   ├── export_golden_keyboard_pcm.py       # Bit-Exact Validation Audio Exporter
│   └── simulate_onchip_inference.py        # Bit-Exact CPU Simulator
│
├── firmware/
│   ├── esp32s3/                            # Seeed Studio XIAO ESP32-S3 Sense Project
│   │   ├── CMakeLists.txt
│   │   ├── prj.conf
│   │   ├── app.overlay
│   │   └── src/ (main.cpp, inference.cpp, audio_preprocessing.cpp, model.espdl)
│   │
│   └── efr32mg24/                          # Silicon Labs xG24-DK2601B Project
│       ├── CMakeLists.txt
│       ├── prj.conf
│       └── src/ (Dual Profiles: Flagship Dense & Sparse CSR Kernels)
│
├── requirements.txt                        # Python Dependencies
├── .gitignore                              # Clean Repository Filters
└── README.md                               # Project Documentation
```

---

## 🚀 Step-by-Step Reproduction Guide

### 1. Environment & Dataset Setup
```bash
git clone https://github.com/egemen-akkoyunlu/tinyml-esc50-acoustic-classifier.git
cd tinyml-esc50-acoustic-classifier
pip install -r requirements.txt

# Download and extract the ESC-50 dataset
wget https://github.com/karolpiczak/ESC-50/archive/master.zip
unzip master.zip
```

### 2. Model Training & Knowledge Distillation
The training pipeline consists of two phases:

1. **Phase 1 (Local Base Training):**  
   Train the initial PhiNet-CRNN model with Quantization-Aware Training (QAT):
   ```bash
   python training/qat_training.py
   ```

2. **Phase 2 (Cloud Distillation on GPU):**  
   Upload `training/train_distill_91_colab.py` or open [`training/train_distill_91_colab.ipynb`](training/train_distill_91_colab.ipynb) in **Google Colab** with a GPU runtime to distill knowledge from a pre-trained ResNet-34 teacher into the 124.9k student. This generates `best_distilled_qat_model.pth` (**90.50% Validation Accuracy**). Move the downloaded checkpoint to `models/best_distilled_qat_model.pth`.

### 3. Target-Specific Model Export & Quantization

Because ESP32-S3 and EFR32MG24 use different embedded acceleration engines, export follows target-specific pipelines:

#### 🔹 For Espressif ESP32-S3 (Monolithic INT8 ESP-DL):
Quantizes the full model into an INT8 binary accelerated by the 128-bit Xtensa PIE SIMD vector engine:
```bash
python export/quantize_esc50_to_espdl.py
# Outputs: firmware/esp32s3/src/model.espdl (146.8 KB)
```

#### 🔹 For Silicon Labs EFR32MG24 (2-Stage Hybrid TFLM + Cortex-M33 FPU):
Splits the model into an INT8 2D CNN backbone for TFLite Micro and a hardware FPU-accelerated GRU classifier:
```bash
# 1. Export INT8 CNN Backbone to C Byte Array Header for TFLite Micro
python export/export_clean_cnn_int8.py
# Outputs: firmware/efr32mg24/src/phinet_features_model_data.h

# 2. (Optional) Export Golden Verification Audio Header
python export/export_golden_keyboard_pcm.py
# Outputs: firmware/efr32mg24/src/golden_keyboard_typing_pcm.h
```

### 4. Build & Flash Firmware

#### 👑 Option A: Espressif ESP32-S3 Sense (Seeed Studio XIAO)

> 💡 **ESP-DL on Zephyr Compatibility Layer:**  
> Espressif's ESP-DL library natively targets ESP-IDF. This repository includes a custom **Zephyr-to-ESP-IDF compatibility shim layer** (`firmware/esp32s3/src/compat/`) that maps FreeRTOS mutexes, heap management, and hardware timers directly to Zephyr RTOS primitives without requiring the full ESP-IDF framework.

1. **(One-time setup) Clone the official ESP-DL library:**
   ```bash
   git clone https://github.com/espressif/esp-dl.git firmware/esp32s3/esp-dl
   ```

2. **Build and flash using Zephyr `west`:**
   ```bash
   cd firmware/esp32s3
   west build -p always -b xiao_esp32s3/esp32s3/procpu/sense
   west flash
   west espressif monitor
   ```

#### 👑 Option B: Silicon Labs EFR32MG24 (xG24-DK2601B)

The EFR32MG24 firmware compiles natively using Zephyr's built-in CMSIS-DSP and standard C++ FPU math with **zero external library dependencies**:

```bash
cd firmware/efr32mg24
west build -p always -b xg24_dk2601b
west flash
screen /dev/ttyACM* 115200
```

---

## 👤 Author & Affiliations

**Egemen Acar Akkoyunlu**  
* 🏛️ **Electrical and Electronics Engineering**, Boğaziçi University, Istanbul, Turkey  
* 🔬 **Research Intern**, Fondazione Bruno Kessler (FBK), Trento, Italy  
* 🐙 **GitHub:** [@egemen-akkoyunlu](https://github.com/egemen-akkoyunlu)

---

## 📜 License
This project is licensed under the Apache 2.0 License.
