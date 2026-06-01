# TinyRUL-CNN: Dual-Core Edge AI Battery Diagnostics on ESP32

![RUL Dashboard](images/dashboard.png)

## Overview
TinyRUL-CNN is an end-to-end predictive maintenance pipeline for calculating the Remaining Useful Life (RUL) of lithium-ion batteries. By extracting complex thermodynamic and electrochemical aging signatures from raw sensor data, a custom Convolutional Neural Network (CNN) was trained, quantized, and deployed to run entirely offline on an ESP32 microcontroller using TensorFlow Lite Micro.

## System Architecture: FreeRTOS Dual-Core Integration
Deploying a CNN alongside a live data stream and a web server typically results in memory crashes or dropped packets. To solve this, the ESP32 firmware was architected using **FreeRTOS** to completely isolate I/O from AI compute:

* **Core 1 (I/O & Web Server):** Dedicated to handling a continuous 115200 baud UART stream of raw battery telemetry while serving an asynchronous, non-blocking AJAX web dashboard.
* **Core 0 (Heavy Compute):** Sleeps until a full 10-cycle sequence buffer is filled. Once triggered via Semaphores, it wakes up, executes C++ feature extraction, and runs the TFLite neural network in total isolation.

### TinyML Constraints & Performance
The PyTorch model was exported via ONNX, translated to TensorFlow, and compiled into a C++ header (`model_finalVersion.h`) for bare-metal execution.
* **Inference Latency:** ~15 ms
* **Flash Memory Footprint:** ~19 KB 
* **SRAM Allocation:** 10 KB Tensor Arena

## Dataset & Feature Engineering
The model was trained using the **Oxford Battery Degradation Dataset** (Cells 2, 3, 5, 6, and 8). Instead of passing raw time-series data to the AI, 11 physics-informed degradation features are computed locally on the edge device for every discharge cycle.

| Feature Type | Extracted Metrics |
| :--- | :--- |
| **Electrical** | Delivered Capacity, Total Energy, Discharge Duration |
| **Thermal** | Average Temp, Max Core Temp, Temp Rise |
| **Statistical** | Mean Voltage, Voltage Standard Deviation, Voltage Entropy |
| **Electrochemical** | Peak dQ/dV, dQ/dV Area (Internal resistance proxy) |

![Feature Correlation Matrix](images/Feature%20Correlation%20Matrix.png)
*Correlation analysis proves that as the battery degrades, Capacity and Energy collapse, while Peak Temperature and internal resistance (dQ/dV) violently spike.*

## Model Architecture & Validation
The predictive engine is a lightweight 1D Convolutional Neural Network optimized for sequential feature data. 

**Model Pipeline:** `Conv1D (16) -> Conv1D (32) -> Global Average Pooling -> Dense (32) -> Dense (16) -> Output (RUL)`

To ensure the model generalizes to completely unseen batteries without data leakage, **Leave-One-Cell-Out Cross Validation (LOOCV)** was utilized during training (e.g., Train on Cells 2,3,5,6; Test on Cell 8).

![LOOCV RMSE](images/Tiny_RMSE_Barplot.png)

## Repository Structure

```text
.
├── dataset_analysis/
│   ├── data_visualisation.m      # Macro/Micro raw data visualization
│   └── Feature_Extraction.m      # Signal processing and physics feature generation
├── LOOCV_Validation/
│   ├── LOOCV.ipynb               # Cross-validation training notebook
│   └── TinyML_LOOCV_Results.csv  # Fold-by-fold RMSE metrics
├── Training/
│   ├── train_tinyrul_cnn.py      # Final production model training script
│   └── export_tflite.py          # PyTorch → ONNX → TFLite Micro conversion
├── esp32_deployment/
│   ├── main.cpp                  # FreeRTOS firmware, AJAX server, and feature math
│   └── model_finalVersion.h      # Quantized CNN payload
└── images/
