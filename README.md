# TinyRUL-CNN: Battery Remaining Useful Life Prediction on ESP32

## Overview

TinyRUL-CNN is an end-to-end TinyML pipeline for Remaining Useful Life (RUL) prediction of lithium-ion batteries using the Oxford Battery Degradation Dataset.

The project combines battery health feature extraction, deep learning, cross-validation, and embedded deployment to run battery health inference directly on an ESP32 microcontroller using TensorFlow Lite Micro.

### Project Highlights

* Oxford Battery Degradation Dataset
* Physics-informed battery degradation features
* Lightweight 1D CNN architecture
* Leave-One-Cell-Out Cross Validation (LOOCV)
* PyTorch → ONNX → TensorFlow Lite conversion
* ESP32 deployment using TensorFlow Lite Micro
* Real-time battery health dashboard

---

# Project Pipeline

```text
Oxford Battery Dataset
        ↓
Dataset Analysis
        ↓
Feature Extraction
        ↓
LOOCV Validation
        ↓
Final Production Training
        ↓
PyTorch Model (.pth)
        ↓
ONNX Export
        ↓
TensorFlow Lite
        ↓
C Header Conversion
        ↓
ESP32 Deployment
```

---

# Dataset

This project uses the Oxford Battery Degradation Dataset, which contains long-term cycling data collected from lithium-ion cells.

### Cells Used

* Cell2
* Cell3
* Cell5
* Cell6
* Cell8

### Raw Signals

For every discharge cycle:

* Voltage (V)
* Charge (Q)
* Temperature (T)
* Time (t)

---

# Dataset Exploration

The first step was understanding battery degradation behaviour through macro and micro-level visualization.

![Dataset Visualization](images/Datasetv%20visualisation.png)

### Macro View

* Capacity fade across battery lifetime
* End-of-Life (EOL) threshold tracking

### Micro View

* Voltage discharge profile
* Temperature evolution during discharge

---

# Feature Engineering

The following degradation features were extracted from each discharge cycle.

| Feature     | Description                            |
| ----------- | -------------------------------------- |
| Capacity    | Delivered discharge capacity           |
| Energy      | Total discharge energy                 |
| Duration    | Discharge duration                     |
| AvgTemp     | Average temperature                    |
| MaxTemp     | Maximum temperature                    |
| VoltageMean | Mean voltage                           |
| VoltageStd  | Voltage standard deviation             |
| dQdVPeak    | Peak differential capacity             |
| dQdVArea    | Area under differential capacity curve |
| Entropy     | Voltage signal entropy                 |

A normalized cycle index was appended during training.

### Total Model Inputs

```text
10 engineered features
+ 1 cycle index
----------------
11 input features
```

---

# Extracted Degradation Features

The extracted features capture electrical, thermal, and electrochemical ageing behaviour.

![Degradation Features](images/Degradation%20Features.png)

---

# Feature Correlation Analysis

A correlation matrix was generated to study relationships between extracted battery health indicators and Remaining Useful Life.

![Feature Correlation Matrix](images/Feature%20Correlation%20Matrix.png)

---

# Model Architecture

## TinyRUL CNN

```text
Input Shape:
(10 timesteps × 11 features)

Conv1D(11 → 16)
BatchNorm
ReLU

Conv1D(16 → 32)
BatchNorm
ReLU

Global Average Pooling

Linear(32 → 32)
ReLU
Dropout(0.2)

Linear(32 → 16)
ReLU

Linear(16 → 1)

Output:
Remaining Useful Life
```

### Training Configuration

| Parameter     | Value      |
| ------------- | ---------- |
| Epochs        | 80         |
| Batch Size    | 16         |
| Optimizer     | Adam       |
| Learning Rate | 0.001      |
| Loss Function | Huber Loss |

---

# Validation Strategy

## Leave-One-Cell-Out Cross Validation (LOOCV)

To evaluate generalization capability, Leave-One-Cell-Out Cross Validation was performed.

For every fold:

```text
Train: Cell2 Cell3 Cell5 Cell6
Test : Cell8

Train: Cell2 Cell3 Cell5 Cell8
Test : Cell6

...
```

This ensures that the model is evaluated on an entirely unseen battery cell.

---

# LOOCV Results

![LOOCV RMSE](images/Tiny_RMSE_Barplot.png)

Detailed fold-by-fold results are available in:

```text
LOOCV_Validation/TinyML_LOOCV_Results.csv
```

---

# TinyML Deployment Pipeline

```text
PyTorch
   ↓
ONNX
   ↓
TensorFlow Lite
   ↓
TensorFlow Lite Micro
   ↓
ESP32
```

The trained PyTorch model was exported to ONNX, converted to TensorFlow Lite, and compiled into a C header file for deployment on the ESP32.

---

# TinyML Constraints & Performance

| Metric                 | Value              |
| ---------------------- | ------------------ |
| Inference Latency      | ~15 ms             |
| Flash Memory Footprint | ~19 KB             |
| SRAM Allocation        | 10 KB Tensor Arena |

### Deployment Target

```text
ESP32
```

### Inference Engine

```text
TensorFlow Lite Micro
```

---

# ESP32 Deployment

Files:

```text
esp32_deployment/
├── main.cpp
├── model_finalVersion.h
```

The deployed model performs completely offline inference on-device without requiring cloud connectivity.

---

# Real-Time Dashboard

The ESP32 outputs battery health metrics and predicted Remaining Useful Life through a lightweight dashboard interface.

![RUL Dashboard](dashboard.png)

### Example Output

```text
Predicted RUL : 48.69 cycles

Capacity Discharged : 699.225 mAh

Maximum Core Temperature : 41°C
```

---

# Repository Structure

```text
.
├── dataset_analysis
│   ├── data_visualisation.m
│   └── Feature_Extraction.m
│
├── LOOCV_Validation
│   ├── LOOCV.ipynb
│   └── TinyML_LOOCV_Results.csv
│
├── Training
│   ├── train_tinyrul_cnn.py
│   └── export_tflite.py
│
├── esp32_deployment
│   ├── main.cpp
│   └── model_finalVersion.h
│
├── images
│   ├── Datasetv visualisation.png
│   ├── Degradation Features.png
│   ├── Feature Correlation Matrix.png
│   └── Tiny_RMSE_Barplot.png
│
├── dashboard.png
│
└── README.md
```

---

# Technologies Used

* Python
* MATLAB
* PyTorch
* NumPy
* SciPy
* Scikit-Learn
* ONNX
* TensorFlow Lite
* TensorFlow Lite Micro
* ESP32

---

# Future Improvements

* Quantized INT8 deployment
* Multi-cell battery pack prediction
* Real-time Battery Management System (BMS) integration
* Transformer-based RUL estimation
* Deployment on ARM Cortex-M devices

---

# Author

**Aditya K.**

Electronics and Communication Engineering (ECE)

Interests:

* TinyML
* Embedded AI
* Battery Health Monitoring
* Edge Machine Learning
