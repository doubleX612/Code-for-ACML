# ACML Integration Library

This repository provides the **Adaptive Channel Mixing Layer (ACML)** module and integration code designed to improve the robustness of EEG-based motor imagery classification. The ACML module has been validated on two standard public datasets — **BCI Competition IV 2a** and **High-Gamma Dataset** — demonstrating performance improvements across five different neural network models.

## 📋 Key Features

- **Adaptive Channel Mixing Layer (ACML)**: A flexible preprocessing layer that can be seamlessly integrated into existing neural network models.
- **Performance Improvements**: The integration of ACML led to consistent enhancements in **accuracy** and **kappa scores** across two datasets and five models.
- **Minimal Overhead**: Easily pluggable into existing architectures with minimal modifications required.

## 📊 Experimental Results

ACML was validated on the following:

- **Datasets**:
  - BCI Competition IV 2a
  - High-Gamma Dataset
  
- **Models**:
  - EEGNet
  - DeepConvNet
  - EEGTCNet
  - EEGTCNet_Fusion
  - ATCNet (State-of-the-Art)

Results indicate that incorporating the ACML module consistently enhances model performance, even on optimized baselines.

