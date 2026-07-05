# Local Outlier Factor (LOF)

## Overview
Local Outlier Factor (LOF) computes the local density deviation of a given data point with respect to its neighbors. Points with a substantially lower density than their neighbors are considered outliers. We use it in *novelty detection* mode (`novelty=True`).

## Intended Use
- **Modality:** Time-Series Sensor Data
- **Primary Use Case:** Local density deviation analysis
- **Dataset:** NAB Machine Temperature

## Features
Operates on the standard 22 causal time-series features defined in `models/feature_columns.json`.

## Performance (Test Set)
- **PR-AUC:** ~0.990
- **Windowed Recall:** 1.0
- **Detection Latency:** 0.0 minutes
- **Inference Time:** ~0.43ms per reading (Highest inference latency among classical models)

## Limitations
- Novelty detection mode in LOF can be memory-intensive because it requires storing the training points to compute distances during inference.
- High inference latency makes it less ideal for high-frequency real-time systems compared to Isolation Forest.
- Performance degrades in high-dimensional feature spaces due to the curse of dimensionality.
