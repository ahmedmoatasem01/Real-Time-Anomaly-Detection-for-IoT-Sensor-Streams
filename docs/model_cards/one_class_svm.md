# One-Class SVM

## Overview
One-Class Support Vector Machine (SVM) learns a decision boundary that encompasses the normal training data, classifying points falling outside this boundary as anomalies. It uses a non-linear RBF kernel to capture complex data distributions.

## Intended Use
- **Modality:** Time-Series Sensor Data
- **Primary Use Case:** Offline/Batch anomaly detection comparison
- **Dataset:** NAB Machine Temperature

## Features
Operates on the standard 22 causal time-series features defined in `models/feature_columns.json`.

## Performance (Test Set)
- **PR-AUC:** ~0.998
- **Windowed Recall:** 1.0
- **Detection Latency:** 0.0 minutes
- **Inference Time:** ~0.09ms per reading (Significantly slower than Isolation Forest)

## Limitations
- Computationally expensive to train on large datasets ($O(n^2)$ to $O(n^3)$).
- Inference latency is an order of magnitude higher than tree-based methods due to support vector evaluations.
- Highly sensitive to hyperparameter scaling ($\nu$ and $\gamma$).
