# Elliptic Envelope

## Overview
Elliptic Envelope assumes the data is generated from a known multivariate Gaussian distribution and fits a robust covariance estimate to the data (using FastMCD). It flags points that are far from the center of this distribution (high Mahalanobis distance).

## Intended Use
- **Modality:** Time-Series Sensor Data
- **Primary Use Case:** Baseline contrast and Gaussian assumption testing
- **Dataset:** NAB Machine Temperature

## Features
Operates on the standard 22 causal time-series features defined in `models/feature_columns.json`.

## Performance (Test Set)
- **PR-AUC:** ~0.317
- **Windowed Recall:** 1.0 (But False Alarm Rate is ~99.97%)
- **Detection Latency:** 0.0 minutes
- **Inference Time:** ~0.0008ms per reading

## Limitations
- **Honest Negative Result:** The fundamental assumption that normal industrial time-series features perfectly follow a single multivariate Gaussian distribution is false. The robust covariance matrix is often singular, forcing a fallback estimator.
- Resulting thresholding causes the model to flag nearly 100% of all data points.
- It is included specifically to contrast against flexible, non-parametric methods like Isolation Forest.
