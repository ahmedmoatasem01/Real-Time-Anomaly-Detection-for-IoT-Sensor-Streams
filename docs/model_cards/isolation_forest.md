# Isolation Forest

## Overview
Isolation Forest is a tree-based ensemble method for anomaly detection. It isolates observations by randomly selecting a feature and then randomly selecting a split value between the maximum and minimum values of the selected feature. Anomalies require fewer splits to be isolated compared to normal points.

## Intended Use
- **Modality:** Time-Series Sensor Data
- **Primary Use Case:** Real-time stream anomaly detection (Production Default)
- **Dataset:** NAB Machine Temperature

## Features
Operates on the standard 22 causal time-series features defined in `models/feature_columns.json`, including lag, rolling mean, rolling standard deviation, EWMA, and Z-score over varying window sizes (5, 15, 60 minutes).

## Performance (Test Set)
- **PR-AUC:** ~0.999
- **Windowed Recall:** 1.0 (Detects 100% of labeled anomaly windows)
- **Detection Latency:** 0.0 minutes
- **Inference Time:** ~0.007ms per reading

## Limitations
- Performance depends highly on the quality and stability of engineered features.
- Can struggle with subtle, low-variance anomalies if not explicitly captured by the feature engineering logic (e.g., small steady drift).
