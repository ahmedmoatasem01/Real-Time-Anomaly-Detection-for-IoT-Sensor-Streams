# Rolling Z-score (Baseline)

## Overview
A simple, rule-based baseline model that evaluates the absolute Z-score of the raw signal over a trailing 15-minute window.

## Intended Use
- **Modality:** Time-Series Sensor Data
- **Primary Use Case:** Simplistic baseline for evaluation contrast
- **Dataset:** NAB Machine Temperature

## Features
Relies solely on the pre-computed `zscore_15` feature.

## Performance (Test Set)
- **PR-AUC:** ~0.154
- **Windowed Recall:** 1.0
- **Detection Latency:** ~20.0 minutes
- **Inference Time:** Negligible

## Limitations
- Suffers from high detection latency. The rolling mean is heavily dragged by the anomaly itself, reducing the Z-score exactly when it needs to be high.
- Has a high false alarm rate as it lacks historical context beyond the 15-minute window.
