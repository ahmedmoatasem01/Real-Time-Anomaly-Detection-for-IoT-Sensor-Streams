## 7. Dataset Explanation and Selection Rationale

To build a truly multi-modal predictive maintenance platform, the architecture must support diverse datasets, each presenting unique statistical and temporal challenges. The datasets selected for this project represent the industry standards for benchmarking industrial AI.

### 7.1 Numenta Anomaly Benchmark (NAB) Machine Temperature Dataset
*   **Source:** Numenta Anomaly Benchmark (NAB) repository.
*   **Purpose:** The primary dataset utilized for the active Live Telemetry MVP module.
*   **Data Description:** This dataset contains real-world temperature readings from a large industrial machine's internal sensor. It consists of thousands of rows containing a `timestamp` and a `value` (temperature).
*   **Temporal Characteristics:** The data is sampled at consistent 5-minute intervals. The sheer volume of normal data heavily outweighs the anomalous data, perfectly mimicking real-world industrial imbalance.
*   **Label Construction (Anomaly Windows):** Unlike basic datasets that provide binary labels for every row, NAB provides "anomaly windows." Mechanical failures don't happen in a single 5-minute snapshot; they unfold over hours. The preprocessing pipeline parses these JSON windows and assigns a binary label `1` to any timestamp falling within the window.
*   **Selection Rationale:** This dataset perfectly simulates low-frequency, univariate continuous telemetry. It forces the Machine Learning models to focus intensely on temporal patterns—specifically variance shifts and rate-of-change—rather than high-dimensional spatial patterns. The low frequency allows for rapid prototyping of the core MLOps pipeline and WebSocket infrastructure.
*   **Current Limitations:** Being univariate, the dataset inherently limits the ability of the models to learn complex cross-sensor correlations (e.g., detecting an anomaly only when Temperature is high *and* Pressure is low).

### 7.2 NASA Bearing Dataset (Vibration Roadmap)
*   **Source:** NASA Ames Prognostics Data Repository.
*   **Purpose:** Designated for the Future Vibration Health Module.
*   **Data Description:** This dataset contains high-frequency accelerometer data. Four bearings were installed on a shaft and run continuously at 2000 RPM until catastrophic failure occurred. 
*   **Sampling Rate:** The data was collected in 1-second snapshots at an incredible 20kHz (20,000 readings per second), creating massive, dense files.
*   **Weak Labeling & Degradation Paths:** The dataset is explicitly "run-to-failure." There are no hard anomaly labels provided. Instead, the assumption is that the bearing begins in a state of absolute health, and degrades monotonically over time as the inner race or rolling elements physically break down. Models trained on this dataset must learn the initial healthy baseline and output a "Degradation Index" (a proxy for Remaining Useful Life - RUL) representing the mathematical distance from the healthy baseline.

### 7.3 MVTec AD Image Dataset (Vision Roadmap)
*   **Source:** MVTec Anomaly Detection dataset.
*   **Purpose:** Designated for the Future Visual Inspection Module.
*   **Data Description:** High-resolution optical images of various industrial products (e.g., pills, cables, metal nuts) and textures (e.g., wood, leather) with localized defects like scratches, contamination, and structural deformations.
*   **Normal-Only Training Concept:** In optical manufacturing inspection, defective samples are exceedingly rare, making supervised image classification nearly impossible. Models must be trained *exclusively* on defect-free images. During inference, if the model (typically a Convolutional Autoencoder) cannot accurately compress and reconstruct an image, the areas of high pixel-wise reconstruction error explicitly represent the spatial location of the defect.

### 7.4 Synthetic Multi-Sensor Data
*   **Purpose:** Generating controlled, mathematically predictable anomalies for rigorous system testing, latency evaluation, and product demonstration.
*   **Selection Rationale:** Real-world anomalies in the NAB dataset are messy and highly varied. By injecting synthetic faults programmatically, the data science team can definitively calculate detection latency and boundary failure points.
*   **Supported Fault Vectors:**
    *   **Spike:** Instantaneous multiplication of the value, simulating an electrical short.
    *   **Drift:** Gradual arithmetic addition over a defined time window, simulating frictional wear.
    *   **Stuck:** Freezing the sensor value, simulating a broken transducer.

<!-- PAGE BREAK -->
<div style="page-break-after: always;"></div>

## 8. Data Processing Pipeline

Data preprocessing acts as the critical bridge between raw, unstructured sensor logs and the strict numeric arrays required for Machine Learning inference. Industrial data is notoriously dirty—sensors drop packets, timestamps drift, and maintenance reboots create massive gaps.

The platform employs a rigorous Python-based pipeline (leveraging `pandas` and `scikit-learn`) to sanitize the data and prevent the cardinal sin of Machine Learning: Data Leakage.

### 8.1 The Processing Sequence
1.  **Raw Data Loading & Verification:** The system attempts to download the dataset via the Kaggle API. If Kaggle credentials are not provided in the environment variables, it gracefully falls back to a locally cached version in the `data/raw/` directory.
2.  **Temporal Alignment:** The `timestamp` column, usually stored as a string, is parsed into localized `datetime` objects. The dataframe is then explicitly sorted by this timestamp. Chronological order is absolutely critical; if the data is shuffled, the rolling window feature engineering will mathematically collapse.
3.  **Missing Value Handling (Imputation):** Industrial sensors occasionally fail to send a reading. Dropping these rows (as is common in static ML) would destroy the temporal continuity required for time-series forecasting. Instead, a Forward-Fill (`ffill`) strategy is used. The last known good value is carried forward, and a boolean `is_imputed` flag is generated to inform the model that the data is synthetic.
4.  **Label Generation:** If ground-truth anomaly windows are provided (as with NAB), the system iterates through the JSON windows. Any dataframe row where the timestamp falls inclusively between the window's start and end times is assigned a binary label of `1`. All other rows are strictly `0`.
5.  **Strict Chronological Split:** The dataset is split into Train (50%), Validation (20%), and Test (30%) sets. Unlike traditional machine learning which uses `train_test_split` to randomly sample rows, this pipeline splits the data strictly chronologically. The first 50% of the timeline is the training set. Random sampling is strictly prohibited, as it would leak "future" data into the "past" training set.
6.  **Parameter Isolation and Scaling:** Machine Learning models (especially distance-based algorithms like SVMs) require normalized data. A `StandardScaler` is initialized and `fit` *exclusively* on the Training split to establish the true mean ($\mu$) and standard deviation ($\sigma$) of the normal operating conditions. The Validation and Test sets are then `transformed` using these exact parameters. If the scaler was fit on the entire dataset, the test set's anomalies would artificially skew the training mean.

### 8.2 Data Processing Pipeline Flow
*(Note: The following flowchart details the temporal isolation and scaling methodology.)*

![Data Processing Pipeline](figures/data_pipeline.png)

<!-- PAGE BREAK -->
<div style="page-break-after: always;"></div>

## 9. Feature Engineering Theory and Implementation

Raw temperature data is simply a scalar float. It is incredibly noisy and lacks the contextual history required for advanced anomaly detection. For instance, if a machine's temperature rises from 50°C to 80°C over 4 hours, it might be normal operation. If it rises from 50°C to 80°C in 5 seconds, it is a catastrophic failure. The raw value (80) is identical; the temporal context is what dictates the anomaly.

The system employs a sliding window (default length: $w = 65$ readings) to extract statistical, temporal, and frequency-domain features in real-time.

### 9.1 Statistical Time-Domain Features
These features capture the localized statistical moments of the signal, providing the models with a smoothed representation of the recent past.

*   **Rolling Mean ($\mu_w$):** Captures the localized central tendency, smoothing out high-frequency noise and jitter. It acts as a low-pass filter.
    *   **Formula:** $\mu_w = \frac{1}{w} \sum_{i=t-w+1}^{t} x_i$
*   **Rolling Standard Deviation ($\sigma_w$):** Captures localized volatility. An unexpected spike in variance is one of the most reliable precursors to mechanical failure. Even if the mean temperature is normal, violent swings up and down indicate instability.
    *   **Formula:** $\sigma_w = \sqrt{\frac{1}{w-1} \sum_{i=t-w+1}^{t} (x_i - \mu_w)^2}$
*   **Exponentially Weighted Moving Average (EWMA):** Gives exponentially greater weight to more recent readings. This allows the model to react much faster to sudden step-changes than a simple moving average, which lags significantly.
    *   **Formula:** $EWMA_t = \alpha \cdot x_t + (1 - \alpha) \cdot EWMA_{t-1}$ (where $\alpha$ is the smoothing factor).
*   **First-Order Derivative (Rate of Change):** Captures the velocity of the signal. A sudden temperature spike will possess a massive rate of change, instantly triggering models that monitor this feature.
    *   **Formula:** $\Delta x = x_t - x_{t-1}$
*   **Rolling Z-Score:** Provides a normalized measure of how anomalous the current data point is relative *only* to its recent local history, rather than the global history.
    *   **Formula:** $Z_t = \frac{x_t - \mu_w}{\sigma_w}$

### 9.2 Temporal Context Features
A machine's baseline shifts throughout the day. By extracting cyclical temporal features from the raw datetime string, we provide the model with essential context.
*   **Hour of Day (0-23):** Equipment runs hotter during midday peak production shifts compared to midnight maintenance windows.
*   **Day of Week (0-6):** Factories often spin down or operate under lighter loads over the weekend.
By appending these integer features, a non-linear model (like an Isolation Forest) can learn disparate baselines for a Tuesday afternoon versus a Sunday morning.

### 9.3 Frequency Domain Features (FFT)
While time-domain features are excellent for trend and volatility detection, they struggle to identify changes in the *periodicity* of a signal. Many industrial processes operate in cycles (e.g., heating up for 10 minutes, cooling for 5 minutes). 
*   **Fast Fourier Transform (FFT) Dominant Frequency:** The system applies a 1D discrete Fourier transform to the sliding window array, converting the time-series signal into the frequency domain. It then extracts the frequency component with the highest magnitude. If the dominant frequency suddenly shifts, it indicates a disruption in the machine's fundamental operating cycle.
    *   **Formula:** $X(k) = \sum_{n=0}^{N-1} x_n \cdot e^{-i 2 \pi k n / N}$

### 9.4 Implementation Complexity: The "On-The-Fly" Challenge
During offline training, calculating these features is trivial using vectorized pandas functions (e.g., `df['value'].rolling(65).mean()`). 

However, during live production inference via the FastAPI backend, the system does not have access to a static dataframe. It only receives a single JSON payload containing one reading. To solve this, the backend implements a highly optimized, stateful `collections.deque` buffer. 
As each reading arrives, it is appended to the right side of the deque. If the deque exceeds 65 items, the oldest item is popped from the left. The `numpy` library is then used to rapidly calculate the mean, std, and FFT across the array in-memory. This guarantees that the real-time inference vector perfectly matches the mathematical structure of the offline training vector, effectively eliminating the primary cause of model degradation in production.
