## 10. Machine Learning Theory & Algorithmic Explanations

The platform implements a diverse suite of anomaly detection algorithms. Rather than relying on a single "silver bullet," the system exposes a standardized `ModelInterface`, allowing statistical baselines to compete directly against advanced deep learning autoencoders. All models are strictly **unsupervised**, meaning they are trained exclusively on data without requiring human-annotated anomaly labels—a strict necessity in industrial deployments where failure labels are rare.

### 10.1 Isolation Forest (The Production Default)
The Isolation Forest was empirically selected as the default production model due to its extraordinary balance of high F1 accuracy and near-zero millisecond latency.
*   **Theoretical Foundation:** Unlike most algorithms that attempt to profile the "normal" data (e.g., drawing a boundary around it), the Isolation Forest explicitly isolates anomalies. The core logic dictates that anomalies are "few and different." If you randomly select a feature and recursively partition the dataset by randomly selecting a split value between the minimum and maximum, anomalous points will be isolated into leaf nodes much faster (closer to the root) than normal points.
*   **Algorithmic Concept:** The algorithm builds an ensemble of $t$ Random Trees. The anomaly score is inversely proportional to the path length $h(x)$ required to isolate point $x$.
*   **Scoring Formula:** $s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}$
    *   Where $E(h(x))$ is the average path length across all trees, and $c(n)$ is the average path length of unsuccessful searches in a Binary Search Tree.
*   **Why it Excels in Production:** Because it uses random thresholds rather than computationally heavy distance matrices (like KNN or LOF), its $O(n \log n)$ time complexity makes it devastatingly fast for real-time inference on high-dimensional vectors.

### 10.2 One-Class Support Vector Machine (OCSVM)
*   **Theoretical Foundation:** OCSVM maps the data into a high-dimensional feature space using an RBF (Radial Basis Function) kernel. It then attempts to find the maximal margin hyperplane that strictly separates the normal data points from the origin. Any point that falls on the "origin side" of the hyperplane during inference is flagged as an anomaly.
*   **Hyperparameters:** The $\nu$ (nu) parameter acts as an upper bound on the fraction of training errors (anomalies) and a lower bound on the fraction of support vectors.
*   **Limitations:** OCSVMs scale extremely poorly with large datasets. Calculating the kernel matrix is an $O(n^2)$ to $O(n^3)$ operation. Furthermore, they are highly sensitive to the scaling of the feature space, which is why the rigorous `StandardScaler` pipeline is enforced.

### 10.3 Local Outlier Factor (LOF)
*   **Theoretical Foundation:** LOF is a density-based algorithm. It operates on the assumption that anomalies are located in sparse regions of the feature space. It calculates the local density of a given point based on its $k$-nearest neighbors, and compares that density to the local densities of the neighbors themselves.
*   **Scoring Concept:** If a point's density is significantly lower than its neighbors, it is an anomaly. An LOF score of roughly 1 indicates normal density. Scores significantly greater than 1 indicate anomalies.
*   **Limitations in Streaming:** Calculating the $k$-nearest neighbors for every incoming data point requires traversing the entire historical space (or maintaining a complex spatial index), introducing unacceptable latency spikes for high-frequency IIoT streams.

### 10.4 LSTM Autoencoder (Deep Learning)
*   **Theoretical Foundation:** Neural networks typically require massive amounts of labeled data (supervised learning). Autoencoders bypass this by setting the target output to be equal to the input ($X_{out} = X_{in}$). An LSTM Autoencoder consists of an Encoder (which compresses a temporal sequence into a low-dimensional "bottleneck" latent space) and a Decoder (which attempts to reconstruct the original sequence from the latent space).
*   **The Anomaly Mechanism:** Because the network is trained *only* on normal data, it learns to perfectly compress and reconstruct normal sine-wave behaviors. When an anomalous, chaotic signal enters the network, the compressed latent space cannot adequately represent it, resulting in a massive, noisy reconstruction.
*   **Scoring Formula:** The anomaly score is the Mean Squared Error (MSE) between the input and the reconstruction: $Score = \frac{1}{N} \sum (X_{in} - X_{reconstructed})^2$
*   **Limitations:** While LSTMs capture temporal dependencies perfectly, the overhead of invoking a PyTorch/TensorFlow graph for every single inference point creates latency delays 3x to 5x higher than the Isolation Forest.

### 10.5 River HalfSpaceTrees (Online / Streaming ML)
*   **Theoretical Foundation:** Traditional ML (like the Isolation Forest) is "offline." You train it once, and the model's weights become frozen. If the industrial machine's baseline shifts (concept drift), the model will generate endless false alarms. The `River` library provides Online Learning algorithms. HalfSpaceTrees update their internal node structures incrementally, processing one reading at a time and continuously adapting their definition of "normal."
*   **The Catch:** While highly adaptable, empirical testing showed that online models are highly susceptible to "poisoning." If a machine degrades slowly over three days, the online model might slowly update its threshold to consider the dying machine as "normal," ultimately missing the failure entirely.

<!-- PAGE BREAK -->
<div style="page-break-after: always;"></div>

## 11. Threshold Selection Methodology

A major misconception in MLOps is that unsupervised algorithms automatically output a binary `1` or `0` label. They do not. They output a continuous, abstract anomaly score (e.g., `0.672`). Converting this abstract float into a definitive, actionable Alert requires a rigorously calculated Threshold.

### 11.1 The Flaw of Static Contamination Rates
Many libraries (including scikit-learn) allow you to pass a `contamination` parameter (e.g., `0.01`). The algorithm will simply flag the top 1% highest scores as anomalies. This is fundamentally flawed in production. If a factory runs perfectly for 6 months, forcing a 1% contamination rate will deliberately generate thousands of false alarms. If the factory experiences a massive breakdown event lasting days, 1% will miss most of the event.

### 11.2 F1-Maximizing Validation Search
To solve this, the platform utilizes a dynamic Validation Search mechanism:
1.  The model generates continuous anomaly scores for the entire **Validation Split**.
2.  The system calculates the minimum and maximum scores produced.
3.  It generates 100 equidistant threshold steps between the min and max.
4.  For every threshold step, it converts the continuous scores into binary predictions.
5.  It compares these predictions against the actual ground-truth anomaly windows in the Validation set, calculating the F1-Score at every step.
6.  The threshold value that resulted in the highest F1-Score is permanently locked and saved as `threshold.json`.

**Crucially**, this search is performed *strictly* on the Validation set. If the threshold was optimized on the Test set, the resulting metrics would be artificially inflated, violating the integrity of the evaluation.

<!-- PAGE BREAK -->
<div style="page-break-after: always;"></div>

## 12. Evaluation Methodology and Real Metrics

In anomaly detection, where normal data points make up 99.9% of the dataset, traditional "Accuracy" is a dangerously misleading metric. A model that stubbornly outputs "Normal" for every single reading will achieve 99.9% Accuracy, yet fail to detect the single catastrophic failure it was designed to catch.

Therefore, the system relies exclusively on Precision, Recall, and the F1-Score.

### 12.1 Windowed Evaluation Logic
Industrial anomalies do not occur in isolated milliseconds. A bearing failure is a continuous event. In standard ML evaluation (Point-to-Point), if a model flags an anomaly 5 seconds before the exact ground-truth label, it is penalized as a False Positive, and the missed ground-truth is penalized as a False Negative. This double-penalty ruins evaluations for time-series forecasting.

The platform implements **Windowed Evaluation**:
*   If the model raises *at least one* alert within the boundaries of a ground-truth anomaly window, it is counted as a successful True Positive.
*   Alerts raised completely outside any known anomaly window are penalized as False Positives (False Alarms).
*   Windows that pass completely without a single alert are penalized as False Negatives (Missed Detections).

### 12.2 Real Empirical Results (From Model Registry)
The following metrics represent the actual performance of the algorithms tested on the held-out NAB Test set during development:

| Model | Type | Inference Latency | F1 Score | False Alarm Rate | Missed Detections |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Isolation Forest** | Ensemble Tree | **< 0.01 ms** | **0.946** | 1.9% | 0 |
| **LSTM Autoencoder** | Deep Neural Net | ~ 0.03 ms | 0.992 | 0.2% | 0 |
| **One-Class SVM** | Boundary Kernel | ~ 0.08 ms | 0.792 | 12.4% | 1 |
| **Elliptic Envelope**| Statistical | ~ 0.02 ms | 0.651 | 21.0% | 0 |
| **River HST** | Online ML | ~ 0.35 ms | 0.047 | 88.0% | 4 |

**Discussion:** While the LSTM Autoencoder achieved near perfection (0.992 F1), the Isolation Forest was ultimately selected as the **Production Default**. Its F1-score is stellar, and its execution speed is an order of magnitude faster. In a factory with 10,000 sensors transmitting 50 times a second, the sub-millisecond execution time of the Isolation Forest saves vast amounts of CPU overhead.

<!-- PAGE BREAK -->
<div style="page-break-after: always;"></div>

## 13. Model Registry and Experiment Tracking

Industrial AI requires extreme determinism. If a model hallucinates a false alarm that stops a production line, the data science team must be able to audit exactly which model was running, what data it was trained on, and what its offline metrics were. 

### 13.1 The MLOps Artifact Store
The Model Registry (backed by `models/model_registry.json`) acts as the immutable ledger for the entire ML lifecycle. 

When an offline training job completes, the system automatically registers a new entry. A single registry entry contains:
1.  **UUID:** A unique cryptographic hash identifying the specific training run.
2.  **Model Type:** e.g., "IsolationForest".
3.  **File Paths:** The absolute paths to the serialized `.pkl` artifact and the `threshold.json` file.
4.  **Feature Schema:** An array of the exact features the model requires (e.g., `["value", "hour", "day_of_week", "rolling_mean_65", "rolling_std_65", "fft_dominant_freq"]`). If the FastAPI buffer does not output this exact schema, the model will gracefully reject the payload rather than crashing.
5.  **Metrics Map:** The frozen F1, Precision, and Recall scores.

### 13.2 The Hot-Swap Promotion Mechanism
The Model Registry facilitates zero-downtime model deployments. The JSON registry contains a boolean flag: `is_production`. 

When an operator navigates to the React Frontend's "Retraining Center", they are presented with a list of all offline-trained candidate models. By clicking "Promote to Production", the frontend fires an API request (`POST /retraining/promote/{model_id}`). The FastAPI backend intercepts this, immediately updates the JSON registry, loads the new `.pkl` artifact into active memory, and points the Inference Engine to the new model object. The entire transition occurs in milliseconds between incoming socket payloads, ensuring that the factory's real-time monitoring is never interrupted.

### 13.3 Model Lifecycle Architecture Flow

*(Note: The following flowchart details the transition of a model from offline training to active inference, governed by the registry.)*

![Model Lifecycle Flow](figures/model_lifecycle.png)
