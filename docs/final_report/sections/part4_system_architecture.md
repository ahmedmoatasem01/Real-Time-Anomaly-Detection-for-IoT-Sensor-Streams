## 14. Real-Time Inference Architecture & API Internals

The core engine of the platform is the Python-based FastAPI backend. It serves as the high-throughput, low-latency real-time bridge connecting the data stream, the predictive machine learning models, and the React frontend. 

### 14.1 The Concurrency Model (Asynchronous I/O)
Industrial data streams do not wait. If the API blocks on a database write or a model inference, incoming HTTP POST payloads will stack up in the TCP buffer, eventually leading to dropped packets and massive latency spikes. 

To solve this, FastAPI was selected over traditional WSGI frameworks like Flask or Django. FastAPI natively supports ASGI (Asynchronous Server Gateway Interface). 
*   **Database Writes:** When an alert is generated, the `INSERT` command to the SQLite database is executed `await`-style using `aiosqlite` or a ThreadPoolExecutor, ensuring the main event loop is never blocked by disk I/O.
*   **Model Execution:** The `IsolationForest.predict()` method is entirely CPU-bound and synchronous. To prevent it from blocking the async event loop, the inference function is dispatched to a background worker thread.

### 14.2 The Stateful Sliding Buffer Engine
REST APIs are, by design, stateless. A `POST /predict` containing `{"value": 75.4}` knows absolutely nothing about the `POST /predict` that occurred 50 milliseconds prior. However, because our Feature Engineering relies entirely on a 65-point rolling window, the backend must forcibly maintain state.

The `InferenceService` implements a thread-safe, in-memory `collections.deque` dictionary. 
*   **Sensor Segregation:** The dictionary uses the `sensor_id` as the key. `buffer_dict = {"Sensor-01": deque(maxlen=65), "Sensor-02": deque(maxlen=65)}`. This ensures that data from Machine A does not contaminate the rolling features of Machine B.
*   **The Warm-Up Phase:** When the system cold-starts, the first 64 readings appended to the deque simply return a `202 Accepted` status. No predictions are made because the buffer is not yet full enough to calculate reliable standard deviations or FFTs.
*   **The Steady State:** On the 65th reading, the buffer reaches capacity. The `deque` automatically pops the oldest value (O(1) time complexity). The array is converted to a NumPy array, and the feature extraction pipeline is executed.

### 14.3 Inference Sequence Diagram
*(Note: The following sequence diagram illustrates the exact asynchronous flow of a single payload through the FastAPI buffer, to the model, and out via WebSockets.)*

![Inference Sequence Diagram](figures/inference_sequence.png)

<!-- PAGE BREAK -->
<div style="page-break-after: always;"></div>

## 15. The Stream Simulator Ecosystem

Because a live, physical industrial CNC machine cannot be easily attached to a development environment, the platform includes a highly robust, multi-threaded Python Stream Simulator.

### 15.1 Historical Replay Dynamics
The simulator is not generating random noise; it is replaying the mathematically scaled `machine_temperature` Test set. 
1.  It loads the CSV into a pandas dataframe.
2.  It iterates row by row. It calculates the actual time delta (`dt`) between `row[n]` and `row[n-1]`.
3.  It applies a `Speed Multiplier` (e.g., `50x`). It then invokes `time.sleep(dt / 50)` to perfectly mimic the temporal spacing of the original industrial sensor, albeit at an accelerated pace for demonstration purposes.

### 15.2 Synthetic Fault Injection (Chaos Engineering)
While replaying historical data proves the model's accuracy, it does not prove the UI's resilience or the operator's workflow. The simulator exposes a `Chaos Mode` that can be toggled via the Frontend's "Demo Control Panel" or the `/faults/inject` REST endpoint.

When a fault is injected, it intercepts the reading *before* the `POST` request is fired:
*   **Spike Fault:** `value = original_value * 3.0`. Simulates an instant catastrophic failure (e.g., a snapped belt). The Rolling Standard Deviation feature will explode instantly, triggering a Critical alert.
*   **Gradual Drift Fault:** `value = original_value + (0.1 * tick)`. Over 60 seconds, the value climbs relentlessly. The rate of change remains relatively small, but the Rolling Mean eventually exceeds the normalized bounds. Simulates slow friction or lubrication loss.
*   **Sensor Stuck Fault:** The simulator ignores the CSV and repeats the exact same float indefinitely. The model detects that the variance has dropped exactly to 0.0, an impossibility in a vibrating machine, and flags it.
*   **High-Frequency Noise:** Injects a massive Gaussian noise array. Simulates RF interference or a loose wire connection.

<!-- PAGE BREAK -->
<div style="page-break-after: always;"></div>

## 16. Alert Management System & Human-In-The-Loop

Generating a mathematically perfect anomaly score is useless if it does not drive a mechanical action on the factory floor. The Alert Management system translates the `float` score into a strict operational workflow.

### 16.1 Severity Scaling
When an anomaly score breaches the validation threshold, the system calculates the magnitude of the breach to assign a severity.
*   **Score < Threshold:** Normal. No action.
*   **Threshold < Score < (Threshold + 10%):** Low Severity. (Yellow). Usually indicative of early-stage gradual drift.
*   **Score > (Threshold + 25%):** Critical Severity. (Red). Indicative of an immediate spike fault. Triggers UI flashing.

### 16.2 The Alert Lifecycle State Machine
Alerts are treated as immutable state-machine entities. An operator cannot arbitrarily delete an alert; they must transition it through its lifecycle. This forces accountability.

1.  **New:** The threshold is breached. The row is inserted into SQLite.
2.  **Acknowledged:** An operator clicks the alert. A timestamp `acknowledged_at` is recorded. This metric tracks operator responsiveness.
3.  **Investigating:** The operator has actively dispatched a maintenance technician to the physical machine.
4.  **Resolved (Terminal):** The technician confirmed and repaired the fault. The machine is healthy.
5.  **False Alarm (Terminal):** The anomaly was algorithmic hallucination or a scheduled reboot.

### 16.3 Operator Feedback Loop
Before an alert can reach a terminal state (Resolved or False Alarm), the system strictly requires the operator to input an `Operator Note` (e.g., "Checked bearing housing; discovered excessive lubrication causing thermal insulation"). 

This qualitative text is saved alongside the quantitative sensor array. This creates a supervised dataset out of an unsupervised environment. In the future, NLP models can be trained on these operator notes to automatically categorize the *type* of mechanical failure based on the waveform shape.

### 16.4 Alert Lifecycle Flowchart
*(Note: The following state diagram illustrates the strict paths an alert must take before resolution.)*

![Alert Lifecycle](figures/alert_lifecycle.png)

<!-- PAGE BREAK -->
<div style="page-break-after: always;"></div>

## 17. Data Drift Detection (Population Stability Index)

Industrial machinery is subject to physical wear and tear. Over five years, a machine's baseline "normal" temperature might permanently shift upward by 3°C due to permanent mechanical tolerances expanding. If the Machine Learning model does not adapt to this new normal, it will begin generating endless False Alarms, rendering the system useless. This phenomenon is known as Data Drift.

### 17.1 Mathematical Implementation of PSI
To detect drift computationally, the backend implements the **Population Stability Index (PSI)**. 
1.  **Baseline Distribution:** During offline training, the system bins the `Training Set` into 10 deciles (buckets) and records the percentage of data points in each bucket.
2.  **Current Distribution:** The `InferenceService` maintains a secondary, larger sliding window (e.g., the last 10,000 readings). It sorts this current data into the exact same 10 baseline buckets.
3.  **Calculation:** The system compares the expected (baseline) percentages against the actual (current) percentages.
    $$PSI = \sum_{i=1}^{10} (\% \text{Actual}_i - \% \text{Expected}_i) \cdot \ln\left(\frac{\% \text{Actual}_i}{\% \text{Expected}_i}\right)$$

### 17.2 Operational Response to Drift
*   **PSI < 0.1:** No significant drift. The current model is perfectly valid.
*   **0.1 < PSI < 0.2:** Moderate drift. The "System Health" UI component turns yellow. An automated warning is issued recommending that the data science team schedule a retraining session in the near future.
*   **PSI > 0.2:** Critical drift. The data distribution has fundamentally changed. The current model's predictions are no longer mathematically sound. The system mandates an immediate offline retraining sequence.

## 18. Automated Retraining & Model Registry Workflow

When PSI exceeds 0.2, the system initiates the MLOps retraining workflow.

1.  **Data Extraction:** The backend extracts the latest massive window of telemetry from the SQLite database.
2.  **Offline Execution:** A separate worker thread (or an Airflow DAG in a larger deployment) instantiates the `scikit-learn` algorithms.
3.  **Candidate Benchmarking:** It trains a new Isolation Forest, a new OCSVM, and a new Elliptic Envelope on this new data. It runs the automated Validation Threshold Search (Section 11) to find the new optimal threshold.
4.  **Shadow Testing:** The newly trained "Candidate" models are evaluated against a holdout test set to generate their F1 metrics.
5.  **Registration:** The Candidate models are serialized and inserted into the JSON Model Registry with `is_production: false`.

At this stage, automation stops. To prevent catastrophic "black-box" overwrites (e.g., an automated system deploying a broken model that halts the factory), the system relies on manual intervention. An operator navigates to the Retraining Center UI, reviews the new F1 metrics, compares them against the current production model, and clicks **"Promote to Production"**, thereby executing the hot-swap detailed in Section 13.2.
