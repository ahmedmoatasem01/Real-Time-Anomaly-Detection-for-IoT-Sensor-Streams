# Real-Time Anomaly Detection for IoT Sensor Streams — Build Spec

> **Purpose of this file.** This is an executable specification for Claude Code. It is written to be handed directly to an agent to scaffold, implement, test, and document a complete ML + software system — not a notebook. Follow it top to bottom. Where a value is prescribed (dataset, model, ports, thresholds), use it. Where a choice is offered, pick the MVP option first and leave the advanced option as a documented stretch.

---

## 0. Executive summary (read first)

- **What we build:** a system that replays historical IoT sensor data as a live stream, scores each reading for anomalies with a trained ML model, persists predictions, raises alerts, and shows a live dashboard.
- **Primary dataset:** **Numenta Anomaly Benchmark (NAB)** — `realKnownCause/machine_temperature_system_failure.csv`. Univariate industrial-machine temperature, ~22,695 rows at 5-min intervals, timestamped, with labeled anomaly windows (planned shutdown → early warning → catastrophic failure). Kaggle: `https://www.kaggle.com/datasets/boltzmannbrain/nab`.
- **Backup dataset:** **NASA Bearing Sensor Data** (run-to-failure vibration, 4 bearings) — Kaggle: `https://www.kaggle.com/datasets/vinayak123tyagi/bearing-dataset`. Multivariate, unlabeled but with a known failure point; good for a harder unsupervised variant.
- **Final model path:** Rolling Z-score (baseline) → **Isolation Forest (MVP model, the one that gets graded)** → **LSTM Autoencoder (advanced)** → optional **River HalfSpaceTrees** online learner (added value).
- **MVP stack:** Python 3.11, pandas, scikit-learn, FastAPI + WebSocket, Streamlit, SQLite. Single `docker compose up`.
- **Advanced stack (stretch, documented not required):** MQTT/Redpanda, TimescaleDB, Redis, MLflow.

Why NAB over SWaT/WADI: SWaT and WADI require signing a request form with iTrust and are **not** freely downloadable from Kaggle — do not build the graded path on them. NAB is public, small enough to iterate fast, has real labels, and its "system_failure" file tells a clean story for a demo (normal → degradation → failure).

---

## 1. Idea & Objectives (10 marks)

**Problem.** Industrial and building IoT sensors produce continuous streams. Failures (bearing wear, overheating, pump faults, intrusion) are rare but expensive. Threshold alarms miss slow drift and gradual degradation and fire false alarms on noise. We need a system that learns *normal* behaviour and flags deviations in real time.

**Why real-time matters.** Detection latency is the cost function. A fault caught at the "early warning" stage is a cheap maintenance ticket; the same fault caught at "catastrophic failure" is downtime plus replacement. The whole point is to shorten time-to-detection.

**Use cases:** predictive maintenance (rotating machinery), smart-building HVAC, industrial process monitoring, cyber-physical intrusion detection, energy/grid monitoring.

**Aim.** Develop a real-time anomaly detection system for IoT sensor streams using machine learning: historical Kaggle sensor data trains the models, the data is then replayed as a live stream, anomalies are scored and stored, alerts are raised, and everything is visualised on a live dashboard.

**Main objective.** Detect abnormal sensor behaviour on a live stream with high recall and controlled false-alarm rate, and surface it fast enough to act on.

**Sub-objectives.**
1. Clean, engineer features from, and window the sensor time series.
2. Train and compare a baseline, a classical ML model, and a deep model.
3. Serve the chosen model behind a low-latency inference API.
4. Simulate a live stream at configurable speed and route it through the API.
5. Persist every reading + score + label; raise alerts with severity.
6. Visualise the live stream, scores, and alert history on a dashboard.

**Expected outcome.** A dockerised, runnable system with a trained model, a scored live dashboard, an alerts table, an evaluation report, and tests.

**Why this is an ML project, not just a dashboard.** The decision boundary is *learned* from data (Isolation Forest / LSTM-AE reconstruction error), not hand-coded. Model selection is driven by precision/recall/PR-AUC on labeled anomalies, and features (rolling stats, lags, rate-of-change) are engineered specifically to make the learning tractable. The dashboard only renders what the model decides.

---

## 2. System Analysis & Requirements / Design (15 marks)

### 2.1 Dataset comparison

| Dataset | Domain | Real/Synth | Rows × Feats | Sensors | Timestamp | Labels | Sup/Unsup | Stream sim | Difficulty | Kaggle |
|---|---|---|---|---|---|---|---|---|---|---|
| **NAB — machine_temperature_system_failure** *(PRIMARY)* | Industrial machine | Real | ~22,695 × 1 | Temperature | Yes (5-min) | Yes (windows) | Both | Excellent | Low–Med | `boltzmannbrain/nab` |
| **NASA Bearing** *(BACKUP)* | Rotating machinery | Real | ~2M+ vib. samples, 4 ch | Vibration/accel | Yes (10-min snapshots) | Weak (known failure point) | Unsup | Good | Med–High | `vinayak123tyagi/bearing-dataset` |
| IoT/Threat Intelligence | Smart-device/IoT security | Synthetic | Large × ~10+ | CPU/mem/net | Yes | Yes (DoS/spoof/inject) | Sup | Good | Med | `ziya07/anomaly-detection-and-threat-intelligence-dataset` |
| AnoML-IoT | Multi-sensor IoT | Real | Med × several | Temp/hum/etc | Yes | Partial | Both | Good | Med | `hkayan/anomliot` |
| SWaT | Water treatment CPS | Real | Large × 51 | Many process | Yes | Yes | Both | Excellent | High | **Not on Kaggle — iTrust form required** |
| WADI | Water distribution CPS | Real | Large × 123 | Many process | Yes | Yes | Both | Excellent | High | **Not on Kaggle — iTrust form required** |
| NASA SMAP/MSL | Spacecraft telemetry | Real | 27+53 series | Telemetry | Yes | Yes | Both | Excellent | High | Third-party mirrors |

**Selection.**
- **Primary: NAB machine_temperature_system_failure.** Public, labeled, small, tells a normal→degradation→failure story that demos beautifully. Univariate keeps the pipeline simple so effort goes into system engineering, not data wrangling.
- **Backup: NASA Bearing.** If a multivariate/harder unsupervised variant is wanted, swap the loader; the rest of the pipeline (features → IF/LSTM-AE → API → dashboard) is unchanged.
- **Reason:** free, reproducible, gradeable, and demo-friendly. SWaT/WADI are academically stronger but gated behind a request form — unacceptable risk for a graded deadline.

### 2.2 System overview

Offline: load → clean → feature-engineer → train → save model + scaler + threshold. Online: stream simulator emits one reading at a time → FastAPI `/predict` scores it → result written to DB and pushed over WebSocket → alert raised if anomalous → Streamlit dashboard renders live.

### 2.3 Roles
- **Operator** — watches dashboard, acknowledges alerts.
- **Data scientist** — trains/evaluates/swaps models.
- **System (automated)** — streams, scores, persists, alerts.

### 2.4 Functional requirements
FR1 load dataset; FR2 preprocess + feature engineer; FR3 train baseline/IF/LSTM-AE; FR4 evaluate + persist metrics; FR5 serve inference API; FR6 simulate stream at configurable speed; FR7 persist every reading+score+label; FR8 raise alerts with severity; FR9 live dashboard; FR10 structured logging.

### 2.5 Non-functional requirements
NFR1 inference latency < 50 ms/reading (IF); NFR2 throughput ≥ 200 readings/s in fast mode; NFR3 reproducible (fixed seeds, pinned deps); NFR4 `docker compose up` one command; NFR5 config-driven (no hard-coded paths); NFR6 graceful error handling on malformed readings.

### 2.6 I/O spec
- **Input reading (JSON):** `{ "timestamp": ISO8601, "sensor_id": str, "value": float }` (multivariate backup: `values: {sensor: float}`).
- **Output prediction (JSON):** `{ "timestamp", "sensor_id", "value", "anomaly_score": float, "is_anomaly": bool, "severity": "low|medium|high", "reason": str, "model": str, "inference_ms": float }`.

### 2.7 Data flow
`CSV → loader → preprocess → feature_engineering → [train: model+scaler+threshold saved] ` and at runtime `stream_simulator → POST /predict → inference_service (scale→features→score→threshold→severity) → DB insert + WebSocket broadcast → dashboard + alert`.

### 2.8 Use-case description
Operator opens dashboard → sees live values + score line + threshold band. When score crosses threshold, a row appears in the alerts panel with severity and the offending sensor/feature. Operator clicks to see the recent window. Data scientist can re-run training and hot-swap the model file.

### 2.9 Architecture
- **MVP:** Streamlit ⇄ FastAPI(WebSocket + REST) ⇄ inference(model.pkl) ; stream_simulator → FastAPI ; FastAPI → SQLite. All in one docker compose.
- **Advanced (stretch):** simulator → MQTT/Redpanda → consumer → FastAPI inference → TimescaleDB + Redis(latest state) → dashboard; MLflow for model registry.

### 2.10 Database design (SQLite MVP; identical schema on Postgres/Timescale)
```sql
CREATE TABLE readings (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL, sensor_id TEXT NOT NULL, value REAL NOT NULL,
  anomaly_score REAL, is_anomaly INTEGER, severity TEXT,
  reason TEXT, model TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_readings_ts ON readings(ts);
CREATE TABLE alerts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  reading_id INTEGER, ts TEXT NOT NULL, sensor_id TEXT,
  severity TEXT, score REAL, reason TEXT, acknowledged INTEGER DEFAULT 0,
  FOREIGN KEY(reading_id) REFERENCES readings(id)
);
CREATE TABLE model_runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  model TEXT, trained_at TEXT, precision REAL, recall REAL,
  f1 REAL, pr_auc REAL, roc_auc REAL, threshold REAL, notes TEXT
);
```

### 2.11 API design (FastAPI)
- `GET /health` → `{status:"ok", model:...}`
- `POST /predict` → body = reading, returns prediction (see 2.6).
- `POST /predict/batch` → list of readings.
- `WS /ws/stream` → server broadcasts each scored reading to connected dashboards.
- `GET /alerts?limit=100` / `POST /alerts/{id}/ack`
- `GET /metrics` → latest `model_runs` row + live counters (throughput, avg latency).

### 2.12 Dashboard design (Streamlit)
Live line chart of value with threshold band; anomaly score subplot with markers on flagged points; KPI row (readings/s, avg latency, alerts count, model name); alerts table (ts, sensor, severity, reason, ack button); model-comparison page (metrics table + PR curves from `reports/`); system-health panel (stream rate, latency, model status).

### 2.13 Stream simulation design
`stream_simulator.py` reads the processed CSV in timestamp order and emits readings via HTTP POST (MVP) or MQTT (advanced). `--speed` multiplier (e.g. `50` = replay 50× faster than real 5-min cadence), `--loop`, `--start-index`. Sleep = `original_interval / speed`.

---

## 3. Algorithms / Methods / Techniques (20 marks)

### 3.1 Preprocessing
Parse `timestamp` to datetime, sort ascending, set as index. Missing values: forward-fill short gaps (≤ 3 steps), else interpolate linearly; flag imputed. Drop exact duplicate timestamps (keep last). Scale with `StandardScaler` fit **on the training split only** (persist scaler). Type/range validation on ingest.

### 3.2 Time-series split (no shuffling)
Chronological 70/15/15 train/val/test. **Fit scaler and unsupervised models on the "normal" region only** (NAB provides an initial burn-in ~15% that is known-normal). Never let test rows leak into scaler/threshold.

### 3.3 Feature engineering (windowed, causal only — no future leakage)
For window `w ∈ {5, 15, 60}` steps: rolling mean, rolling std, rolling min, rolling max, EWMA. Plus: lag features (`t-1,t-2,t-3`), rate-of-change (`diff`), rolling z-score `(x - roll_mean)/roll_std`. Multivariate backup adds pairwise rolling correlation between sensors. Drop rows with NaN warm-up window.

### 3.4 Methods compared

**Baselines.** Static threshold (mean ± k·std on train); Rolling z-score (flag |z| > 3); EWMA control chart. Cheap, interpretable, O(1) per reading, great real-time fit; weak on multivariate/seasonal.

**Classical ML.**
- **Isolation Forest (MVP model).** Isolates points via random splits; anomalies isolate in fewer splits → higher score. Unsupervised, fast train, ~sub-ms inference, handles the engineered feature vector well. Output: `-score_samples` as anomaly score; threshold via validation PR curve (or `contamination`). **This is the model that gets graded.**
- One-Class SVM — learns a boundary around normal; RBF kernel; slower, sensitive to scaling; good small-data alternative.
- LOF — local density deviation; strong for local anomalies; batch-oriented, less natural for streaming.
- Random Forest / XGBoost / LightGBM — only if using labels as supervised classification; strong when labels exist but NAB labels are sparse windows, so treat as secondary.

**Deep learning.**
- **LSTM Autoencoder (advanced model).** Train to reconstruct normal windows; high reconstruction error ⇒ anomaly. Input: sequences of the scaled value + features. Threshold on reconstruction-error distribution of normal data (e.g. 99th percentile). Captures temporal dependence the IF ignores. Heavier to train, needs GPU-ideal but CPU-OK at this size.
- GRU-AE / Temporal-CNN-AE / Transformer-AE — same recipe, documented as alternatives; only build if time allows.

**Streaming / online.**
- **River `HalfSpaceTrees`** (added value) — true online anomaly detector, updates per reading, adapts to drift; pair with an **adaptive threshold** (rolling quantile of scores). Demonstrates online learning without retraining.

### 3.5 Recommended path
1. Rolling z-score — baseline sanity + demo contrast.
2. **Isolation Forest — MVP graded model.**
3. **LSTM Autoencoder — advanced model.**
4. River HalfSpaceTrees + adaptive threshold — optional added value.

### 3.6 Evaluation
Report on the labeled test region: **Precision, Recall, F1, Confusion matrix, ROC-AUC, PR-AUC, false-alarm rate, missed-anomaly rate, detection latency (steps from anomaly onset to first flag), inference time, stream throughput.** Because anomalies are a tiny fraction, **accuracy is misleading** (a model predicting "never anomaly" scores >99% accuracy) — **PR-AUC, recall, and false-alarm rate are the real criteria.** Use NAB-style windowed scoring (a hit anywhere inside the labeled window counts) alongside point metrics.

---

## 4. Implementation & Coding / Results (20 marks)

### 4.1 Repository structure
```
real-time-iot-anomaly-detection/
├── data/{raw,processed,sample_stream}/
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_preprocessing_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation.ipynb
├── src/
│   ├── data/{data_loader.py,preprocessing.py}
│   ├── features/feature_engineering.py
│   ├── models/{train_baseline.py,train_isolation_forest.py,train_lstm_autoencoder.py,evaluate.py,predict.py}
│   ├── streaming/{stream_simulator.py,mqtt_publisher.py}
│   ├── api/{main.py,schemas.py,inference_service.py}
│   ├── database/database.py
│   ├── dashboard/dashboard.py
│   └── utils/{config.py,logger.py}
├── models/{isolation_forest.pkl,scaler.pkl,threshold.json,lstm_autoencoder.pt}
├── reports/{figures/,evaluation_results.csv,final_report.pdf}
├── tests/{test_preprocessing.py,test_features.py,test_api.py,test_model.py,test_stream.py,test_database.py}
├── docker/Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
└── .gitignore
```

### 4.2 File responsibilities

| File | Purpose | Inputs | Outputs | Key functions/classes |
|---|---|---|---|---|
| `data/data_loader.py` | download/load raw CSV | Kaggle path/URL | raw DataFrame | `load_nab()`, `load_bearing()` |
| `data/preprocessing.py` | clean, split, scale | raw df | processed df + `scaler.pkl` | `clean()`, `time_split()`, `fit_scaler()`, `transform()` |
| `features/feature_engineering.py` | rolling/lag/roc/z features | processed df | feature matrix | `make_features(df, windows)`, `FeaturePipeline` |
| `models/train_baseline.py` | z-score/EWMA baseline | features | threshold json | `fit_zscore()`, `fit_ewma()` |
| `models/train_isolation_forest.py` | train IF | features | `isolation_forest.pkl`, `threshold.json` | `train_if()`, `pick_threshold()` |
| `models/train_lstm_autoencoder.py` | train LSTM-AE | sequences | `lstm_autoencoder.pt` | `LSTMAutoencoder`, `train_ae()`, `recon_threshold()` |
| `models/evaluate.py` | metrics + plots | model + test | `evaluation_results.csv`, figures | `evaluate()`, `nab_score()` |
| `models/predict.py` | load model, score one reading | reading + state | prediction dict | `Predictor.score(reading)` |
| `streaming/stream_simulator.py` | replay CSV as stream | processed csv | HTTP/MQTT emissions | `simulate(speed, loop)` |
| `api/main.py` | FastAPI app + WS | HTTP/WS | JSON/WS | routes in §2.11 |
| `api/schemas.py` | pydantic models | — | — | `Reading`, `Prediction`, `Alert` |
| `api/inference_service.py` | orchestrate scoring | reading | prediction | `InferenceService` (holds model, scaler, feature buffer) |
| `database/database.py` | DB layer | prediction/alert | rows | `init_db()`, `insert_reading()`, `insert_alert()`, `get_alerts()` |
| `dashboard/dashboard.py` | Streamlit UI | WS/DB | charts | `main()`, chart builders |
| `utils/config.py` | central config | env/yaml | settings | `Settings` |
| `utils/logger.py` | structured logging | — | logs | `get_logger()` |

**Stateful inference note:** the API must keep a rolling buffer per `sensor_id` so it can compute rolling/lag features for a single incoming reading. `InferenceService` owns a `deque(maxlen=max_window)` per sensor.

### 4.3 Expected results/artifacts
EDA plots (raw series with labeled anomaly windows shaded); model performance table (baseline vs IF vs LSTM-AE across §3.6 metrics); confusion matrix; anomaly-score-over-time plot with flagged points; PR + ROC curves; live dashboard screenshots; example API request/response; sample DB rows; short demo clip.

---

## 5. Testing & Documentation (10 marks)

### 5.1 Tests (pytest)
- `test_preprocessing.py` — no NaNs post-clean, chronological order, scaler fit only on train, no leakage.
- `test_features.py` — feature values match hand-computed on a tiny fixture; no future leakage (feature at t uses only ≤ t).
- `test_model.py` — model loads; `score()` returns float; known-anomalous fixture scores higher than known-normal.
- `test_api.py` — `/health` 200; `/predict` valid schema; malformed body → 422.
- `test_stream.py` — simulator emits in timestamp order; speed scaling respected.
- `test_database.py` — insert then read back; alert FK integrity.
- **Latency test** — assert avg IF inference < 50 ms. **Stress test** — 10k readings, assert no drops and throughput ≥ target. **Error handling** — NaN/negative/missing fields handled gracefully.

### 5.2 Documentation
README (overview, architecture diagram, dataset description + link, setup, train/run/dashboard/docker instructions, model explanation, testing report, results table, screenshots). Plus `reports/final_report.pdf`.

### 5.3 Commands
```bash
pip install -r requirements.txt
python -m src.data.data_loader          # fetch/prepare data
python -m src.models.train_isolation_forest
python -m src.models.evaluate
uvicorn src.api.main:app --reload        # API on :8000
python -m src.streaming.stream_simulator --speed 50
streamlit run src/dashboard/dashboard.py # dashboard on :8501
pytest -q
docker compose up --build
```

---

## 6. Innovation & Added Value (10 marks)

Build these **5** (achievable, high-visibility for grading):
1. **Real-time stream simulation** from historical Kaggle data with configurable speed.
2. **Explainable anomaly reason** — report which feature/sensor drove the score (top contributor via feature deviation / IF path or AE per-feature error).
3. **Severity levels** — low/medium/high from score bands, shown on dashboard + stored.
4. **Model-comparison page** — baseline vs IF vs LSTM-AE metrics + PR curves side by side.
5. **Online learning + adaptive threshold** (River HalfSpaceTrees) with **drift detection** — the standout differentiator.

Also cheap to add: alert history with acknowledge, system-health panel (stream rate/latency/model status), dockerised deployment, API-based inference (not just a notebook).

---

## 7. Presentation & Demo (10 marks)

**Slides.** 1 Title · 2 Problem & motivation · 3 Dataset (NAB, labeled failure story) · 4 System architecture · 5 ML pipeline · 6 Models (baseline→IF→LSTM-AE→online) · 7 Results & evaluation (PR-AUC, recall, latency) · 8 Live dashboard · 9 Innovation · 10 Demo flow · 11 Challenges & solutions · 12 Conclusion & future work.

**Demo script (≈4 min).**
1. "Here's the raw machine-temperature stream — normal for weeks, then a failure." (show EDA plot)
2. Start API + simulator at 50× — dashboard fills live.
3. Point at the score line hugging zero during normal operation.
4. As the degradation window hits, score rises → point turns red → **alert row appears with severity + reason**.
5. Open alerts table, acknowledge one.
6. Flip to model-comparison page: IF vs LSTM-AE PR-AUC/recall.
7. Toggle the online River model, show adaptive threshold tracking drift.
8. Close with metrics table + limitations (univariate, sparse labels, single machine).

---

## 8. Final Submission (5 marks)

Package: source code, dataset link + `sample_stream/` sample, trained `models/*`, README, `requirements.txt`, `docker-compose.yml`, EDA + training notebooks, `reports/evaluation_results.csv`, dashboard screenshots, `reports/final_report.pdf`, slides, demo video. Zip as `LastName_IoT_Anomaly_Detection/` mirroring the repo tree, with README at root and a one-page `SUBMISSION.md` checklist.

---

## 9. Final answers (the 13 requested items)

1. **Best Kaggle dataset:** NAB `machine_temperature_system_failure.csv` (`boltzmannbrain/nab`).
2. **Backup:** NASA Bearing (`vinayak123tyagi/bearing-dataset`).
3. **Final architecture:** MVP — Streamlit ⇄ FastAPI(REST+WS) ⇄ IsolationForest, simulator→API→SQLite, one docker compose. Advanced (documented): MQTT/Redpanda + TimescaleDB + Redis + MLflow.
4. **Final ML model path:** Rolling z-score → **Isolation Forest (graded MVP)** → **LSTM Autoencoder (advanced)** → River HalfSpaceTrees + adaptive threshold (added value).
5. **Implementation roadmap:** data→preprocess→features→train IF→evaluate→API→simulator→DB→dashboard→alerts→LSTM-AE→online model→tests→docker→docs.
6. **Repo structure:** §4.1.
7. **Dashboard features:** live value+threshold, score subplot with markers, KPI row, alerts table w/ ack, model-comparison page, system-health panel.
8. **Testing plan:** §5.1 (unit + integration + latency + stress + error handling).
9. **Innovation features:** stream sim, explainable reason, severity, model-comparison, online+drift (§6).
10. **Presentation/demo:** §7.
11. **Submission checklist:** §8.
12. **2-week schedule:** below.
13. **3-week schedule:** below.

### 2-week schedule (tight, MVP-focused)
- **D1–2:** repo scaffold, config/logger, loader, EDA notebook.
- **D3–4:** preprocessing + feature engineering + tests.
- **D5–6:** train baseline + Isolation Forest, evaluate, save artifacts.
- **D7:** DB layer + schema + tests.
- **D8–9:** FastAPI (`/predict`, `/health`, WS) + inference service (rolling buffer) + tests.
- **D10:** stream simulator + wire end-to-end.
- **D11:** Streamlit dashboard (live chart + alerts).
- **D12:** severity + explainable reason + model-comparison page.
- **D13:** docker compose, latency/stress tests, README.
- **D14:** slides, demo video, final report, package.

### 3-week schedule (adds the advanced/added-value)
- **Week 1 (D1–7):** everything through DB + baseline + Isolation Forest + evaluation, fully tested.
- **Week 2 (D8–14):** FastAPI + WS, simulator, dashboard, severity, explainable reason, model-comparison, docker, integration tests.
- **Week 3 (D15–21):** LSTM Autoencoder (train/eval/integrate), River online model + adaptive threshold + drift detection, system-health panel, stress/latency hardening, polish dashboard, slides + demo video + final report + submission package.

---

## 10. Build order for the agent (do this literally)

1. Scaffold repo (§4.1), write `requirements.txt`, `.gitignore`, `utils/config.py`, `utils/logger.py`.
2. `data_loader.py` (support Kaggle CLI download + local fallback) → `preprocessing.py` → `feature_engineering.py`; write their tests; commit.
3. `train_baseline.py`, `train_isolation_forest.py`, `evaluate.py`; save `isolation_forest.pkl`, `scaler.pkl`, `threshold.json`; generate `evaluation_results.csv` + figures.
4. `database/database.py` + schema (§2.10) + tests.
5. `api/schemas.py`, `api/inference_service.py` (rolling buffer per sensor), `api/main.py` (REST+WS); tests.
6. `streaming/stream_simulator.py`; wire simulator → API → DB → WS end-to-end.
7. `dashboard/dashboard.py` (live chart, score, KPIs, alerts, model-comparison, health).
8. Add severity + explainable reason across inference + dashboard.
9. `train_lstm_autoencoder.py` + integrate as selectable model; re-evaluate.
10. River online model + adaptive threshold + drift.
11. `docker/Dockerfile` + `docker-compose.yml` (api + dashboard + optional db); latency/stress tests.
12. README + notebooks + final report + slides + package.

**Guardrails:** fixed random seeds everywhere; no future leakage in features; scaler/threshold fit on train/normal only; config-driven paths; every module has a test; keep IF as the always-working default so the demo never depends on the LSTM finishing.
