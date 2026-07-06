# IMPLEMENTATION_PLAN_START_FROM_ZERO.md
## Real-Time Anomaly Detection for IoT Sensor Streams — Zero-to-Running Plan

> Hand this file to a developer (or to Claude Code). Follow it top to bottom. Every command is meant to be copy-pasted. The dataset, models, ports, and schema are fixed — do not substitute them.

---

## 1. Project overview

We build a system that takes historical industrial-machine sensor data, **replays it as if it were a live IoT stream**, scores every incoming reading for anomalies with a trained machine-learning model, stores each reading and its score, raises alerts when something looks wrong, and shows it all on a **live dashboard**.

Plain-terms flow:

```
CSV (historical)  →  stream simulator (replays row by row, fast)
                  →  FastAPI /predict  (loads model, scores reading)
                  →  SQLite            (saves reading + score + alert)
                  →  WebSocket + Streamlit dashboard (live charts + alerts)
```

The "brain" is a **machine-learning model that learns what normal looks like** and flags deviations — not a hand-coded threshold. Isolation Forest is the stable main model; an LSTM Autoencoder and a River online detector are advanced add-ons.

---

## 2. Final selected dataset

**Use this and only this for the main build:**

- **Dataset:** Numenta Anomaly Benchmark (NAB) — Kaggle `boltzmannbrain/nab`
- **File:** `realKnownCause/machine_temperature_system_failure.csv`

**Expected columns (raw):**

| Column | Type | Meaning |
|---|---|---|
| `timestamp` | datetime string | reading time, 5-minute cadence |
| `value` | float | machine temperature reading |

Around **22,695 rows**, spanning ~Dec 2013 → Feb 2014. It contains three known anomalous periods: a planned shutdown, an early warning sign, and a catastrophic failure.

**Why selected:** public, free, timestamped, real, small enough to iterate fast, has **known labeled anomaly windows**, and tells a clean **normal → degradation → failure** story that demos perfectly. Univariate keeps data-wrangling minimal so effort goes into the *system*.

**Backup dataset (only if a harder multivariate variant is wanted):** NASA Bearing Sensor Data — Kaggle `vinayak123tyagi/bearing-dataset`. Swap the loader; everything downstream stays the same.

**NAB known anomaly windows for `machine_temperature_system_failure`** (hard-code these to build the label; times are the labeled anomalous intervals):

```
2013-12-10 06:25:00  →  2013-12-12 05:35:00
2013-12-15 17:50:00  →  2013-12-17 17:00:00
2014-01-27 14:20:00  →  2014-01-29 13:30:00
2014-02-07 14:55:00  →  2014-02-09 14:05:00
```

Any reading whose timestamp falls inside any window ⇒ `label = 1`, else `0`.

---

## 3. Required tools before coding

Install:
- **Python 3.11** (exactly; check `python --version`)
- **VS Code** or **Cursor**
- **Git** + a **GitHub** account
- **Kaggle account** + **API token** (`kaggle.json`)
- **Docker Desktop** (for the final containerized run)

System dependencies:
- Windows: nothing extra beyond Python + Docker Desktop (WSL2 backend recommended).
- macOS/Linux: `build-essential`/Xcode CLT for any wheels that compile.

Optional (advanced sections only):
- **PyTorch** (LSTM Autoencoder), **River** (online model), **MLflow** (experiment tracking), **Mosquitto/MQTT**, **TimescaleDB**.

---

## 4. Environment setup from zero

Run these in order. (Windows PowerShell shown; macOS/Linux notes inline.)

```powershell
# 4.1 Create the project folder
mkdir real-time-iot-anomaly-detection
cd real-time-iot-anomaly-detection
git init

# 4.2 Create virtual environment (Python 3.11)
py -3.11 -m venv .venv
# macOS/Linux:  python3.11 -m venv .venv

# 4.3 Activate it — Windows PowerShell
.\.venv\Scripts\Activate.ps1
# Windows cmd:  .\.venv\Scripts\activate.bat
# macOS/Linux:  source .venv/bin/activate

# 4.4 Upgrade pip
python -m pip install --upgrade pip
```

**4.5 Create `requirements.txt`:**

```text
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.1
joblib==1.4.2
fastapi==0.111.0
uvicorn[standard]==0.30.1
pydantic==2.8.2
websockets==12.0
streamlit==1.37.0
plotly==5.22.0
sqlalchemy==2.0.31
requests==2.32.3
python-dotenv==1.0.1
pytest==8.2.2
httpx==0.27.0
# --- advanced (optional) ---
# torch==2.3.1
# river==0.21.1
# mlflow==2.14.1
```

```powershell
# 4.6 Install
pip install -r requirements.txt
```

**4.7 Set up Kaggle API token:**
1. Kaggle → Account → *Create New API Token* → downloads `kaggle.json`.
2. Place it:
   - Windows: `C:\Users\<YOU>\.kaggle\kaggle.json`
   - macOS/Linux: `~/.kaggle/kaggle.json` then `chmod 600 ~/.kaggle/kaggle.json`
3. `pip install kaggle`

**4.8 Download the dataset:**

```powershell
kaggle datasets download -d boltzmannbrain/nab -p data/raw --unzip
# The file we use ends up at:
# data/raw/realKnownCause/machine_temperature_system_failure.csv
```

If Kaggle CLI is blocked, download manually from the dataset page and drop the CSV into `data/raw/`.

**4.9 Create `.env`:**

```text
RAW_CSV=data/raw/realKnownCause/machine_temperature_system_failure.csv
PROCESSED_CSV=data/processed/nab_processed.csv
DB_URL=sqlite:///reports/anomaly.db
API_URL=http://localhost:8000
MODEL_DIR=models
SENSOR_ID=machine_temperature
```

**4.10 Create `.gitignore`:**

```text
.venv/
__pycache__/
*.pyc
.env
data/raw/
data/processed/
*.db
models/*.pkl
models/*.pt
models/*.json
reports/figures/
.ipynb_checkpoints/
.DS_Store
```

---

## 5. Final repository structure

```
real-time-iot-anomaly-detection/
├── data/
│   ├── raw/
│   ├── processed/
│   └── sample_stream/
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_preprocessing_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation.ipynb
├── src/
│   ├── data/
│   │   ├── data_loader.py
│   │   └── preprocessing.py
│   ├── features/
│   │   └── feature_engineering.py
│   ├── models/
│   │   ├── train_baseline.py
│   │   ├── train_isolation_forest.py
│   │   ├── train_lstm_autoencoder.py
│   │   ├── evaluate.py
│   │   └── predict.py
│   ├── streaming/
│   │   ├── stream_simulator.py
│   │   └── mqtt_publisher.py
│   ├── api/
│   │   ├── main.py
│   │   ├── schemas.py
│   │   └── inference_service.py
│   ├── database/
│   │   └── database.py
│   ├── dashboard/
│   │   └── dashboard.py
│   └── utils/
│       ├── config.py
│       └── logger.py
├── models/
├── reports/
│   ├── figures/
│   └── evaluation_results.csv
├── tests/
├── docker/
│   └── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
└── .gitignore
```

Add empty `__init__.py` to every folder under `src/` so imports work as packages.

---

## 6. File-by-file implementation plan

| File | Purpose | Inputs | Outputs | Main functions/classes | Depends on | When |
|---|---|---|---|---|---|---|
| `utils/config.py` | central config from `.env` | `.env` | `Settings` object | `Settings`, `get_settings()` | dotenv | Step 0 |
| `utils/logger.py` | consistent logging | — | logger | `get_logger(name)` | logging | Step 0 |
| `data/data_loader.py` | load raw NAB CSV | `RAW_CSV` | DataFrame | `load_nab()`, `load_bearing()` | pandas | Step 1 |
| `data/preprocessing.py` | clean, label, split, scale | raw df | `nab_processed.csv`, `scaler.pkl` | `add_labels()`, `clean()`, `time_split()`, `fit_scaler()`, `transform()` | pandas, sklearn, joblib | Step 3 |
| `features/feature_engineering.py` | causal rolling/lag/z features | processed df | feature matrix | `make_features(df, windows)`, `FeaturePipeline` | pandas | Step 4 |
| `models/train_baseline.py` | rolling z-score baseline | features | `threshold_zscore.json` | `fit_zscore()`, `score_zscore()` | numpy | Step 5 |
| `models/train_isolation_forest.py` | **main model** | features (train/normal) | `isolation_forest.pkl`, `threshold.json` | `train_if()`, `pick_threshold()` | sklearn, joblib | Step 6 |
| `models/evaluate.py` | metrics + plots | model + labeled test | `evaluation_results.csv`, figures | `evaluate()`, `plot_all()` | sklearn, plotly | Step 7 |
| `models/predict.py` | score one reading | reading + buffer | prediction dict | `Predictor.score()` | joblib | Step 11 |
| `models/train_lstm_autoencoder.py` | advanced model | sequences | `lstm_autoencoder.pt`, `ae_threshold.json` | `LSTMAutoencoder`, `train_ae()` | torch | advanced |
| `database/database.py` | SQLite layer | prediction/alert | rows | `init_db()`, `insert_reading()`, `insert_alert()`, `get_alerts()`, `ack_alert()`, `latest_metrics()` | sqlalchemy | Step 9 |
| `api/schemas.py` | pydantic models | — | — | `Reading`, `Prediction`, `Alert` | pydantic | Step 10 |
| `api/inference_service.py` | orchestrate scoring | reading | prediction | `InferenceService` (per-sensor buffer) | predict, features | Step 11 |
| `api/main.py` | FastAPI app + WS | HTTP/WS | JSON/WS | routes §13 | fastapi, db, inference | Step 10 |
| `streaming/stream_simulator.py` | replay CSV as stream | processed csv | POSTs to API | `simulate(speed, loop, start)` | requests | Step 12 |
| `streaming/mqtt_publisher.py` | advanced transport | processed csv | MQTT msgs | `publish()` | paho-mqtt | advanced |
| `dashboard/dashboard.py` | Streamlit UI | API/DB | charts | `main()`, page fns | streamlit, plotly | Step 14 |

---

## 7. Build order from zero

```
Step 0  : scaffold repo + config + logger + __init__.py files
Step 1  : download/load dataset
Step 2  : explore dataset (notebook 01)
Step 3  : preprocess data (label, clean, split, scale)
Step 4  : feature engineering (causal features)
Step 5  : rolling z-score baseline
Step 6  : Isolation Forest (main model)
Step 7  : evaluate
Step 8  : save model / scaler / threshold
Step 9  : create SQLite database
Step 10 : FastAPI backend (health/predict/ws)
Step 11 : inference service (rolling buffer)
Step 12 : stream simulator
Step 13 : connect simulator → API → DB → WS end-to-end
Step 14 : Streamlit dashboard (live monitoring)
Step 15 : alerts + severity
Step 16 : model comparison page
Step 17 : tests
Step 18 : Dockerize
Step 19 : README + report
Step 20 : demo prep
(advanced) LSTM Autoencoder, River online model, drift detection
```

**Rule:** keep Isolation Forest working as the default at all times, so the demo never depends on the LSTM finishing.

---

## 8. Commands to run at every stage

```bash
# setup
pip install -r requirements.txt

# data + training
python -m src.data.data_loader
python -m src.data.preprocessing
python -m src.features.feature_engineering
python -m src.models.train_baseline
python -m src.models.train_isolation_forest
python -m src.models.evaluate

# run system (3 terminals)
uvicorn src.api.main:app --reload                    # terminal 1: API :8000
python -m src.streaming.stream_simulator --speed 50  # terminal 2: stream
streamlit run src/dashboard/dashboard.py             # terminal 3: dashboard :8501

# advanced
python -m src.models.train_lstm_autoencoder

# tests
pytest -q

# docker (one command runs everything)
docker compose up --build
```

---

## 9. Data processing logic (NAB, exact)

In `preprocessing.py`, in this order:

1. **Load** raw CSV via `data_loader.load_nab()`.
2. **Parse timestamp:** `df["timestamp"] = pd.to_datetime(df["timestamp"])`.
3. **Sort** ascending by `timestamp`; reset index.
4. **Rename** if needed so the value column is exactly `value`.
5. **Add `sensor_id`** column = `"machine_temperature"` (from `.env SENSOR_ID`).
6. **Add label** using the four NAB windows from §2: `label = 1` if timestamp is inside any window else `0`.
7. **Missing values:** forward-fill gaps ≤ 3 steps; longer gaps → linear interpolate; add `imputed` flag column.
8. **Duplicates:** drop duplicate timestamps, keep last.
9. **Chronological split** (no shuffling): first 70% train, next 15% validation, last 15% test. Save split boundaries.
10. **Fit `StandardScaler` on TRAIN ONLY** (and specifically on the known-normal burn-in region — the first labeled-normal ~15%). Persist as `models/scaler.pkl`.
11. Transform `value` → `value_scaled`.
12. **Save** the full processed frame (with `timestamp,value,value_scaled,sensor_id,label,split,imputed`) to `data/processed/nab_processed.csv`. Also copy the test slice to `data/sample_stream/` for the simulator.

**Leakage guard:** the scaler and every threshold are fit on train/normal only; test rows never influence them.

---

## 10. Feature engineering logic (causal only)

In `feature_engineering.py`, windows = **5, 15, 60** readings. All operations use only past/current values (`.rolling(w)`, `.shift(k)`) — never center or look ahead.

For each window `w`:
- `roll_mean_w`  = `value.rolling(w).mean()`
- `roll_std_w`   = `value.rolling(w).std()`
- `roll_min_w`   = `value.rolling(w).min()`
- `roll_max_w`   = `value.rolling(w).max()`
- `ewma_w`       = `value.ewm(span=w).mean()`
- `zscore_w`     = `(value - roll_mean_w) / (roll_std_w + 1e-9)`

Non-windowed:
- `lag_1, lag_2, lag_3` = `value.shift(1/2/3)`
- `roc` (rate of change) = `value.diff()`
- optional time features: `hour`, `dayofweek` (only if they help; NAB is machine data so usually minor).

Drop the warm-up rows containing NaN (first ~60). Feature list is the model input vector; persist the exact column order to `models/feature_columns.json` so the API builds vectors identically at inference.

---

## 11. ML training plan

### 11.1 Rolling z-score baseline (`train_baseline.py`)
- **Input:** `value` + `roll_mean_15`, `roll_std_15`.
- **Train data:** train split.
- **Logic:** flag anomaly if `|zscore_15| > 3`.
- **Threshold:** k=3 (tune on validation for best F1); save `{"k":3,"window":15}` → `threshold_zscore.json`.
- **Predict:** score = `|zscore_15|`; label = score > k.
- **Output:** baseline metrics for comparison.

### 11.2 Isolation Forest — MAIN MODEL (`train_isolation_forest.py`)
- **Input features:** full engineered vector (§10), `feature_columns.json` order.
- **Train data:** train split, preferably known-normal rows only (unsupervised).
- **Model:** `IsolationForest(n_estimators=200, contamination="auto", random_state=42, n_jobs=-1)`.
- **Anomaly score:** `score = -model.score_samples(X)` (higher = more anomalous).
- **Threshold:** compute scores on validation, pick the threshold that maximizes F1 vs labels (or the 99th percentile of normal scores if unlabeled). Save `{"threshold": <float>, "model":"isolation_forest"}` → `threshold.json`.
- **Save:** `joblib.dump(model, "models/isolation_forest.pkl")`.
- **Expected output:** `isolation_forest.pkl`, `threshold.json`, validation PR curve.

### 11.3 LSTM Autoencoder — ADVANCED (`train_lstm_autoencoder.py`)
- **Input:** sequences (e.g. length 30) of scaled value (+ features optionally).
- **Train data:** normal windows only.
- **Model:** encoder LSTM → latent → decoder LSTM; MSE reconstruction loss.
- **Threshold:** reconstruction error distribution on normal data → 99th percentile → `ae_threshold.json`.
- **Predict:** anomaly score = reconstruction error of the window.
- **Save:** `torch.save(model.state_dict(), "models/lstm_autoencoder.pt")`.

### 11.4 River HalfSpaceTrees — OPTIONAL online (added value)
- **Input:** one scaled reading (+ light features) at a time.
- **Model:** `river.anomaly.HalfSpaceTrees(seed=42)`; `.learn_one()` then `.score_one()` per reading.
- **Threshold:** adaptive — rolling quantile (e.g. 98th pct) of recent scores; enables **drift detection**.
- **Save:** pickle the River model periodically; it updates live, no batch retrain.

---

## 12. Evaluation plan (`evaluate.py`)

Compute on the labeled **test** split:
- Precision, Recall, F1-score
- Confusion matrix
- ROC-AUC, PR-AUC (PR-AUC is the headline metric — data is highly imbalanced)
- **False alarm rate** = FP / (FP + TN)
- **Detection latency** = steps from anomaly-window onset to first correct flag
- **Inference time** = mean ms per `score()` call
- **Throughput** = readings/second the pipeline sustains

Save a tidy `reports/evaluation_results.csv` with one row per model (baseline, IF, LSTM-AE). Generate and save to `reports/figures/`:
- `raw_series_with_windows.png` (value with anomaly windows shaded)
- `anomaly_score_timeline.png` (score line + red flagged points + threshold)
- `confusion_matrix_if.png`
- `pr_curve.png`, `roc_curve.png`
- `model_comparison_bar.png`

**Why not accuracy:** anomalies are <1% of rows, so a "never anomaly" model scores >99% accuracy while catching nothing. Grade on recall, PR-AUC, and false-alarm rate.

---

## 13. API plan (FastAPI, `api/main.py`)

Routes:

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness + loaded model name |
| POST | `/predict` | score one reading |
| POST | `/predict/batch` | score a list |
| GET | `/alerts?limit=100` | recent alerts |
| POST | `/alerts/{id}/ack` | acknowledge an alert |
| GET | `/metrics` | latest model metrics + live counters |
| WS | `/ws/stream` | broadcast each scored reading to dashboards |

**`POST /predict` request:**
```json
{ "timestamp": "2014-01-27T14:25:00", "sensor_id": "machine_temperature", "value": 72.13 }
```
**Response:**
```json
{
  "timestamp": "2014-01-27T14:25:00",
  "sensor_id": "machine_temperature",
  "value": 72.13,
  "anomaly_score": 0.83,
  "is_anomaly": true,
  "severity": "high",
  "reason": "zscore_15=6.2; roll_std_15 spike",
  "model": "isolation_forest",
  "inference_ms": 1.4
}
```
**`GET /health`:** `{ "status": "ok", "model": "isolation_forest" }`

On each `/predict`: build features from the per-sensor buffer, score, derive severity, insert into `readings`, insert `alerts` if anomalous, broadcast over `/ws/stream`.

---

## 14. Database plan (SQLite via SQLAlchemy)

`DB_URL=sqlite:///reports/anomaly.db`. Three tables:

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

- **readings** — every scored reading (the full history the dashboard charts).
- **alerts** — one row per anomalous reading, with severity and ack state.
- **model_runs** — evaluation metrics per training run (populated by `evaluate.py`), powering the model-comparison page.

---

## 15. Stream simulator plan (`stream_simulator.py`)

Replays `data/processed/nab_processed.csv` (or the `sample_stream` slice) as a live feed.

- Read rows in timestamp order.
- For each row, `POST {timestamp, sensor_id, value}` to `API_URL/predict`.
- **`--speed`** multiplier: real cadence is 5 min; `--speed 50` sleeps `300/50 = 6 s`; use large values (e.g. 300) for a fast demo.
- **`--loop`** restart at end; **`--start-index N`** begin partway (jump near an anomaly window for demos).
- Print a log line per reading: `ts | value | score | is_anomaly | severity`.
- Handle API errors: retry with backoff, skip on repeated failure, never crash.

CLI: `python -m src.streaming.stream_simulator --speed 50 --loop --start-index 0`

---

## 16. Dashboard plan (Streamlit, multi-page)

**Page 1 — Live monitoring:** live temperature line (Plotly), anomaly-score line below it, **red markers** on flagged points, threshold band, auto-refresh from WS/DB.
**Page 2 — Alert history:** table (ts, sensor, severity, score, reason, ack button), filter by severity.
**Page 3 — Model comparison:** metrics table from `model_runs` + PR/ROC images from `reports/figures`.
**Page 4 — System health:** stream rate (readings/s), avg inference latency, model status, DB row count.
**Page 5 — Project explanation:** short write-up of problem, dataset, pipeline, innovation (doubles as demo narration).

**KPI cards (top of Page 1):** total readings · total anomalies · current model name · avg inference latency · stream rate · current severity.

---

## 17. Testing plan (`tests/`, pytest)

- `test_preprocessing.py` — no NaNs post-clean; chronological order; labels match windows; scaler fit on train only.
- `test_features.py` — features match hand-computed values on a small fixture; **no future leakage** (feature at t uses only ≤ t).
- `test_model.py` — model loads; `score()` returns float; a known-anomalous fixture scores higher than a known-normal one.
- `test_api.py` — `/health` 200; `/predict` returns valid schema; malformed body → 422.
- `test_database.py` — insert then read back; alert FK integrity; ack updates flag.
- `test_stream.py` — emits in timestamp order; speed scaling respected.
- **Latency test** — assert mean IF inference < 50 ms.
- **Stress test** — push 10k readings, assert no drops and throughput ≥ target.

Run: `pytest -q`

---

## 18. Docker plan

**`docker/Dockerfile`:**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000 8501
```

**`docker-compose.yml`:**
```yaml
services:
  api:
    build: { context: ., dockerfile: docker/Dockerfile }
    command: uvicorn src.api.main:app --host 0.0.0.0 --port 8000
    ports: ["8000:8000"]
    volumes: ["./models:/app/models", "./reports:/app/reports", "./data:/app/data"]
    environment: ["DB_URL=sqlite:///reports/anomaly.db"]

  dashboard:
    build: { context: ., dockerfile: docker/Dockerfile }
    command: streamlit run src/dashboard/dashboard.py --server.port 8501 --server.address 0.0.0.0
    ports: ["8501:8501"]
    depends_on: ["api"]
    environment: ["API_URL=http://api:8000"]

  # optional advanced:
  # db:
  #   image: timescale/timescaledb:latest-pg16
  #   environment: [POSTGRES_PASSWORD=pass]
  #   ports: ["5432:5432"]
```

Run everything: `docker compose up --build`. (Stream simulator can run from host against `localhost:8000`, or add it as a fourth service.)

---

## 19. README plan

README.md must contain: project **title**; one-paragraph **description**; **architecture** diagram (the flow in §1); **dataset** description + Kaggle link + the anomaly windows; **setup** (env + Kaggle token + download); **training** commands; **running the API**; **running the dashboard**; **running the simulator**; **Docker** one-liner; **testing** command; **results** table (from `evaluation_results.csv`); **screenshots** of the dashboard; **demo instructions**.

---

## 20. Final demo plan (≈4 min)

1. Open dashboard Page 5, state the problem + dataset in two sentences.
2. `docker compose up` (or start API + dashboard) — show `/health` green.
3. Start simulator at high speed near a normal region — score line hugs zero.
4. `--start-index` jump to a degradation window — score climbs, **points turn red**, an **alert row appears with severity + reason**.
5. Go to Alert history, acknowledge one alert; show DB has the rows.
6. Model comparison page: IF vs baseline (and LSTM-AE if built) — PR-AUC/recall.
7. (If built) toggle River online model + adaptive threshold tracking drift.
8. Close with results table and limitations (univariate, sparse labels, single machine). State the innovation: real-time replay, explainable reason, severity, online learning.

---

## 21. Two-week schedule (MVP-focused)

- **D1** repo scaffold, config, logger, requirements, Kaggle download.
- **D2** data_loader + EDA notebook 01.
- **D3** preprocessing (label/clean/split/scale) + its tests.
- **D4** feature engineering + tests.
- **D5** baseline + Isolation Forest training, save artifacts.
- **D6** evaluate.py + figures + evaluation_results.csv.
- **D7** database.py + schema + tests.
- **D8** FastAPI (`/health`, `/predict`, WS) + schemas.
- **D9** inference_service (rolling buffer) + `/predict` tests.
- **D10** stream simulator + end-to-end wiring.
- **D11** Streamlit Page 1 (live monitoring).
- **D12** alerts + severity + explainable reason + Alert history page.
- **D13** model-comparison page, docker compose, latency/stress tests.
- **D14** README, slides, demo video, final report, package.

## 22. Three-week schedule (adds advanced)

- **Week 1 (D1–7):** everything through DB + baseline + Isolation Forest + evaluation, fully tested.
- **Week 2 (D8–14):** FastAPI + WS, simulator, dashboard Pages 1–2, severity, explainable reason, model-comparison page, docker, integration tests.
- **Week 3:**
  - **D15–16** LSTM Autoencoder: train, evaluate, integrate as selectable model.
  - **D17–18** River HalfSpaceTrees online model + adaptive threshold + drift detection.
  - **D19** System-health page + KPI polish.
  - **D20** stress/latency hardening; full pytest pass; docker final.
  - **D21** slides + demo video + final report + submission package.

---

## 23. Final submission checklist

- [ ] Source code (full repo, §5 structure)
- [ ] Dataset link (`boltzmannbrain/nab`) + `data/sample_stream/` sample
- [ ] Processed sample data (`nab_processed.csv` slice)
- [ ] Trained models (`isolation_forest.pkl`, `scaler.pkl`, `threshold.json`; optional `lstm_autoencoder.pt`)
- [ ] Evaluation report (`reports/evaluation_results.csv` + figures)
- [ ] Dashboard screenshots (all 5 pages)
- [ ] README.md
- [ ] requirements.txt
- [ ] docker-compose.yml + Dockerfile
- [ ] Tests (`tests/`, passing `pytest -q`)
- [ ] Final report (PDF)
- [ ] Slides
- [ ] Demo video

Package as `LastName_IoT_Anomaly_Detection/` mirroring the repo tree, README at root, plus a one-page `SUBMISSION.md` reproducing this checklist ticked.

---

### Guardrails (do not violate)
- Fixed random seeds everywhere (`random_state=42`).
- No future leakage in features; scaler/threshold fit on train/normal only.
- Config-driven paths (`.env` + `config.py`), never hard-coded.
- Isolation Forest stays the always-working default model.
- Every `src/` module has a matching test.
