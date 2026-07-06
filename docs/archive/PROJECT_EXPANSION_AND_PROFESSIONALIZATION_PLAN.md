# PROJECT_EXPANSION_AND_PROFESSIONALIZATION_PLAN.md
## Real-Time Anomaly Detection for IoT Sensor Streams → Professional ML Platform

> A senior-engineer redesign document. This is analysis and decisions, not just a task list. It builds **on top of** the working MVP; nothing functional is removed. Isolation Forest stays the stable production default; more models are added as experiment candidates.

---

## 1. Current Project Understanding

I read `REPO_STATUS_FOR_CHATGPT.md`, `IMPLEMENTATION_SPEC.md`, and `IMPLEMENTATION_PLAN_START_FROM_ZERO.md`. Here is the honest state.

### What already exists and works
- **Backend:** FastAPI with `/health`, `/predict`, `/predict/batch`, `/alerts`, `/alerts/{id}/ack`, `/metrics`, `/readings`, and `WS /ws/stream`. Working.
- **Inference:** `inference_service.py` holds a stateful rolling buffer per sensor and scores each reading. Working.
- **Data pipeline:** `data_loader` (Kaggle + GitHub fallback), `preprocessing` (clean, impute-flag, chronological split), `feature_engineering` (rolling, lag, ROC, EWMA). Working.
- **Models:** Rolling Z-score baseline + Isolation Forest (production default). Trained, thresholded, saved.
- **Database:** SQLite via SQLAlchemy — `readings`, `alerts`, `model_runs`. Tested.
- **Frontend:** React + TS + Vite + Tailwind + shadcn/ui + Recharts. Single-page dashboard with live charts, KPIs, WS reconnection, alert acknowledge. Builds in ~6 s.
- **Docker:** two services (api:8000, frontend:3000), volume binds. `docker compose up --build` works.
- **Tests:** six pytest modules covering API, DB, features, model, preprocessing, stream.

### What is weak or broken (be honest)
1. **Metrics contradiction — must fix first.** `reports/evaluation_results.csv` reports Isolation Forest **F1 = 0.9468, Recall = 1.0, PR-AUC = 0.9994**, but `models/threshold.json` records the *same model* at **F1 = 0.6097, Recall = 0.9841, Precision = 0.4417**. These cannot both describe the production threshold. One is the threshold-selection run on validation, the other the final evaluation on test — but they're presented as if interchangeable. **A grader who notices this reads it as fabricated or careless results.** This is the single most important thing to reconcile before adding anything.
2. **Only 2 models.** The brief and good grading demand ≥4 with real comparison. Right now the "comparison" is baseline vs one model — thin.
3. **`pytest` fails on bare invocation** (`ModuleNotFoundError: No module named 'src'`). Works via `python -m pytest`, but a grader running `pytest -q` sees six collection errors. Looks broken even though it isn't. Trivial fix (`pyproject.toml`/`conftest.py`), high perception cost.
4. **Single-page frontend.** It's a *dashboard*, not a *platform*. One route, one view. Nothing communicates "system" — no model lab, no data explorer, no experiment history.
5. **No model registry.** Models are loose `.pkl`/`.json` files. No single source of truth for "what's in production, trained when, with what metrics."
6. **Frontend bundle warning** (Recharts >500 kB). Cosmetic, but easy to silence with code-splitting.

### What makes it look *small*
- One page. One real model. Metrics that don't reconcile. No experiment trail. No notion of "product." It reads as a competent homework demo, not a system.

### What must change to look *large and professional*
- Multi-model training with a **unified, reproducible evaluation** that produces one coherent results table.
- A **model registry** as the backbone (every artifact tracked with metadata + metrics).
- A **multi-page web platform** where the live dashboard is *one* of eight sections.
- **Experiment tracking** the frontend can read and render.
- Clean, reconciled numbers and a green `pytest -q`.

---

## 2. Professional Project Vision

**Redefinition:** *An end-to-end real-time IoT anomaly detection platform* — data ingestion and validation, a reusable feature pipeline, a multi-model training engine with experiment tracking and a model registry, real-time inference with selectable production model and explainability, a stream simulator, an alert management system, a professional multi-section web application, and a reports/export module — all dockerized and tested.

**Why this is more than a dashboard:** a dashboard *renders state*. A platform *manages the ML lifecycle*: it trains and compares many models, records every experiment, promotes one to production through a registry, serves it with explainable low-latency inference, manages alerts as first-class objects, and exposes all of it through a web product with distinct operational surfaces. The live monitor is the tip; the registry, experiment store, model lab, and data explorer are the iceberg.

Module set: Data Ingestion · Dataset Preparation · Feature Engineering Pipeline · Multi-Model Training Engine · Experiment Tracking · Model Comparison · Model Registry · Real-Time Inference API · Stream Simulator · Alert Management · Monitoring Web App · Reports/Export · Testing & Docs · Docker Deployment.

---

## 3. System Modules

### A. Data Layer
Raw NAB CSV, processed CSV, `sample_stream/` slice, data validation (schema, range, monotonic timestamps), anomaly labels from the four NAB windows, and **dataset metadata** (`data/dataset_metadata.json`: row counts, date range, split boundaries, label prevalence). *Add validation + metadata; rest exists.*

### B. Feature Engineering Layer
Rolling (mean/std/min/max) at windows 5/15/60, EWMA, lag 1–3, rate-of-change, rolling z-score, optional time features — behind a **reusable `FeaturePipeline`** that serializes its exact output column order to `models/feature_columns.json` so training and inference build identical vectors. *Mostly exists; formalize as a class + persisted column order.*

### C. Model Training Layer
Baseline (Z-score/EWMA), classical unsupervised (Isolation Forest, One-Class SVM, LOF, Elliptic Envelope), deep (LSTM Autoencoder), online (River HalfSpaceTrees). Light hyperparameter tuning per model on validation. Uniform artifact saving through the registry writer. *Add OCSVM/LOF/Elliptic + optional deep/online.*

### D. Evaluation Layer
Precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix, false-alarm rate, detection latency, inference time, throughput, and **model ranking**. One `evaluate_all.py` scores every registered model on the *same* test split and writes one coherent table. *Refactor current evaluate to loop all models.*

### E. Model Registry Layer
`models/model_registry.json` — for each model: artifact path, scaler path, threshold, feature set, training/val/test row counts, all metrics, training date, `is_production` flag, notes. Single source of truth. *New.*

### F. Inference Layer
FastAPI predict/batch, **model selection** (serve any registered model), explainability (top-contributing feature/sensor), severity bands, WebSocket broadcast. *Add model selection + registry-driven loading.*

### G. Alert Management Layer
Alert creation, severity, reason, acknowledgement, filtering, history, and **alert statistics** (counts by severity, rate over time). *Extend existing alerts.*

### H. Web Platform Layer
Eight sections (Overview, Live Monitoring, Alert Center, Model Lab, Experiment Results, Data Explorer, System Health, Demo Control Panel). Dashboard = one of them. *Expand from single page to routed app.*

### I. Deployment & Quality Layer
Docker (api + frontend, optional db), tests (green on `pytest -q`), structured logging, README, final report, demo script. *Fix pytest discovery; rest exists.*

---

## 4. Multi-Model ML Strategy

Implement **at least these four** with real evaluation, all unsupervised (fit on the known-normal region, scored on the labeled test split):

| # | Model | Type | Input features | Train data | Threshold selection | Prediction | Advantages | Limitations | Real-time | Artifact |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **Rolling Z-score / EWMA** | Baseline, unsup | value + roll_mean_15/std_15 | train | k tuned on val (F1) | \|z\|>k | trivial, interpretable, O(1) | misses multivariate/slow drift | Excellent | `threshold_zscore.json` |
| 2 | **Isolation Forest** *(production default)* | Unsup ensemble | full feature vector | normal rows | max-F1 on val PR curve | `-score_samples` > τ | fast, robust, sub-ms | less temporal awareness | Excellent | `isolation_forest.pkl` |
| 3 | **One-Class SVM** | Unsup boundary | full vector (scaled) | normal rows | ν + max-F1 on val | decision_function < 0 | strong non-linear boundary | slow on large N, scale-sensitive | Good | `one_class_svm.pkl` |
| 4 | **Local Outlier Factor** | Unsup density | full vector | normal rows (novelty=True) | contamination / val | `-score_samples` > τ | catches *local* anomalies | batch-ish, k-sensitive | Moderate | `lof.pkl` |

**Also include (cheap, strengthens the set):**
- **Elliptic Envelope** — Gaussian covariance outlier model; near-free to add; good contrast point. Artifact `elliptic_envelope.pkl`.

**Advanced (do if time; never block the demo):**
- **5. LSTM Autoencoder** — reconstruct normal windows; error > 99th-pct-normal ⇒ anomaly. Captures temporal structure the others miss. Heavier train. Artifact `lstm_autoencoder.pt` + `ae_threshold.json`.
- **6. River HalfSpaceTrees** — true online detector, updates per reading, adaptive rolling-quantile threshold, enables **drift detection**. The standout differentiator. Artifact `river_hst.pkl` (periodic snapshot).

### Should Random Forest / XGBoost be added? — Honest analysis
**Recommendation: No for the graded core; optional experiment at most, clearly caveated.**
- **Are NAB labels strong enough for supervised classification?** Weakly. Labels are a handful of *windows*, not per-point ground truth. Positive class is a tiny fraction.
- **Too small / univariate?** Yes for supervised. One raw feature (temperature) expanded into engineered features from ~22k rows, with only ~4 anomaly windows, gives very few *independent* positive examples.
- **Overfitting risk?** High. A tree ensemble will happily memorize the four windows and post inflated metrics that don't generalize — exactly the kind of result a sharp grader distrusts.
- **Verdict:** the project's honesty is a selling point. Frame it as **unsupervised anomaly detection** (correct for this data) and, if you want to show you considered supervision, add **one** XGBoost run explicitly labeled *"supervised baseline — included to demonstrate why supervised framing overfits sparse-window labels; not a production candidate."* That earns more credit than a suspiciously perfect classifier.

---

## 5. Experiment Tracking and Model Registry

**Reconcile the metrics contradiction here.** Establish one rule: **threshold is chosen on validation; all reported metrics come from a single `evaluate_all.py` run on the held-out test split.** `threshold.json` stores only the threshold + the validation number used to pick it (clearly labeled `selection_metrics`); `evaluation_results.csv` stores the authoritative test metrics. Never present validation and test numbers as the same thing again.

Artifacts to create/maintain:
- `reports/evaluation_results.csv` — one row per model, test metrics (rewritten by `evaluate_all.py`).
- `reports/model_comparison.json` — ranked comparison + metadata for the frontend.
- `models/model_registry.json` — the registry (see §3E).
- `reports/figures/` — PR/ROC per model + `model_comparison_bar.png` + confusion matrices.

Each experiment record fields: model name, model type, feature set, training rows, validation rows, test rows, threshold, precision, recall, F1, ROC-AUC, PR-AUC, false-alarm rate, detection latency, avg inference time, model file path, scaler path, creation time, notes. The **Model Lab** page reads `model_comparison.json`.

---

## 6. Website / Frontend Expansion

Convert the single page into a routed platform (React Router). Keep the current dashboard as the Live Monitoring page. Design language per `SKILL.md`: React + TypeScript + Vite + Tailwind + shadcn/ui, **professional dark monitoring UI** — muted slate/zinc base, one restrained accent (e.g. cyan or amber) for anomalies, dense data-ink, no purple gradients, no generic AI look.

- **Page 1 — Platform Overview:** project summary, data source, active production model, total readings processed, total anomalies, system status, quick demo-start guide.
- **Page 2 — Live Monitoring** *(existing):* live temperature chart, anomaly-score chart, live markers, stream + WS status, active model, current severity, latest alerts.
- **Page 3 — Alert Center:** alert table, filters, severity badges, acknowledge, details drawer with anomaly reason and a timeline around the anomaly.
- **Page 4 — Model Lab:** compare all trained models — model cards, ranking, metric charts, confusion matrices, PR-AUC/ROC-AUC comparison, selected production model (with a **promote** action if `/models/select` is built).
- **Page 5 — Data Explorer:** dataset summary, the four anomaly windows, feature list, label distribution, train/val/test split, sample rows, preprocessing explanation.
- **Page 6 — Experiment Results:** experiment table, feature-set comparison, threshold comparison, saved figures, export-report button.
- **Page 7 — System Health:** API/DB/WS status, inference latency, stream rate, recent backend log tail, Docker service status if available.
- **Page 8 — Demo Control Panel:** start/stop simulator instructions, stream-speed explanation, normal vs anomaly scenario, demo checklist.

Fix the bundle warning with route-level `React.lazy` + dynamic import (also naturally splits Recharts out of the initial chunk).

---

## 7. Backend / API Expansion

Existing endpoints stay. Add:

| Endpoint | Purpose | Required? |
|---|---|---|
| `GET /models` | list trained models + basic metrics | **Required** (Model Lab) |
| `GET /models/registry` | full registry records | **Required** |
| `GET /models/comparison` | ranked comparison JSON | **Required** (Model Lab) |
| `POST /models/select/{model_name}` | hot-swap production model | Recommended |
| `GET /experiments` | experiment history | **Required** (Experiment Results) |
| `GET /data/summary` | dataset metadata | **Required** (Data Explorer) |
| `GET /data/features` | feature list + descriptions | Recommended |
| `GET /data/splits` | split boundaries + counts | Recommended |
| `GET /system/status` | API/DB/WS/latency/stream health | **Required** (System Health) |
| `GET /reports/summary` | headline results for Overview/export | Recommended |

Most are thin readers over registry/CSV/DB files — low effort, high perceived surface area. `POST /models/select` is the only one needing real logic (reload model in `InferenceService`).

---

## 8. Database Expansion

Keep `readings`, `alerts`, `model_runs`. Add:

| Table | Purpose | Key columns | Required? |
|---|---|---|---|
| `experiments` | one row per training/eval run | id, model, feature_set, train/val/test rows, threshold, all metrics, created_at, notes | **MVP-required** (backs Experiment Results; can also be a CSV-first, DB-later) |
| `model_registry` | production/candidate registry | id, model, artifact_path, scaler_path, threshold, metrics, is_production, trained_at | **MVP-required** (or JSON file first) |
| `system_events` | audit log (model swaps, stream start/stop, errors) | id, ts, event_type, detail | Advanced |
| `stream_sessions` | one row per simulator run | id, started_at, ended_at, speed, readings_sent, anomalies | Advanced |

Pragmatic path: implement `experiments` and `model_registry` as **JSON/CSV first** (fast, matches existing file-based artifacts), mirror into SQLite tables only if time allows. `system_events` and `stream_sessions` are polish.

---

## 9. Professional Repository Upgrade (additive only)

```
real-time-iot-anomaly-detection/
├── src/
│   ├── models/
│   │   ├── train_one_class_svm.py      # NEW
│   │   ├── train_lof.py                # NEW
│   │   ├── train_elliptic_envelope.py  # NEW (optional)
│   │   ├── train_lstm_autoencoder.py   # NEW (advanced)
│   │   ├── train_river_online.py       # NEW (advanced)
│   │   └── evaluate_all.py             # NEW (unified evaluation)
│   ├── experiments/                    # NEW
│   │   └── experiment_tracker.py       # writes experiments + comparison json
│   ├── registry/                       # NEW
│   │   └── model_registry.py           # read/write model_registry.json, promote
│   └── reports/                        # NEW
│       └── report_builder.py           # export PDF/HTML summary
├── frontend/src/
│   ├── pages/                          # NEW: Overview, LiveMonitoring, AlertCenter,
│   │                                   #      ModelLab, ExperimentResults, DataExplorer,
│   │                                   #      SystemHealth, DemoControl
│   ├── components/                     # shared cards, charts, badges
│   └── lib/api.ts                      # typed API client
├── configs/                            # NEW: model_configs.yaml, app_config.yaml
├── scripts/                            # NEW: train_all.sh, run_demo.sh, reset_db.sh
├── docs/                               # NEW: architecture.md, api.md, models.md
├── models/model_registry.json          # NEW
├── reports/model_comparison.json       # NEW
└── data/dataset_metadata.json          # NEW
```
Nothing existing is deleted or moved.

---

## 10. What To Implement Next (prioritized)

**Priority 1 — Make current MVP bulletproof (do first, ~1 day)**
- **Reconcile the metrics** (§5) — one honest test table. *Highest priority; it's a credibility issue.*
- Fix `pytest -q` discovery: add `pyproject.toml` with `[tool.pytest.ini_options] pythonpath=["."]` (or a root `conftest.py`). Green on bare `pytest`.
- Confirm `predict.py` present (it is), silence the sklearn feature-name warning at the source (pass DataFrame with columns into `.transform`), pin any missing deps.
- Code-split frontend to clear the chunk warning.
- Re-verify `docker compose up --build`.

**Priority 2 — Multi-model training (~2–3 days)**
- Add One-Class SVM, LOF, Elliptic Envelope (all follow the IF recipe).
- Write `evaluate_all.py` — same test split, one results table, per-model figures.
- (If time) LSTM Autoencoder.

**Priority 3 — Registry & comparison (~1–2 days)**
- `model_registry.py` + `experiment_tracker.py`; populate `model_registry.json`, `model_comparison.json`.
- Expose `/models`, `/models/registry`, `/models/comparison`, `/experiments`.

**Priority 4 — Website expansion (~3–4 days)**
- Router + shared layout; build Overview, Model Lab, Data Explorer, Experiment Results, Demo Control Panel; keep Live Monitoring; add Alert Center + System Health.

**Priority 5 — Advanced product features (time-permitting)**
- `POST /models/select` production swap; River online model + drift detection; alert timeline; export report.

---

## 11. Implementation Decision Table

| Feature | Value for grading | Difficulty | Priority | Implement now? | Notes |
|---|---|---|---|---|---|
| Reconcile metrics | Critical (credibility) | Low | P1 | **Yes** | Must precede everything |
| pytest green on `pytest -q` | High (perception) | Low | P1 | **Yes** | `pyproject.toml` pythonpath |
| 4 ML models (Zscore, IF, OCSVM, LOF) | Very high (required) | Med | P2 | **Yes** | Core of the brief |
| Elliptic Envelope (5th) | Medium | Low | P2 | Yes | Nearly free |
| Unified `evaluate_all.py` | Very high | Med | P2 | **Yes** | Produces the one true table |
| Model registry | High | Low–Med | P3 | **Yes** | JSON-first |
| Experiment tracking | High | Low–Med | P3 | **Yes** | Backs Model Lab / Experiments |
| Model Lab page | Very high (looks pro) | Med | P4 | **Yes** | Biggest "large system" signal |
| Data Explorer page | High | Low–Med | P4 | **Yes** | Cheap, impressive |
| Experiment Results page | Medium–High | Low | P4 | Yes | Reads existing JSON/CSV |
| Alert Center page | Medium | Low–Med | P4 | Yes | Extends existing alerts |
| System Health page | Medium | Low | P4 | Yes | Thin readers |
| Demo Control Panel | Medium (smooth demo) | Low | P4 | Yes | Helps you present |
| Docker (already works) | Required | — | — | Keep | Verify only |
| LSTM Autoencoder | High (advanced credit) | High | P2/P5 | If time | Never block demo |
| River online + drift | High (differentiator) | Med–High | P5 | If time | Standout feature |
| Supervised classifier (XGB) | Low/negative | Low | Opt | Only as caveated experiment | Overfits sparse labels |
| TimescaleDB | Low (overkill) | Med | — | **No** | SQLite is fine at this scale |
| MQTT/Kafka | Low (overkill) | High | — | **No** | HTTP stream sim is enough |

---

## 12. Risks and Professional Decisions

- **NAB is univariate.** Real limit: no cross-sensor correlation features, and density/boundary models have less to work with. **Mitigation:** rich causal feature engineering (windows 5/15/60 + lags + ROC + z-score) turns one raw signal into a meaningful vector, and the *platform* (multi-model, registry, experiments, web product) is what demonstrates seniority — not signal count.
- **Add NASA Bearing later?** Yes — as a **secondary dataset** to prove the pipeline is dataset-agnostic (swap loader, everything else holds). Multivariate vibration also unlocks correlation features and a harder unsupervised story. Do it *after* the four-model platform is solid; don't let it destabilize the working MVP.
- **Is 4 models enough?** Yes for the requirement; 5–6 (add Elliptic Envelope, then LSTM-AE) reads as thorough. Beyond 6 is diminishing returns.
- **React or Streamlit?** **React** — you already have a working React app and `SKILL.md` mandates it. A multi-page React platform looks far more like a product than Streamlit.
- **Best achievable scope:** four unsupervised models + registry + experiment tracking + 8-page React platform + Docker + green tests + honest reconciled results. LSTM-AE and River as advanced credit.
- **Exclude to avoid overengineering:** TimescaleDB, Kafka/MQTT, user auth, complex alert routing. They add infra, not grade.

---

## 13. Final Recommended Direction

**Final scope:** an end-to-end real-time IoT anomaly detection *platform* — multi-model training, experiment tracking, model registry, real-time explainable inference with selectable production model, alert management, and an eight-section professional React web app — dockerized, tested, with one honest results table. Isolation Forest remains the stable production default; other models are experiment candidates.

**Exact 4 models to implement first:** Rolling Z-score/EWMA → Isolation Forest (production) → One-Class SVM → Local Outlier Factor. (Add Elliptic Envelope as an easy 5th; LSTM Autoencoder + River as advanced.)

**Exact website pages to build:** Platform Overview, Live Monitoring (existing), Alert Center, Model Lab, Experiment Results, Data Explorer, System Health, Demo Control Panel.

**Exact API additions:** `/models`, `/models/registry`, `/models/comparison`, `/experiments`, `/data/summary`, `/system/status` (required); `/models/select/{name}`, `/data/features`, `/data/splits`, `/reports/summary` (recommended).

**Exact database additions:** `experiments`, `model_registry` (JSON-first, mirror to SQLite if time); `system_events`, `stream_sessions` (advanced).

**Exact deliverables:** four-model reconciled `evaluation_results.csv` + `model_comparison.json` + `model_registry.json`, per-model PR/ROC + comparison figures, eight-page React platform, expanded FastAPI, green `pytest -q`, working `docker compose up --build`, updated README + final report + demo script + slides.

**Exact next coding steps (in order):**
1. Reconcile metrics; define validation-vs-test rule; rewrite results honestly.
2. Add `pyproject.toml` pythonpath; make `pytest -q` green; silence sklearn warning; code-split frontend.
3. Implement `train_one_class_svm.py`, `train_lof.py`, `train_elliptic_envelope.py`.
4. Implement `evaluate_all.py` (one test split, one table, all figures).
5. Implement `model_registry.py` + `experiment_tracker.py`; write registry + comparison JSON.
6. Add the six required API readers + optional `/models/select`.
7. Introduce React Router + shared dark layout; build Model Lab, Data Explorer, Overview, Experiment Results, Demo Control Panel; keep Live Monitoring; add Alert Center + System Health.
8. (Advanced) LSTM Autoencoder, then River online + drift detection, then export-report.

---

### Non-negotiable guardrails
- Do not remove or regress the working MVP.
- Isolation Forest stays the stable production default.
- No faked or inflated results; validation and test numbers are labeled distinctly and never conflated.
- Fixed seeds (`random_state=42`); no future leakage; scaler/threshold fit on train/normal only.
- Additive repo changes only; every new `src/` module gets a test.

*End of document. Awaiting approval before coding.*
