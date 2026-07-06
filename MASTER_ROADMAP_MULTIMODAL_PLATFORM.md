# MASTER_ROADMAP_MULTIMODAL_PLATFORM.md
# Real-Time Industrial Anomaly Detection Platform for Multi-Modal IoT Monitoring
### The single master build document — current MVP → full multi-modal platform

> **What this is.** One ordered roadmap that folds together every plan from this project: the implementation spec, the professionalization plan, the execution playbook, the evaluation audit, the GUI enhancement, and the extension roadmap. It sequences all of it into phases you execute top to bottom. Hand it to Claude Code and work phase by phase; each task has a **Definition of Done (DoD)** that must pass before the next begins.
>
> **Product one-liner.** A multi-modal industrial condition-monitoring platform detecting machine failures from sensor streams, vibration signals, and visual-inspection images — with multi-model ML, experiment tracking, a model registry, real-time FastAPI inference, WebSocket monitoring, alert management, operator feedback, and a professional React platform.
>
> **Non-negotiable principles (apply to every task):**
> 1. **Additive only** — never regress the working MVP.
> 2. **Honest metrics** — one real NAB test split; validation picks thresholds, test reports results; never conflate them; no leakage.
> 3. **Reproducible** — `random_state=42`, pinned deps, no future leakage, scaler/threshold fit on train-normal only.
> 4. **Three modalities are separate modules sharing infrastructure** — no fusion claims.
> 5. **Isolation Forest stays the production default** until challengers are proven on the same honest split.
> 6. **`pytest -q` and `npm run build` stay green** after every phase.

---

## Table of phases

| Phase | Theme | Depends on | Outcome |
|---|---|---|---|
| **0** | Stabilize the MVP | — | Green tests, valid docker, clean build |
| **1** | Fix evaluation credibility | 0 | Honest, defensible metrics table |
| **2** | Multi-model core (classical) | 1 | ≥5 models + registry + experiments + API |
| **3** | GUI enhancement (platform feel) | 2 | Modern dark 8-page platform |
| **4** | Deep + online models | 2 | LSTM-AE, River online, drift, ensemble |
| **5** | Product features | 3,4 | Alert lifecycle, feedback, incident PDF, faults, demo panel |
| **6** | Second modality — vibration | 2 | NASA Bearing → Vibration Health Lab |
| **7** | Third modality — vision (stretch) | 6 | MVTec AD → Visual Inspection Lab |
| **8** | Asset Center + final integration | 6(7) | Unified platform framing |
| **9** | Testing, docs, demo, submission | all | Graded deliverable |

**Scope guidance:** Phases 0–5 = a high-grade project (1–2 weeks). +Phase 6 = capstone-level multi-modal (2 weeks). +Phase 7 = ultimate three-modality (1 month, only with buffer).

---

# TECH STACK (final, locked)

**Backend:** Python 3.11 · FastAPI · Uvicorn · Pydantic v2 · SQLAlchemy · SQLite · pandas · NumPy · scikit-learn · joblib · **PyTorch** (LSTM-AE, CNN-AE) · **River** (online) · **scipy** (FFT/vibration) · **torchvision** (image embeddings) · reportlab or weasyprint (incident PDF).
**Frontend:** React + TypeScript + Vite · Tailwind · shadcn/ui · Recharts · **react-router-dom** · **@tanstack/react-query** · **framer-motion** · **lucide-react** · **sonner** · clsx + tailwind-merge.
**Infra:** Docker + Docker Compose · Git/GitHub · pytest.
**Deliberately excluded:** Kafka, TimescaleDB, Redis, Kubernetes, auth, CI/CD, MLflow server, real email/SMS. (Future-work slide only.)

---

# DATA — what to install, where, and how (best practice)

All datasets live under `data/` with a registry at `data/dataset_metadata.json`. Never commit raw data (gitignore `data/raw/`).

### Dataset 1 — NAB machine temperature (PRODUCTION STREAM) — already installed
- **Kaggle:** `boltzmannbrain/nab` · **File:** `realKnownCause/machine_temperature_system_failure.csv`
- **Install:**
```bash
pip install kaggle
# put kaggle.json in ~/.kaggle/ (chmod 600)
kaggle datasets download -d boltzmannbrain/nab -p data/raw --unzip
```
- **Columns:** `timestamp` (5-min), `value` (temperature). ~22,695 rows.
- **Labels (hard-code these NAB windows):**
```
2013-12-10 06:25 → 2013-12-12 05:35
2013-12-15 17:50 → 2013-12-17 17:00
2014-01-27 14:20 → 2014-01-29 13:30
2014-02-07 14:55 → 2014-02-09 14:05
```

### Dataset 2 — NASA Bearing (VIBRATION) — Phase 6
- **Kaggle:** `vinayak123tyagi/bearing-dataset`
```bash
kaggle datasets download -d vinayak123tyagi/bearing-dataset -p data/raw/bearing --unzip
```
- **Nature:** run-to-failure vibration; snapshots of 20,480 points @ 20 kHz per file, multiple bearings. **Best practice:** treat each snapshot as a sample; extract time + frequency features (below); the *trend over snapshots* is the degradation signal. Weak labels (known failure at end of run) → unsupervised + trend, not supervised classification.

### Dataset 3 — MVTec AD (VISION) — Phase 7 (stretch)
- **Source:** MVTec (free for research; register/download). Start with **one category** (e.g. `bottle` or `hazelnut`), not all 15.
```bash
# after download:
mkdir -p data/raw/mvtec && tar -xf mvtec_ad.tar.xz -C data/raw/mvtec
```
- **Structure:** `train/good/` (normal only), `test/<defect>/`, `ground_truth/` masks. **Best practice:** train on normal images only (unsupervised AD), evaluate on the labeled test set; use provided masks only for heatmap validation.

### Dataset 4 — Synthetic multi-sensor generator (DEMO CONTROL) — Phase 5
- Not downloaded — **generated** by `src/synthetic/multi_sensor_generator.py`. Emits temperature, vibration-RMS, pressure, current, RPM with injectable faults (spike, drift, stuck, dropout, overheating). Best demo-control per hour spent.

### `data/dataset_metadata.json` (create in Phase 2)
```json
{ "datasets": [
  {"name":"NAB Machine Temperature","modality":"time_series","source":"kaggle:boltzmannbrain/nab","rows":22695,"labels":"anomaly windows","used_for":"live stream anomaly detection"},
  {"name":"NASA Bearing","modality":"vibration","source":"kaggle:vinayak123tyagi/bearing-dataset","labels":"run-to-failure (weak)","used_for":"vibration predictive maintenance"},
  {"name":"MVTec AD","modality":"image","source":"mvtec","labels":"normal/defect + masks","used_for":"visual inspection"}
]}
```

---

# PHASE 0 — Stabilize the MVP 🔴 (gate: nothing new until all DoD pass)

### 0.1 Fix `pytest` discovery
`pyproject.toml`:
```toml
[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
addopts = "-q"
```
Add root `conftest.py` (`sys.path.insert(0, os.path.dirname(__file__))`) and `__init__.py` in every `src/*` package.
✅ **DoD:** both `pytest -q` and `python -m pytest` run with 0 collection errors.

### 0.2 Fix sklearn feature-name warning at source
Build a named `pd.DataFrame([features], columns=FEATURE_COLUMNS)` before `scaler.transform()`; remove any blanket warning filter.
✅ **DoD:** `/predict` works, no sklearn `UserWarning` in logs.

### 0.3 Frontend build + chunk fix
`vite.config.ts` manual chunks (`react`, `charts`); prep lazy routes.
✅ **DoD:** `npm run build` clean, no >500 kB warning.

### 0.4 Backend + Docker sanity
✅ **DoD:** `uvicorn ... /health` ok; `docker compose config` valid; `up --build` starts api+frontend.

**Commit:** `chore(stabilize): pytest config, sklearn warning fix, chunk split, docker verify`

---

# PHASE 1 — Fix evaluation credibility 🔴 (the number-one grading risk)

**Problem (real, from your CSV):** four models show **Recall = 1.000, FN = 0**; Elliptic Envelope flags 99.97% of normal points yet still "catches everything." Unexplained perfection reads as fabricated. Cause: NAB labels are **windows** and the failure spike is large → point-wise recall is trivially 1.0.

### 1.1 Implement NAB-style windowed scoring
Add `nab_window_score()` to evaluation: a detection anywhere inside a labeled window counts as a hit for that window; report **windowed precision/recall/F1 alongside point metrics**, clearly labeled.

### 1.2 Make detection latency the tiebreaker
Compute latency in **minutes from window onset to first flag**. When models tie at recall, latency + false-alarm rate rank them. Surface this prominently.

### 1.3 Rule out leakage (assert it)
Test: scaler and every threshold fit on train/normal indices only; assert no test index participates. 

### 1.4 Keep Elliptic Envelope as an honest negative result
Label it: "flags ~100% of points, unusable on this data; included for contrast." That *raises* credibility.

### 1.5 Reconcile all result files
Re-run `evaluate_all.py`; make `evaluation_results.csv`, `model_comparison.json`, and any status docs agree; `threshold.json` stores validation `selection_metrics` separately from test metrics.
✅ **DoD:** results table shows windowed + point metrics + latency; leakage test green; every metric has context; no bare Recall=1.0 without explanation.

**Commit:** `fix(eval): NAB windowed scoring + latency tiebreaker + leakage test; reconcile results`

---

# PHASE 2 — Multi-model core (classical) 🟡

Models (unsupervised; fit on normal, score on labeled test): Rolling Z-score, **Isolation Forest (prod)**, One-Class SVM, LOF (novelty), Elliptic Envelope. *(Most exist — this phase formalizes the contract, registry, experiments, API.)*

### 2.1 Freeze feature contract
`feature_engineering.py` writes ordered `models/feature_columns.json`; training + inference both read it. Add a test asserting identical order.

### 2.2–2.4 Ensure each classical trainer saves: artifact `.pkl` + `threshold_<m>.json` (validation `selection_metrics`). Tune on validation (OCSVM `nu`, LOF `n_neighbors`, Elliptic `contamination`).

### 2.5 `evaluate_all.py` — one true table
Same test split for all; compute point + windowed metrics, ROC-AUC, PR-AUC, false-alarm, latency (min), inference ms, throughput; rank by PR-AUC then latency; write CSV + `model_comparison.json` + per-model PR/ROC/confusion figures; **print every number**.

### 2.6 Model registry — `src/registry/model_registry.py` → `models/model_registry.json`
Per model: name, family, modality, dataset, artifact/scaler/threshold/feature paths, train/val/test rows, all metrics, trained_at, `is_production` (IF=true), limitations, notes. Functions: `register/list/get/set_production`.

### 2.7 Experiment tracking — `src/experiments/experiment_tracker.py` → `reports/experiments.json` (append-only).

### 2.8 API additions (thin readers): `GET /models`, `/models/registry`, `/models/comparison`, `/experiments`, `/data/summary`, `/system/status`; `POST /models/select/{name}` (hot-swap into `InferenceService`).

### 2.9 Model cards — `docs/model_cards/<model>.md` (intended use, data, features, metrics, limitations, latency). Cheap documentation marks.
✅ **DoD:** ≥5 models in registry with real reconciled metrics; endpoints return real data; `select` swaps production and back; tests green.

**Commit:** `feat(ml): feature contract, registry, experiments, model endpoints, model cards`

---

# PHASE 3 — GUI enhancement (platform feel) 🟢

### 3.0 Install: `react-router-dom framer-motion @tanstack/react-query lucide-react sonner clsx tailwind-merge @fontsource/inter`.

### 3.1 Design system (do first)
`tokens.css`: dark base (`#0A0E14`/`#111721`), **one accent cyan `#22D3EE`**, semantic warn/crit/ok; Inter + tabular-nums; 8px grid, `rounded-xl`, `border-white/6`. **No purple gradients, no neon, no generic AI look.** Primitives: `StatCard, StatusDot, SeverityBadge, MetricSparkline, DataTable, SectionHeader, Skeleton, Drawer`.

### 3.2 Motion: page transitions (fade/slide 180ms), KPI count-up, alert enter/exit (`AnimatePresence`), anomaly pulse. All ≤250ms; disable chart animation on live tick.

### 3.3 react-query: one typed hook per endpoint; polling for status (3s) and alerts (5s); skeleton + error states everywhere.

### 3.4 Shell: `AppShell` — collapsible sidebar (grouped Monitoring/ML/System) + top status bar (WS dot, active model chip, clock). Route-level `lazy()` + `<Suspense>`.

### 3.5 Pages (keep behavior, elevate): **Live Monitoring** (dual synced charts, threshold band, pulsing anomaly dots, KPI row, latest-alerts side panel, replay-speed readout) · **Overview** (hero status, StatCards, top-3 leaderboard, start-demo card) · **Model Lab** (model cards, ranking sortable by latency/false-alarm since recall ties, confusion heatmaps, PR/ROC, **Promote** → confirm → toast) · **Data Explorer** (label donut, split bars, anomaly-window timeline, feature chips, sample rows) · **Alert Center** (filterable DataTable, ack, details Drawer + mini timeline) · **Experiment Results** (history table, figures, export) · **System Health** (status dots, latency/rate sparklines, log tail, polling) · **Demo Control Panel** (copy-paste commands, scenarios, checklist).
✅ **DoD:** clean build; 8 cohesive animated pages; skeletons/errors; Model Lab ranks by latency/false-alarm; dark pro UI.

**Commit:** `feat(frontend): design system, motion, react-query, 8-page platform`

---

# PHASE 4 — Deep + online models 🟡 (depth)

### 4.1 LSTM Autoencoder (`src/models/train_lstm_autoencoder.py`, PyTorch)
Sequences (len ~30) of scaled value+features; encoder-decoder LSTM; MSE recon loss; train on **normal** windows; threshold = 99th pct of normal recon error → `ae_threshold.json`; save `lstm_autoencoder.pt`; register; appears in Model Lab automatically. CPU-fine at this size; seed everything.

### 4.2 River HalfSpaceTrees (`src/models/train_river_online.py`)
`river.anomaly.HalfSpaceTrees(seed=42)`; per-reading `learn_one`/`score_one`; **adaptive threshold** = rolling quantile of scores.

### 4.3 Drift detection
Track rolling mean/variance shift + score-distribution shift; raise a **drift warning** + "retraining recommended" flag; expose `GET /system/drift`.

### 4.4 Ensemble model agreement
Score a reading with all production-eligible models; report vote count + ensemble score + confidence ("4 of 6 agree → anomaly, 75%"). Endpoint `POST /predict/ensemble`; show as a Model Lab panel + Live Monitor badge.
✅ **DoD:** LSTM-AE + River registered and compared on the same honest split; drift warning fires on injected drift; ensemble endpoint returns vote breakdown; IF still default until challengers proven.

**Commit:** `feat(ml): LSTM autoencoder, River online, drift detection, ensemble agreement`

---

# PHASE 5 — Product features 🟢 (makes it feel real)

### 5.1 Alert lifecycle
Extend `alerts`: `status` (new/acknowledged/investigating/resolved/false_alarm), `operator_note`, `resolved_at`. Endpoint `POST /alerts/{id}/status`. Alert Center reflects lifecycle with status chips.

### 5.2 Operator feedback loop
`POST /alerts/{id}/feedback` (true_anomaly/false_alarm/unsure) → stored for future supervised use. Buttons in the Alert drawer.

### 5.3 Anomaly explanation (upgrade "reason")
Top contributing feature + current value + normal range + deviation (σ) + last-10 readings. Example: "temp 3.1σ above rolling baseline + high rate-of-change."

### 5.4 Anomaly replay
Select an alert → replay ±30 steps → show when the model first detected it and severity climbing. Great demo moment.

### 5.5 Synthetic fault injection (`src/synthetic/`)
Generator + `POST /faults/inject` (spike/drift/stuck/dropout/overheating). Lets you trigger anomalies on demand instead of waiting for a window.

### 5.6 Incident report PDF (`src/reports/incident_report.py`)
`GET /reports/incident/{id}` → PDF: timestamp, values, score, severity, top features, chart, operator notes, rule-based recommendation.

### 5.7 Maintenance recommendation (rule-based)
Map reason → action (high z-score→"inspect cooling"; high RMS→"inspect bearing"; visual defect→"reject product").
✅ **DoD:** alert moves through lifecycle; feedback stored; replay works; fault injection triggers a live anomaly; incident PDF downloads; recommendations show.

**Commit:** `feat(product): alert lifecycle, feedback, explanation, replay, fault injection, incident PDF`

---

# PHASE 6 — Second modality: Vibration (NASA Bearing) 🟡 (best data extension)

### 6.1 Loader (`src/vibration/data_loader.py`)
Read Bearing snapshots in run order; each snapshot = one sample; keep the sequence for trend.

### 6.2 Feature extraction (`src/vibration/features.py`)
**Time-domain:** RMS, peak-to-peak, kurtosis, skewness, crest factor, variance. **Frequency-domain (scipy.fft):** dominant frequency, spectral centroid, spectral entropy, band energy, (optional) envelope spectrum.

### 6.3 Models (`src/vibration/train.py`)
Isolation Forest on features (MVP) + **1D-CNN Autoencoder** (PyTorch) on raw/feature windows (advanced); threshold on healthy-region recon error; register under modality=`vibration`.

### 6.4 Degradation trend
Rolling health index (e.g. normalized RMS/kurtosis) → normal/warning/critical bands over the run.

### 6.5 API + page
`GET /vibration/sample`, `POST /vibration/analyze`; **Vibration Health Lab** page: waveform, FFT spectrum, health-indicator cards, anomaly score, degradation-trend chart.
✅ **DoD:** Bearing loads; features computed; model detects late-run degradation; page shows waveform+FFT+trend; registered model appears in Model Lab (filtered by modality).

**Commit:** `feat(vibration): NASA Bearing loader, time+freq features, models, Vibration Health Lab`

---

# PHASE 7 — Third modality: Vision (MVTec AD) 🟢 stretch (1-month only)

### 7.1 Data: one category, normal-only training set.
### 7.2 Embedding extractor (`src/image/embeddings.py`): torchvision ResNet18 (pretrained), extract features from `train/good`.
### 7.3 Model (MVP): Isolation Forest / k-NN distance on embeddings → normal/defect + score. **Advanced:** CNN autoencoder → reconstruction-error **heatmap**; validate against ground-truth masks.
### 7.4 API + page: `POST /image/analyze`; **Visual Inspection Lab**: upload → result + score + heatmap + normal/abnormal gallery.
✅ **DoD:** normal-only training; test images classified with scores; heatmap (if AE) localizes defect; page functional. Only pursue with two weeks of buffer — a half-built image module hurts more than none.

**Commit:** `feat(vision): MVTec embeddings + image AD + Visual Inspection Lab`

---

# PHASE 8 — Asset Center + integration 🟢

### 8.1 `assets` table + `GET /assets`: rows for Machine-01 (temperature), Bearing-01 (vibration), Product-Inspection-01 (image), Simulated-Machine-02 (synthetic).
### 8.2 **Asset Center** page: cards per asset (modality, status, last anomaly) linking to the right module. This is the framing that makes it read as *one platform*.
### 8.3 Overview rollup: datasets connected, models trained, readings processed, anomalies detected, latest critical alert — across modalities.
✅ **DoD:** Asset Center lists all assets; each links to its module; Overview aggregates cross-modality stats.

**Commit:** `feat(platform): asset registry + Asset Center + cross-modal overview`

---

# PHASE 9 — Testing, docs, demo, submission ✅

### 9.1 Tests: unit (features/models/registry/vibration/image), integration (API endpoints incl. new), latency (<50 ms IF), stress (10k readings), leakage assertion. Green on bare `pytest -q`.
### 9.2 Docs: README (architecture, all modalities, setup, run, docker, results), model cards, `docs/architecture.md`, final report PDF.
### 9.3 Demo script (≈6 min): Overview → Live Monitor (inject fault → anomaly → alert) → Model Lab (ranking by latency, promote) → ensemble agreement → Vibration Lab (degradation trend) → (Visual Lab if built) → Alert lifecycle + incident PDF → drift warning → limitations & future work.
### 9.4 Submission package: source, dataset links + samples, trained models, evaluation report, figures, screenshots, README, requirements, docker-compose, tests, final report, slides, demo video. Mirror repo tree; root `SUBMISSION.md` checklist.

**Final gauntlet:**
```bash
# train everything
python -m src.data.data_loader && python -m src.data.preprocessing && python -m src.features.feature_engineering
python -m src.models.train_baseline && python -m src.models.train_isolation_forest
python -m src.models.train_one_class_svm && python -m src.models.train_lof && python -m src.models.train_elliptic_envelope
python -m src.models.train_lstm_autoencoder && python -m src.models.train_river_online
python -m src.models.evaluate_all
python -m src.vibration.train      # phase 6
# verify
pytest -q && python -m pytest -q
uvicorn src.api.main:app --reload
python -m src.streaming.stream_simulator --speed 50 --start-index <near window>
cd frontend && npm run build && npm run dev
docker compose up --build
```
✅ **Final DoD:** green tests; reconciled honest metrics (windowed+point+latency); all committed modules functional; multi-modal platform runs under docker; demo video recorded; no fabricated numbers anywhere.

---

# MASTER EXECUTION ORDER (do literally)

```
PHASE 0  stabilize:   pytest config → sklearn fix → chunk split → backend/docker
PHASE 1  credibility: windowed scoring → latency tiebreaker → leakage test → reconcile files   [GATE]
PHASE 2  ml core:     feature contract → classical trainers → evaluate_all → registry → experiments → API → model cards
PHASE 3  gui:         design system → motion → react-query → shell → 8 pages
PHASE 4  deep/online: LSTM-AE → River → drift → ensemble agreement
PHASE 5  product:     alert lifecycle → feedback → explanation → replay → fault injection → incident PDF
PHASE 6  vibration:   loader → time+freq features → IF/1D-CNN-AE → trend → Vibration Health Lab
PHASE 7  vision:      (stretch) ResNet embeddings → image AD → heatmap → Visual Inspection Lab
PHASE 8  platform:    assets table → Asset Center → cross-modal overview
PHASE 9  ship:        tests → docs → demo script → submission package
```

**Time-boxed scopes:**
- **1 week (high grade):** Phases 0–5 (no new datasets). Six models incl. deep, honest metrics, controllable demo, pro UI, MLOps evidence.
- **2 weeks (capstone):** + Phase 6 (vibration) + Phase 8. Genuinely multi-modal (sensor + vibration).
- **1 month (ultimate):** + Phase 7 (vision). Three modalities. Only with buffer.

### Guardrails (restate, never violate)
- Phase 1 (credibility) is a gate — do not expand on distrusted numbers.
- Additive only; MVP never regresses; Isolation Forest stays production default until challengers proven on the honest split.
- Modalities are separate modules sharing infrastructure — no fusion claims, no pretending image data is the same machine.
- One real NAB test split; validation vs test never conflated; leakage asserted by test.
- Dark pro UI, one accent, tasteful motion; no purple gradients, no generic AI look.
- Clearest working story beats longest technology list; a half-built module is worse than none.
