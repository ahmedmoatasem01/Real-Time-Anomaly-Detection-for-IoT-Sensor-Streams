# EXECUTION_PLAYBOOK_FOR_CLAUDE_CODE.md
## Real-Time IoT Anomaly Detection — Professional Upgrade, Executed Step by Step

> Hand this to Claude Code. Execute **in order**. Each task has: goal, exact files, exact code/diff intent, exact commands, and a **Definition of Done (DoD)** that must pass before moving on. Do not skip a DoD. Do not fake numbers — every metric comes from a real run on the real NAB test split. Do not remove or regress the working MVP. Isolation Forest stays the production default.

**Legend:** 🔴 blocker · 🟡 required · 🟢 enhancement · ✅ DoD (must pass)

---

# PART 0 — Ground rules (read once, apply always)

1. **Additive only.** Never delete a working module. New files sit alongside old ones.
2. **One evaluation truth.** Threshold is chosen on **validation**; every reported metric comes from **`evaluate_all.py` on the held-out test split**. `threshold.json` stores `selection_metrics` (validation) with a label; `evaluation_results.csv` stores authoritative **test** metrics. Never conflate them.
3. **Reproducibility.** `random_state=42` everywhere; no future leakage; scaler + thresholds fit on train/normal only.
4. **Commit per task** with the message shown. Keep commits small so a regression is easy to bisect.
5. **After every code change**, run the task's DoD before proceeding.

---

# PART 1 — PRIORITY 1: Stabilize the MVP 🔴

Goal: bare `pytest -q` green from repo root, backend runs, frontend builds, docker config valid, sklearn warning gone, frontend chunk warning gone. **No new features until every DoD in Part 1 passes.**

## Task 1.1 — Fix `pytest` module resolution 🔴

**Problem:** `pytest -q` fails with `ModuleNotFoundError: No module named 'src'` because the repo root isn't on `sys.path` on bare invocation.

**Do this (preferred — single source of truth):** create `pyproject.toml` at repo root. If one exists, merge these keys.

```toml
[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
addopts = "-q"
filterwarnings = [
    "ignore::DeprecationWarning",
]
```

**Also add a root `conftest.py`** (belt-and-suspenders, guarantees discovery even on older pytest):

```python
# conftest.py  (repo root)
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
```

**Ensure package markers exist:** every folder under `src/` (and `src/` itself) has an `__init__.py`. Create any that are missing:
```
src/__init__.py
src/api/__init__.py
src/data/__init__.py
src/features/__init__.py
src/models/__init__.py
src/database/__init__.py
src/streaming/__init__.py
src/utils/__init__.py
tests/__init__.py   # optional but harmless
```

**Commands:**
```bash
pytest -q
python -m pytest -q
```
✅ **DoD 1.1:** both commands collect and run the same tests with **0 collection errors**. If any test *logic* fails, fix the test or code — but the `ModuleNotFoundError` must be gone from both.

**Commit:** `fix(tests): add pyproject pytest config + conftest for src import resolution`

---

## Task 1.2 — Silence the sklearn feature-name warning at the source 🟡

**Problem:** `inference_service.py` calls `scaler.transform()` on a raw array/np, triggering `UserWarning: X does not have valid feature names`. Currently *silenced*, not *fixed*.

**Fix properly:** build a one-row `pandas.DataFrame` with the exact training feature columns before `.transform()`, so column names match what the scaler saw at fit time.

- Load persisted `models/feature_columns.json` (create it in training if absent — see Task 2.1) — the ordered list of columns the scaler/model expect.
- In `InferenceService`, when scoring: assemble features into `pd.DataFrame([feature_dict], columns=FEATURE_COLUMNS)` then `scaler.transform(df)`.
- Remove any blanket `warnings.filterwarnings("ignore")` that was masking this.

✅ **DoD 1.2:** run `uvicorn src.api.main:app --reload`, POST one reading to `/predict`; response is valid; **no sklearn UserWarning in logs**.

**Commit:** `fix(inference): pass named DataFrame to scaler.transform to remove sklearn warning`

---

## Task 1.3 — Confirm backend runs 🔴

**Commands:**
```bash
uvicorn src.api.main:app --reload
# in another shell:
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"timestamp":"2014-01-27T14:25:00","sensor_id":"machine_temperature","value":72.13}'
```
✅ **DoD 1.3:** `/health` returns `{"status":"ok","model":"isolation_forest"}`; `/predict` returns a valid `Prediction` JSON with `anomaly_score`, `is_anomaly`, `severity`, `inference_ms`.

---

## Task 1.4 — Confirm frontend builds + kill the chunk warning 🟡

**Problem:** Recharts pushes the main chunk >500 kB.

**Fix:** in `frontend/vite.config.ts` add manual chunking, and prepare for route-level lazy loading (used in Part 3).

```ts
// vite.config.ts — inside defineConfig
build: {
  chunkSizeWarningLimit: 900,
  rollupOptions: {
    output: {
      manualChunks: {
        react: ['react', 'react-dom', 'react-router-dom'],
        charts: ['recharts'],
      },
    },
  },
},
```

**Commands:**
```bash
cd frontend
npm install
npm run build
cd ..
```
✅ **DoD 1.4:** `npm run build` succeeds; **no chunk-size warning** (charts split into their own chunk).

**Commit:** `chore(frontend): manual chunks to split recharts and clear bundle warning`

---

## Task 1.5 — Confirm Docker config 🔴

**Commands:**
```bash
docker compose config
docker compose up --build -d
docker compose ps
docker compose down
```
✅ **DoD 1.5:** `config` prints valid YAML for `api` + `frontend`; `up --build` starts both; `ps` shows healthy; `down` clean.

---

## Task 1.6 — Reconcile the metrics contradiction 🔴 (credibility-critical)

**Problem:** `evaluation_results.csv` (IF F1≈0.947) and `threshold.json` (IF F1≈0.610) disagree for the same model.

**Fix now, before adding models:**
1. Decide the rule (Part 0 #2): validation picks threshold; test produces reported metrics.
2. In `train_isolation_forest.py`, when saving `threshold.json`, store the threshold plus a **`selection_metrics`** block clearly labeled `"split":"validation"`.
3. Ensure `evaluate.py` (soon `evaluate_all.py`) computes on the **test** split only and is the sole writer of `evaluation_results.csv`, with a `"split":"test"` note column.
4. Re-run training + evaluation so both files are internally consistent and traceable to real runs.

```bash
python -m src.models.train_isolation_forest
python -m src.models.evaluate
```
✅ **DoD 1.6:** `threshold.json` shows `selection_metrics` labeled validation; `evaluation_results.csv` shows test metrics; the two are no longer presented as equal. Numbers come from actual runs printed in logs.

**Commit:** `fix(eval): separate validation threshold-selection from test evaluation; reconcile metrics`

> **STOP.** All of DoD 1.1–1.6 must pass. Only then continue to Part 2.

---

# PART 2 — PRIORITY 2: Multi-model training (real NAB metrics) 🟡

Goal: at least **4 working unsupervised models** + a **unified evaluation** on the same real test split + a **model registry** + **experiment tracking**, exposed via API. No faked metrics.

## Model selection decision (final)
Implement, in this order:
1. **Rolling Z-score / EWMA** — baseline (exists).
2. **Isolation Forest** — production default (exists).
3. **One-Class SVM** — non-linear boundary around normal.
4. **Local Outlier Factor** (novelty=True) — local-density anomalies.
5. **Elliptic Envelope** — Gaussian covariance outliers (cheap 5th, strengthens comparison).
- **Advanced (only after 1–5 solid, never blocking demo):** LSTM Autoencoder, then River HalfSpaceTrees + drift.
- **Supervised (XGBoost):** NOT a production candidate. Optionally add **one** run labeled *"supervised baseline — demonstrates overfitting on sparse-window labels"*. Skip if time-constrained.

All classical models are **unsupervised**: fit on the known-normal region, scored on the labeled test split. Same feature vector, same scaler, same test split for every model — that's what makes the comparison honest.

---

## Task 2.1 — Persist the canonical feature contract 🟡

Before adding models, freeze the feature interface so every model + the API agree.

- In `feature_engineering.py`, after building features, write the ordered column list to `models/feature_columns.json`.
- Add `make_feature_matrix(df) -> (X: DataFrame, columns: list)` returning columns in this exact frozen order.
- Training scripts and `InferenceService` both load `feature_columns.json`.

✅ **DoD 2.1:** `models/feature_columns.json` exists; training and inference both read it; feature vector order is identical in both paths (add a test asserting this).

**Commit:** `feat(features): freeze feature_columns.json as the canonical feature contract`

---

## Task 2.2 — One-Class SVM 🟡

**File:** `src/models/train_one_class_svm.py`

- **Input:** full scaled feature matrix (`feature_columns.json` order).
- **Train data:** known-normal rows of the train split.
- **Model:** `OneClassSVM(kernel="rbf", gamma="scale", nu=<tuned>)`. Tune `nu ∈ {0.01,0.03,0.05,0.1}` on validation for best F1.
- **Score:** anomaly score = `-decision_function(X)` (higher = more anomalous); threshold at 0 or tuned on validation PR.
- **Save:** `joblib.dump(model, "models/one_class_svm.pkl")`; write threshold + `selection_metrics` (validation) to `models/threshold_ocsvm.json`.

✅ **DoD 2.2:** script runs, saves artifacts, logs validation selection metrics; a known-anomalous fixture scores higher than a known-normal one.

**Commit:** `feat(models): add One-Class SVM trainer with validation-tuned nu`

---

## Task 2.3 — Local Outlier Factor 🟡

**File:** `src/models/train_lof.py`

- **Model:** `LocalOutlierFactor(n_neighbors=<tuned>, novelty=True)` so it supports `.predict`/`.score_samples` on unseen data.
- **Train:** fit on known-normal train rows. Tune `n_neighbors ∈ {20,35,50}` on validation.
- **Score:** `-score_samples(X)`; threshold tuned on validation.
- **Save:** `models/lof.pkl` + `models/threshold_lof.json` (with `selection_metrics`).

✅ **DoD 2.3:** runs, saves artifacts, sane validation metrics logged; fixture ordering holds.

**Commit:** `feat(models): add LOF (novelty mode) trainer`

---

## Task 2.4 — Elliptic Envelope 🟢

**File:** `src/models/train_elliptic_envelope.py`

- **Model:** `EllipticEnvelope(contamination=<small>, random_state=42, support_fraction=None)`.
- **Train:** known-normal rows. **Score:** `-score_samples(X)`; threshold on validation.
- **Save:** `models/elliptic_envelope.pkl` + `models/threshold_elliptic.json`.
- If covariance is singular (univariate risk), catch and log; fall back to a higher `support_fraction`. Document any instability honestly in notes.

✅ **DoD 2.4:** runs and saves, or logs a clear honest reason it's unstable on this data (acceptable — that's a real finding, not a failure to hide).

**Commit:** `feat(models): add Elliptic Envelope trainer with singular-covariance guard`

---

## Task 2.5 — Unified evaluation `evaluate_all.py` 🔴 (the one true table)

**File:** `src/models/evaluate_all.py`

- Discover all trained models (baseline, IF, OCSVM, LOF, Elliptic) via their artifacts + thresholds.
- Load the **same** processed **test** split for all.
- For each model compute on test: **Precision, Recall, F1, ROC-AUC, PR-AUC, false-alarm rate, detection latency (steps), avg inference time (ms), throughput (readings/s)**.
- Rank by **PR-AUC** (headline for imbalanced data), tiebreak by recall then low false-alarm rate.
- Write:
  - `reports/evaluation_results.csv` — one row per model (authoritative test metrics, `split=test`).
  - `reports/model_comparison.json` — ranked list + metadata for the frontend.
  - `reports/figures/pr_<model>.png`, `roc_<model>.png`, `confusion_<model>.png`, and `model_comparison_bar.png`.
- **Print every number to the log** so results are auditable and obviously real.

**Command:** `python -m src.models.evaluate_all`
✅ **DoD 2.5:** CSV + JSON + figures generated; ≥4 models present with real, distinct metrics; ranking sensible (IF likely top; baseline weak — which is the honest, expected result). No hard-coded numbers anywhere.

**Commit:** `feat(eval): unified evaluate_all across all models on real test split`

---

## Task 2.6 — Model registry 🟡

**Files:** `src/registry/model_registry.py`, output `models/model_registry.json`

- Registry writer records per model: name, type, feature_set (`feature_columns.json` hash), train/val/test row counts, threshold, all **test** metrics (from `evaluate_all`), artifact path, scaler path, `trained_at`, `is_production` (default **isolation_forest = true**), notes.
- Functions: `register(model_meta)`, `list_models()`, `get(name)`, `set_production(name)`.
- `evaluate_all.py` calls `register(...)` for each model at the end.

✅ **DoD 2.6:** `models/model_registry.json` lists all trained models; `isolation_forest` flagged production; entries match `evaluation_results.csv`.

**Commit:** `feat(registry): model_registry.json as single source of truth for artifacts+metrics`

---

## Task 2.7 — Experiment tracking 🟡

**Files:** `src/experiments/experiment_tracker.py`, output `reports/experiments.json` (+ optional `experiments` SQLite table)

- One record per training/eval run: id, model, feature_set, train/val/test rows, threshold, all metrics, created_at, notes.
- `log_experiment(record)` appends; never overwrites history.

✅ **DoD 2.7:** `reports/experiments.json` grows by one record per model run; readable by the frontend.

**Commit:** `feat(experiments): append-only experiment tracker`

---

## Task 2.8 — Expose models/experiments via API 🟡

**File:** `src/api/main.py` (+ small readers)

Add endpoints (thin readers over registry/CSV/JSON/DB):

| Endpoint | Returns | Required |
|---|---|---|
| `GET /models` | list + basic metrics | 🟡 |
| `GET /models/registry` | full registry | 🟡 |
| `GET /models/comparison` | ranked comparison JSON | 🟡 |
| `GET /experiments` | experiment history | 🟡 |
| `GET /data/summary` | dataset metadata | 🟡 |
| `GET /system/status` | api/db/ws/latency/stream health | 🟡 |
| `POST /models/select/{name}` | hot-swap production model in `InferenceService` | 🟢 |
| `GET /data/features` · `/data/splits` · `/reports/summary` | explorer/export helpers | 🟢 |

`POST /models/select` reloads the chosen artifact + threshold into the live `InferenceService` and updates `is_production` in the registry.

✅ **DoD 2.8:** each required endpoint returns real data (curl-verified); `/models/select/one_class_svm` then `/health` shows the swapped model; swapping back to `isolation_forest` works.

**Commit:** `feat(api): model registry/comparison/experiments/data/system endpoints + model select`

---

## Task 2.9 — Tests for new surface 🟡

Add: `tests/test_models_multi.py` (each model loads + fixture ordering), `tests/test_registry.py`, `tests/test_evaluate_all.py` (writes all 3 outputs), `tests/test_api_models.py` (new endpoints 200 + schema).

**Command:** `pytest -q`
✅ **DoD 2.9:** full suite green on bare `pytest -q`, including new tests.

**Commit:** `test: cover multi-model training, registry, evaluate_all, model endpoints`

> **Advanced (optional, only after 2.1–2.9 green):** `train_lstm_autoencoder.py` (PyTorch, reconstruction-error threshold on normal), then `train_river_online.py` (HalfSpaceTrees + rolling-quantile adaptive threshold + drift flag). Each registers itself and appears in the comparison automatically. Never let these block the demo.

---

# PART 3 — Professional dynamic frontend 🟢 (modern, not a student dashboard)

Goal: turn the single page into a routed, animated, modern monitoring **platform**. Keep the working dashboard as the Live Monitoring page. The dashboard is one of eight sections.

## Design system (apply globally)
- **Stack:** React + TypeScript + Vite + Tailwind + shadcn/ui + Recharts + **react-router-dom** + **framer-motion** (motion) + **lucide-react** (icons) + **@tanstack/react-query** (data fetching/caching).
- **Aesthetic:** professional **dark monitoring UI**. Base slate/zinc (`#0b0f14`–`#111827`), surfaces with subtle borders (`border-white/5`), **one** restrained accent — cyan/teal for healthy, amber for warning, red for critical. Dense data-ink. **No** purple gradients, **no** generic AI-hero look, **no** neon everything.
- **Motion (tasteful):** framer-motion for page transitions (fade/slide 150–200ms), KPI number count-up, alert row enter/exit, chart mount. Motion communicates *liveness*, never distracts.
- **Layout:** persistent left sidebar (icons + labels, collapsible) + top status bar (WS dot, active model, stream rate, clock). Content area routes below.
- **Components:** reusable `<StatCard>`, `<StatusDot>`, `<SeverityBadge>`, `<MetricSparkline>`, `<DataTable>` (sortable/filterable), `<Drawer>`, `<ModelCard>`, `<SectionHeader>`.

## Task 3.1 — Install deps + router shell 🟢
```bash
cd frontend
npm install react-router-dom framer-motion lucide-react @tanstack/react-query
```
- Add `main.tsx` providers: `QueryClientProvider`, `BrowserRouter`.
- Create `src/layouts/AppShell.tsx` (sidebar + top status bar + `<Outlet/>`).
- Create `src/lib/api.ts`: typed client for every endpoint; `src/lib/ws.ts`: reusable WebSocket hook with the existing exponential-backoff logic.
- Set up **route-level lazy loading**: `const ModelLab = lazy(() => import('./pages/ModelLab'))` etc., wrapped in `<Suspense>` with a skeleton — this also enforces code-splitting.

✅ **DoD 3.1:** app boots with sidebar + top bar; routes render placeholders; `npm run build` succeeds with split chunks.

**Commit:** `feat(frontend): app shell, router, query client, typed api + ws hooks`

## Task 3.2 — Page: Live Monitoring (migrate existing) 🟡
Move current `App.tsx` dashboard into `pages/LiveMonitoring.tsx`. Keep live temperature chart, anomaly-score chart, live red markers, WS status, active model, current severity, latest alerts. Add framer-motion mount + KPI count-up. **Nothing regresses.**
✅ **DoD 3.2:** live page behaves exactly as before, now inside the shell, animated.

**Commit:** `feat(frontend): Live Monitoring page migrated into platform shell`

## Task 3.3 — Page: Platform Overview 🟢
Hero status band (system OK/degraded), `<StatCard>`s: total readings processed, total anomalies, active production model, uptime/stream status; quick "start demo" guide; mini comparison of top model. Data from `/system/status`, `/metrics`, `/models/comparison`.
✅ **DoD 3.3:** live numbers, animated count-up, no hard-coded values.

## Task 3.4 — Page: Model Lab 🟡 (biggest "pro" signal)
Read `/models/comparison`. Show: `<ModelCard>` per model (metrics, PR-AUC/ROC-AUC, badge for production), **ranking table**, metric bar charts, confusion-matrix images, PR/ROC images from `reports/figures`. If `/models/select` built, a **Promote to production** button (with confirm) → toast → refetch.
✅ **DoD 3.4:** all trained models visible with real metrics; production model clearly marked; promote works if enabled.

**Commit:** `feat(frontend): Model Lab with comparison, ranking, promote`

## Task 3.5 — Page: Data Explorer 🟢
From `/data/summary` (+ `/data/features`,`/data/splits`): dataset summary, the four anomaly windows, feature list, label distribution, train/val/test split sizes, sample rows, preprocessing explanation. Small charts (label balance, split sizes).
✅ **DoD 3.5:** accurate dataset facts, no invented stats.

## Task 3.6 — Page: Alert Center 🟡
Sortable/filterable alert `<DataTable>` from `/alerts`; severity badges; acknowledge button (`POST /alerts/{id}/ack`); details `<Drawer>` with anomaly reason + a mini timeline chart around the anomaly.
✅ **DoD 3.6:** filter/sort/ack all work against real API.

## Task 3.7 — Page: Experiment Results 🟢
Table from `/experiments`; threshold + feature-set columns; embedded figures; **export report** button (calls `/reports/summary` or downloads the CSV/JSON).
✅ **DoD 3.7:** history renders; export downloads a real file.

## Task 3.8 — Page: System Health 🟢
From `/system/status`: API/DB/WS status dots, inference latency, stream rate, recent log tail, docker service status if exposed. Auto-refresh via react-query polling.
✅ **DoD 3.8:** live health, polling updates.

## Task 3.9 — Page: Demo Control Panel 🟢
Start/stop simulator instructions, `--speed` explanation, normal-vs-anomaly scenario (with `--start-index` to jump to a failure window), a demo checklist component.
✅ **DoD 3.9:** clear, copy-pasteable demo guidance.

## Task 3.10 — Polish + build 🟢
Consistent spacing, skeleton loaders, empty states, error boundaries, responsive down to laptop width. Final:
```bash
npm run build
```
✅ **DoD 3.10:** clean build, no warnings; all 8 pages navigable; dark pro UI cohesive; motion tasteful.

**Commit:** `feat(frontend): complete 8-section platform, polish, skeletons, error boundaries`

---

# PART 4 — Final verification gauntlet (run before demo) ✅

```bash
# backend + models
python -m src.data.data_loader
python -m src.data.preprocessing
python -m src.features.feature_engineering
python -m src.models.train_baseline
python -m src.models.train_isolation_forest
python -m src.models.train_one_class_svm
python -m src.models.train_lof
python -m src.models.train_elliptic_envelope
python -m src.models.evaluate_all      # writes the one true table + registry + experiments

# tests (both must be green)
pytest -q
python -m pytest -q

# run system
uvicorn src.api.main:app --reload          # :8000
python -m src.streaming.stream_simulator --speed 50 --start-index <near a window>
cd frontend && npm run build && npm run dev # :5173/:3000

# docker
docker compose config
docker compose up --build
```
✅ **Final DoD:**
- `pytest -q` green from root (no `src` import error).
- `evaluation_results.csv` has ≥4 models, real distinct metrics, `split=test`; matches `model_registry.json` and `model_comparison.json`.
- `threshold.json` clearly separates validation `selection_metrics` from test results.
- Backend endpoints (old + new) return real data.
- Frontend builds clean, 8 pages, dark pro UI, animated, Model Lab shows real comparison.
- `docker compose up --build` runs the whole platform.
- No fabricated numbers anywhere; every metric traceable to a logged run.

---

# PART 5 — Execution order summary (do literally)

```
P1  1.1 pytest config → 1.2 sklearn fix → 1.3 backend → 1.4 frontend build+chunks
    → 1.5 docker → 1.6 reconcile metrics        [ALL DoD pass before P2]
P2  2.1 feature contract → 2.2 OCSVM → 2.3 LOF → 2.4 Elliptic
    → 2.5 evaluate_all → 2.6 registry → 2.7 experiments → 2.8 API → 2.9 tests
    (optional: LSTM-AE → River+drift)
P3  3.1 shell/router → 3.2 Live Monitoring → 3.3 Overview → 3.4 Model Lab
    → 3.5 Data Explorer → 3.6 Alert Center → 3.7 Experiments → 3.8 Health
    → 3.9 Demo Panel → 3.10 polish
P4  full verification gauntlet
```

## Guardrails (never violate)
- Additive only; MVP never regresses; Isolation Forest stays production default.
- Real metrics only, from the real NAB test split; validation and test never conflated.
- `random_state=42`; no future leakage; scaler/threshold fit on train/normal only.
- Every new `src/` module gets a test; both `pytest -q` and `python -m pytest` stay green.
- Modern dark monitoring UI: one accent, tasteful motion, no purple gradients, no generic AI look.
