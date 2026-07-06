# BUILD_SPEC_MLOPS_PRODUCT_FEATURES.md
## Five Product/MLOps Features — Implementation Spec for Claude Code

> Execute against the live repository, in order. Keep `pytest -q` and `npm run build` green after **every** feature. Do not restart the project, do not break the MVP, keep Isolation Forest the production default. Do **not** implement NASA Bearing or the image module. Commit after each feature. At the end, fill in the **Final Report** section with real outputs — never fabricate.

**Order:** F1 Fault Injection → F2 Drift Detection → F3 Alert Lifecycle → F4 Incident PDF → F5 Retraining. (Ordered so later features can use earlier ones — drift consumes injected faults; incident PDF consumes lifecycle fields; retraining consumes drift status.)

**Before starting — recon (do this literally):**
1. Open and read: `src/api/main.py`, `src/api/schemas.py`, `src/api/inference_service.py`, `src/database/database.py`, `src/streaming/stream_simulator.py`, `models/feature_stats.json`, `models/feature_columns.json`, `frontend/src/lib/api.ts`, `frontend/src/pages/` (AlertCenter, SystemHealth, DemoPanel).
2. Note the exact SQLAlchemy style (sync vs async — the walkthrough says async SQLite via threadpool; match it), the Pydantic v2 schema patterns, how routes are registered, and how the frontend api client is shaped.
3. Match existing conventions exactly. Every new module gets an `__init__.py` and a test.

---

# FEATURE 1 — Synthetic Fault Injection 🟢

**Goal:** an injectable fault source whose readings flow through the *real* `/predict` pipeline so the dashboard and alerts react naturally. No separate fake path.

### Files to create
```
src/synthetic/__init__.py
src/synthetic/fault_generator.py
src/synthetic/synthetic_stream.py
tests/test_fault_injection.py
```

### `fault_generator.py` — pure functions, no I/O (easy to test)
Implement a `FaultGenerator` that, given a base value and a step index within the fault, returns the modified value. Fault types:

| type | behavior (relative to baseline `b`, magnitude `m`, step `i`/`n`) |
|---|---|
| `spike` | single/short burst: `b + m` for i in first few steps, else `b` |
| `gradual_drift` | ramp: `b + m * (i/n)` |
| `sensor_stuck` | hold last real value constant for n steps |
| `missing_values` | emit `None`/NaN (pipeline must handle; see note) |
| `noise_burst` | `b + gauss(0, m)` per step |
| `overheating` | monotonic climb that accelerates: `b + m * (i/n)**2` |
| `vibration_fault` | **placeholder** — raise `NotImplementedError("reserved for Phase 6 NASA Bearing")` or return baseline with a `reserved=True` flag; do not fake vibration |

Signature sketch:
```python
FAULT_TYPES = ["spike","gradual_drift","sensor_stuck","missing_values","noise_burst","overheating","vibration_fault"]

class FaultGenerator:
    def __init__(self, fault_type, duration_steps, magnitude, sensor_id, baseline=None): ...
    def next_value(self, current_real_value: float) -> float | None:
        # returns modified value; advances internal step; sets self.finished when step>=duration
    @property
    def finished(self) -> bool: ...
    def status(self) -> dict:  # {fault_type, sensor_id, step, duration_steps, magnitude, active}
```

### `synthetic_stream.py` — orchestration + global state
- Holds the single **active fault** (module-level singleton or a small manager class the API imports).
- `start_fault(cfg) -> status`, `stop_fault() -> status`, `get_status() -> status`, `apply(current_value) -> value` (called by the injection point).
- **Injection point:** simplest robust design — the fault manager exposes `apply(value)`; the stream simulator (or a synthetic feed loop) calls `apply()` on each reading before POSTing to `/predict`. That way injected readings traverse the identical inference path. Document clearly which loop calls `apply()`.
- `missing_values`: when `apply()` returns `None`, the simulator should still exercise the pipeline's missing-value handling (send a payload the API treats as missing, or skip+log per existing preprocessing rules). Match whatever `preprocessing.py` already does for gaps — don't invent new behavior.

### `schemas.py` additions (Pydantic v2)
```python
class FaultInjectRequest(BaseModel):
    fault_type: Literal[*FAULT_TYPES]
    duration_steps: int = Field(gt=0, le=100000)
    magnitude: float = 0.0
    sensor_id: str = "machine_temperature"

class FaultStatus(BaseModel):
    active: bool
    fault_type: str | None = None
    sensor_id: str | None = None
    step: int = 0
    duration_steps: int = 0
    magnitude: float = 0.0
```

### API endpoints (`main.py`)
| Method | Path | Body/── | Returns |
|---|---|---|---|
| GET | `/faults/types` | — | `{"types": FAULT_TYPES, "descriptions": {...}}` |
| POST | `/faults/inject` | `FaultInjectRequest` | `FaultStatus` (active=true) |
| POST | `/faults/stop` | — | `FaultStatus` (active=false) |
| GET | `/faults/status` | — | `FaultStatus` |

- `vibration_fault` in `/faults/inject` → **422** with a clear message that it's reserved for the future vibration module (don't silently succeed).

### Tests (`test_fault_injection.py`)
- Each fault type: assert value behavior (spike raises value; drift increases monotonically to ~`b+m`; stuck holds constant; overheating accelerates; noise varies around b; missing returns None).
- `/faults/inject` sets status active; `/faults/status` reflects it; `/faults/stop` clears it.
- `vibration_fault` returns 422 (reserved).

✅ **DoD F1:** endpoints work; injecting `gradual_drift` produces rising readings that pass through `/predict` and eventually trip an anomaly on the live dashboard; tests green; `npm run build` still clean (no FE change yet).
**Commit:** `feat(synthetic): fault injection module + /faults endpoints + tests`

---

# FEATURE 2 — Drift Detection 🟡

**Goal:** compare live feature distribution against training baseline stats and report drift with a PSI score and a retraining recommendation.

### Files
```
src/drift/__init__.py
src/drift/drift_detector.py     # pure math, testable
src/drift/drift_service.py      # holds rolling live window + state, persistence
tests/test_drift.py
```

### Baseline source
Load `models/feature_stats.json` (per-feature train mean/std, and if present, histogram bins/quantiles). If histograms aren't stored, compute PSI against a bucketization derived from train quantiles — **read what feature_stats.json actually contains first** and adapt; do not assume fields that aren't there.

### `drift_detector.py`
```python
def population_stability_index(expected_pct, actual_pct, eps=1e-6) -> float:
    # PSI = Σ (a - e) * ln(a / e)  over bins
def mean_shift_sigma(baseline_mean, baseline_std, live_mean) -> float:
    # (live_mean - baseline_mean) / (baseline_std + eps)
def classify(psi, mean_shift) -> str:
    # stable  : psi < 0.1  and |shift| < 2
    # warning : 0.1<=psi<0.25 or 2<=|shift|<3
    # critical: psi>=0.25 or |shift|>=3
```
(PSI thresholds 0.1 / 0.25 are the standard industry bands — cite in the model card.)

### `drift_service.py`
- Maintains a rolling window of recent live readings/features (e.g. last N=500 via `deque`), fed from the inference path (hook where readings are already processed).
- `check() -> DriftStatus`: compute live means + PSI vs baseline for each feature; aggregate; classify; build recommendation string.
- Persist latest to `reports/drift_status.json`; append history entries (timestamp + summary) for `/drift/history`.

### Response shape
```json
{
  "baseline_mean": 0.0, "live_mean": 0.0, "mean_shift_sigma": 0.0,
  "psi": 0.0, "affected_features": ["roll_mean_15", "..."],
  "status": "stable|warning|critical",
  "recommendation": "No action | Monitor | Retraining recommended",
  "checked_at": "ISO8601"
}
```

### API (`main.py`)
- `GET /drift/status` → latest from state/`drift_status.json`.
- `POST /drift/check` → force recompute now.
- `GET /drift/history` → recent drift records.

### Tests (`test_drift.py`)
- **PSI correctness:** identical distributions → PSI≈0; shifted distribution → PSI grows; known toy arrays → expected PSI within tolerance.
- **no-drift:** live≈baseline → status `stable`.
- **warning:** moderate shift → `warning`.
- **critical:** large shift (feed a drift fault or synthetic shifted array) → `critical` + retraining recommendation.

✅ **DoD F2:** `/drift/check` after injecting `gradual_drift` (F1) moves status toward warning/critical; `drift_status.json` written; tests green.
**Commit:** `feat(drift): PSI + mean-shift detector, drift service, /drift endpoints + tests`

---

# FEATURE 3 — Alert Lifecycle 🟡

**Goal:** alerts become stateful objects operators move through a lifecycle, with notes and feedback.

### DB migration (`database.py`)
Add columns to `alerts` (nullable, backward-compatible — existing rows keep working):
- `status TEXT DEFAULT 'new'`  (enum in app: new/acknowledged/investigating/resolved/false_alarm)
- `operator_note TEXT`
- `feedback TEXT`  (true_anomaly/false_alarm/unsure)
- `resolved_at TEXT`

**Migration approach:** since it's SQLite, add an idempotent `ALTER TABLE ... ADD COLUMN` guarded by a "column exists?" check in `init_db()` (or a tiny `migrate()` run at startup). Do **not** drop/recreate the table (would lose data). Keep the existing `acknowledged` field working; `status` supersedes it (set `status='acknowledged'` when the old ack path is used, for consistency).

### Schemas
```python
AlertStatus = Literal["new","acknowledged","investigating","resolved","false_alarm"]
Feedback   = Literal["true_anomaly","false_alarm","unsure"]

class AlertStatusUpdate(BaseModel):
    status: AlertStatus
    operator_note: str | None = None

class AlertFeedback(BaseModel):
    feedback: Feedback
```

### API
- `POST /alerts/{id}/status` — body `AlertStatusUpdate`; validates transition; sets `resolved_at` when status∈{resolved,false_alarm}; returns updated alert. Invalid id → 404.
- `POST /alerts/{id}/feedback` — body `AlertFeedback`; stores feedback; 404 on bad id.
- Allowed transitions (enforce): `new→acknowledged→investigating→{resolved,false_alarm}`; allow `new→false_alarm`, `acknowledged→resolved`; reject nonsense (e.g. `resolved→new`) with 409.

### Tests
- Valid transition chain persists and returns correct status + `resolved_at`.
- Invalid transition → 409; invalid id → 404.
- Feedback stored and retrievable.

✅ **DoD F3:** lifecycle transitions persist; old ack path still works; tests green.
**Commit:** `feat(alerts): lifecycle status + operator note + feedback endpoints + migration + tests`

---

# FEATURE 4 — Incident Report PDF 🟢

**Goal:** one-click PDF for an alert.

### Files
```
src/reports/__init__.py
src/reports/incident_report.py   # builds the PDF (reportlab)
src/reports/report_service.py    # gathers data for a given alert_id
tests/test_incident_report.py
```
Add `reportlab` to `requirements.txt`.

### `report_service.py`
`gather(alert_id) -> dict`: join `alerts` + related `readings` (the reading that triggered it + a window around it if retrievable), pull model name + reason/top-feature from the stored prediction, operator note + feedback + status, and a rule-based maintenance recommendation (reuse the mapping: high z-score→inspect cooling; high rate-of-change→check load; etc.). 404 if alert missing.

### `incident_report.py`
`build_pdf(data) -> bytes` with reportlab: header (alert id, generated-at), fields (timestamp, sensor id, value, anomaly score, severity, model, reason/top feature), a small table or matplotlib chart of recent readings around the alert (embed as image if you already generate figures; otherwise a values table), operator note, feedback/status, maintenance recommendation. Keep it one–two pages, clean.

### API
- `GET /reports/incident/{alert_id}` → `Response(content=pdf_bytes, media_type="application/pdf", headers={"Content-Disposition": f'attachment; filename="incident_{alert_id}.pdf"'})`. Bad id → 404.

### Tests
- Valid alert → 200, `content-type: application/pdf`, non-empty body starting with `%PDF`.
- Invalid alert id → 404.

✅ **DoD F4:** PDF downloads for a real alert with real fields; invalid id 404; tests green.
**Commit:** `feat(reports): incident PDF export via reportlab + /reports/incident + tests`

---

# FEATURE 5 — Retraining Workflow / Retraining Center 🟡

**Goal:** train a candidate model on available processed/new readings, compare to production, and promote only via explicit endpoint. **Never auto-overwrite production.**

### Files
```
src/retraining/__init__.py
src/retraining/retraining_service.py
src/retraining/compare_models.py
tests/test_retraining.py
models/candidates/               # candidate artifacts land here
```

### DB table (`database.py`) — `retraining_runs`
```
id INTEGER PK, started_at TEXT, finished_at TEXT, model_name TEXT,
status TEXT,            -- running | completed | failed
old_model_f1 REAL, new_model_f1 REAL,
old_model_pr_auc REAL, new_model_pr_auc REAL,
promoted INTEGER DEFAULT 0, notes TEXT
```

### `retraining_service.py`
- `start() -> run_id`: create a `retraining_runs` row (status=running); train a **candidate Isolation Forest** on available processed data (reuse existing training code path; same feature contract, `random_state=42`); save to `models/candidates/isolation_forest_<run_id>.pkl` + candidate threshold; evaluate candidate on the **same held-out test split** used by `evaluate_all`; write old vs new F1 + PR-AUC; status=completed. Long op → run in threadpool/background so the event loop doesn't block (match existing async pattern).
- **Never** touches `models/isolation_forest.pkl` or the registry production flag.
- `get_status()`, `list_runs()`, `compare(run_id)`.

### `compare_models.py`
`compare(run_id) -> {production: {...metrics}, candidate: {...metrics}, delta: {...}, recommend_promote: bool}` — recommend promote only if candidate ≥ production on PR-AUC **and** not worse on false-alarm/latency beyond a small tolerance.

### Promotion
- `promote(run_id)`: copy candidate artifact over the production path **and** update `model_registry.json` production flag through the existing registry API; set `promoted=1`. Explicit only.

### API
- `GET /retraining/status` · `POST /retraining/start` · `GET /retraining/runs` · `GET /retraining/compare/{run_id}` · `POST /retraining/promote/{run_id}`.

### Tests
- `start` creates a run, produces a candidate artifact under `models/candidates/`, records old+new metrics, does **not** change production.
- `compare` returns both metric sets + recommendation.
- `promote` flips production only after explicit call; registry reflects it; a second model can be promoted back (restore IF).

✅ **DoD F5:** candidate trains + evaluates on the real test split; production untouched until promote; tests green.
**Commit:** `feat(retraining): candidate train/eval/compare/promote + retraining_runs + tests`

---

# FRONTEND (consistent with existing React/TS/Tailwind/shadcn)

Match the current design system (dark, one accent, shadcn components). Add a typed method to `frontend/src/lib/api.ts` for every new endpoint. Keep `npm run build` green after each page.

### Demo Control Panel (fault injection UI)
Buttons: inject spike / gradual_drift / sensor_stuck / missing_values / noise_burst / overheating; a **Stop Fault** button; an **active fault status** card (type, step/duration progress bar, magnitude) polling `GET /faults/status`. Each inject button opens a small form (duration, magnitude) → `POST /faults/inject` → toast. `vibration_fault` shown **disabled** with a "coming with vibration module" tooltip.

### System Health (add drift panel)
Cards: drift `status` (stable/warning/critical with color), `mean_shift_sigma`, `psi`, affected features list, `recommendation`; a "Check now" button → `POST /drift/check`; poll `GET /drift/status`.

### Alert Center (lifecycle + feedback + export)
Per-alert: status badge; buttons Acknowledge / Investigating / Resolved / False Alarm (`POST /alerts/{id}/status`, optimistic + toast); operator-note textarea (saved with status); feedback buttons True anomaly / False alarm / Unsure (`POST /alerts/{id}/feedback`); **Export Incident Report** button → `GET /reports/incident/{id}` (download PDF). Enforce sensible button enable/disable per current status.

### Retraining Center (new page/route)
Cards: current production model + last-trained date (from registry); new-readings-collected count; drift status (reuse); **Trigger Retraining** button → `POST /retraining/start` (show running state); candidate-vs-production comparison table (F1, PR-AUC, false-alarm, latency, delta) from `GET /retraining/compare/{run_id}`; **Promote Candidate** button (confirm dialog) → `POST /retraining/promote/{run_id}`; retraining history table from `GET /retraining/runs`.

✅ **DoD FE:** all four surfaces functional against real endpoints; `npm run build` clean; design consistent.
**Commit:** `feat(frontend): demo panel, drift panel, alert lifecycle UI, retraining center`

---

# VERIFICATION (run after each feature and at the end)

```bash
pytest -q
python -m pytest
cd frontend && npm run build && cd ..
docker compose config
docker compose up --build -d   # if safe
```

**Manual end-to-end (record actual results):**
1. Start API + frontend.
2. Demo Panel → inject `gradual_drift` (duration 120, magnitude 8) → confirm rising readings + a live anomaly + alert appears.
3. System Health → Check drift → status moves toward warning/critical.
4. Alert Center → acknowledge → investigating → resolved; add note; add feedback.
5. Export Incident Report → PDF downloads with real fields.
6. Retraining Center → Trigger retraining → candidate trains → compare vs production → (optional) promote → verify registry, then promote IF back.
7. Stop fault → status clears.

---

# FINAL REPORT (fill with REAL outputs — do not fabricate)

Report after implementation:
1. **Files created** — list actual paths.
2. **Files modified** — list actual paths + one-line reason each.
3. **API endpoints added** — method + path + one-line purpose.
4. **Database changes** — columns added to `alerts`, new `retraining_runs` table, migration method used.
5. **Frontend pages/sections added** — routes/components.
6. **Tests added** — file names + count + what they cover.
7. **Commands run and outputs** — paste real `pytest -q` summary line, `npm run build` result, `docker compose config` result. If something fails, report it honestly and what you did.
8. **Known limitations** — e.g. single active fault at a time; drift window size assumptions; retraining uses same test split (no fresh labels); PDF chart source.
9. **Exact demo steps** — the working click-path you verified.

---

## Guardrails (restate)
- Additive only; MVP never regresses; Isolation Forest stays production default; promotion is explicit only.
- No NASA Bearing, no image module now; `vibration_fault` is a reserved placeholder (422 / disabled).
- Real metrics only; retraining evaluates on the same honest test split; no leakage.
- SQLite migrations are additive `ADD COLUMN` guarded by existence checks — never drop/recreate.
- `pytest -q` and `npm run build` green after every feature; commit per feature.
- Match existing async/Pydantic-v2/shadcn conventions; every new module has a test.
