# PROJECT_SCAN_AND_CLEANUP_PLAN.md

**Scan date:** 2026-07-06
**Method:** Direct execution of `pytest`, `docker compose config`, `npm run build`; direct reading of source files; two read-only sub-agent audits (backend wiring/duplication, frontend wiring/build) with file:line citations. No files were modified to produce this report.

---

## 1. Current Project Reality

### What is implemented and genuinely working
- The core pipeline is real and wired end-to-end: NAB CSV → `src/data/` preprocessing → `src/features/feature_engineering.py` → 7 trained models in `models/*.pkl` → `src/api/inference_service.py` → SQLite (`src/database/database.py`) → WebSocket broadcast → React frontend.
- All 7 ML models (Isolation Forest, Rolling Z-score, One-Class SVM, LOF, Elliptic Envelope, LSTM Autoencoder, River HST) exist as trained artifacts, and the F1 scores in `reports/evaluation_results.csv` match the README table exactly.
- Drift detection, synthetic fault injection, incident-report PDF generation, and the alert lifecycle are all wired into `src/api/main.py` and exercised by dedicated passing tests.
- The two multi-modal extensions are real, not stubs: `src/vibration/train.py` → `models/vibration_iforest.pkl` → `vibration_router.py` → frontend `VibrationLab.tsx` (live WebSocket), and `src/image/train.py` → `models/vision_iforest.pkl` → `image_router.py` → frontend `VisionLab.tsx` (real `fetch` calls to `/image/gallery` and `/image/analyze`). Asset Center similarly calls a real `/assets/` endpoint.
- All 12 frontend pages are routed in `App.tsx` and reference real components — no orphaned pages found.

### What is partially implemented
- **Retraining**: `POST /models/retrain` exists and calls `src/retraining/retrain_pipeline.py` in a background thread — this part is real. But the frontend's "promote" action and the README's documented `POST /retraining/promote/{model_id}` do not call this pipeline at all; they call `/models/select/{name}`, which just flips the active model pointer in `model_registry.py`. There is no tested code path that trains a candidate, shadow-tests it, and promotes it automatically — a human still has to trigger `/models/retrain` and then manually `/models/select` the result.
- **Model documentation**: `docs/model_cards/` has 5 files (elliptic_envelope, isolation_forest, lof, one_class_svm, rolling_zscore) but **no card for LSTM Autoencoder or River HST**, even though both are trained, evaluated, and listed as "✅ Implemented" in the README's model table.
- **Multi-modal test coverage**: vibration, image, asset, model-registry, and retraining endpoints have zero dedicated pytest coverage. `tests/test_api.py` only exercises `/health` and `/predict` (confirmed by direct read — 2 test functions total). Everything else in that file's surface area is untested by the suite, even though it's used in tests indirectly through other test modules for drift/alerts/faults/incident.

### What is roadmap/future work only (correctly labeled as such)
- TimescaleDB/Postgres migration, Kafka integration, distributed edge deployment, RL-based control — all correctly listed under README §18 "Roadmap" as not-yet-built, and no code exists for any of them. No overclaim here.

### What is claimed in README but not actually implemented
- **`POST /retraining/promote/{model_id}`** (README §12, API Documentation) — this route does not exist anywhere in `src/api/main.py`. The real, functioning equivalent is `POST /models/select/{name}`. This is a factual documentation error, not a missing feature (the underlying capability exists under a different name).
- **"Verify Frontend TypeScript/Vite build: `npm run build`"** (README §15, Testing) — this command **currently fails** on a clean checkout (see §5 below). The README presents it as a passing verification step; right now it is not.
- **"The platform relies on rigorous unit testing... `pytest -q`"** (README §15) — currently **4 tests fail** on a full-suite run (see §5). Three of the four pass in isolation, meaning the suite is order-dependent and not currently green as a whole.

### Files that are outdated or duplicated
- `src/models/evaluate.py` — superseded by `src/models/evaluate_all.py` (the actual entrypoint named in README §14). No other file imports `evaluate.py`; it is dead.
- `src/models/predict.py` — a thin `Predictor` wrapper around `InferenceService`. Zero imports of it found anywhere in the repo. Dead.
- `src/api/drift_detector.py` — a second, unrelated `DriftDetector` class (river ADWIN-based) that duplicates the *name* of `src/drift/drift_detector.py` (the one actually used by `drift_service.py`). The `src/api/` copy has no importers anywhere. Dead.
- `models/lstm_autoencoder.pth` — a raw PyTorch `state_dict` saved alongside `lstm_autoencoder.pkl`. Only the `.pkl` is ever loaded (`evaluate_all.py`); no `torch.load` call exists anywhere in `src/`. This file is also the **only model artifact tracked in git** (`git ls-files` confirms it — all other `.pkl`/`.json` model files are correctly gitignored). It looks like an accidental commit: the `.gitignore` excludes `models/*.pkl` and `models/*.pt` but not `models/*.pth`, so this one slipped through.
- `reports/figures/confusion_matrix_if.png`, `pr_curve.png`, `roc_curve.png` — generic, non-model-suffixed figure names sitting alongside the current per-model convention (`confusion_isolation_forest.png`, `pr_isolation_forest.png`, etc.). These look like leftovers from an earlier single-model version of the eval pipeline. Not confirmed dead (not exhaustively diffed against current `evaluate_all.py` output logic), but suspicious — flagged for manual confirmation before deletion.

### What code is unused
- Backend: `evaluate.py`, `predict.py`, `src/api/drift_detector.py` (whole files, see above). Unused imports: `BaseModel, ConfigDict` in `main.py`, `ModelRun` in `asset_router.py`, `Session` in `retrain_pipeline.py` and `incident_report.py`.
- Frontend: two `package.json` dependencies with zero imports anywhere in `src/` — `date-fns` and `sonner` (a toast library installed but no `<Toaster/>` ever mounted).

### What code is risky to delete
- `models/lstm_autoencoder.pth` — looks unused today, but before deleting, confirm no external script (outside `src/`, e.g. a notebook or a teammate's local tool) depends on the raw `state_dict` format rather than the joblib-wrapped `.pkl`.
- `src/models/evaluate.py` — safe to delete by import evidence, but it may still be referenced in an unlisted doc or a grader's rubric expectation (this project has academic submission history — several planning docs reference "evaluate.py" by name in early drafts). Grep the planning docs before removing, not just the code.
- `reports/figures/confusion_matrix_if.png` / `pr_curve.png` / `roc_curve.png` — these are referenced by name in `docs/final_report/` sections or the PDF; deleting without checking those references first could break the already-generated academic report's embedded image links.
- Frontend hardcoded `localhost:8000` URLs — not risky to fix, but must be fixed consistently across all 5 files at once (`AlertCenter.tsx`, `Overview.tsx`, `AssetCenter.tsx`, `VisionLab.tsx`, `VibrationLab.tsx`) rather than piecemeal, or the app ends up half on env-config, half hardcoded, which is worse than either extreme.

---

## 2. Feature Status Matrix

| Feature | Implemented? | Evidence/File | Tested? | README claim correct? | Action needed |
|---|---|---|---|---|---|
| NAB temperature stream | Yes | `data/raw/realKnownCause/machine_temperature_system_failure.csv`, `src/data/data_loader.py` | Yes — `test_preprocessing.py`, `test_leakage.py` | Correct | None |
| Preprocessing | Yes | `src/data/preprocessing.py`, `data/processed/nab_processed.csv` | Yes — `test_preprocessing.py` | Correct | None |
| Feature engineering | Yes | `src/features/feature_engineering.py`, `models/feature_columns.json` | Yes — `test_features.py` | Correct | None |
| Stream simulator | Yes | `src/streaming/stream_simulator.py` | Yes — `test_stream.py` | Correct | None |
| FastAPI API | Yes | `src/api/main.py` (~24 routes + 3 included routers) | Partial — only `/health`,`/predict` covered by `test_api.py` | Mostly, except one wrong endpoint name | Add coverage for `/models/*`, `/experiments`, `/data/summary`; fix README endpoint name |
| WebSocket | Yes | `main.py` `/ws/stream`; `vibration_router.py` `/vibration/ws/stream` | No dedicated test found | Correct | Add a WS smoke test |
| SQLite | Yes | `src/database/database.py`, `reports/anomaly.db` | Yes, but currently failing in full-suite run | Correct (works), suite is red | Fix test isolation (§3) |
| React frontend | Yes | `frontend/src`, 12 pages routed in `App.tsx` | Build only, and build currently fails | Currently false (`npm run build` fails) | Fix `tsconfig` TS6 error (§5) |
| Model comparison | Yes | `reports/evaluation_results.csv`, `reports/model_comparison.json`, `evaluate_all.py` | Values cross-checked consistent | Correct | Remove dead `evaluate.py` duplicate |
| Model registry | Yes | `src/registry/model_registry.py`, `models/model_registry.json`, `/models/registry` | No dedicated test | Correct | Add test |
| Isolation Forest | Yes | `train_isolation_forest.py`, F1 0.9468 in CSV | Yes — `test_model.py` | Correct (0.946) | None |
| Rolling Z-Score | Yes | `train_baseline.py`, F1 0.2119 in CSV | Yes (generic model tests) | Correct (0.211) | None |
| One-Class SVM | Yes | `train_one_class_svm.py`, F1 0.7922 | Yes | Correct (0.792) | None |
| LOF | Yes | `train_lof.py`, F1 0.5443 | Yes | Correct (0.544) | None |
| Elliptic Envelope | Yes | `train_elliptic_envelope.py`, F1 0.2594 | Yes | Correct (0.259, honestly framed as a negative result) | None |
| LSTM Autoencoder | Yes | `train_lstm_autoencoder.py`, `lstm_wrapper.py`, F1 0.9921 | Yes — `test_model.py` | F1 correct (0.992), but ROC-AUC in the same CSV row is `1.2e-05` (near-zero/inverted) and unremarked anywhere | Investigate score-sign bug behind the ROC-AUC anomaly; add missing model card |
| River online model | Yes | `train_river_online.py`, F1 0.0472 | Yes | Correct (0.047, honestly framed as weakest) | Add missing model card |
| Drift detection | Yes | `src/drift/drift_detector.py` + `drift_service.py`, wired at `main.py` | Yes — `test_drift.py` | Correct | Delete unused duplicate `src/api/drift_detector.py` |
| Retraining workflow | Partial | `src/retraining/retrain_pipeline.py`, wired via `POST /models/retrain` | No dedicated test | README's `/retraining/promote/{model_id}` doesn't exist — actual path is `/models/select/{name}` | Fix README; add test; consider auto-promotion after shadow test if that was the original intent |
| Synthetic fault injection | Yes | `src/synthetic/*`, wired via `/faults/*` | Yes — `test_fault_injection.py` | Correct | None |
| Incident PDF | Yes | `src/reports/incident_report.py`, `/reports/incident/{alert_id}` | Yes — `test_incident_report.py` | Correct | Frontend link hardcodes `localhost:8000` (§3) |
| Alert lifecycle | Yes | `main.py` alert routes, `database.Alert` | Yes, but 3/3 fail in full-suite run | Correct (works), suite is red | Fix test isolation (§3) |
| NASA Bearing vibration | Yes | `src/vibration/*`, `vibration_router.py`, `models/vibration_iforest.pkl` | No dedicated test | Correct | Add test; route frontend through shared API client |
| MVTec image module | Yes | `src/image/*`, `image_router.py`, `models/vision_iforest.pkl` | No dedicated test | Correct | Add test |
| Visual Inspection Lab | Yes | `frontend/src/pages/VisionLab.tsx` — real fetches to `/image/gallery`, `/image/analyze` | Manual only | Correct | Route through `src/lib/api.ts` |
| Asset Center | Yes | `AssetCenter.tsx` + `asset_router.py` `/assets/*` | Manual only | Correct | Route through `src/lib/api.ts` |
| Final report | Yes | `docs/final_report/FINAL_PROJECT_REPORT.pdf`/`.md`, `sections/part1-6_*.md` | N/A | Correct | None |
| Figures/graphs | Yes | `docs/final_report/figures/*.png`, `reports/figures/*.png` | N/A | Mostly correct | Confirm and remove suspected stale generic figures (§1) |

---

## 3. Code Quality Review

**Dead code (whole files):**
- `src/models/evaluate.py` — superseded by `evaluate_all.py`, zero importers.
- `src/models/predict.py` — thin wrapper, zero importers.
- `src/api/drift_detector.py` — unused duplicate of `src/drift/drift_detector.py`.

**Unused imports:**
- `src/api/main.py:8` — `BaseModel, ConfigDict` never used.
- `src/api/asset_router.py:3` — `ModelRun` never used.
- `src/retraining/retrain_pipeline.py:4` — `Session` never used.
- `src/reports/incident_report.py:3` — `Session` never used.

**Unused frontend dependencies:** `date-fns`, `sonner` (installed, zero imports, no `<Toaster/>` mounted).

**Old Streamlit code:** None found — a repo-wide grep for `streamlit` in `.py` files returned no matches. The migration to React appears complete with no leftovers.

**Duplicated functions/logic:**
- `evaluate.py` vs `evaluate_all.py` overlap in PR/ROC-AUC/latency/plotting logic (see §1).
- Two files both named `drift_detector.py` with unrelated implementations, one dead (see §1).
- `lstm_autoencoder.pkl` + `.pth` both produced by `train_lstm_autoencoder.py`, only `.pkl` consumed.

**Hard-coded paths:**
- `src/retraining/retrain_pipeline.py:70,74` — hardcodes `".venv\Scripts\python.exe"` in a `subprocess.run` call. Breaks on Linux/macOS and inside the Docker image (which has no `.venv`). This is a real portability bug, not just style.
- `src/utils/config.py:8` — `API_URL` default is `http://localhost:8000` (acceptable as a default, but worth confirming it's overridden correctly in `docker-compose.yml`, which it is via `environment:`).
- Five backend modules (`vibration_router.py:15`, `image_router.py:14`, `asset_router.py:9`, `vibration/train.py:17`, `image/train.py:13`) independently hardcode `MODELS_DIR = "models"` instead of reading `get_settings().MODEL_DIR`. Same value today, but a silent drift risk if `MODEL_DIR` is ever changed via `.env`.
- Frontend: `AlertCenter.tsx:215`, `Overview.tsx:52`, `AssetCenter.tsx:14`, `VisionLab.tsx:15,34`, `VibrationLab.tsx:29` all hardcode `http://localhost:8000` / `ws://localhost:8000` instead of using the `VITE_API_URL`/`VITE_WS_URL` env vars that `src/lib/api.ts` and `ws.ts` correctly use elsewhere.

**Broken/stale references and version drift:**
- **Model/dependency version mismatch**: every `.pkl` model raises `sklearn.exc.InconsistentVersionWarning` on load — trained with scikit-learn **1.7.2**, but `requirements.txt` pins **1.5.1** (confirmed installed version matches the pin). This is a real reproducibility risk: a fresh `pip install -r requirements.txt` followed by loading the committed models produces version-mismatch warnings and an explicit "use at your own risk" from sklearn. Either the models need re-training under 1.5.1, or `requirements.txt` needs bumping to match what actually produced the artifacts.
- **`docker-compose.yml`** uses the obsolete top-level `version:` key — cosmetic, `docker compose config` runs fine but emits a deprecation warning.
- **`git ls-files` generated-file audit**: only 9 files under `models/`, `reports/`, `data/` are tracked, which is mostly correct given `.gitignore`, except `models/lstm_autoencoder.pth` (accidental commit, see §1) and `data/iot_anomaly.db` (tracked but empty — 0 bytes — looks like an intentional placeholder, not a mistake, but worth confirming its purpose).

**Stale metrics / unremarked anomaly:** LSTM Autoencoder's ROC-AUC of `1.2259e-05` in `reports/evaluation_results.csv` is inconsistent with its F1 of 0.992 — this pattern (near-zero AUC, high F1) usually indicates an inverted anomaly-score sign convention specific to reconstruction-error-based scoring not being handled the same way as the other models' scores when computing AUC. Not fixed here per instructions, but it is a genuine correctness question, not just a formatting nitpick.

**README overclaims found:** the `/retraining/promote/{model_id}` endpoint name, and both testing commands in README §15 (`pytest -q`, `npm run build`) currently failing as-is on a clean run.

**Future features presented as complete:** None found — cross-checking the "Current Implementation Status" table against actual code, every ✅ row does correspond to real, wired code (even if some corners, like retraining auto-promotion, are thinner than the README prose implies).

---

## 4. Repository Structure Review

The current structure is already close to the target layout described in `README.md` §17 and mostly matches what exists on disk. Recommended homes, none of which require moving currently-working files except where noted:

- `docs/` — keep as the root for all documentation; already correct.
- `docs/final_report/` — keep final report + sections here; already correct.
- `docs/final_report/figures/` — keep architecture/ERD/sequence diagrams (mermaid-sourced) here; already correct. Add the two missing model cards to `docs/model_cards/` (`lstm_autoencoder.md`, `river_hst.md`) rather than a new location.
- `src/models/` — keep all training scripts and `evaluate_all.py` here; **remove** `evaluate.py` and `predict.py` from here once confirmed safe to delete (§6 Priority 2).
- `src/api/` — keep routers and `main.py` here; **remove** `src/api/drift_detector.py` (dead duplicate) — the real drift logic already lives correctly in `src/drift/`.
- `src/reports/` — keep `incident_report.py` here; already correct, no change.
- `src/registry/` — keep `model_registry.py` here; already correct.
- `src/drift/` — keep as the single source of truth for drift logic after removing the `src/api/` duplicate.
- `src/retraining/` — keep `retrain_pipeline.py` here; fix the hardcoded `.venv` path (§6 Priority 1) but no relocation needed.
- `src/synthetic/` — keep as-is; already correctly separates fault generation from the rest of the pipeline.
- `frontend/src/pages/` — keep all 12 pages; no relocation needed, they're already well-organized and fully routed.
- `frontend/src/components/` — keep `ui/` primitives here; consider adding a `frontend/src/services/` (or extending `src/lib/`) directory specifically so `AssetCenter.tsx`, `VisionLab.tsx`, `VibrationLab.tsx`, `Overview.tsx`, `AlertCenter.tsx` can import typed API functions instead of inlining raw `fetch`/`WebSocket` calls with hardcoded URLs — this is the single biggest frontend structural gap (`src/lib/api.ts` doesn't cover asset/image/vibration routes at all).
- `scripts/` — keep `demo_reset.py`, `generate_diagrams.py`, `generate_final_report.py`; already correctly separated from `src/`.

No files need to move between top-level directories — the structural problems here are internal (dead files, missing client coverage, hardcoded values), not misplacement.

---

## 5. Validation Command Plan

Commands run during this scan, with actual results:

| Command | Result | Detail |
|---|---|---|
| `pytest -q` | **FAIL** | 4 failed: `test_alert_lifecycle.py::test_update_alert_status`, `::test_update_alert_feedback`, `::test_resolve_alert` (all `sqlalchemy.exc.OperationalError` — but **pass individually in isolation**, confirming order-dependent test pollution from sharing a real file-backed SQLite DB `test_alerts.db` across the suite instead of an isolated/in-memory DB per test module); `test_model.py::test_explain_anomaly` (stale string assertion — code returns `"20.0 std devs"`, test expects `"20.0σ"`) |
| `python -m pytest` | Same as above (equivalent invocation) |
| `python -m src.models.evaluate_all` | Not re-run (would overwrite `reports/evaluation_results.csv`/figures) — confirmed via code read to be the real, current entrypoint; `evaluate.py` is the dead alternative |
| `cd frontend && npm run build` | **FAIL** | `tsc -b` exits with `TS5101: Option 'baseUrl' is deprecated` under TypeScript 6.0.3 (pinned `~6.0.2` in `package.json`); `vite build` never runs because `tsc -b` fails first. The committed `frontend/dist/` predates this TypeScript version. |
| `docker compose config` | **PASS** (with warning) | Valid config resolved for both `api` and `frontend` services; only a cosmetic warning that the top-level `version:` key in `docker-compose.yml` is obsolete |
| `docker compose up --build` | Not run in this scan (would launch containers — held per "do not modify the system yet") |

---

## 6. Cleanup Plan (proposed — awaiting approval, nothing executed)

**Priority 1 — Safe fixes (no behavior change, restore green CI):**
1. Fix `tests/test_alert_lifecycle.py` (or shared test fixture setup) so the SQLite test database is isolated per test module (in-memory `sqlite://` or a unique temp file per test run), removing order-dependence.
2. Fix `tests/test_model.py::test_explain_anomaly` — align the assertion with the actual `explain_anomaly` output format (or fix the format if the σ symbol was the intended UX and the string was accidentally changed — needs a judgment call before editing).
3. Fix `frontend/tsconfig.app.json` TS6 `baseUrl` deprecation (add `"ignoreDeprecations": "6.0"` or migrate to the new `paths`-only config) so `npm run build` passes again.
4. Fix `src/retraining/retrain_pipeline.py`'s hardcoded `.venv\Scripts\python.exe` path — use `sys.executable` instead so it works cross-platform and inside Docker.
5. Correct README §12 to document `POST /models/select/{name}` instead of the nonexistent `/retraining/promote/{model_id}` (or add a real `/retraining/promote/{model_id}` alias if that name was the intended contract — needs your decision).

**Priority 2 — Remove unused files (verify no external references first):**
1. Delete `src/models/evaluate.py` (after grepping planning docs for direct references, per §1 risk note).
2. Delete `src/models/predict.py`.
3. Delete `src/api/drift_detector.py`.
4. Remove `models/lstm_autoencoder.pth` from git tracking (keep locally-generated, add `*.pth` to `.gitignore`).
5. Remove unused imports: `BaseModel`/`ConfigDict` in `main.py`, `ModelRun` in `asset_router.py`, `Session` in `retrain_pipeline.py` and `incident_report.py`.
6. Remove `date-fns` and `sonner` from `frontend/package.json` (or wire up `sonner`'s `<Toaster/>` if toast notifications were actually intended).
7. Confirm and remove stale `reports/figures/confusion_matrix_if.png`, `pr_curve.png`, `roc_curve.png` if they're not referenced by the already-generated final report PDF/markdown.

**Priority 3 — Restructure docs:**
1. Add missing model cards: `docs/model_cards/lstm_autoencoder.md`, `docs/model_cards/river_hst.md`.
2. No directory moves needed (see §4) — this priority is additive documentation only.

**Priority 4 — Improve README:**
1. Fix the retraining endpoint name (Priority 1.5, tracked here too since it's a README-specific edit).
2. Add a short note next to the LSTM Autoencoder row in §8 acknowledging the ROC-AUC anomaly pending investigation, or fix the underlying scoring bug first and then update the number.
3. Update §15 Testing section to reflect actual current pass/fail state until Priority 1 fixes land.

**Priority 5 — Regenerate figures/report (only after Priority 1–2 land):**
1. Re-run `python -m src.models.evaluate_all` if the LSTM AUC scoring bug is fixed, to refresh `reports/evaluation_results.csv`, `reports/model_comparison.json`, and all `reports/figures/*.png` with corrected numbers.
2. Regenerate `docs/final_report/figures/*.png` via `scripts/generate_diagrams.py` only if any architecture actually changed (not needed for the fixes above).

**Priority 6 — Final validation:**
1. `pytest -q` → expect all green.
2. `cd frontend && npm run build` → expect success.
3. `docker compose config` → expect no warnings (after removing obsolete `version:` key).
4. `docker compose up --build` → manual smoke test of the full stack before considering cleanup complete.

---

**Nothing has been deleted, edited, or regenerated as part of producing this document.** Awaiting your approval before executing any Priority level above.
