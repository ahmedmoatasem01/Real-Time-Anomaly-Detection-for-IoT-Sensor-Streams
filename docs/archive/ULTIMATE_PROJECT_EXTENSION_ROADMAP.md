# ULTIMATE_PROJECT_EXTENSION_ROADMAP.md
## Real-Time IoT Anomaly Detection → Multi-Modal Industrial Platform — Strategic Roadmap

> Strategy, not code. The memo you received is a strong idea-bank; my job here is the harder one — deciding **what to build, what to cut, and in what order**, given this is a graded academic project with a deadline, not a startup with a runway. I'll be blunt where blunt is useful.

---

## 0. The one decision that governs everything

**You have a working, complete, gradeable platform right now.** Five models, registry, hot-swap, 8-page React app, WebSockets, Docker, 13 tests. The failure mode from here is not "too small" — it's **breaking a working system by bolting on half-finished modules before the demo.** Every recommendation below is filtered through: *does this raise the grade more than it raises the risk?*

There is also an unresolved credibility issue from the last review that **outranks every expansion idea**: four of your five models report Recall = 1.000 with zero false negatives, and Elliptic Envelope flags 99.97% of normal points yet still "catches everything." Until the evaluation is reframed (windowed vs point scoring, latency as tiebreaker, leakage ruled out), **adding a vibration lab or an image lab just adds more surface for a grader to distrust.** Fix the numbers first. Expansion second.

---

## 1. Best final project vision

A **multi-modal industrial condition-monitoring platform**: a live sensor-stream anomaly detector (production path), extended with a vibration predictive-maintenance module and a visual-inspection module, unified by one ML backbone (registry, experiments, multi-model comparison), real-time inference, alert lifecycle, and a professional operator-facing web platform. The live temperature dashboard is one module of several — that's what makes it read as a *platform* rather than a dashboard.

**Crucial honesty constraint:** the three data tracks are **separate modules that share infrastructure**, not a fused multi-modal model. Do **not** claim sensor + vibration + image fusion — that's a research problem you don't need and can't defend. The correct, defensible story is: *"one platform, one ML backbone, three industrial asset types, each with its own appropriate detector."*

## 2. Best project title

> **Real-Time Industrial Anomaly Detection Platform for Multi-Modal Asset Monitoring**

Subtitle for the report: *sensor streams, vibration signals, and visual inspection — unified multi-model ML, live inference, and alert management.*

## 3. Recommended modules

| Module | Status | Verdict |
|---|---|---|
| Live IoT Monitoring (NAB temperature) | exists | **Keep — production demo** |
| Model Lab (multi-model + registry) | exists | **Keep — strongest ML evidence** |
| Data Explorer | exists | **Keep** |
| Alert Center | exists | **Keep — extend to lifecycle** |
| Experiment Results | exists | **Keep** |
| System Health | exists | **Keep** |
| Demo Control Panel | planned | **Build — cheap, big demo payoff** |
| Vibration Health Lab (NASA Bearing) | new | **Build if ≥2 weeks — best data extension** |
| Visual Inspection Lab (MVTec AD) | new | **Only if ≥1 month — highest effort/risk** |
| Synthetic Fault Injection | new | **Build — best demo control per hour spent** |

## 4. Recommended datasets (in priority order)

1. **NAB machine temperature** *(keep as production stream)* — already implemented, gives the real-time story.
2. **NASA Bearing** *(vibration_signal)* — the single best extension. Still sensor/industrial/predictive-maintenance, so it *fits the same narrative*. Run-to-failure vibration → time + frequency features → degradation trend. Difficulty medium-high. **Implement in the 2-week scope.**
3. **Synthetic multi-sensor generator** — not a Kaggle set, but the best *demo* addition: you control faults on demand (spike, drift, stuck sensor, missing data). Difficulty medium. **Implement early — it de-risks the live demo.**
4. **MVTec AD** *(image)* — genuinely impressive and multi-modal, but a different ML stack (CNN embeddings/autoencoder) and a different failure surface. Difficulty high. **Only in the 1-month scope, and only after everything else is stable.**
5. **Synthetic maintenance logs** *(text)* — cheap flavor for an alert/incident story; optional. Skip unless time is abundant.
6. **Semiconductor wafer** — **cut.** It drags the project from "IoT streaming" toward "manufacturing process analytics" and dilutes the narrative. Don't.

## 5. Recommended models

- **Classical (done):** Rolling Z-score, Isolation Forest (production), One-Class SVM, LOF, Elliptic Envelope. *Keep; fix their evaluation framing.*
- **Deep (do next):** **LSTM Autoencoder** on the temperature stream — the one genuine depth gap; expected at this level. Then a **1D-CNN Autoencoder** for vibration if the Bearing module is built.
- **Online (strong added value):** **River HalfSpaceTrees** + adaptive threshold → enables **drift detection**. Medium difficulty, high "MLOps" signal.
- **Image (1-month scope only):** ResNet18 embeddings + Isolation Forest first; CNN autoencoder + reconstruction heatmap if time. Do **not** start with PatchCore.
- **Supervised (XGBoost/RF):** **do not make it core.** NAB labels are sparse windows; supervised models overfit and post suspiciously perfect numbers. Include at most one run explicitly labeled *"experimental supervised baseline — overfits sparse-window labels, not a production candidate."*
- **Ensemble "model agreement":** high impact, low cost — show "4 of 6 models agree → anomaly, 75% confidence." Build it; it's a great slide.

## 6. Recommended frontend pages

Keep the 8 you have; add in this order of value: **Demo Control Panel** (cheap, presentation marks) → **Vibration Health Lab** (waveform + FFT + degradation trend) → **Visual Inspection Lab** (upload → normal/defect → heatmap, 1-month only). Everything shares the existing dark design system and sidebar so new pages feel native, not bolted on.

## 7. Recommended backend APIs

Existing predict/batch/alerts/metrics/ws/models/registry/comparison/experiments stay. Add only what a built module needs:
- `GET /assets` — asset registry (machine/bearing/product) powering the "Asset Center" framing.
- `GET /vibration/sample`, `POST /vibration/analyze` — if Bearing module built.
- `POST /image/analyze` — if Visual Inspection built.
- `POST /faults/inject` — synthetic fault injection for the demo.
- `POST /alerts/{id}/status`, `POST /alerts/{id}/feedback` — alert lifecycle + operator feedback.
- `GET /reports/incident/{id}` — incident PDF export.
Don't add endpoints for modules you haven't committed to building.

## 8. Recommended database tables

Additive to `readings`/`alerts`/`model_runs`:
- `assets` (id, name, type, modality, status) — cheap, enables the platform framing. **Build.**
- **Extend `alerts`** with `status` (new/ack/investigating/resolved/false_alarm), `operator_note`, `feedback`, `resolved_at` — the alert-lifecycle story. **Build.**
- `datasets` (registry) — or keep as `data/dataset_metadata.json` (simpler). **JSON is fine.**
- `vibration_runs`, `image_inspections` — only if those modules are built.
- `maintenance_logs` — only if the text track is built. Likely skip.

## 9. Recommended repository additions

```
src/vibration/        # loader, feature_extraction (RMS/kurtosis/FFT), train, analyze   [2-wk]
src/image/            # embedding_extractor, train_image_ad, analyze                     [1-mo]
src/synthetic/        # multi_sensor_generator, fault_injector                           [early]
src/reports/          # incident_report.py (PDF export)                                  [opt]
data/dataset_metadata.json                                                               [now]
docs/model_cards/*.md                                                                    [now — cheap doc marks]
frontend/src/pages/   # DemoPanel, VibrationLab, VisualInspectionLab                     [as built]
```
Additive only; nothing existing moves.

## 10. Recommended implementation phases

- **Phase 0 — Stabilize (non-negotiable, do first):** fix evaluation framing (windowed scoring + latency tiebreaker + leakage test), reconcile all result files, green tests, clean build.
- **Phase 1 — Depth on existing data:** LSTM Autoencoder + River online + drift + ensemble agreement. All on NAB. No new data, low risk, high ML marks.
- **Phase 2 — Product polish:** alert lifecycle + operator feedback + incident PDF + Demo Control Panel + synthetic fault injection + model cards.
- **Phase 3 — Second modality:** NASA Bearing → Vibration Health Lab (time + FFT features, 1D-CNN AE or IF, degradation trend).
- **Phase 4 — Third modality (stretch):** MVTec AD → Visual Inspection Lab (ResNet embeddings + IF, then heatmap).

## 11. What to implement first

1. **Fix the Recall=1.0 evaluation framing** (credibility gate — nothing else matters if the numbers look faked).
2. **GUI enhancement** (you already asked for it; highest demo visibility).
3. **LSTM Autoencoder** (closes the deep-model gap).
4. **Synthetic fault injection + Demo Control Panel** (makes the live demo controllable and reliable).
5. **Ensemble model agreement** (cheap, impressive).

## 12. What to avoid

Kafka, TimescaleDB, Redis, Kubernetes, full auth, CI/CD, a real MLflow server, real email/SMS alerting, the semiconductor dataset, and **multi-modal fusion claims.** Also avoid starting the image module before Phases 0–2 are solid. The strongest project has the clearest working story, not the most technologies.

## 13. Realistic for 1 week

Phase 0 + GUI enhancement + LSTM Autoencoder + synthetic fault injection + Demo Control Panel + model cards. **No new datasets.** This alone is a high-grade project: multi-model (6 incl. deep), honest evaluation, controllable demo, professional UI, MLOps evidence.

## 14. Realistic for 2 weeks

Everything in the 1-week scope, plus **River online + drift detection + ensemble agreement** (Phase 1 complete), **alert lifecycle + operator feedback + incident PDF** (Phase 2), and the **NASA Bearing Vibration Health Lab** (Phase 3). This is a clearly multi-modal industrial platform — sensor + vibration — and reads as capstone-level.

## 15. Realistic for 1 month

Everything above plus the **MVTec Visual Inspection Lab** (Phase 4), the Asset Center framing, and optional synthetic maintenance logs. Now it's genuinely a three-modality platform. Only pursue if Phases 0–3 are rock-solid with two weeks of buffer; a half-built image module hurts more than no image module.

## 16. Final recommended scope for a high-grade demo

**Target the 2-week scope. It is the sweet spot: unmistakably large and professional, still achievable, still stable.**

- **Data:** NAB temperature (live) + NASA Bearing (vibration) + synthetic multi-sensor (demo control).
- **Models:** Z-score, IF (prod), OCSVM, LOF, Elliptic, **LSTM-AE**, **River+drift**, + **ensemble agreement**; 1D-CNN-AE for vibration.
- **Backend:** existing + assets, vibration, faults/inject, alert-lifecycle, incident-report endpoints.
- **Frontend:** existing 8 pages (GUI-enhanced) + Demo Control Panel + Vibration Health Lab.
- **Product:** severity, explainability, alert lifecycle, operator feedback, incident PDF, synthetic fault injection, model promotion, drift warnings.
- **Evidence:** honest windowed+point metrics with latency tiebreaker, model cards, experiment history, green tests, Docker, README + final report.

Cut the image module unless you reach the 1-month runway with buffer. Keep Isolation Forest the stable production default until LSTM-AE and River are proven on the same honest test split.

---

## The exact order to hand Claude Code

```
0. Fix evaluation framing (windowed scoring, latency tiebreaker, leakage test); reconcile result files
1. GUI enhancement (design system, motion, react-query, page upgrades)
2. LSTM Autoencoder on NAB → register → appears in Model Lab
3. Synthetic fault injection + Demo Control Panel
4. Ensemble model agreement (vote count + confidence)
5. River online + drift detection
6. Alert lifecycle + operator feedback + incident PDF export
7. NASA Bearing → Vibration Health Lab (time+FFT features, 1D-CNN AE / IF, degradation trend)
8. (1-month only) MVTec AD → Visual Inspection Lab (ResNet+IF, then heatmap)
```

### Guardrails
- Fix credibility (Recall=1.0 framing) before any expansion.
- Additive only; never regress the working platform; Isolation Forest stays production default until challengers are proven.
- Three modalities are **separate modules sharing infrastructure** — no fusion claims, no pretending image data comes from the same machine.
- Real metrics only, one honest NAB test split, no leakage.
- Prefer the clearest working story over the longest technology list; a half-built module is worse than no module.
