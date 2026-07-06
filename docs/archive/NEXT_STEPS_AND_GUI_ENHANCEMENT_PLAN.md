# NEXT_STEPS_AND_GUI_ENHANCEMENT_PLAN.md
## Real-Time IoT Anomaly Detection — Status Review, Evaluation Audit, and GUI Upgrade

> Two parts. **Part A** is an honest audit of where the project stands and exactly what to do next (with one issue in the evaluation you must fix *before* the demo). **Part B** is a deep, build-ready GUI enhancement plan for Claude Code.

---

# PART A — Status review & next steps

## A1. What is genuinely done (and good)
The platform is real and complete against the brief: five models trained (Isolation Forest production + One-Class SVM, LOF, Elliptic Envelope, Rolling Z-score), a unified `evaluate_all.py` producing one test-split table, a model registry with hot-swap, experiment tracking, a full FastAPI surface with WebSocket streaming, SQLite persistence, an 8-page React/TS/Vite/Tailwind platform, 13 passing tests, and a valid Docker setup. That is a strong, gradeable system. Good work.

## A2. The evaluation problem you must fix first 🔴 (credibility)
Look at the real numbers in `evaluation_results.csv`:

| Model | Recall | FN | FP | False-alarm | F1 | PR-AUC |
|---|---|---|---|---|---|---|
| Isolation Forest | **1.000** | **0** | 57 | 1.97% | 0.947 | 0.9994 |
| One-Class SVM | **1.000** | **0** | 266 | 9.19% | 0.792 | 0.9988 |
| LOF | **1.000** | **0** | 849 | 29.3% | 0.544 | 0.9904 |
| Elliptic Envelope | **1.000** | **0** | 2895 | 99.97% | 0.259 | 0.318 |
| Rolling Z-score | 0.438 | 285 | 1366 | 47.2% | 0.212 | 0.155 |

**Four models catch 100% of anomalies with zero misses.** Elliptic Envelope flags *everything* (FP=2895, TN=1) yet still posts Recall=1.0. That pattern is a **tell**: a perfect recall with zero false negatives across four very different algorithms almost never happens on real data unless the evaluation is set up so anomalies are trivially separable. A sharp examiner will see "Recall = 1.000" four times and distrust the whole table. **This is the single most important thing to address before grading — not because the code is wrong, but because unexplained perfection reads as faked even when it isn't.**

**Most likely cause (investigate in this order):**
1. **Point-wise scoring on window labels.** NAB labels are *windows*. If the whole window (507 points) is marked positive and the models fire anywhere in the elevated-temperature region, you get FN=0 by construction. This is the usual culprit.
2. **Threshold tuned on the same distribution as test**, or test anomalies sitting far outside the normal range (the machine-failure spike is large and obvious), making them easy.
3. **Label leakage** — anomaly windows influencing the scaler/threshold.

**What to do (honest fixes, pick what applies):**
- **Report NAB-style windowed scoring alongside point metrics**, and *say so*: "a detection anywhere inside a labeled window counts as a hit; point-metrics shown for reference." That reframes Recall=1.0 as expected and defensible, not suspicious.
- **Add detection latency in real terms** (minutes from window onset to first flag) — this becomes the discriminating metric when everyone hits Recall=1.0. It's already a column; make it meaningful and put it front-and-centre in Model Lab.
- **Confirm no leakage**: scaler + thresholds fit on train/normal only; test never touches them. Add a one-line assertion/test.
- **Keep Elliptic Envelope in the table as an honest negative result** — "flags 99.97% of points, unusable; included to show not all models suit this data." That *increases* credibility.
- Re-run `evaluate_all.py` and make the CSV, `model_comparison.json`, and the numbers in `REPO_STATUS` all agree.

✅ **Done when:** the results table distinguishes windowed vs point scoring, latency is the tiebreaker, leakage is ruled out by a test, and no metric is presented without context that explains why it's high.

## A3. Prioritized next steps

**P1 — Fix the evaluation framing (A2).** Highest priority; it's the difference between "impressive" and "unbelievable" to a grader.

**P2 — GUI enhancement (Part B).** Your ask, and the highest-visibility upgrade for the demo. Detailed below.

**P3 — One advanced model for depth: LSTM Autoencoder.** You have five classical models; a deep temporal model adds real range and is expected at this level. Reconstruction-error thresholded on normal windows. Register it so it appears in Model Lab automatically. (River online + drift is a nice-to-have after.)

**P4 — Demo hardening.** Seed script that jumps the simulator to a failure window (`--start-index`), a reset script, and a scripted 4-minute walkthrough.

**Explicitly skip for now (overengineering):** Kafka, AWS/GCP deploy, auth. They add infrastructure, not marks. Mention them only as "future work" on a slide.

---

# PART B — GUI enhancement plan (deep, build-ready)

Goal: elevate the 8-page app from "clean and functional" to "modern product." The bones are right (React/TS/Vite/Tailwind, WebSockets, code-split). This part adds motion, a real design system, richer data-viz, and the polish that makes it look like a 2025-era monitoring SaaS — without a rewrite.

## B0. Install the enhancement layer
```bash
cd frontend
npm install framer-motion @tanstack/react-query lucide-react \
  clsx tailwind-merge sonner @radix-ui/react-tooltip @radix-ui/react-dialog \
  recharts d3-scale
```
- **framer-motion** — page/element motion.
- **@tanstack/react-query** — caching, polling, loading/error states (replaces ad-hoc fetches).
- **sonner** — toast notifications (model promote, alert fired).
- **lucide-react** — consistent icon set.
- **clsx + tailwind-merge** — clean conditional styling (`cn()` helper).

## B1. Design system (do this first — everything else inherits it)
Create `frontend/src/styles/tokens.css` and a Tailwind theme extension.

**Palette — dark monitoring, one accent:**
```
--bg:        #0A0E14   /* app background */
--surface:   #111721   /* cards */
--surface-2: #161D28   /* elevated */
--border:    rgba(255,255,255,0.06)
--text:      #E6EDF3
--muted:     #8B98A9
--accent:    #22D3EE   /* cyan — healthy/active */
--warn:      #F59E0B   /* amber — medium severity */
--crit:      #EF4444   /* red — high severity / anomaly */
--ok:        #34D399   /* green — system healthy */
```
Rules: **one** accent (cyan), semantic colors only for state, generous whitespace, `border` + subtle shadow for depth (no heavy drop-shadows), 8px spacing grid, `rounded-xl` cards, `font-medium` headings. **No purple gradients, no neon-on-black, no glassmorphism overload.**

**Typography:** Inter or Geist (`npm i @fontsource/inter`), tabular-nums for all metrics so numbers don't jitter as they update.

**Reusable primitives to build (`src/components/ui/`):**
- `StatCard` — label, big tabular number, delta, sparkline, status dot.
- `StatusDot` — ok/warn/crit pulse animation.
- `SeverityBadge` — low/medium/high pill.
- `MetricSparkline` — tiny inline Recharts line.
- `DataTable` — sortable, filterable, sticky header, zebra rows, empty state.
- `SectionHeader` — title + subtitle + right-slot actions.
- `Skeleton` — shimmer loader for every async panel.
- `Drawer` / `Dialog` — Radix-based, for alert details.

✅ **Done when:** tokens applied globally; all pages use the same cards, headers, badges; numbers use tabular-nums.

## B2. Motion & liveness (framer-motion)
- **Page transitions:** wrap route content in `<motion.div>` fade+slide (opacity 0→1, y 8→0, 180ms).
- **KPI count-up:** animate stat numbers from previous→new value (150–250ms) so live updates feel alive.
- **Alert enter/exit:** `AnimatePresence` on alert rows — new alerts slide in, acknowledged ones fade out.
- **Anomaly pulse:** when a new anomaly arrives on Live Monitor, pulse the red marker + flash the KPI card border once.
- **Chart mount:** Recharts `isAnimationActive` on first render only; disable on live tick to avoid jitter.
Keep everything ≤250ms and eased. Motion signals real-time, never distracts.

✅ **Done when:** navigation, KPIs, and alerts all animate subtly; live ticks don't cause layout jitter.

## B3. Data fetching with react-query (replaces manual fetch)
- Wrap app in `QueryClientProvider`.
- One typed hook per endpoint in `src/lib/queries.ts`: `useModels()`, `useComparison()`, `useAlerts()`, `useSystemStatus()`, `useExperiments()`, `useDataSummary()`.
- Polling where it matters: `useSystemStatus({refetchInterval: 3000})`, `useAlerts({refetchInterval: 5000})`.
- Built-in `isLoading`/`isError` → drive `Skeleton` and error states everywhere. No more blank flashes.

✅ **Done when:** every page shows a skeleton while loading and a friendly error card on failure; live pages poll.

## B4. Page-by-page upgrades

**Live Monitor** (keep behavior, elevate visuals): dual synced charts (temperature + anomaly score) sharing an x-axis; threshold band shaded; anomaly points as pulsing red dots; a live "now" cursor; top KPI row (readings/s, latency, active model, current severity) with count-up; a compact "latest 5 alerts" side panel with slide-in. Add a **speed/scale** readout so viewers know it's a replay.

**Overview** (make it a real landing): hero status strip (system OK/degraded with animated dot), four `StatCard`s (total readings, total anomalies, production model, uptime), a mini leaderboard (top-3 models by PR-AUC/latency), and a "Start demo in 3 steps" card. First impression = product.

**Model Lab** (your strongest page — make it shine): model **cards** with metric chips + PR-AUC/ROC-AUC mini-curves; a **ranking table** (sortable by any metric — and since Recall ties at 1.0, make **detection latency + false-alarm rate the visible differentiators**); confusion-matrix heatmaps (use a proper color scale, not raw PNGs where possible); a **Promote to production** button → confirm dialog → `sonner` toast → refetch → production badge moves. This page sells the "multi-model platform" story.

**Data Explorer:** label-distribution donut, train/val/test split bar, the four anomaly windows on a timeline, feature list with type chips, sample-rows table, a short "how preprocessing works" explainer card. Charts, not walls of text.

**Alert Center:** `DataTable` with severity filter chips, time-range filter, acknowledge action (optimistic update + toast), and a details **Drawer** showing the reason + a mini timeline chart of the ±30-step window around the anomaly.

**Experiment Log:** sortable history table; columns for model, feature set, threshold, key metrics, timestamp; embedded PR/ROC thumbnails; an **Export** button (download CSV/JSON).

**System Health:** status dots for API/DB/WS with pulse; latency + stream-rate sparklines; a live **log tail** panel (monospace, auto-scroll, color by level); DB size + row counts. Polls every 3s.

**Demo Panel:** copy-paste commands with a one-click copy button; `--speed` and `--start-index` explained; a normal→anomaly scenario walkthrough; a checkable demo checklist (state persisted in memory only — no localStorage in artifacts, but fine in your real app).

✅ **Done when:** all 8 pages share the design system, animate, show skeletons/errors, and each has at least one real chart or interactive element beyond plain text.

## B5. Global shell polish
- **Sidebar:** icon + label, active-route highlight (accent left-border + subtle bg), collapsible to icons, section grouping (Monitoring / ML / System).
- **Top bar:** WS status dot + text, active production model chip, live clock, global search-command (optional `cmdk`).
- **Responsive:** graceful down to laptop (1280px); sidebar collapses; charts reflow.
- **Empty/error/loading states** everywhere — the mark of a finished product.
- **Favicon + title + meta** so browser tab looks intentional.

✅ **Done when:** the app feels like one cohesive product, not eight separate pages.

## B6. Build & verify
```bash
cd frontend
npm run build      # clean, no chunk warnings (charts + vendor split)
npm run dev        # visual pass across all 8 pages
```
✅ **Final GUI DoD:** clean build; 8 cohesive pages; one-accent dark theme; tasteful motion; skeletons + error states; Model Lab ranks by latency/false-alarm (not just tied recall); no purple gradients, no generic AI look.

---

# PART C — Execution order (literal)

```
P1  Fix evaluation framing (A2):
    1. add windowed (NAB-style) scoring alongside point metrics
    2. surface detection latency (minutes) as the tiebreaker
    3. add a no-leakage assertion/test
    4. keep Elliptic Envelope as an honest negative result
    5. re-run evaluate_all; reconcile CSV / comparison.json / status doc
P2  GUI enhancement (Part B):
    B0 deps → B1 design system → B2 motion → B3 react-query
    → B4 page upgrades → B5 shell polish → B6 build+verify
P3  LSTM Autoencoder (advanced depth); register → appears in Model Lab
P4  Demo hardening: seed-to-window script, reset script, 4-min walkthrough
Skip: Kafka, cloud deploy, auth (future-work slide only)
```

## Guardrails
- Additive only; don't regress the working platform; Isolation Forest stays production default.
- **No metric shown without context** — windowed vs point scoring stated explicitly; Recall=1.0 explained, not hidden.
- Real numbers only, from the real NAB test split; no leakage (assert it).
- One accent, tasteful motion, dark pro UI; no purple gradients, no generic AI look.
- `pytest -q` and `npm run build` stay green after every change.
