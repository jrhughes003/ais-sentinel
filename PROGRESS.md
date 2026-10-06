# PROGRESS

## Plan summary (set 2026-10-05; full details in PLAN.md)

**What:** a maritime domain awareness system built on public MarineCadastre AIS data. It
covers the **Detroit–St. Clair corridor** (Port Huron to western Lake Erie,
41.4–43.1 °N, 83.6–82.3 °W) for **May–Oct 2023**, about 10 M reports.

**Pillars:**
1. CV Kalman filter and an IMM (stationary / cruising / coordinated-turn EKF), validated
   on simulated ground truth.
2. Prediction at 15/30/60/120 min, with uncertainty. Baselines: dead reckoning, KF and IMM
   extrapolation, and historical kNN routes. ML: a GRU with Gaussian and mixture heads,
   plus an optional small transformer.
3. Anomaly detectors: gaps, kinematic jumps, loitering, route deviation and rendezvous.
   They are evaluated by synthetic injection and backed by real case studies.

The results go on a static Vite + TypeScript + MapLibre site on GitHub Pages.

**Splits:**

| Split | Dates (2023) |
|---|---|
| train | May–Aug |
| validation | Sep 2–30 |
| **locked test** | Oct 2–31 |

There are 1-day buffers between splits, and each voyage falls entirely inside one split.
Test results are also reported for vessels never seen in training.

**Success criteria (fixed before results):**

| Area | Target |
|---|---|
| Data | `run-all` reproduces the dataset; data-quality (DQ) tests pass; ≥ 5 M rows |
| Tracking (simulation) | IMM ≥ 20% lower RMSE than the tuned CV-KF in manoeuvres; ≤ 10% worse on straights |
| Tracking (NEES) | Inside the 95% bounds for ≥ 80% of steps |
| Tracking (real NIS) | 2–10% of values exceed the χ²₂(0.95) threshold |
| Prediction | Best ML model beats the best baseline at 60 and 120 min (paired CI excludes 0) and is ≥ 15% better at 120 min; 90% ellipse coverage in [85, 95]% |
| Anomalies, precision/recall | gap 0.9/0.9, jump 0.9/0.9, loiter 0.8/0.8, rendezvous 0.8/0.8, route deviation 0.7/0.7, each on its stated magnitude bucket |
| Anomalies, case studies | ≥ 3 real |
| Site | Every view works; Playwright passes at desktop and mobile widths; 0 serious axe violations; initial load ≤ 2 MB; site ≤ 50 MB |
| Engineering | CI green; ≥ 85% coverage on core modules; README complete |

## Current state
- Phase 0 (planning plus skeleton) in progress.

## Next up
1. Finish the skeleton (package, CI, Pages workflow, README). Create the GitHub repo.
2. Phase 1 vertical slice: download 3 sample days, clean, CV-KF, dead reckoning, gap rule,
   minimal map page.

## Blocked / needs my input
- (none)

## Log
### 2026-10-05
- Step 0: `ontario-grid-mpc` had no git repo and only planning docs (PLAN, CLAUDE,
  docs/SETUP, docs/KICKOFF_PROMPT), so it was deleted as instructed.
- Created `ais-sentinel` and initialised git.
- Environment check:
  - No CUDA GPU (Intel Iris Xe), 16 GB RAM, about 32 GB free disk.
  - Python 3.12.10, Node 24, torch 2.14 CPU.
- Research: MarineCadastre formats and URLs (verified by HEAD requests and in-memory
  sampling), the literature, and the web stack. Written up in docs/RESEARCH.md.
- Wrote PLAN.md, CLAUDE.md, DECISIONS.md (D1–D7) and this file.
