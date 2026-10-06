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
- **Repo:** https://github.com/jrhughes003/ais-sentinel (public). Pages:
  https://jrhughes003.github.io/ais-sentinel/ (GitHub Actions source). Created automatically
  with `gh`, on the owner's authorisation; see GITHUB_SETUP.md.
- **Phase 1 vertical slice works end to end:**
  `download → build → track → (samples, baselines, metrics) → export → site map`.
- **Full download running in the background:** `logs/download.log` and
  `data/interim/manifest.json`. Expect about 3 h. Re-running `ais-sentinel download` retries
  failed days and skips finished ones.
- **First look on 8 May/July days (not a result, a smoke test):**
  - Dead reckoning mean error: about 1.1 km at 15 min, 8.0 km at 60 min, 19.7 km at 120 min.
  - Cargo ships alone: 0.7 km at 15 min, 5.7 km at 60 min. The river bends hurt straight-line
    prediction.
  - The CV-KF re-initialises often on ferries' hard manoeuvres, which motivates the IMM.

## Next up
1. Python tests for samples, metrics, gaps and export (coverage).
2. Once the download finishes: rerun `build`/`track` on all six months and write
   `reports/data_summary.md`.
3. **Phase 3:** AIS-like simulator, then IMM (stationary / CV / CT-EKF), then the simulation
   study.
4. **Phase 4:** IMM extrapolation, kNN route baseline, calibration on validation, and the
   evaluation report.

## Blocked / needs my input
- (none)
- FYI: a helper Claude session ("diag-d6") reported clearing temp files outside this project
  (WSL crash dumps, swap vhdx) to free disk space. This session did not do that or ask for
  it; please confirm you're happy with it.

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
- Step 0 continued: the owner authorised `gh`, so I created the public repo and enabled
  Pages (workflow build type). The first deploy succeeded.
- Data:
  - Downloader verified on 2023-07-12: 9.96 M national rows → 59,861 AOI rows (1.3 MB).
  - Fixed CSV parsing: names contain literal quotes, so quoting is disabled; rare ragged
    lines are dropped.
  - Added single-worker prefetch.
- Built geo (ENU via ECEF), clean, segment, build, CV-KF plus RTS, the track stage,
  prediction samples, B0/B1 baselines, metrics with cluster-bootstrap CIs, the gap detector
  and the export.
- Site: MapLibre map (fixed the v6 worker URL under Vite). Playwright smoke tests pass at
  1280×800 and 390×844.
- Gotchas recorded:
  - Python `write_text` on Windows defaults to cp1252 and CRLF. Use `PYTHONUTF8=1` and LF
    (ruff is pinned to LF).
  - `.gitignore` `data/` matched `site/public/data`; it is now anchored as `/data/`.
