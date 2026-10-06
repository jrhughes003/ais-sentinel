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
  https://jrhughes003.github.io/ais-sentinel/. Both were created automatically with `gh`; see
  GITHUB_SETUP.md.
- **Data:** the full six-month download is running in the background
  (`logs/download.log`). It is about 40% done and download-bound at about 1.5 min/day.
  Re-running `ais-sentinel download` retries failed days and skips finished ones.
- **Pipeline stages working:**
  - `download → build → sim-study → track → evaluate → export`;
  - plus `holdout`, explicit only.
  - `evaluate` and `holdout` have only been smoke-tested on synthetic data so far; they
    still need to run on real data.
- **Tracking (simulation study, `reports/tracking.md`):**
  - CV-KF and IMM are implemented and tested.
  - Median errors are about 5–6 m (raw noise radial median about 5.9 m), and outlier
    recall is about 0.9.
  - **PLAN §9.2 criteria not met yet.** See "Blocked" and DECISIONS D15.
- **Prediction:**
  - B0 dead reckoning;
  - B1 CV-KF and B2 IMM extrapolation from stored filter states;
  - B3 kNN route analogs;
  - M1/M2 GRU (Gaussian and MDN), as deep ensembles;
  - calibration on validation;
  - reports with voyage-cluster bootstrap CIs.
- **Anomalies:** only the gap rule exists so far.
- **Site:** a minimal map page (raw vs smoothed, rejected outliers, gaps). Playwright smoke
  tests pass at desktop and mobile widths in CI.

## Next up
1. When the download finishes:
   - run `ais-sentinel download` again (retry failures), then `build`, `track` and
     `evaluate`;
   - inspect the validation results;
   - then `holdout` once.
2. **Phase 6 anomaly suite:** reception grid and port zones; jump, loiter,
   route-deviation and rendezvous detectors; synthetic injection evaluation; case studies.
3. **Phase 7 front end:** landing page, tracks, prediction, anomalies, results.
4. Real-data NIS consistency check (PLAN §9.2, last bullet).

## Blocked / needs my input
- **Tracking criterion (PLAN §9.2), after 3 attempts.**
  - The IMM's RMSE is not 20% below the tuned CV-KF in manoeuvres. RMSE is dominated by
    about 20 rare events per 36 k fixes, so tuning and verdicts are unstable (DECISIONS D15).
  - **Options:**
    - (a) A track-confirmation scheme that retro-corrects the coast after a rejected
      post-gap fix.
    - (b) Tune on many more seeds (about 1 h of CPU).
    - (c) Accept the result as-is. RMSE stays the criterion and is reported as not met,
      with robust metrics alongside.
  - I'll keep going with (c) for now and may try (a) later if time allows. Your call if you
    prefer otherwise.
- FYI: a helper Claude session ("diag-d6") reported clearing temp files outside this project
  (WSL crash dumps, swap vhdx). This session did not do or request that.

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
- Tracking:
  - Built the IMM, the AIS-like simulator and the simulation-study stage.
  - Found and fixed three IMM gating problems (combined gate, any-mode gate, then the
    clutter likelihood ratio).
  - The CV and IMM re-initialise after 2 consistent rejections.
  - The tracking criterion is still not met (see Blocked).
- Prediction:
  - B1/B2 from per-mode snapshots, with frozen-mode IMM forecasts (D12).
  - kNN route analogs (B3).
  - Causal features, and GRU Gaussian/MDN with ensembles (D13).
  - The `evaluate`/`holdout` stages (D14) pass the synthetic end-to-end smoke test.
- Gotcha: PyYAML reads `2e-07` as a string. Write floats with a decimal point.
