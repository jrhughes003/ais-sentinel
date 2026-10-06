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

## Current state (2026-10-06 ~01:00)
- **Repo:** https://github.com/jrhughes003/ais-sentinel. **Live site:**
  https://jrhughes003.github.io/ais-sentinel/
  - All five views are live: overview, tracks, predictions, anomalies, results.
  - The site currently shows a clearly bannered **development preview**, built from
    `configs/dev.yaml` (train May 1–24, validate May 26–31, test June 2–10).
- **Download:** about 100 of 184 days done; `logs/download.log`. Afterwards, re-run
  `ais-sentinel download` once to retry failures.
- **Dev run, real data:** the whole pipeline worked end to end.

  | Stage | Dev-run result |
  |---|---|
  | Tracking | 3.55 M fixes in about 14 min |
  | Prediction (dev test) | GRU-MDN vs kNN: +17% at 15 min, tie at 60 min, −17% at 120 min. The pre-registered target (significant win at 60 *and* 120) would be **not met**. Coverage of 0.87–0.89 is within target. On vessels never seen in training the gain persists |
  | Anomalies (dev test) | All five types meet their targets after two bug fixes; the base alarm rate is reported |

- **Not yet done:**
  - the full-data run;
  - the locked October test;
  - case studies on final data (`configs/case_studies.yaml`);
  - README results table and screenshot;
  - final summary.

## Next up
1. When the download finishes, run on the full data:

   ```
   ais-sentinel download
   ais-sentinel build
   ais-sentinel sim-study
   ais-sentinel track
   ais-sentinel evaluate
   ais-sentinel anomaly
   ais-sentinel holdout    # once
   ais-sentinel export
   ```

   Training budget is about 1.5 h (D16).
2. Write 3+ case studies from real events. Candidates seen in the dev data:
   - cargo MMSI 246824000 "jumping" 87–125 km within about a minute (shared MMSI or GPS);
   - a tanker holding position for 127 h outside learned port zones;
   - tug pairs "meeting" for 12–18 h (likely a tug and its barge).
3. README: results table and screenshot. Final PROGRESS summary.
4. Optional: ONNX in-browser model; tracking option (a) from Blocked.

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

### 2026-10-06
- Anomaly suite:
  - context grids (reception, ports, traffic);
  - the jump, loiter, deviation and rendezvous detectors;
  - synthetic injection evaluation.
- Bugs fixed:
  - injection eligibility (`min(initial=0)`);
  - parked vessels flagged as dark;
  - gap ranking.
- Front end: five views, lazy-loaded maps and charts, a validated palette, axe-clean.
  Playwright passes 24/24 at desktop and mobile widths.
- Dev pipeline on real May–June data: the experiment, export and site all work end to end.
- kNN grid widened (the dev run chose the old grid's edges). ML compute budget set (D16).
- Note: anomaly rule refinements were made on dev data (June, which falls inside the real
  training period). The October locked test has not been touched.
