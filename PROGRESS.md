# PROGRESS

## Final summary

**Built:**
- An end-to-end maritime domain awareness system on 11.1 M public AIS reports. It covers
  the Detroit–St. Clair corridor (Canada–US border waterways), May–October 2023.
- The pipeline is reproducible with one command (`ais-sentinel run-all`, then
  `ais-sentinel holdout` once):
  - streaming download and cleaning;
  - Kalman and IMM tracking;
  - six forecasting models with calibrated uncertainty;
  - five explainable anomaly detectors, evaluated by synthetic injection;
  - a static five-view website: https://jrhughes003.github.io/ais-sentinel/
- The website runs the neural forecaster live **in the browser**. It is verified to match
  Python to within 1 cm.
- Engineering: 80 Python tests (95% coverage), 26 browser tests including accessibility
  checks, and green CI.

**How it performs, on the locked October test, evaluated once:**
- **Prediction:**
  - The best ML model (GRU with a mixture head) predicts positions **2.2 km off at
    60 min** and **5.2 km off at 120 min** on average.
  - Straight-line dead reckoning is 7.2 and 18.5 km off at the same horizons.
  - The strongest non-ML baseline, a "where did ships here go next" route model, is 2.4 and
    5.8 km off. The ML model beats it by 7–10% at 60–120 min (statistically significant),
    and by about 17% on vessels it never saw in training.
  - The 90% uncertainty ellipses contain the truth 89–91% of the time.
- **Anomalies:**
  - Impossible jumps, loitering and rendezvous are found 96–99% of the time with no false
    alarms on injected tests.
  - Real case studies found:
    - a cargo ship whose MMSI was also used by another transmitter;
    - a cargo ship going dark mid-lake while every vessel around it stayed visible;
    - a "rendezvous" that was really tug assistance;
    - a fishing boat that tripped the off-lane rule by working its grounds.

**What didn't meet the pre-registered targets** (5 of 11 targets met; misses reported, not
hidden):
- **Tracking**, after two attempts, both reported (D15, D19):
  - Attempt 2 tuned measurement noise and added gap-aware process noise for both filters,
    evaluated on fresh simulation seeds. The IMM now **ties the Kalman filter on straight
    legs** (target met) and is **7% better in manoeuvres**, short of the 20% target.
  - On real data both filters remain underconfident (NIS 0.9% against 2–10%). The cause is
    process noise tuned on simulated manoeuvres harsher than real lakers make. Measurement
    noise turned out not to matter.
- **Prediction:** a 15% gain at 120 min was the bar; the result was 9.5%.
- **Anomalies:**
  - Gap recall was 0.85 against a 0.90 target. The reception mask deliberately ignores
    silences in weak-coverage cells, trading recall for precision 1.00.
  - Route-deviation recall was 0.47 against 0.70. Four months of traffic, pleasure craft
    included, leave few empty cells in the wide lakes, so an absolute "≤ 2 voyages" rule
    stops working.

**Suggested next steps:**
1. **Tracking:** fit the process noise, or the simulator's manoeuvre statistics, to real data,
   for example by maximising the one-step predictive likelihood on the training split. Then
   re-run the simulation study on fresh seeds.
2. **Deviation detector:** commercial-only lane density, a relative threshold, and
   tolerance for short interruptions. Evaluate on a *new* test month, since October has
   now been seen.
3. **Prediction:**
   - A larger training budget; validation loss was still falling at 20 epochs.
   - Destination-aware or graph-of-channels models.
   - A second region (Danish DMA data) to test transfer.
4. **Site:** an animated GIF for the README; per-vessel-type filters in the Anomalies view.

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

## Current state (2026-10-06 ~07:00)
- **Complete.** The full-data run, the locked test (run once, logged) and the export are
  done. The site is live with final data, and the README results are filled in.
- Post-result analyses, on validation or as diagnosis only, with no reported numbers
  changed:
  - `reports/knn_sensitivity.md`: k = 80 was effectively optimal, so the baseline was not
    under-tuned;
  - the anomaly miss diagnosis, in DECISIONS D18.

## Next up
- See "Suggested next steps" above. Any change to the deviation detector or the models needs
  a fresh test period (October has been used).

## Blocked / needs my input
- (none). Tracking attempt 2 was done as agreed (D19). The next step is listed in the final
  summary if you want to pursue it.
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
- Full-data run:
  - track: 11.0 M fixes in 34 min;
  - evaluate: 402 k training samples, 5 GRU ensemble members, about 3.5 h;
  - anomaly;
  - holdout: once, logged.
- Case studies were written from verified raw data (`configs/case_studies.yaml`).
- README results and screenshots added. Final summary written.
- A partial source day (2023-10-29) was found with help from the helper session. Gaps
  touching it are excluded (D17).
- Tracking attempt 2 (agreed with the owner):
  - tuned measurement noise plus gap-aware process noise, on fresh seeds 1000–1199;
  - straight-leg criterion now met, manoeuvre criterion 7.1% (target 20%);
  - real-data noise calibration showed the underconfidence comes from process noise.
