# Results in detail

The README gives the headline numbers. This page has the full tables and the tracking
story. Everything here is copied from the pipeline's own reports, which are the source of
truth:
- prediction: [reports/prediction_test.md](../reports/prediction_test.md)
- tracking: [reports/tracking.md](../reports/tracking.md) and
  [reports/tracking_calibration.md](../reports/tracking_calibration.md)
- anomalies: [reports/anomaly.md](../reports/anomaly.md)
- data: [reports/data_summary.md](../reports/data_summary.md)
- the log of locked-test runs: [reports/holdout_runs.md](../reports/holdout_runs.md)

## Evaluation design

- **Time-based splits:** train May–August 2023, validate September, **locked test October
  2023**. One-day buffers separate the splits, and each voyage sits entirely inside one
  split.
- **The locked test is a separate command** (`ais-sentinel holdout`). It is not part of
  `run-all`, and every run of it is appended to
  [reports/holdout_runs.md](../reports/holdout_runs.md). It was run **once**, after every
  model, setting and calibration had been fixed on May–September data.
- **Targets were written down before any results** ([PLAN.md](../PLAN.md) §9). Missed
  targets are reported as missed.
- **Confidence intervals** come from a voyage-cluster bootstrap. Forecasts from the same
  voyage are correlated, so the voyage, not the individual forecast, is resampled.
- **Vessel-unseen subset:** results are also reported for ships whose MMSI never appears
  in training.

## Data

- 11,100,530 AIS reports inside the area of interest, cleaned to 11,088,073.
- 184 days, 1,109 distinct vessels, 50,235 voyages; 24,664 voyages remain after length
  filters.
- 44,431 underway forecast samples in the test month (fewer at long horizons, because the
  voyage has to last long enough to be scored).

## Trajectory prediction (locked October 2023 test)

Mean error in km, with 95% voyage-cluster bootstrap intervals for the chosen model.

| Model | 15 min | 30 min | 60 min | 120 min |
|---|---:|---:|---:|---:|
| B0 Dead reckoning | 0.94 | 2.79 | 7.24 | 18.52 |
| B1 Kalman (CV) extrapolation | 0.99 | 2.86 | 7.31 | 18.62 |
| B2 IMM extrapolation | 1.14 | 3.13 | 7.71 | 17.68 |
| B3 Route analogs (kNN) | **0.42** | **0.97** | 2.37 | 5.76 |
| M1 GRU, Gaussian (3-member ensemble) | 0.44 | 1.01 | 2.29 | 5.50 |
| **M2 GRU, mixture (2-member ensemble)** | **0.42** | **0.97** | **2.20** [2.08, 2.33] | **5.21** [4.94, 5.47] |

M2 was chosen on validation. Against the strongest baseline (kNN route analogs), with a
paired voyage-cluster bootstrap:

| Horizon | M2 minus kNN (km) [95% CI] | Relative | Reading |
|---|---|---:|---|
| 15 min | +0.01 [−0.01, +0.02] | +1.6% | tie |
| 30 min | +0.00 [−0.03, +0.03] | +0.2% | tie |
| 60 min | −0.17 [−0.24, −0.10] | −7.1% | M2 better |
| 120 min | −0.55 [−0.78, −0.33] | −9.5% | M2 better |

- **Vessels never seen in training:** M2 4.11 km against kNN 4.95 km at 120 min, about 17%
  lower.
- **Uncertainty:** M2's calibrated 90% ellipses contain the true position 91%, 91%, 90% and
  89% of the time at the four horizons.
- **Per vessel type** (no confidence intervals) is in the full report.

## Tracking (simulation study and real data)

The pre-registered tracking targets ask the 3-mode IMM to beat a well-tuned constant-velocity
Kalman filter (CV-KF) in a simulator with known ground truth. There were four attempts,
all reported ([DECISIONS.md](../DECISIONS.md) D15, D19, D20, D21). The latest attempt is
reported, not the best one.

**Attempt 4** (final). The simulator and both filters were fitted to real ship behaviour.
It was evaluated on fresh seeds 3000–3199:

| Segment | CV-KF RMSE | IMM RMSE | Median CV / IMM |
|---|---:|---:|---:|
| Straight legs | 7.3 m | 9.7 m | 1.8 / 1.7 m |
| Manoeuvres | 4.5 m | 9.0 m | 1.8 / 1.8 m |
| Stopped | 2.1 m | 1.7 m | 1.7 / 1.4 m |

- Medians and 95th percentiles are essentially equal. The RMSE differences come from a
  handful of tail fixes.
- NEES (is the filter's stated uncertainty honest?): IMM inside the 95% band on 17% of
  steps, CV-KF on 0%. The target was 80%.

**How the attempts went:**

| Attempt | What changed | Main result |
|---|---|---|
| 1 | Original design | IMM no better in manoeuvres (−0.6%), much worse on straights |
| 2 | Tuned measurement noise, gap-aware process noise | IMM 7.1% better in manoeuvres (target 20%); straights criterion met |
| 3 | Simulator and both filters fitted to real AIS data | Manoeuvres tied (1.4 m each). Straight-leg RMSE 63.5 m against 1.1 m, caused by **two** bad fixes out of 24,057 after long gaps |
| 4 | Restart the filter when the first fix after a long gap fails the gate | Straight-leg IMM RMSE 63.5 → 9.7 m; CV-KF still slightly ahead |

**On real data** (validation split, attempt-3 settings):
- The IMM predicts each next real fix far better than the CV-KF: mean log-likelihood −7.38
  against −10.40 per fix.
- NIS consistency: 1.6% of IMM values exceed the 95% threshold, up from 0.9% before
  fitting the noise to real data. The target was 2–10%.
- The attempt-4 real-data NIS has not been computed yet. It is part of the pending v2
  run.

**Conclusion:** in a simulator that matches real ships, manoeuvres are gentle enough that a
well-tuned constant-velocity filter is as accurate as the IMM. The IMM's advantage shows up
in real-data predictive likelihood instead.

**Scope note:** the website's tracks and the B1/B2 prediction baselines in the locked test
were produced with the attempt-1 tracker. The locked test was not re-run after later
attempts.

## Anomaly detection (synthetic injection on October 2023)

Each anomaly type was injected into real commercial voyages, at a range of magnitudes.
Precision counts only detections that were not already present before injection.

| Type | Target bucket | n | Precision | Recall | Target |
|---|---|---:|---:|---:|---|
| AIS gap (going dark) | ≥ 45 min | 192 | 1.00 | 0.85 | 0.90 / 0.90 |
| Impossible jump | ≥ 3 km | 156 | 1.00 | 0.99 | 0.90 / 0.90 |
| Loitering | ≥ 3 h | 159 | 1.00 | 0.96 | 0.80 / 0.80 |
| Route deviation | ≥ 1.5 km for ≥ 20 min | 169 | 1.00 | 0.47 | 0.70 / 0.70 |
| Rendezvous | ≥ 1.5 h | 179 | 1.00 | 0.96 | 0.80 / 0.80 |

Why the two misses happened (DECISIONS D18):
- **Gap:** most missed silences end in cells with weak receiver coverage. The detector
  ignores those on purpose, trading recall for precision.
- **Route deviation:** four months of traffic, pleasure craft included, leave few empty
  cells in the wide lakes, so an absolute "≤ 2 voyages per cell" rule stops firing.

The fixes were tuned on September only and have **not** been re-scored on October, because
that would be tuning on the test set. They wait for the v2 test month.

Real-world case studies (see the Anomalies view on the site):
- a cargo ship whose MMSI was also used by another transmitter;
- a cargo ship that went dark mid-lake while every vessel around it stayed visible;
- a "rendezvous" that was really tug assistance;
- a fishing boat that tripped the off-lane rule by working its grounds.

## v2: a fresh test month (pending)

October 2023 has been used, so any changed model or detector needs a new test period. v2
uses **October 2024** (MarineCadastre's newer Zstandard `csv2` format, also supported by
the downloader). Train and validation months are unchanged. The design is in DECISIONS
D21, and the run is `scripts/run_v2.ps1`. v1 results stay published either way.
