# Tracking: simulation study

**Fixed mode (attempt 4: post-gap restart, repeats skipped; DECISIONS D21).** Simulator calibrated to real AIS behaviour, and both filters' noise chosen by one-step predictive likelihood on real training data (reports/tracking_calibration.md). Nothing was tuned in simulation. Evaluation seeds: 3000–3199 (200 voyages), never used before. Simulated ships turn log-uniformly at 0.05–0.61 °/s, report every 70 s, with 1.5 m GPS noise and 0.05% gross outliers. The CV-KF baseline is the better (on tuning seeds) of a CV-KF tuned by real-data likelihood (tuning RMSE 37.4 m; on evaluation it rejects 4051 good fixes) and one tuned for accuracy in this simulator (tuning RMSE 2.1 m): **simulator tuned**, q = 0.03.

## Position error (m): CV-KF vs IMM

| segment | fixes | CV RMSE | IMM RMSE | IMM gain | median CV / IMM | p95 CV / IMM |
|---|---:|---:|---:|---:|---:|---:|
| straight | 23,267 | 7.3 | 9.7 | -33.3% | 1.8 / 1.7 | 3.7 / 3.5 |
| manoeuvre (turns, speed changes) | 10,428 | 4.5 | 9.0 | -102.4% | 1.8 / 1.8 | 3.7 / 3.7 |
| stopped | 1,528 | 2.1 | 1.7 | +20.0% | 1.7 / 1.4 | 3.7 / 2.9 |
| all moving | 33,695 | 6.5 | 9.5 | -45.3% | 1.8 / 1.7 | 3.7 / 3.6 |

## Consistency (NEES, 4-D [e, n, ve, vn], expected value 4)

| filter | mean ANEES | median ANEES | 95% band | steps in band |
|---|---:|---:|---|---:|
| CV-KF | 0.27 | 0.27 | [3.62, 4.40] | 0% |
| IMM | 3.51 | 3.10 | [3.62, 4.40] | 17% |

## Outlier rejection (injected gross outliers)

| filter | precision | recall | good fixes rejected |
|---|---:|---:|---:|
| CV-KF | 0.82 | 1.00 | 2 |
| IMM | 0.35 | 1.00 | 17 |

## Success criteria (PLAN §9.2, set before results)

| criterion | target | result | met? |
|---|---|---|---|
| IMM manoeuvre RMSE vs best-tuned CV-KF | >= 20% lower | -102.4% lower | ❌ |
| IMM straight-segment RMSE vs CV-KF | <= 10% worse | +33.3% | ❌ |
| IMM NEES steps inside 95% band | >= 80% | 17% | ❌ |

## Tuned parameters

- CV-KF: `{"q_accel": 0.03, "r_pos_m": 5.0, "init_vel_std": 5.0, "sogcog_vel_std": 0.5, "gate_prob": 0.99999, "max_consecutive_rejects": 2, "gap_q": 0.05, "gap_s": 180.0, "gap_reinit_s": 300.0, "skip_repeats": true}`
- IMM: `{"r_pos_m": 1.5, "q_stationary": 0.0001, "tau_stationary_s": 30.0, "q_cruise": 0.0001, "q_turn": 0.02, "q_omega": 2e-08, "sojourn_s": [1800.0, 600.0, 300.0], "init_probs": [0.2, 0.6, 0.2], "init_vel_std": 5.0, "sogcog_vel_std": 0.5, "init_omega_std": 0.017453292519943295, "clutter_density": 5e-11, "gap_q": 0.0, "gap_s": 180.0, "gap_reinit_s": 300.0, "skip_repeats": true, "max_consecutive_rejects": 2}`
