# Tracking: simulation study

**Attempt 3 (fixed mode).** Simulator calibrated to real AIS behaviour, and both filters' noise chosen by one-step predictive likelihood on real training data (reports/tracking_calibration.md). Nothing was tuned in simulation. Evaluation seeds: 2000–2199 (200 voyages), never used before. Simulated ships turn log-uniformly at 0.05–0.61 °/s, report every 70 s, with 0.75 m GPS noise and 0.05% gross outliers. The CV-KF baseline is the better (on tuning seeds) of a CV-KF tuned by real-data likelihood (tuning RMSE 113.4 m; on evaluation it rejects 3515 good fixes) and one tuned for accuracy in this simulator (tuning RMSE 1.1 m): **simulator tuned**, q = 0.1.

## Position error (m): CV-KF vs IMM

| segment | fixes | CV RMSE | IMM RMSE | IMM gain | median CV / IMM | p95 CV / IMM |
|---|---:|---:|---:|---:|---:|---:|
| straight | 24,057 | 1.1 | 63.5 | -5677.9% | 0.9 / 0.9 | 1.8 / 1.8 |
| manoeuvre (turns, speed changes) | 10,114 | 1.4 | 1.4 | -0.4% | 0.9 / 0.9 | 1.8 / 1.8 |
| stopped | 1,347 | 46.5 | 1.0 | +97.9% | 0.9 / 0.8 | 1.8 / 1.7 |
| all moving | 34,171 | 1.2 | 53.3 | -4402.8% | 0.9 / 0.9 | 1.8 / 1.8 |

## Consistency (NEES, 4-D [e, n, ve, vn], expected value 4)

| filter | mean ANEES | median ANEES | 95% band | steps in band |
|---|---:|---:|---|---:|
| CV-KF | 4.36 | 0.07 | [3.62, 4.40] | 0% |
| IMM | 2.60 | 2.58 | [3.62, 4.40] | 1% |

## Outlier rejection (injected gross outliers)

| filter | precision | recall | good fixes rejected |
|---|---:|---:|---:|
| CV-KF | 1.00 | 0.92 | 0 |
| IMM | 0.87 | 1.00 | 2 |

## Success criteria (PLAN §9.2, set before results)

| criterion | target | result | met? |
|---|---|---|---|
| IMM manoeuvre RMSE vs best-tuned CV-KF | >= 20% lower | -0.4% lower | ❌ |
| IMM straight-segment RMSE vs CV-KF | <= 10% worse | +5677.9% | ❌ |
| IMM NEES steps inside 95% band | >= 80% | 1% | ❌ |

## Tuned parameters

- CV-KF: `{"q_accel": 0.1, "r_pos_m": 5.0, "init_vel_std": 5.0, "sogcog_vel_std": 0.5, "gate_prob": 0.999, "max_consecutive_rejects": 2, "gap_q": 0.3, "gap_s": 180.0}`
- IMM: `{"r_pos_m": 0.75, "q_stationary": 0.0001, "tau_stationary_s": 30.0, "q_cruise": 0.0005, "q_turn": 0.05, "q_omega": 2e-08, "sojourn_s": [1800.0, 600.0, 300.0], "init_probs": [0.2, 0.6, 0.2], "init_vel_std": 5.0, "sogcog_vel_std": 0.5, "init_omega_std": 0.017453292519943295, "clutter_density": 5e-11, "gap_q": 0.0, "gap_s": 180.0, "max_consecutive_rejects": 2}`
