# Tracking: simulation study

Tuning seeds: 0–49 (50 voyages). Evaluation seeds: 100–299 (200 voyages, disjoint). Each voyage is 4 h with 60 s AIS-like reporting, dropouts, bursts of missing reports, 5 m GPS noise and 0.5% gross outliers.

## Position error (m): CV-KF vs IMM

| segment | fixes | CV RMSE | IMM RMSE | IMM gain | median CV / IMM | p95 CV / IMM |
|---|---:|---:|---:|---:|---:|---:|
| straight | 29,696 | 16.3 | 43.7 | -168.0% | 5.9 / 5.4 | 12.4 / 11.5 |
| manoeuvre (turns, speed changes) | 6,009 | 26.5 | 26.6 | -0.6% | 5.9 / 6.5 | 12.4 / 14.3 |
| stopped | 1,625 | 116.3 | 199.7 | -71.7% | 6.0 / 3.1 | 12.4 / 8.4 |
| all moving | 35,705 | 18.4 | 41.3 | -124.3% | 5.9 / 5.6 | 12.4 / 12.0 |

## Consistency (NEES, 4-D [e, n, ve, vn], expected value 4)

| filter | mean ANEES | median ANEES | 95% band | steps in band |
|---|---:|---:|---|---:|
| CV-KF | 1.74 | 0.58 | [3.62, 4.40] | 0% |
| IMM | 2.67 | 1.22 | [3.62, 4.40] | 1% |

## Outlier rejection (injected gross outliers)

| filter | precision | recall | good fixes rejected |
|---|---:|---:|---:|
| CV-KF | 0.95 | 0.93 | 9 |
| IMM | 0.93 | 0.91 | 14 |

## Success criteria (PLAN §9.2, set before results)

| criterion | target | result | met? |
|---|---|---|---|
| IMM manoeuvre RMSE vs best-tuned CV-KF | >= 20% lower | -0.6% lower | ❌ |
| IMM straight-segment RMSE vs CV-KF | <= 10% worse | +168.0% | ❌ |
| IMM NEES steps inside 95% band | >= 80% | 1% | ❌ |

## Tuned parameters

- CV-KF: `{"q_accel": 0.1, "r_pos_m": 10.0, "init_vel_std": 5.0, "sogcog_vel_std": 0.5, "gate_prob": 0.9999, "max_consecutive_rejects": 2}`
- IMM: `{"r_pos_m": 10.0, "q_stationary": 0.0001, "tau_stationary_s": 30.0, "q_cruise": 0.002, "q_turn": 0.2, "q_omega": 2e-07, "sojourn_s": [1800.0, 600.0, 300.0], "init_probs": [0.2, 0.6, 0.2], "init_vel_std": 5.0, "sogcog_vel_std": 0.5, "init_omega_std": 0.017453292519943295, "clutter_density": 5e-11, "max_consecutive_rejects": 2}`
