# Tracking: simulation study

Tuning seeds: 0–49 (50 voyages). Evaluation seeds: 1000–1199 (200 voyages, disjoint). Each voyage is 4 h with 60 s AIS-like reporting, dropouts, bursts of missing reports, 5 m GPS noise and 0.5% gross outliers.

## Position error (m): CV-KF vs IMM

| segment | fixes | CV RMSE | IMM RMSE | IMM gain | median CV / IMM | p95 CV / IMM |
|---|---:|---:|---:|---:|---:|---:|
| straight | 29,844 | 33.4 | 33.4 | -0.2% | 5.8 / 5.4 | 12.3 / 11.4 |
| manoeuvre (turns, speed changes) | 6,157 | 86.7 | 80.5 | +7.1% | 5.9 / 5.9 | 12.5 / 12.5 |
| stopped | 1,543 | 221.6 | 169.5 | +23.5% | 5.9 / 5.1 | 12.2 / 11.4 |
| all moving | 36,001 | 47.0 | 45.1 | +4.0% | 5.8 / 5.5 | 12.3 / 11.6 |

## Consistency (NEES, 4-D [e, n, ve, vn], expected value 4)

| filter | mean ANEES | median ANEES | 95% band | steps in band |
|---|---:|---:|---|---:|
| CV-KF | 8.83 | 0.58 | [3.62, 4.40] | 2% |
| IMM | 35.69 | 3.13 | [3.62, 4.40] | 8% |

## Outlier rejection (injected gross outliers)

| filter | precision | recall | good fixes rejected |
|---|---:|---:|---:|
| CV-KF | 0.90 | 0.89 | 19 |
| IMM | 0.91 | 0.94 | 17 |

## Success criteria (PLAN §9.2, set before results)

| criterion | target | result | met? |
|---|---|---|---|
| IMM manoeuvre RMSE vs best-tuned CV-KF | >= 20% lower | 7.1% lower | ❌ |
| IMM straight-segment RMSE vs CV-KF | <= 10% worse | +0.2% | ✅ |
| IMM NEES steps inside 95% band | >= 80% | 8% | ❌ |

## Tuned parameters

- CV-KF: `{"q_accel": 0.1, "r_pos_m": 10.0, "init_vel_std": 5.0, "sogcog_vel_std": 0.5, "gate_prob": 0.9999, "max_consecutive_rejects": 2, "gap_q": 0.0, "gap_s": 180.0}`
- IMM: `{"r_pos_m": 5.0, "q_stationary": 0.0001, "tau_stationary_s": 30.0, "q_cruise": 0.0005, "q_turn": 0.05, "q_omega": 2e-07, "sojourn_s": [1800.0, 600.0, 300.0], "init_probs": [0.2, 0.6, 0.2], "init_vel_std": 5.0, "sogcog_vel_std": 0.5, "init_omega_std": 0.017453292519943295, "clutter_density": 5e-11, "gap_q": 0.05, "gap_s": 180.0, "max_consecutive_rejects": 2}`
