# Tracker calibration on real AIS (training split)

Real manoeuvre statistics from 1,806,374 one-minute steps of moving commercial training vessels, mapped to the simulator; filter noise chosen by one-step predictive log-likelihood on 150 seeded training voyages (55,654 fixes).

## Real behaviour vs simulator

| statistic | real | calibrated simulator |
|---|---:|---:|
| speed 5–95% (kn) | 4.8–19.8 | 4.8–19.8 |
| share of time turning (> 0.05 °/s) | 20.4% | 18.8% |
| turn rate upper bound (99th pct, °/s) | 0.61 | 0.61 (log-uniform) |
| acceleration 90–99% (m/s²) | 0.0034–0.0689 | 0.0034–0.0689 (log-uniform) |
| median report interval (s) | 70 | 70 |
| share of gaps > 3 min | 0.53% | p_burst 0.0053 |
| outlier share | 0.04% | 0.05% |
| position noise σ (m) | – | 0.75 (IMM's likelihood-chosen R) |

## Chosen filter noise (max predictive log-likelihood)

- CV-KF: `{"q_accel": 0.0001, "r_pos_m": 5.0, "gap_q": 0.0}`
- IMM: `{"q_cruise": 0.0005, "q_turn": 0.05, "q_omega": 2e-08, "r_pos_m": 0.75, "gap_q": 0.0}`
