# Measurement-noise calibration on real AIS (training split)

400 seeded training voyages. Share of NIS above χ²₂(0.95) (nominal 5%).
'no repeats' excludes fixes identical to the previous one (mostly moored vessels).

| r_pos_m | CV all | CV no repeats | IMM all | IMM no repeats |
|---:|---:|---:|---:|---:|
| 1.5 | 0.4% | 0.4% | 1.2% | 1.4% |
| 2 | 0.4% | 0.4% | 1.2% | 1.4% |
| 3 | 0.4% | 0.4% | 1.2% | 1.4% |
| 5 | 0.4% | 0.4% | 1.3% | 1.4% |
| 10 | 0.4% | 0.4% | 1.3% | 1.5% |

Chosen (closest to 5% without repeats): CV-KF r = 1.5 m, IMM r = 10 m.
