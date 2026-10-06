# kNN sensitivity to k (validation split only)

Scales fixed at the validation choice (1200 m, 20°). Mean error in km; the chosen k was 80 (upper edge of the tuning grid).

| k | 15 min | 30 min | 60 min | 120 min | DR fallback |
|---:|---:|---:|---:|---:|---:|
| 40 | 0.530 | 1.180 | 2.600 | 5.963 | 3.5% |
| 80 | 0.539 | 1.196 | 2.633 | 5.876 | 2.3% |
| 160 | 0.556 | 1.231 | 2.693 | 5.882 | 1.5% |
| 320 | 0.581 | 1.283 | 2.782 | 5.950 | 1.1% |
