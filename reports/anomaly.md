# Anomaly detection: synthetic-injection evaluation

Real October 2023 (locked test period) commercial voyages with one injected anomaly each. Precision counts only detections that were *not* present before injection; see `ais_sentinel/anomaly/run.py` for the matching rule.

## By type and magnitude

| type | magnitude | n | precision | recall |
|---|---:|---:|---:|---:|
| gap | 20 | 51 | nan | 0.00 |
| gap | 30 | 57 | 1.00 | 0.86 |
| gap | 45 | 55 | 1.00 | 0.85 |
| gap | 60 | 48 | 1.00 | 0.83 |
| gap | 90 | 55 | 1.00 | 0.87 |
| gap | 180 | 34 | 1.00 | 0.82 |
| jump | 0.5 | 53 | nan | 0.00 |
| jump | 1 | 44 | 1.00 | 0.09 |
| jump | 2 | 47 | 1.00 | 0.96 |
| jump | 3 | 39 | 1.00 | 1.00 |
| jump | 5 | 39 | 1.00 | 0.97 |
| jump | 10 | 38 | 1.00 | 1.00 |
| jump | 30 | 40 | 1.00 | 1.00 |
| loiter | 1 | 44 | nan | 0.00 |
| loiter | 1.5 | 53 | nan | 0.00 |
| loiter | 2 | 44 | nan | 0.00 |
| loiter | 3 | 53 | 1.00 | 0.96 |
| loiter | 4 | 46 | 1.00 | 0.96 |
| loiter | 6 | 60 | 1.00 | 0.95 |
| deviation | 0.5 | 47 | 1.00 | 0.02 |
| deviation | 1 | 62 | 1.00 | 0.21 |
| deviation | 1.5 | 38 | 1.00 | 0.32 |
| deviation | 2 | 49 | 1.00 | 0.45 |
| deviation | 3 | 55 | 1.00 | 0.47 |
| deviation | 5 | 49 | 1.00 | 0.51 |
| rendezvous | 0.5 | 66 | nan | 0.00 |
| rendezvous | 1 | 55 | 1.00 | 0.96 |
| rendezvous | 1.5 | 70 | 1.00 | 0.96 |
| rendezvous | 2 | 54 | 1.00 | 0.98 |
| rendezvous | 3 | 55 | 1.00 | 0.95 |

Magnitude units: gap = minutes of deleted reports, jump = km displacement, loiter and rendezvous = hours, deviation = km peak sideways offset.

## Success criteria (PLAN §9.4, set before results)

| type | bucket | n | precision (target) | recall (target) | met? |
|---|---|---:|---|---|---|
| gap | ≥ 45 min | 192 | 1.00 (≥ 0.90) | 0.85 (≥ 0.90) | ❌ |
| jump | ≥ 3 km | 156 | 1.00 (≥ 0.90) | 0.99 (≥ 0.90) | ✅ |
| loiter | ≥ 3 h | 159 | 1.00 (≥ 0.80) | 0.96 (≥ 0.80) | ✅ |
| deviation | ≥ 1.5 km for ≥ 20 min | 169 | 1.00 (≥ 0.70) | 0.47 (≥ 0.70) | ❌ |
| rendezvous | ≥ 1.5 h | 179 | 1.00 (≥ 0.80) | 0.96 (≥ 0.80) | ✅ |

## Base alarm rate on unmodified test data (alarms per 1,000 vessel-hours)

Some of these may be real anomalies; they are reported, not treated as errors.

| type | vessel group | alarms | vessel-hours | per 1,000 h |
|---|---|---:|---:|---:|
| deviation | cargo | 18 | 8,847 | 2.03 |
| deviation | fishing | 7 | 81 | 86.16 |
| deviation | other | 27 | 3,666 | 7.36 |
| deviation | passenger | 8 | 5,005 | 1.60 |
| deviation | pleasure | 24 | 20,952 | 1.15 |
| deviation | tug_tow | 12 | 9,884 | 1.21 |
| deviation | unknown | 2 | 2,136 | 0.94 |
| gap | cargo | 18 | 8,847 | 2.03 |
| gap | passenger | 3 | 5,005 | 0.60 |
| gap | pleasure | 6 | 20,952 | 0.29 |
| gap | tanker | 1 | 1,627 | 0.61 |
| gap | tug_tow | 9 | 9,884 | 0.91 |
| jump | cargo | 1 | 8,847 | 0.11 |
| jump | pleasure | 4 | 20,952 | 0.19 |
| jump | unknown | 5 | 2,136 | 2.34 |
| loiter | cargo | 22 | 8,847 | 2.49 |
| loiter | other | 3 | 3,666 | 0.82 |
| loiter | pleasure | 11 | 20,952 | 0.53 |
| loiter | tanker | 4 | 1,627 | 2.46 |
| loiter | tug_tow | 22 | 9,884 | 2.23 |

Total events detected across the season: 1,929.

Data completeness: days with a partial national source file (< 60% of the median row count) are treated as unavailable, and no gap touching them is flagged: 2023-10-29.
