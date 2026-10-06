# Anomaly-detector tuning on the validation month (September 2023)

## gap

v1 settings `{"min_reception": 0.8}`: P 1.00 / R 0.84, base rate 1.78 alarms per 1,000 commercial vessel-hours.

| settings | precision | recall | base rate /1000 h | eligible |
|---|---:|---:|---:|---|
| `{"min_reception": 0.5}` | 1.00 | 0.92 | 2.64 | no |
| `{"min_reception": 0.6}` | 1.00 | 0.92 | 2.57 | no |
| `{"min_reception": 0.7}` | 1.00 | 0.90 | 2.00 | no |
| `{"min_reception": 0.8}` | 1.00 | 0.84 | 1.78 | yes |

**Chosen:** `{"min_reception": 0.8}`.

## deviation

v1 settings `{"traffic_groups": "all", "max_cell_voyages": 2, "bridge_min": 0.0}`: P 1.00 / R 0.45, base rate 1.50 alarms per 1,000 commercial vessel-hours.

| settings | precision | recall | base rate /1000 h | eligible |
|---|---:|---:|---:|---|
| `{"traffic_groups": "all", "max_cell_voyages": 2, "bridge_min": 0.0}` | 1.00 | 0.45 | 1.14 | yes |
| `{"traffic_groups": "all", "max_cell_voyages": 2, "bridge_min": 3.0}` | 1.00 | 0.50 | 1.53 | no |
| `{"traffic_groups": "all", "max_cell_voyages": 5, "bridge_min": 0.0}` | 1.00 | 0.53 | 2.57 | no |
| `{"traffic_groups": "all", "max_cell_voyages": 5, "bridge_min": 3.0}` | 1.00 | 0.57 | 3.35 | no |
| `{"traffic_groups": "commercial", "max_cell_voyages": 2, "bridge_min": 0.0}` | 1.00 | 0.66 | 4.56 | no |
| `{"traffic_groups": "commercial", "max_cell_voyages": 2, "bridge_min": 3.0}` | 1.00 | 0.70 | 5.60 | no |
| `{"traffic_groups": "commercial", "max_cell_voyages": 5, "bridge_min": 0.0}` | 1.00 | 0.73 | 6.99 | no |
| `{"traffic_groups": "commercial", "max_cell_voyages": 5, "bridge_min": 3.0}` | 1.00 | 0.78 | 8.13 | no |

**Chosen:** `{"traffic_groups": "all", "max_cell_voyages": 2, "bridge_min": 0.0}`.
