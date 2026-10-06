# Data summary

Days processed: **4**. Clean points: **188,246**. Distinct vessels: **258**. Voyages: **714**.

## Cleaning

| step | rows |
|---|---:|
| rows_in | 188,480 |
| dropped_null_keys | 0 |
| dropped_invalid_mmsi | 231 |
| dropped_duplicates | 3 |
| nulled_sog | 1,713 |
| nulled_cog | 27,794 |
| nulled_heading | 82,617 |
| flagged_spikes | 0 |
| rows_out | 188,246 |

## Usable voyages by split

| split | voyages | vessels | points | hours |
|---|---:|---:|---:|---:|
| train | 385 | 229 | 187,223 | 6,332 |

## Usable voyages by vessel group

| group | voyages | vessels | points |
|---|---:|---:|---:|
| tug_tow | 53 | 37 | 52,601 |
| cargo | 65 | 51 | 44,338 |
| pleasure | 144 | 81 | 31,759 |
| passenger | 46 | 20 | 28,063 |
| other | 33 | 19 | 12,599 |
| tanker | 13 | 10 | 10,458 |
| unknown | 29 | 10 | 7,338 |
| fishing | 2 | 1 | 67 |
