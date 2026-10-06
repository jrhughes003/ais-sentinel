# Data summary

Days processed: **7**. Clean points: **327,832**. Distinct vessels: **312**. Voyages: **1,184**.

## Cleaning

| step | rows |
|---|---:|
| rows_in | 328,071 |
| dropped_null_keys | 0 |
| dropped_invalid_mmsi | 231 |
| dropped_duplicates | 8 |
| nulled_sog | 4,036 |
| nulled_cog | 48,782 |
| nulled_heading | 143,134 |
| flagged_spikes | 0 |
| rows_out | 327,832 |

## Usable voyages by split

| split | voyages | vessels | points | hours |
|---|---:|---:|---:|---:|
| train | 655 | 283 | 326,106 | 10,807 |

## Usable voyages by vessel group

| group | voyages | vessels | points |
|---|---:|---:|---:|
| tug_tow | 98 | 42 | 94,533 |
| cargo | 108 | 65 | 77,118 |
| pleasure | 250 | 109 | 53,286 |
| passenger | 78 | 22 | 48,978 |
| other | 49 | 20 | 19,989 |
| tanker | 19 | 11 | 17,719 |
| unknown | 45 | 12 | 14,110 |
| fishing | 8 | 2 | 373 |
