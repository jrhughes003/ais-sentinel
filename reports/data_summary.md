# Data summary

Days processed: **184**. Clean points: **11,088,073**. Distinct vessels: **1,109**. Voyages: **50,235**.

## Cleaning

| step | rows |
|---|---:|
| rows_in | 11,100,530 |
| dropped_null_keys | 0 |
| dropped_invalid_mmsi | 10,004 |
| dropped_malformed | 0 |
| dropped_duplicates | 2,453 |
| nulled_sog | 131,571 |
| nulled_cog | 2,064,730 |
| nulled_heading | 6,022,867 |
| flagged_spikes | 259 |
| rows_out | 11,088,073 |

## Usable voyages by split

| split | voyages | vessels | points | hours |
|---|---:|---:|---:|---:|
| none | 679 | 297 | 516,182 | 19,851 |
| test | 2,700 | 431 | 1,424,114 | 48,824 |
| train | 17,058 | 952 | 7,170,952 | 251,924 |
| val | 4,227 | 589 | 1,897,839 | 70,105 |

## Usable voyages by vessel group

| group | voyages | vessels | points |
|---|---:|---:|---:|
| pleasure | 13,792 | 583 | 2,883,812 |
| tug_tow | 1,982 | 77 | 2,396,411 |
| cargo | 2,216 | 205 | 2,176,063 |
| passenger | 2,316 | 40 | 1,742,068 |
| other | 1,918 | 67 | 787,545 |
| unknown | 2,049 | 48 | 619,384 |
| tanker | 261 | 25 | 384,758 |
| fishing | 130 | 5 | 19,046 |
