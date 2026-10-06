# Data summary

Days processed: **215**. Clean points: **12,567,690**. Distinct vessels: **1,247**. Voyages: **56,471**.

## Cleaning

| step | rows |
|---|---:|
| rows_in | 12,580,383 |
| dropped_null_keys | 0 |
| dropped_invalid_mmsi | 10,053 |
| dropped_malformed | 0 |
| dropped_duplicates | 2,640 |
| nulled_sog | 131,571 |
| nulled_cog | 2,064,730 |
| nulled_heading | 6,022,867 |
| flagged_spikes | 260 |
| rows_out | 12,567,690 |

## Usable voyages by split

| split | voyages | vessels | points | hours |
|---|---:|---:|---:|---:|
| none | 3,562 | 540 | 2,052,258 | 73,351 |
| test | 3,118 | 504 | 1,359,467 | 51,968 |
| train | 17,058 | 952 | 7,170,952 | 251,924 |
| val | 4,227 | 589 | 1,897,839 | 70,105 |

## Usable voyages by vessel group

| group | voyages | vessels | points |
|---|---:|---:|---:|
| pleasure | 15,350 | 656 | 3,235,607 |
| tug_tow | 2,430 | 82 | 2,777,931 |
| cargo | 2,627 | 231 | 2,501,479 |
| passenger | 2,733 | 46 | 1,932,685 |
| other | 2,185 | 85 | 862,734 |
| unknown | 2,102 | 52 | 656,402 |
| tanker | 333 | 33 | 469,955 |
| fishing | 205 | 16 | 43,723 |
