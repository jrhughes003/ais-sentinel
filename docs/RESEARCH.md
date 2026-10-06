# Research notes (2026-10-05)

These notes back PLAN.md. Tags: **[V]** checked against a primary source or by direct
measurement; **[A]** abstract or secondary source only; **[U]** unverified or an
engineering estimate.

## 1. MarineCadastre AIS

**Legacy daily zip CSV (2015–2023)**
- URL pattern: `https://coast.noaa.gov/htdata/CMSP/AISDataHandler/YYYY/AIS_YYYY_MM_DD.zip`.
- [V] Measured `AIS_2023_06_15.zip`:
  - 328.8 MB zipped, inflating to 896 MB of CSV;
  - about 8.3 M rows/day (about 107 bytes per row).
- [V] Other days measured were 285–335 MB.
- [V] The 2024 legacy zips are listed but return 404.

**csv2 Zstandard (2015–2026-06)**
- URL pattern: `https://noaaocm.blob.core.windows.net/ais/csv2/csvYYYY/ais-YYYY-MM-DD.csv.zst`.
- The schema changes to snake_case with lon before lat, e.g.
  `mmsi,base_date_time,longitude,latitude,sog,cog,heading,vessel_name,imo,call_sign,vessel_type,status,length,width,draft,cargo,transceiver`.
- Sentinels become nulls.
- [V] The 2024-06-15 file is 296 MB.

**GeoParquet (2024)**
- URL pattern: `https://ocmgeodatastor1.blob.core.windows.net/marinecadastre/ais2024/ais-2024-MM-DD.parquet`.
- [V] About 313 MB per day.
- Per the readme: WKB points, UTC, minute-downsampled.

**Legacy header (≤ 2023)**
`MMSI,BaseDateTime,LAT,LON,SOG,COG,Heading,VesselName,IMO,CallSign,VesselType,Status,Length,Width,Draft,Cargo,TransceiverClass`
- `BaseDateTime` is ISO format without a time-zone suffix, and is UTC per InPort.

**Sampling and coverage:** downsampled to 1 per minute. US EEZ plus the Great Lakes;
Alaska has been removed (InPort 77594).

**Data quality** [V] on a 2023 sample:
- Heading 511 in 52% of rows.
- SOG ≥ 102 in 0.3% of rows.
- 30% of rows are Class B.

**Licence:** CC0 1.0. NOAA "as-is" disclaimer. Citation: Martin, Brass, Dornback &
Fontenault (2025), NOAA OCM.

**Vessel type codes**
- 2020 table: `docs/vessel-type-codes-2020.pdf` in the ocm-marinecadastre repo.
- 2018 table (with the 1001–1025 AVIS codes): `docs/vessel-type-codes-2018.pdf`.
- Groups:

| Codes | Group |
|---|---|
| 30 | fishing |
| 31, 32, 52 | tug/tow |
| 36, 37 | pleasure/sailing |
| 60–69 | passenger |
| 70–79 | cargo |
| 80–89 | tanker |

**robots.txt** for coast.noaa.gov does not disallow `/htdata/`.

**Great Lakes coverage:** USCG NAIS receivers along the Great Lakes were completed in
June 2015 [A]. Distinct MMSIs observed in the first 30 min of 2023-06-15 [V]:

| Area | Distinct MMSIs |
|---|---|
| Whole Great Lakes box | 729 |
| Detroit / St. Clair / W. Lake Erie | 92 |
| Lake Ontario | 91 |
| US-side St. Lawrence | 7 |

**Volume estimates** [U, scaled from the 30-min sample]:

| Area | Share of national rows | Rows/day |
|---|---|---|
| Detroit / St. Clair / W. Erie | 0.69% | about 57 k |
| Juan de Fuca / Puget Sound | 9.3% | about 775 k |

**Download speed** measured from this machine: about 5.3 MB/s.

## 2. Other AIS sources
- **DMA (Denmark):** raw daily CSV in an S3 bucket, 2006 onward, 580–755 MB/day zipped.
  Free under the PSI act; no named licence.
- **Kystverket (Norway):** NLOD 2.0. Download service for ships > 45 m within 12 nm.
- **AISHub:** real-time only; you must contribute a receiver feed.
- **Global Fishing Watch:** CC BY-NC 4.0, API token, derived events only.
- **Canada (DFO):** density rasters only, under OGL-Canada.

## 3. Trajectory prediction: literature numbers

**TrAISformer** (Nguyen & Fablet 2021/2024, arXiv 2109.03958) [V]
- Data: DMA cargo/tanker, Jan–Mar 2019.
- Errors at 1/2/3 h: 0.48 / 0.94 / 1.64 nm. These are **best-of-16 sampled futures** (an
  oracle metric).
- The paper's LSTM seq2seq baseline scores 5.83 / 8.39 / 11.64 nm.

**Leakage-aware re-evaluation** (arXiv 2609.25827, Sep 2026) [A]
- Errors on DMA at 1/2/3 h:

| Model | 1 h (km) | 2 h (km) | 3 h (km) |
|---|---|---|---|
| CV, deterministic | 3.56 | 9.47 | 16.53 |
| TrAISformer, greedy | 2.23 | 5.08 | 8.46 |

- On US Gulf data the pattern is similar.
- Best-of-N inflates the apparent gain by 2–3×.
- Sharing vessels between train and test cuts the 1 h error by 23–25%.

**Capobianco et al. 2021** (IEEE TAES, arXiv 2101.02486) [V]
- Data: DMA tankers, 15-min resampling.
- Linear model: 1.51 nm at 1 h, 7.12 nm at 3 h.
- Attention encoder–decoder: 0.78 nm at 1 h, 3.66 nm at 3 h.

**Capobianco et al. 2022** (uncertainty-aware model, arXiv 2205.05404) [V]
- Errors: 0.57 / 1.14 / 1.90 nm at 1/2/3 h.
- Uncertainty: Gaussian NLL plus MC dropout.
- Reports no coverage or calibration metrics, which is a gap we address.

**Takeaway:** realistic deterministic ranges are:

| Model | 15 min | 30 min | 60 min | 120 min |
|---|---|---|---|---|
| CV | 0.1–0.4 km [U] | 0.3–1.2 km [U] | 1.5–3.6 km | 5–9.5 km |
| ML | — | — | 1.0–2.2 km | 2–5 km |

ML gains come mostly at turns and waypoints.

## 4. Uncertainty
- Methods: Gaussian NLL heads, MDNs (Bishop 1994), deep ensembles (Lakshminarayanan et al.
  2017), proper scoring rules (Gneiting & Raftery 2007), and conformal regions for
  trajectories (Lindemann et al. 2023, arXiv 2210.10254).
- Metrics: NLL, coverage of 50/90% ellipses (χ²₂ radius), and sharpness (area).

## 5. Anomaly detection
- **Welch et al. 2022** (Sci. Adv., going dark) [V]
  - Gaps ≥ 12 h, > 50 nm from shore, with satellite reception > 10 positions/day.
  - Precision 0.86, F0.5 0.739, against exactEarth labels.
- **Miller et al. 2018** (Front. Mar. Sci., transshipment) [V]
  - Encounter: < 500 m, < 2 kn, ≥ 2 h, ≥ 10 km from an anchorage.
  - Loitering: < 2 kn for ≥ 8 h, ≥ 20 nm offshore.
  - No precision or recall reported; only threshold sensitivity sweeps.
- **GeoTrackNet** (Nguyen et al. 2021, IEEE T-ITS) [A]
  - Method: VRNN likelihood plus a-contrario detection.
  - Validation was qualitative, by experts.
- **Kinematic checks:**
  - An implied-speed cap of 40 kn is used in TrAISformer preprocessing.
  - IMM-based spoofing screening appears in arXiv 2603.11055 (2026) [A].
  - Note: the SeaSpoofFinder preprint was **withdrawn**; do not cite it.
- **Evaluation without labels:** synthetic injection plus PR/F1 is the common practice
  (e.g. Mu et al. 2026, Chinese J. Ship Research) [A].

## 6. Tracking
- **NEES/NIS tests:** Bar-Shalom, Li & Kirubarajan (2001). NEES needs ground truth, so
  it is measured in simulation; NIS works on real data.
- **IMM gain:** Yang et al. 2022 (IMM square-root cubature KF on AIS tracks) report RMSE
  30% lower than a single-model EKF on manoeuvring tracks [A]. No canonical "IMM vs CV"
  figure exists, so we set our own target.
- **AIS reporting intervals** (ITU-R M.1371 / NAVCEN), Class A:
  - 10 s at 2–14 kn;
  - 3.33 s when changing course;
  - 6 s at 14–23 kn;
  - 2 s above 23 kn;
  - 3 min at anchor.
- Class B-SO: 30 s. Class B-CS: 30 s.
- Positional accuracy flag 1 means ≤ 10 m.
- **Preprocessing conventions (TrAISformer):**
  - drop SOG ≥ 30 kn and implied speeds > 40 kn;
  - split tracks at 2 h gaps;
  - require ≥ 4 h;
  - resample to 10 min.
- The 2026 protocol paper splits at 30-min gaps.

## 7. Web stack
- **GitHub Pages limits:**
  - 1 GB site;
  - 100 GB/month soft bandwidth;
  - 100 MB file hard limit, 50 MB warning;
  - 10-minute deploy timeout;
  - public repos only on the Free plan.
- **Deploy:** `actions/upload-pages-artifact@v5` and `actions/deploy-pages@v5`, with
  permissions `pages: write` and `id-token: write`. Pages serves HTTP range requests.
- **Basemap:** OpenFreeMap (no key, no limits stated). Attribution: "© OpenFreeMap ©
  OpenMapTiles © OpenStreetMap contributors". CARTO now requires a key.
- **Libraries:**

| Library | Version | Licence | Size |
|---|---|---|---|
| MapLibre GL JS | 6.12 | BSD-3 | about 150 KB gz |
| deck.gl | 9.4 | MIT | 575 KB gz full; modular imports are much smaller |
| Observable Plot | 0.6.17 | ISC | about 70 KB gz |
| onnxruntime-web | 1.30 | MIT | 3.7 MB gz wasm |

- **ONNX in the browser:** GRU is supported only on the WASM execution provider, not on
  WebGPU. There are no threads, because Pages cannot set the COOP/COEP headers.
- **Routing:** hash routing is the most robust choice on Pages.
- **Testing:** Playwright 1.63 with `@axe-core/playwright` 4.13.
