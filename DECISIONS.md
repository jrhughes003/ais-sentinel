# Decisions

Short records of significant technical decisions. Newest at the bottom.

### D1 — Project name: `ais-sentinel` (2026-10-05)
The name says what the data is (AIS) and what the system does: it watches for and flags
behaviour. "vessel-track" covers only one of the three pillars.

### D2 — Region and window: Detroit–St. Clair corridor, May–Oct 2023 (2026-10-05)
- Estimated about 57 k rows/day (about 10 M rows over six months): laptop-sized.
- It is a Canada–US border waterway. Its curving channels make prediction non-trivial.
- 2023 legacy zips are verified downloadable; the 2024 legacy URLs return 404.
- Rejected Puget Sound (14× larger) and the St. Lawrence (thin US-receiver coverage).
- Details are in PLAN.md §3.2.

### D3 — Data access: stream-filter national daily zips, delete after use (2026-10-05)
- The AOI makes up about 0.7% of each 330 MB national file.
- AccessAIS custom extracts would be smaller, but the process is manual and not
  reproducible.
- Range-reading GeoParquet only exists for 2024.
- Process: download sequentially, filter with polars, keep only the AOI as Parquet, then
  delete the zip. Peak disk use is about 1.3 GB.

### D4 — Split design: time split with 1-day buffers + vessel-unseen subset (2026-10-05)
- Train is May–Aug, validation is Sep 2–30 and the locked test is Oct 2–31.
- A voyage belongs to a split only if its whole span lies inside it.
- A recent leakage study (arXiv 2609.25827) shows that shared vessels flatter results, so
  we also report a vessel-unseen test subset rather than forcing vessel-disjoint splits.
  Forcing them would discard most of the regular lakers.

### D5 — Front-end stack: Vite + vanilla TypeScript + MapLibre GL + Observable Plot, hash routing (2026-10-05)
- **Vite:** fast, builds cleanly in Actions, and `base` handles the Pages sub-path.
- **Vanilla TS** with a tiny router: no framework runtime and the fewest concepts for a
  Python-first maintainer. Five views do not justify React.
- **MapLibre (BSD-3):** vector basemaps plus GeoJSON layers are enough for about 100
  curated tracks. deck.gl stays in reserve if animated trails need it.
- **OpenFreeMap tiles:** no API key. CARTO now requires one.
- **Hash routing:** Pages has no rewrites, and deep links must not return 404.

### D6 — Coordinates: local ENU via ECEF, no projection library (2026-10-05)
- The exact geodetic → ECEF → ENU conversion is about 20 lines of numpy and is testable.
- Anchoring a frame per track (tracking) or per sample (prediction) keeps distortion
  negligible at ≤ 150 km.
- This avoids a pyproj dependency.

### D7 — Rules-first anomaly detectors (2026-10-05)
- Analysts need explanations, and there are no labels.
- Transparent thresholds grounded in GFW definitions, plus learned context (reception
  grid, port zones, traffic grid), are explainable and evaluable by injection.
- A learned likelihood model (GeoTrackNet-style) is out of scope.

### D8 — Cleaning flags kinematic spikes instead of deleting them (2026-10-05)
Isolated implausible fixes are kept with `spike = true`. The tracker gates them out and the
anomaly detector reports them. Deleting them during cleaning would hide exactly the
"impossible jump" behaviour pillar 3 is meant to find.

### D9 — Prediction evaluation uses underway anchors only (SOG ≥ 2 kn) (2026-10-05)
Moored and anchored vessels are trivially predictable, and they would dominate sample
counts and make every model look good. Following common practice (TrAISformer removes
moored vessels), primary metrics use anchors where the vessel is underway. Truth is
interpolated between fixes no more than 3 min from the target time.

### D10 — Gap events require data availability on every day they span (2026-10-05)
A silence that spans a day missing from our download is a hole in our data, not a vessel
going dark. Events are kept only if every UTC day they touch is in the manifest.

### D11 — IMM outlier test: mixture likelihood vs a uniform clutter density (2026-10-05)
Two simpler gates failed in the simulation study:

- Gating on the moment-matched combined prediction rejected the first fixes of every turn,
  because the cruising mode dominated: 535 good fixes were rejected.
- An "inside any mode's χ² gate" rule let a wide, improbable turning mode admit km-scale
  outliers on stopped vessels.

The adopted test accepts a fix when Σⱼ cⱼ N(ν; Sⱼ) ≥ λ, i.e. when "from the target" is more
probable than "outlier", as in probabilistic data association.

### D12 — IMM forecasts freeze mode probabilities by default (2026-10-05)
The filter's Markov prior (a cruising sojourn of about 10–15 min) is right for tracking, but
over a 2 h forecast with no measurements it moves almost all probability into
stationary/turning modes. On a straight line, the averaged mean then lands about 50% short.
The switching variant is kept and compared on validation (`imm_switching`); validation
picks the variant used on test.

### D13 — ML targets are corrections to dead reckoning (2026-10-05)
The GRU predicts (truth − dead reckoning) in km, scaled per horizon. A zero output equals
the physics baseline, so the network only has to learn where vessels deviate from straight
lines (channel bends, approaches). That is data-efficient on a laptop-sized dataset, and it
makes "ML vs DR" a test of whether the learned corrections help.

### D14 — The locked test is a separate, explicit CLI stage (2026-10-05)
`ais-sentinel holdout` is excluded from `run-all` and appends a dated row (commit plus config
hash) to `reports/holdout_runs.md`. Reproducing the pipeline can never silently re-run the
locked test.

### D15 — Tracking criterion: RMSE is dominated by a handful of events (2026-10-05, open)
In the simulation study, about 95–98% of the squared position error comes from roughly 20 of
about 36 k fixes. These are coasting after a rejected first-fix-after-a-gap, and outliers
admitted by a wide turning mode. Both the tuning objective and the IMM-vs-CV verdict are
therefore decided by luck on rare events, and the result swings between runs.

- The PLAN §9.2 criteria are **kept unchanged**. The current result is reported as not met.
- Median and p95 errors are reported alongside RMSE, and there the IMM is equal or better.
- See PROGRESS "Blocked" for the options.

### D16 — ML compute budget: smaller ensembles, sparser training anchors (2026-10-06)
On the dev run, one GRU epoch took about 30 s per 112 k samples on the 4-core CPU. The PLAN
setup would take about 10 h on the full training months: anchors every 3 min, a 5-seed
Gaussian plus 3-seed MDN ensemble, up to 30 epochs.

Changes:
- Training anchors every **6 min**. Adjacent anchors overlap almost entirely in history and
  future, so little information is lost.
- **3** Gaussian and **2** MDN ensemble members.
- At most **20 epochs**, patience 3.

The budget is now about 1.5 h. This changes the *method*, not any success target. The
ensemble size is reported in the results.
