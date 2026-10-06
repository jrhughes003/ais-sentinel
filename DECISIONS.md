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

### D17 — Partial source days are "unavailable" for gap detection (2026-10-06)
The MarineCadastre file for 2023-10-29 has 4.58 M national rows, 52% of the median day (the
next-lowest day is at 78%). It yields 20.6 k AOI rows, against 40–80 k elsewhere. This looks
like a partial NOAA file. It falls in the locked test month.

- Any day below 60% of the median national row count is treated as **incomplete**.
- The anomaly stage passes only complete days to the gap detector. D10's availability rule
  had not been wired into the stage until now, and this fixes that.
- The excluded days are listed in `reports/anomaly.md`.
- Its tracks and forecasts are kept: fewer fixes do not bias a forecast's truth, which is
  interpolated only between fixes no more than 3 min from the target time.

(Spotted by a helper session that inspected the manifest.)

### D18 — Final outcomes, and analyses run after the locked test (2026-10-06)
The locked October test was evaluated once (`reports/holdout_runs.md`).

**Prediction:** the best ML model (M2) beats kNN by 7.1% at 60 min and 9.5% at 120 min (both
significant). The 15% bar is **not met**. Coverage target **met**.

**Anomaly targets:** jump, loiter and rendezvous **met**; gap (R 0.85) and deviation (R 0.47)
**not met**.

Analyses afterwards that change no reported number:
- **kNN k-sensitivity on validation** (`reports/knn_sensitivity.md`): k = 80 sits at the grid
  edge but is effectively optimal. Larger k is worse or equal, so the ML comparison is fair.
- **Gap misses:** mostly injected silences ending in cells below 80% normal reception. That
  is the precision-for-recall trade-off chosen in the design.
- **Deviation misses:** with four months of all-vessel traffic, many cells near the lanes
  hold more than 2 training voyages, so the absolute density rule stops firing.
  - Proposed fix: commercial-only density, a relative threshold, and short-interruption
    tolerance.
  - It is **not** applied here. Re-scoring October after seeing its results would be tuning
    on the test set. A fix needs a fresh test period.

### D19 — Tracking attempt 2: one bounded, pre-announced iteration (2026-10-06)
Agreed with the owner: one principled attempt at the two diagnosed weaknesses, with the
PLAN §9.2 targets **unchanged**, and the result reported either way.

**Changes (offered to both filters equally, tuned on the same seeds 0–49):**
- Measurement noise `r_pos_m` ∈ {5, 10} became a tuned parameter. Attempt 1 had fixed it at
  10 m, double the simulator's true 5 m.
- Gap-aware process noise `gap_q` ∈ {0, 0.05, 0.3}, i.e. extra acceleration noise on steps
  longer than 3 min.
- Evaluation moved to **fresh seeds 1000–1199**, because seeds 100–299 had been looked at.
  Attempt-1 reports are kept as `reports/tracking_attempt1.*`.

**Result:**

| Criterion | Attempt 1 | Attempt 2 | Met? |
|---|---:|---:|---|
| IMM manoeuvre RMSE vs CV-KF | −0.6% | 7.1% lower | ❌ (target 20%) |
| IMM straight-leg RMSE vs CV-KF | +168% | +0.2% | ✅ |
| NEES in band | 1% | 8% | ❌ |
| Real-data NIS above threshold | 0.9% | 0.9% | ❌ (2–10%) |

**Attribution:**
- Almost all of the gain came from correct measurement noise. The IMM chose 5 m.
- Gap noise barely moved the tuning score (14.63 against 14.61 m), and larger values hurt.
- The CV-KF kept q = 0.1, which already absorbs gaps.

**Real-data finding** (`reports/r_calibration.md`, training split):
- NIS is insensitive to the measurement noise over 1.5–10 m. The underconfidence comes
  from **process noise** tuned on simulated manoeuvres that are harsher (up to 1°/s turns,
  0.05 m/s² speed changes) than real lakers make.
- The principled fix is to fit the simulator's manoeuvre statistics, or the process noise,
  to real data, for example by maximising the one-step predictive likelihood on the
  training split. That is left as the next step. It would be a second tuning iteration, not
  part of this bounded attempt.

**Scope:**
- The default config now holds the attempt-2 IMM settings.
- The website's tracks and the prediction baselines B1/B2 in the locked test were produced
  with the attempt-1 settings (commit 303cd90). `tracks.parquet` was not regenerated, so
  the locked test is untouched.
- The validation NIS for attempt 2 was computed separately (`scripts/nis_val.py`).

### D20 — Tracking attempt 3: calibrate the simulator and both filters to real data (2026-10-06)
Requested by the owner as the next step after D19. The PLAN §9.2 targets are unchanged.

**What changed (training split only):**
- **Simulator recalibrated** to real moving commercial vessels (1.8 M one-minute steps):
  - speeds 4.8–19.8 kn;
  - log-uniform turn rates 0.05–0.61 °/s (real ships turn for 20% of the time);
  - log-uniform accelerations 0.0034–0.069 m/s²;
  - 70 s reporting, 0.53% gaps over 3 min, 0.05% outliers.

  The old simulator drew turn rates uniformly up to 1 °/s.
- **Both filters tuned by one-step predictive likelihood** on 150 seeded training voyages.
  This is the prediction-error method, with the clutter mixture used in the objective. Two
  issues came up during tuning:
  - The optimum first sat on grid edges. That exposed a **loophole**: fixes where the
    filter re-initialises were not scored, so an overly stiff CV-KF, restarting on 8% of
    fixes, looked best. Restart fixes are now scored, with a regression test.
  - The remaining edge values were confirmed as optima by a local refinement.
- **Results of the tuning:**
  - CV-KF: q = 1e-4, R = 5 m.
  - IMM: q_cruise 5e-4, q_turn 0.05, q_omega 2e-8, R = 0.75 m.
  - Both chose gap_q = 0.
- **Evaluation:** fixed mode on fresh seeds 2000–2199. The IMM is not tuned in simulation.
  - The CV-KF baseline is the **better** (on tuning seeds) of the real-likelihood CV-KF
    and one tuned for accuracy in the calibrated simulator. This guards against a strawman:
    the real-likelihood CV-KF rejected 3,515 good fixes in simulation and would have handed
    the IMM a meaningless "99% better".

**Results:**
- In the realistic simulator, the simulator-tuned CV-KF tracks real-like manoeuvres as well
  as the IMM: 1.4 m RMSE each, **manoeuvre criterion −0.4%, not met**.
- The IMM's straight-leg RMSE (63.5 m against 1.1 m) comes from **two** fixes out of 24,057.
  They are the first fixes after 20-minute gaps in which the vessel turned, the same
  failure mode as attempt 1. **Not met.** Median and 95th-percentile errors are identical
  (0.9 / 1.8 m).
- NEES: 1% of steps in band. **Not met.**
- Real data (validation):
  - IMM NIS above threshold: **1.6%**, up from 0.9% (1.9% excluding repeat positions), so
    **not met**, but close.
  - The CV-KF is at 3.8%, but it rejects 3.8% of real fixes.
  - The IMM predicts real fixes far better: mean log-likelihood −7.38 against −10.40 per fix.

**Scorecard:**
- The final configuration is attempt 3, so the tracking score is 0 of 4. Attempt 2 had 1 of
  4, but against a weaker, mis-specified simulator. The latest attempt is reported, not the
  best one.
- This is the third attempt on this criterion. Per the project guardrail, iteration stops
  here.

**Remaining known fix, not applied:** re-initialise immediately when the first fix after a
long gap fails the gate.

**Scope:** the config holds the attempt-3 parameters. The site's tracks and the locked-test
prediction baselines B1/B2 use the attempt-1 tracker. The attempt-1 and attempt-2 reports
are kept in `reports/`.

## D21 — Tracking attempt 4 and a v2 evaluation on a fresh test month (2026-10-06)

**Context:** the owner asked whether the missed targets were fixable. They are happy to report
misses, as long as no obvious fix is left untried. Two obvious fixes were left:
- the tracker's post-gap failure (D20);
- the gap and route-deviation detectors (PROGRESS, "Suggested next steps").

October 2023 had already been used as the locked test, so the changed detectors and models
need a test month that has never been looked at.

**Tracking attempt 4** (owner-authorised; a fourth attempt is a logged exception to the
three-attempt guardrail):
- **Post-gap restart.** When the first fix after a gap of more than 300 s fails the gate,
  the filter re-initialises at that fix, instead of rejecting it and coasting.
- **Repeats skipped.** A fix whose position is identical to the one before it is treated as
  "no new measurement": predict only, no update, no NIS.
- Both filters were recalibrated on real training data. Restart fixes are scored, so a stiff
  filter cannot hide its misses (test in `tests/test_calibrate.py`).
- Evaluated on fresh simulator seeds 3000–3199, with the best-of-two CV-KF baseline (D20).

**Attempt-4 results (simulation):**
- The IMM's straight-leg RMSE fell from 63.5 m to 9.7 m, but the CV-KF is at 7.3 m: **+33%,
  not met**.
- Manoeuvres: CV-KF 4.5 m, IMM 9.0 m. **Not met.**
- Medians and 95th percentiles are equal (1.7–1.8 m and 3.5–3.7 m). The RMSE differences
  come from a handful of tail fixes.
- NEES: 17% of steps in band, up from 1%. **Not met.**
- Real-data NIS for attempt 4 is pending (`scripts/nis_val.py`, part of the overnight run).

**Conclusion:** no obvious tracking fix remains. In a simulator calibrated to real ships, the
manoeuvres are gentle enough that a well-tuned constant-velocity filter is as accurate as the
IMM, and the IMM's advantage shows up in real-data predictive likelihood instead. Tracking
iteration stops here.

**v2 evaluation design:**
- **Test month:** October 2024. MarineCadastre's 2024 files are Zstandard `csv2`, with a
  different column order; the downloader and schema normaliser handle both formats.
- **Train and validation months:** unchanged (May–August and September 2023).
- **Configuration:** everything lives in `configs/v2.yaml` (it extends the default). v2
  outputs go to `data/v2`, `reports/v2` and `models/v2`. v1 results stay published alongside.
- **Detectors tuned on September only** (`ais-sentinel anomaly-tune`,
  `reports/anomaly_tuning.md`). The rule is to maximise recall subject to precision ≥ 0.9.
  The owner chose the "modest budget" operating point:
  - **Gap:** reception mask 0.7, up from 0.8. September recall 0.84 → 0.90, base alarm
    rate 1.78 → 2.00 per 1,000 commercial vessel-hours.
  - **Route deviation:** a lane model from commercial traffic only, ≤ 2 voyages per cell,
    3-minute bridging of short interruptions, fishing vessels skipped. September recall
    0.45 → 0.70, base alarm rate 1.5 → 5.6 per 1,000 h.
  - This deviation setting breaks the tuning rule's "base rate no higher than v1" cap.
    The owner accepted that trade explicitly, and it is reported as such.
- **Prediction models** are retrained under v2: same architecture, 40 epochs, patience 4.
  They are evaluated once on October 2024 by the `holdout` stage, which appends a row to
  `reports/holdout_runs.md`.
