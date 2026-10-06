# Methods explained

This guide is for a reader who knows Kalman filters, state-space models and Bayesian
reasoning, and wants the *extra* ideas this project uses. Each section ends with the
place in the code where the idea lives.

---

## 1. Local East-North-Up (ENU) frames

**Why not use degrees?** Latitude/longitude are angles, not distances. At 42 °N one degree of
longitude is about 82 km, but one degree of latitude is about 111 km.

**The conversion:** WGS84 → ECEF (Earth-centred Cartesian) → rotate into a tangent plane at
an anchor point. This is exact. Within ~150 km, horizontal distances in the plane match
ground distances to better than 0.05%.

**The anchor depends on the job:**
- Each **track** is filtered in a frame anchored at its first fix.
- Each **forecast** works in a frame anchored at its last observed fix.

**Gotcha:** a line of constant latitude is not straight in a tangent plane. It curves
poleward by about e² tan φ / (2R), roughly 5 cm after 1 km. The tests check this rather than
pretend it away.

*Code:* `src/ais_sentinel/geo.py`, `tests/test_geo.py`.

---

## 2. Irregular sampling in a Kalman filter

**Why rebuild the matrices?** AIS reports arrive every 2 s to 3 min, so F(dt) and Q(dt) are
rebuilt for every step.

**Process noise:** for the nearly-constant-velocity model, the per-axis block of Q is
q·[[dt³/3, dt²/2], [dt²/2, dt]]. This comes from integrating white acceleration noise of
spectral density q (m²/s³).

**Long gaps:** large dt therefore inflates uncertainty in the right way. A vessel unseen for
10 minutes has a much wider predicted cloud than one seen 10 seconds ago.

*Code:* `tracking/models.py`, `tracking/kf.py`.

---

## 3. Interacting Multiple Model (IMM) filter

A single filter has one motion model. Ships alternate between sitting still, cruising and
turning. An IMM runs **one filter per mode** and tracks the probability μⱼ of each mode.

**Mode switching** is a Markov chain:
- Over a step dt, mode i persists with probability exp(−dt/τᵢ), where τᵢ is its mean
  duration.
- Otherwise it switches to mode j with weight W[i,j].
- The dt-dependence matters with irregular AIS: after a 10-minute gap a mode change is much
  more plausible than after 10 seconds.

**One IMM cycle:**
1. **Mixing.** Each mode filter starts from a blend of all modes' previous estimates. The
   blend weights are P(was in mode i | now in mode j) = Π_ij μᵢ / cⱼ, where
   cⱼ = Σᵢ Π_ij μᵢ is the predicted probability of mode j.
2. **Mode-matched predict and update.** An ordinary KF step for each mode. The turning
   mode is nonlinear (the coordinated-turn model has the turn rate ω inside sin/cos), so it
   uses an **EKF**: the model is linearised with its Jacobian at the current estimate.
3. **Probability update.** Bayes' rule: μⱼ ∝ cⱼ · N(innovationⱼ; 0, Sⱼ). A mode whose
   prediction explains the new fix better gains probability.
4. **Output.** The mixture is collapsed to one Gaussian by *moment matching* (matching mean
   and covariance). The covariance includes the spread between mode means:
   P = Σ μⱼ (Pⱼ + (xⱼ − x)(xⱼ − x)ᵀ).

*Code:* `tracking/imm.py` (it has a longer docstring); tests in `tests/test_imm.py`.

---

## 4. Rejecting outliers: likelihood ratio against "clutter"

**The baseline gate.** Each fix is compared with the prediction through its **normalised
innovation squared**, NIS = νᵀ S⁻¹ ν. If the filter is right, NIS follows χ²₂. The
CV-KF rejects a fix when NIS exceeds the 99.99% quantile.

**Why the IMM needs something else.** Two simpler options failed in testing (DECISIONS D11):
- Gating on the combined prediction rejected the first fixes of every turn, because the
  cruising mode dominated.
- Accepting a fix inside *any* mode's gate let an unlikely, wide mode admit kilometre-scale
  outliers.

**The adopted test** compares two hypotheses for each fix:
- the fix comes from the vessel, with likelihood Σⱼ cⱼ N(ν; 0, Sⱼ);
- the fix is an outlier ("clutter") spread uniformly over the area, with density λ.

The fix is rejected when the outlier explanation is more likely. This is the same idea as
probabilistic data association in radar tracking.

**Recovery.** After 2 consecutive rejections the filter assumes the vessel really moved, for
example after a manoeuvre during a gap, and re-initialises. The new velocity is estimated
from those two fixes.

*Code:* `tracking/imm.py`, `tracking/kf.py`.

---

## 5. Checking a filter is honest: NEES and NIS

**NEES** = eᵀ P⁻¹ e, where e is the *true* error. It needs ground truth, so it is computed in
simulation. For a consistent filter it averages to the state dimension.

**ANEES** averages NEES over R Monte Carlo runs at each time step; R·ANEES ~ χ²(n·R). The
report shows the share of steps inside the 95% band.

**NIS** needs no truth. It works on real data: about 5% of values should exceed χ²₂(0.95).
- Far fewer means the filter is **underconfident** (its covariance is too large).
- Far more means it is **overconfident**.

*Code:* `tracking/sim_study.py` (NEES); `tracking/pipeline.py::nis_consistency` (real data).

---

## 6. Prediction samples without leakage

**Leakage** means test information sneaking into training, which makes results look better
than they will be in use. The guards:

1. **Time split with buffer days.** A voyage belongs to a split only if its *whole* span is
   inside that split (`data/segment.py`, with tests).
2. **Causal inputs.** A forecast made at time t₀ uses only fixes at or before t₀. A test
   perturbs the future and checks the features do not change.
3. **Training-only libraries.** The kNN library and anomaly context use the train split only.
4. **Locked test.** October is scored once per model version, and every run is logged
   (`reports/holdout_runs.md`).
5. **Vessel-unseen check.** Ferries repeat the same route every day, so a model could
   "memorise" a vessel. Results are also reported on test vessels that never appear in
   training.

---

## 7. Route analogs (kNN baseline)

**Idea:** in a river, the best guess of where a ship is going is where *other ships at the
same place, heading the same way* went next.

**Library:** every 5 minutes of every training voyage contributes an entry: position,
course, speed, and the true future displacement every 5 minutes up to 3 h.

**Query:** a KD-tree finds neighbours within ~1 km and ~30°. Course is stored as a scaled
unit vector, so 359° and 1° count as close.

**Speed adjustment:** a slower ship follows the same path more slowly. Each neighbour's
future is read at time h·(v_query / v_neighbour).

**Prediction and fallback:** the forecast is the weighted average of the neighbours. Their
spread gives the uncertainty. With too few neighbours it falls back to dead reckoning.

*Code:* `prediction/knn.py`.

---

## 8. The neural network: GRU + Gaussian or mixture head

**GRU.** A gated recurrent unit reads the last 60 minutes, one step per minute. Each step
carries position offset, speed, course (as sin/cos) and a gap flag. Gates decide what to
remember: a learned, nonlinear cousin of a filter's state update.

**Context.** Position in the region (so it can learn where channels bend), speed, course,
vessel type and length.

**Target: a correction, not a position.** It predicts *truth minus dead reckoning*. If it
learns nothing, it equals the physics baseline.

**Gaussian head.** For each horizon it outputs a 2-D Gaussian:
- a mean;
- two log standard deviations;
- a correlation ρ = 0.99·tanh(r), which stays in (−1, 1).

**Mixture density network (MDN).** K = 3 weighted Gaussians. The model can then say "either
it turns into the channel or it goes straight on", instead of averaging the two into a
position nobody reaches.

**Training by negative log-likelihood (NLL).** The loss is −log p(truth | prediction). Unlike
squared error, it **rewards honest uncertainty**: overconfidence on a miss is penalised
heavily, and needless vagueness is penalised too.

**Deep ensemble.** Several networks are trained from different random seeds. Their
disagreement estimates *model* uncertainty. The combined covariance is the average
predicted covariance plus the spread of the members' means.

*Code:* `prediction/features.py`, `prediction/ml.py`.

---

## 9. Calibration and coverage

**Coverage.** A 90% ellipse should contain the truth 90% of the time. With truth y and
forecast (μ, Σ), compute d² = (y − μ)ᵀ Σ⁻¹ (y − μ). The truth lies inside the 90% ellipse
when d² ≤ χ²₂(0.9) = 4.61.

**One number per horizon.** Each model's covariance is multiplied by a single scale per
horizon, chosen on the **validation** month to make coverage exactly 90%. The scale is
s = quantile₀.₉(d²) / 4.61. This is *split-conformal calibration*, one of the simplest
honest calibration methods. The **test** month then shows whether the calibration holds out
of sample.

*Code:* `evaluation/metrics.py::fit_cov_scale`.

---

## 10. Confidence intervals with a cluster bootstrap

**The problem.** Forecasts from the same voyage are strongly correlated: ten anchors on one
straight run are not ten independent tests. The textbook bootstrap resamples individual
samples, which treats them as independent and gives intervals that are too narrow.

**The fix.** Resample **whole voyages** with replacement, recompute the metric, repeat 1,000
times, and take the 2.5th and 97.5th percentiles.

**Comparing models.** "Model A beats model B" uses the *paired* difference on the same
samples, again resampled by voyage. A test confirms the cluster interval is much wider than
the naive one on correlated data.

*Code:* `evaluation/metrics.py`.

---

## 11. Evaluating anomaly detectors without labels

Real AIS has no list of "true anomalies", so we create our own ground truth:

1. Take a real **test-month** commercial voyage.
2. Inject one anomaly of known type, time span and magnitude:
   - delete reports (gap);
   - displace fixes (jump);
   - splice in a slow drift (loiter);
   - bend the path sideways (deviation);
   - add a partner vessel that comes alongside (rendezvous).
3. Run the detector *before* and *after* injection, and count only **new** detections.
   Alarms that already existed belong to the real data. They are reported separately as a
   **base alarm rate** per 1,000 vessel-hours.
4. Score:
   - **recall** = injected anomalies found;
   - **precision** = new alarms that match an injection.

   Both are reported by magnitude, which gives *detectability curves*: for example, how
   large a deviation must be before the detector notices it.

**Caveat:** this measures how detectable *these injected patterns* are. Real deception is
rarer and more varied, so the real-data case studies matter too.

*Code:* `anomaly/inject.py`, `anomaly/run.py`.
