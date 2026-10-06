"""Real-data calibration helpers: simulator fitting and the predictive-likelihood objective."""

from __future__ import annotations

import numpy as np

from ais_sentinel.tracking.calibrate import _score, fit_sim_config, tune_on_real
from ais_sentinel.tracking.kf import CVParams, run_cv_filter
from ais_sentinel.tracking.simulate import simulate_track

STATS = {
    "speed_p5_kn": 5.0,
    "speed_p95_kn": 18.0,
    "turn_share": 0.2,
    "turn_p99_deg_s": 0.6,
    "accel_p90": 0.003,
    "accel_p99": 0.07,
    "report_median_s": 70.0,
    "gap_share_180s": 0.005,
    "rejected_share": 0.002,
}


def test_fit_sim_config_matches_turn_share_and_uses_log_uniform() -> None:
    cfg, share = fit_sim_config(STATS, pos_sigma_m=3.0)
    assert cfg.log_uniform
    assert cfg.turn_rate_deg_s == (0.05, 0.6)
    assert cfg.report_s == 70.0
    assert abs(share - 0.2) < 0.08


def _sim_arrays(sigma: float, n: int = 6, manoeuvres: bool = True) -> list[tuple[np.ndarray, ...]]:
    from dataclasses import replace

    out = []
    base = replace(fit_sim_config(STATS, sigma)[0], duration_s=3600)
    if not manoeuvres:
        base = replace(base, p_turn=0.0, p_speed=0.0, p_stop=0.0, p_outlier=0.0)
    for s in range(n):
        tr = simulate_track(s, base)
        rep = np.zeros(len(tr.t), dtype=bool)
        out.append((tr.t, tr.z, np.full_like(tr.z, np.nan), rep))
    return out


def test_likelihood_prefers_the_true_measurement_noise() -> None:
    # Straight-line motion and tiny process noise, so measurement noise dominates the
    # innovations. With large process noise R is only weakly identifiable (seen on real data).
    arrays = _sim_arrays(sigma=3.0, manoeuvres=False)
    scores = {
        r: _score(("cv", CVParams(q_accel=1e-6, r_pos_m=r), arrays, 5e-11))
        for r in (1.0, 3.0, 12.0)
    }
    assert max(scores, key=scores.get) == 3.0  # type: ignore[arg-type]


def test_tune_on_real_returns_best_by_loglik_sequentially() -> None:
    arrays = _sim_arrays(sigma=3.0, n=3)
    best, hist = tune_on_real("cv", CVParams(), {"r_pos_m": [1.0, 3.0]}, arrays, 5e-11, workers=1)
    assert best.r_pos_m == max(hist, key=lambda h: h["loglik"])["params"]["r_pos_m"]
    assert len(hist) == 2


def test_restart_fixes_are_scored_so_stiff_filters_pay_for_them() -> None:
    t, z = np.arange(40) * 60.0, np.zeros((40, 2))
    z[:, 0] = 5.0 * t
    z[20:] += [20_000.0, 0.0]  # a real jump forces rejections and a restart
    res = run_cv_filter(t, z, CVParams(max_consecutive_rejects=2))
    assert res.reinit[21]
    assert np.isfinite(res.loglik[1:]).all()  # restart fixes keep their predictive likelihood
    assert np.isnan(res.loglik[0])
