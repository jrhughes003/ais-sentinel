"""IMM filter and simulator tests."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from ais_sentinel.tracking.imm import IMMParams, run_imm, transition_matrix
from ais_sentinel.tracking.kf import CVParams, rts_smooth, run_cv_filter
from ais_sentinel.tracking.models import ct_predict
from ais_sentinel.tracking.simulate import SimConfig, simulate_track


@given(st.floats(0.0, 1e5))
def test_transition_matrix_is_stochastic(dt: float) -> None:
    pi = transition_matrix(dt, IMMParams())
    assert np.allclose(pi.sum(axis=1), 1.0)
    assert (pi >= 0).all()


def test_transition_matrix_limits() -> None:
    p = IMMParams()
    assert np.allclose(transition_matrix(0.0, p), np.eye(3))
    assert np.allclose(transition_matrix(1e9, p), p.switch)


def _fixes(states: np.ndarray, rng: np.random.Generator, sigma: float = 5.0) -> np.ndarray:
    return states[:, :2] + rng.normal(0, sigma, (len(states), 2))


def test_mode_probabilities_identify_motion() -> None:
    rng = np.random.default_rng(0)
    t = np.arange(120) * 60.0
    # Moored: position jitter only.
    still = np.zeros((120, 5))
    # Straight: 5 m/s east.
    straight = np.zeros((120, 5))
    straight[:, 0] = 5.0 * t
    straight[:, 2] = 5.0
    # Steady turn: 0.3 deg/s at 5 m/s.
    turn = np.zeros((120, 5))
    x = np.array([0.0, 0.0, 5.0, 0.0, np.radians(0.3)])
    for i in range(120):
        turn[i] = x
        x, _ = ct_predict(x, 60.0)
    p = IMMParams()
    mu_still = run_imm(t, _fixes(still, rng), p).mu[30:].mean(axis=0)
    mu_straight = run_imm(t, _fixes(straight, rng), p).mu[30:].mean(axis=0)
    mu_turn = run_imm(t, _fixes(turn, rng), p).mu[30:].mean(axis=0)
    assert mu_still.argmax() == 0
    assert mu_straight.argmax() == 1
    assert mu_turn.argmax() == 2


def test_imm_tracks_turn_better_than_a_cv_filter_with_its_cruising_noise() -> None:
    rng = np.random.default_rng(1)
    t = np.arange(90) * 60.0
    x = np.array([0.0, 0.0, 6.0, 0.0, np.radians(0.25)])
    truth = np.zeros((90, 5))
    for i in range(90):
        truth[i] = x
        x, _ = ct_predict(x, 60.0)
    z = _fixes(truth, rng)
    p = IMMParams()
    full = run_imm(t, z, p)
    # Reference: a plain CV Kalman filter with the IMM's cruising-mode noise.
    cruise_only = run_cv_filter(t, z, CVParams(q_accel=p.q_cruise, r_pos_m=p.r_pos_m))

    def err(x: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.sum((x[20:, :2] - truth[20:, :2]) ** 2, axis=1))))

    assert err(full.x) < 0.8 * err(cruise_only.x)
    assert full.mu[40:, 2].mean() > 0.5  # the turning mode explains a steady turn


@settings(max_examples=25, deadline=None)
@given(st.integers(0, 10_000))
def test_imm_covariances_spd_and_probabilities_normalised(seed: int) -> None:
    tr = simulate_track(seed, SimConfig(duration_s=3600))
    res = run_imm(tr.t, tr.z, IMMParams())
    assert np.allclose(res.mu.sum(axis=1), 1.0)
    for P in res.P[::5]:
        assert np.allclose(P, P.T, atol=1e-6 * np.abs(P).max())
        assert np.linalg.eigvalsh(P).min() > -1e-9
    xs, _ = rts_smooth(res)
    assert np.all(np.isfinite(xs))


def test_imm_rejects_isolated_outlier() -> None:
    rng = np.random.default_rng(2)
    t = np.arange(60) * 60.0
    truth = np.zeros((60, 5))
    truth[:, 0] = 4.0 * t
    z = _fixes(truth, rng)
    z[30] += [2500.0, 1500.0]
    res = run_imm(t, z, IMMParams())
    assert not res.accepted[30]
    assert res.accepted[31:].all()


def test_simulator_reproducible_and_aislike() -> None:
    a, b = simulate_track(7), simulate_track(7)
    assert np.array_equal(a.t, b.t)
    assert np.array_equal(a.z, b.z)
    dt = np.diff(a.t)
    assert 55 <= np.median(dt) <= 65
    assert set(np.unique(a.segment)) <= {"straight", "turn", "speed", "stopped"}
    clean = ~a.outlier
    resid = np.hypot(*(a.z[clean] - a.truth[clean, :2]).T)
    assert np.median(resid) < 10  # 5 m per-axis noise -> ~5.9 m median radial error


def test_gap_noise_accepts_first_fix_after_unseen_turn() -> None:
    """A ship turns 90 deg during a 15-minute gap: without gap noise the first fix after the
    gap is rejected and the filter coasts on a wrong estimate; with it the fix is accepted."""
    rng = np.random.default_rng(3)
    t = np.concatenate([np.arange(30) * 60.0, 30 * 60.0 + 900 + np.arange(20) * 60.0])
    truth = np.zeros((len(t), 2))
    for i, ti in enumerate(t):
        if ti <= 29 * 60:
            truth[i] = [5.0 * ti, 0.0]  # east at 5 m/s
        else:
            truth[i] = [5.0 * 29 * 60 + 300.0, 5.0 * (ti - 29 * 60 - 60)]  # then north
    z = truth + rng.normal(0, 5, truth.shape)
    k = 30  # first fix after the gap
    for run in (run_cv_filter, run_imm):
        # Restart rule off, so this tests the gap-noise mechanism alone.
        base = (
            CVParams(q_accel=0.003, gap_reinit_s=0.0)
            if run is run_cv_filter
            else IMMParams(gap_reinit_s=0.0)
        )
        off = run(t, z, base)
        on = run(t, z, replace(base, gap_q=0.3))
        assert not off.accepted[k], run.__name__
        assert on.accepted[k], run.__name__
        assert np.hypot(*(on.x[k, :2] - truth[k])) < 0.2 * np.hypot(*(off.x[k, :2] - truth[k]))


def _turn_during_gap() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(3)
    t = np.concatenate([np.arange(30) * 60.0, 30 * 60.0 + 900 + np.arange(20) * 60.0])
    truth = np.zeros((len(t), 2))
    for i, ti in enumerate(t):
        truth[i] = (
            [5.0 * ti, 0.0] if ti <= 29 * 60 else [5.0 * 29 * 60 + 300.0, 5.0 * (ti - 29 * 60 - 60)]
        )
    return t, truth + rng.normal(0, 5, truth.shape), truth


def test_post_gap_restart_uses_the_first_fix_after_an_unseen_turn() -> None:
    t, z, truth = _turn_during_gap()
    k = 30
    for run, base in ((run_cv_filter, CVParams(q_accel=0.003)), (run_imm, IMMParams())):
        on = run(t, z, base)  # gap_reinit_s = 300 by default
        off = run(t, z, replace(base, gap_reinit_s=0.0))
        assert on.reinit[k] and on.accepted[k], run.__name__
        assert np.hypot(*(on.x[k, :2] - truth[k])) < 30  # restarted at the measured fix
        assert np.hypot(*(off.x[k, :2] - truth[k])) > 1000  # old behaviour: coasted far off


def test_repeated_identical_fix_is_not_a_measurement() -> None:
    t = np.arange(20) * 60.0
    z = np.zeros((20, 2))
    z[10:] = [0.4, -0.3]  # a moored vessel's reported position steps once, then repeats
    for run, base in ((run_cv_filter, CVParams()), (run_imm, IMMParams())):
        res = run(t, z, base)
        assert np.isnan(res.nis[11:]).all(), run.__name__  # repeats: no NIS
        assert np.isnan(res.loglik[11:]).all()
        assert np.isfinite(res.nis[10])  # the genuinely new fix is used
        assert res.accepted.all()
        legacy = run(t, z, replace(base, skip_repeats=False))
        assert np.isfinite(legacy.nis[11:]).all()
