"""Kalman filter and motion-model tests (unit, property and Monte Carlo consistency)."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.stats import chi2

from ais_sentinel.tracking.kf import CVParams, rts_smooth, run_cv_filter
from ais_sentinel.tracking.models import (
    ct_predict,
    cv_process_noise,
    cv_transition,
    stationary_predict,
)


def simulate_cv(
    rng: np.random.Generator, n: int, q: float, r: float, dt_mean: float = 60.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Truth from the exact CV model (so the filter's assumptions hold) plus noisy fixes."""
    t = np.cumsum(rng.uniform(0.5, 1.5, n) * dt_mean)
    x = np.array([0.0, 0.0, 4.0, 2.0])
    truth = np.zeros((n, 4))
    truth[0] = x
    for i in range(1, n):
        dt = t[i] - t[i - 1]
        x = cv_transition(dt) @ x + rng.multivariate_normal(np.zeros(4), cv_process_noise(dt, q))
        truth[i] = x
    z = truth[:, :2] + rng.normal(0, r, (n, 2))
    return t, z, truth


@settings(max_examples=50, deadline=None)
@given(st.lists(st.floats(1.0, 1800.0), min_size=2, max_size=40), st.floats(1e-5, 1.0))
def test_covariance_stays_symmetric_positive_definite(dts: list[float], q: float) -> None:
    rng = np.random.default_rng(0)
    t = np.concatenate([[0.0], np.cumsum(dts)])
    z = rng.normal(0, 50, (len(t), 2)) + np.outer(t, [3.0, 1.0])
    res = run_cv_filter(t, z, CVParams(q_accel=q))
    for P in res.P:
        assert np.allclose(P, P.T, atol=1e-6 * np.abs(P).max())
        assert np.linalg.eigvalsh(P).min() > 0


def test_nees_is_consistent_on_matching_model() -> None:
    """Average NEES over Monte Carlo runs lies inside the two-sided 95% chi-square band."""
    q, r, n, runs = 0.003, 10.0, 120, 60
    p = CVParams(q_accel=q, r_pos_m=r, gate_prob=0.999999)
    nees = np.zeros((runs, n))
    for k in range(runs):
        t, z, truth = simulate_cv(np.random.default_rng(k), n, q, r)
        res = run_cv_filter(t, z, p)
        err = res.x - truth
        nees[k] = np.einsum("ni,nij,nj->n", err, np.linalg.inv(res.P), err)
    avg = nees[:, 20:].mean(axis=0)  # skip initial transient (velocity prior is not truth)
    lo, hi = chi2.ppf([0.025, 0.975], df=4 * runs) / runs
    inside = np.mean((avg > lo) & (avg < hi))
    assert inside > 0.85, f"only {inside:.0%} of steps inside the NEES band"
    assert 3.4 < avg.mean() < 4.6


def test_filter_reduces_error_and_smoother_reduces_it_further() -> None:
    t, z, truth = simulate_cv(np.random.default_rng(1), 200, 0.003, 10.0)
    res = run_cv_filter(t, z, CVParams())
    xs, _ = rts_smooth(res)

    def rmse(est: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.sum((est[20:] - truth[20:, :2]) ** 2, axis=1))))

    raw, filt, smooth = rmse(z), rmse(res.x[:, :2]), rmse(xs[:, :2])
    assert filt < raw
    assert smooth < filt


def test_isolated_outlier_is_rejected_and_track_continues() -> None:
    t, z, _ = simulate_cv(np.random.default_rng(2), 60, 0.003, 10.0)
    z[30] += [3000.0, -2000.0]
    res = run_cv_filter(t, z, CVParams())
    assert not res.accepted[30]
    assert res.accepted[31:].all()
    assert not res.reinit[25:].any()


def test_sustained_jump_triggers_reinitialisation() -> None:
    t, z, _ = simulate_cv(np.random.default_rng(3), 60, 0.003, 10.0)
    z[30:] += [20_000.0, 0.0]  # track teleports 20 km and stays there
    res = run_cv_filter(t, z, CVParams(max_consecutive_rejects=3))
    assert not res.accepted[30:32].any()
    assert res.reinit[32]
    assert res.accepted[33:].all()
    xs, _ = rts_smooth(res)  # smoother must not blend across the re-init
    assert abs(xs[31, 0] - res.x[31, 0]) < 1e-9


def test_rejects_non_increasing_time() -> None:
    import pytest

    with pytest.raises(ValueError, match="increasing"):
        run_cv_filter(np.array([0.0, 0.0]), np.zeros((2, 2)), CVParams())


@settings(max_examples=100, deadline=None)
@given(
    st.floats(-5.0, 5.0),
    st.floats(-5.0, 5.0),
    st.floats(-0.05, 0.05),
    st.floats(1.0, 600.0),
)
def test_ct_jacobian_matches_finite_differences(ve: float, vn: float, w: float, dt: float) -> None:
    x = np.array([100.0, -50.0, ve, vn, w])
    _, F = ct_predict(x, dt)
    J = np.zeros((5, 5))
    for j in range(5):
        h = 1e-6 if j == 4 else 1e-4
        dx = np.zeros(5)
        dx[j] = h
        J[:, j] = (ct_predict(x + dx, dt)[0] - ct_predict(x - dx, dt)[0]) / (2 * h)
    assert np.allclose(F, J, rtol=1e-3, atol=1e-3 * max(1.0, dt))


@given(st.floats(-5.0, 5.0), st.floats(-5.0, 5.0), st.floats(1.0, 600.0))
def test_ct_with_zero_turn_rate_equals_cv(ve: float, vn: float, dt: float) -> None:
    x = np.array([10.0, 20.0, ve, vn, 0.0])
    xp, _ = ct_predict(x, dt)
    assert np.allclose(xp[:4], cv_transition(dt) @ x[:4])


def test_ct_preserves_speed_and_turns_counter_clockwise() -> None:
    w = np.radians(1.0)  # 1 deg/s, counter-clockwise
    x = np.array([0.0, 0.0, 5.0, 0.0, w])  # heading east
    xp, _ = ct_predict(x, 90.0)  # quarter turn
    assert np.isclose(np.hypot(xp[2], xp[3]), 5.0)
    assert np.allclose(xp[2:4], [0.0, 5.0], atol=1e-9)  # now heading north


def test_stationary_mode_decays_velocity() -> None:
    x = np.array([0.0, 0.0, 2.0, 0.0, 0.01])
    xp, _ = stationary_predict(x, 600.0, tau_s=60.0)
    assert abs(xp[2]) < 1e-3
    assert xp[4] == 0.0
    assert np.isclose(xp[0], 2.0 * 60.0 * (1 - np.exp(-10)))
