"""Constant-velocity Kalman filter for irregularly sampled AIS positions, with outlier gating
and a Rauch–Tung–Striebel (RTS) smoother.

Gating: each new fix is tested by its **normalised innovation squared** (NIS),
``ν' S⁻¹ ν``, where ν is the innovation (measured minus predicted position) and S its
predicted covariance. If the filter is consistent, NIS follows a χ² distribution with 2
degrees of freedom. A fix whose NIS exceeds the ``gate_prob`` quantile is rejected (the
filter coasts on its prediction). After ``max_consecutive_rejects`` rejections in a row, the
filter assumes it has lost the target and re-initialises at the latest fix. Causes include
a manoeuvre during a reporting gap, a real jump, or two vessels sharing one MMSI. The new
velocity comes from the last two (mutually consistent) rejected fixes. An isolated
outlier is still rejected, because the fix after it agrees with the existing track.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import chi2

from ais_sentinel.tracking.models import cv_process_noise, cv_transition

Array = NDArray[np.float64]

H_POS = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])


@dataclass(frozen=True)
class CVParams:
    """Tuning parameters of the CV Kalman filter."""

    q_accel: float = 0.003
    """White-noise acceleration PSD (m²/s³)."""
    r_pos_m: float = 10.0
    """Position measurement standard deviation per axis (m)."""
    init_vel_std: float = 5.0
    """Initial velocity standard deviation when SOG/COG are unavailable (m/s)."""
    sogcog_vel_std: float = 0.5
    """Initial velocity standard deviation when initialised from SOG/COG (m/s)."""
    gate_prob: float = 0.999
    max_consecutive_rejects: int = 2


def two_point_velocity(t: Array, z: Array, i: int) -> Array:
    """Finite-difference velocity from fixes i-1 and i (m/s)."""
    return (z[i] - z[i - 1]) / max(float(t[i] - t[i - 1]), 1.0)


@dataclass
class FilterResult:
    """Per-fix outputs of a filter run (N fixes, state dimension d)."""

    t: Array  # (N,) seconds
    x: Array  # (N, d) filtered state (posterior; prior for rejected fixes)
    P: Array  # (N, d, d)
    x_pred: Array  # (N, d) one-step prior
    P_pred: Array  # (N, d, d)
    F: Array  # (N, d, d) transition used to reach fix i from fix i-1 (identity at i=0)
    nis: Array  # (N,) NaN at (re)initialisation
    accepted: NDArray[np.bool_]  # (N,)
    reinit: NDArray[np.bool_]  # (N,) True where the filter (re)started


def initial_state(z: Array, vel: Array | None, p: CVParams) -> tuple[Array, Array]:
    """Initial 4-D state and covariance from a position fix and optional velocity."""
    x = np.zeros(4)
    x[:2] = z
    P = np.diag([p.r_pos_m**2, p.r_pos_m**2, p.init_vel_std**2, p.init_vel_std**2])
    if vel is not None and np.all(np.isfinite(vel)):
        x[2:] = vel
        P[2, 2] = P[3, 3] = p.sogcog_vel_std**2
    return x, P


def run_cv_filter(
    t: Array,
    z: Array,
    params: CVParams,
    vel_meas: Array | None = None,
) -> FilterResult:
    """Run the CV Kalman filter over one voyage.

    Args:
        t: (N,) strictly increasing times in seconds.
        z: (N, 2) ENU positions in metres.
        params: tuning parameters.
        vel_meas: optional (N, 2) ENU velocity from SOG/COG, NaN where missing. Used only for
            (re)initialisation.
    """
    n = len(t)
    if n == 0:
        raise ValueError("empty track")
    if np.any(np.diff(t) <= 0):
        raise ValueError("times must be strictly increasing")
    gate = float(chi2.ppf(params.gate_prob, df=2))
    R = np.eye(2) * params.r_pos_m**2
    xs, Ps = np.zeros((n, 4)), np.zeros((n, 4, 4))
    xps, Pps = np.zeros((n, 4)), np.zeros((n, 4, 4))
    Fs = np.tile(np.eye(4), (n, 1, 1))
    nis = np.full(n, np.nan)
    accepted = np.zeros(n, dtype=bool)
    reinit = np.zeros(n, dtype=bool)

    def vel_at(i: int) -> Array | None:
        return None if vel_meas is None else vel_meas[i]

    x, P = initial_state(z[0], vel_at(0), params)
    xs[0], Ps[0], xps[0], Pps[0] = x, P, x, P
    accepted[0] = reinit[0] = True
    rejects = 0
    for i in range(1, n):
        dt = float(t[i] - t[i - 1])
        F = cv_transition(dt)
        x_pred = F @ x
        P_pred = F @ P @ F.T + cv_process_noise(dt, params.q_accel)
        Fs[i], xps[i], Pps[i] = F, x_pred, P_pred
        nu = z[i] - H_POS @ x_pred
        S = H_POS @ P_pred @ H_POS.T + R
        S_inv = np.linalg.inv(S)
        nis[i] = float(nu @ S_inv @ nu)
        if nis[i] <= gate:
            K = P_pred @ H_POS.T @ S_inv
            x = x_pred + K @ nu
            # Joseph form keeps P symmetric positive definite despite rounding.
            IKH = np.eye(4) - K @ H_POS
            P = IKH @ P_pred @ IKH.T + K @ R @ K.T
            accepted[i] = True
            rejects = 0
        else:
            x, P = x_pred, P_pred
            rejects += 1
            if rejects >= params.max_consecutive_rejects:
                x, P = initial_state(z[i], two_point_velocity(t, z, i), params)
                # Two-point velocity variance: 2 R / dt^2 per axis.
                dt_v = max(float(t[i] - t[i - 1]), 1.0)
                P[2, 2] = P[3, 3] = 2 * params.r_pos_m**2 / dt_v**2 + 0.05
                reinit[i] = True
                accepted[i] = True
                nis[i] = np.nan
                rejects = 0
        xs[i], Ps[i] = x, P
    return FilterResult(t, xs, Ps, xps, Pps, Fs, nis, accepted, reinit)


def rts_smooth(res: FilterResult) -> tuple[Array, Array]:
    """Rauch–Tung–Striebel fixed-interval smoother.

    The smoother runs backwards from the last fix, folding information from *future* fixes
    into each state estimate. That is ideal for display and anomaly analysis, but must
    **never** feed a forecast, because it uses data after the forecast time. Smoothing
    restarts at each re-initialisation, since segments either side are independent.

    Returns:
        (x_smooth (N, d), P_smooth (N, d, d)).
    """
    n = len(res.t)
    xs, Ps = res.x.copy(), res.P.copy()
    for i in range(n - 2, -1, -1):
        if res.reinit[i + 1]:
            continue  # segment boundary: keep the filtered estimate
        P_pred = res.P_pred[i + 1]
        C = res.P[i] @ res.F[i + 1].T @ np.linalg.inv(P_pred)
        xs[i] = res.x[i] + C @ (xs[i + 1] - res.x_pred[i + 1])
        Ps[i] = res.P[i] + C @ (Ps[i + 1] - P_pred) @ C.T
    return xs, Ps
