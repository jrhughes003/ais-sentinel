"""Motion models for vessel tracking in a local ENU frame.

State conventions:

* CV (4-D):  ``x = [e, n, ve, vn]``      (m, m, m/s, m/s)
* IMM (5-D): ``x = [e, n, ve, vn, ω]``  (ω = turn rate in rad/s, positive counter-clockwise)

The **nearly-constant-velocity** (white-noise acceleration) model assumes acceleration is
white noise with power spectral density ``q`` (m²/s³). Discretising over an interval dt gives
the standard ``Q`` below. Because AIS reports arrive at irregular times, F and Q are rebuilt
for each step's dt instead of being fixed matrices.

The **coordinated-turn (CT)** model moves the vessel on a circular arc at constant speed and
turn rate ω. It is nonlinear in ω, so it is propagated with its Jacobian (EKF style); see
Bar-Shalom, Li & Kirubarajan (2001), §11.7.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

#: Below this |ω·dt| the CT model is evaluated by its Taylor limit (straight-line motion).
_OMEGA_EPS = 1e-6


def cv_transition(dt: float) -> Array:
    """CV state transition matrix F(dt) for the 4-D state."""
    return np.array(
        [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def cv_process_noise(dt: float, q: float) -> Array:
    """Discrete white-noise-acceleration covariance Q(dt) for the 4-D CV state.

    Per axis, the [position, velocity] block is ``q * [[dt³/3, dt²/2], [dt²/2, dt]]``.
    """
    a, b, c = q * dt**3 / 3, q * dt**2 / 2, q * dt
    return np.array(
        [
            [a, 0.0, b, 0.0],
            [0.0, a, 0.0, b],
            [b, 0.0, c, 0.0],
            [0.0, b, 0.0, c],
        ]
    )


def ct_predict(x: Array, dt: float) -> tuple[Array, Array]:
    """Propagate a 5-D state through the coordinated-turn model.

    Returns:
        (x_pred, F), where F is the Jacobian ∂f/∂x evaluated at ``x``.
    """
    e, n, ve, vn, w = x
    wt = w * dt
    if abs(wt) < _OMEGA_EPS:
        # Limit ω → 0: first-order Taylor expansion in ω, so the prediction and its Jacobian
        # stay continuous across the branch boundary.
        x_pred = np.array(
            [
                e + ve * dt - 0.5 * w * vn * dt**2,
                n + vn * dt + 0.5 * w * ve * dt**2,
                ve - w * vn * dt,
                vn + w * ve * dt,
                w,
            ]
        )
        F = np.eye(5)
        F[0, 2] = F[1, 3] = dt
        F[0, 4] = -0.5 * vn * dt**2
        F[1, 4] = 0.5 * ve * dt**2
        F[2, 4] = -vn * dt
        F[3, 4] = ve * dt
        return x_pred, F
    s, c = np.sin(wt), np.cos(wt)
    a = s / w  # ∫cos
    b = (1 - c) / w  # ∫sin
    x_pred = np.array(
        [
            e + a * ve - b * vn,
            n + b * ve + a * vn,
            c * ve - s * vn,
            s * ve + c * vn,
            w,
        ]
    )
    # Derivatives of a and b with respect to ω.
    da = (wt * c - s) / w**2
    db = (wt * s - (1 - c)) / w**2
    F = np.array(
        [
            [1.0, 0.0, a, -b, da * ve - db * vn],
            [0.0, 1.0, b, a, db * ve + da * vn],
            [0.0, 0.0, c, -s, -dt * (s * ve + c * vn)],
            [0.0, 0.0, s, c, dt * (c * ve - s * vn)],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )
    return x_pred, F


def ct_process_noise(dt: float, q_accel: float, q_omega: float) -> Array:
    """Process noise for the 5-D CT state: white acceleration plus a random-walk turn rate."""
    Q = np.zeros((5, 5))
    Q[:4, :4] = cv_process_noise(dt, q_accel)
    Q[4, 4] = q_omega * dt
    return Q


def stationary_predict(x: Array, dt: float, tau_s: float) -> tuple[Array, Array]:
    """'Stationary' mode: velocity decays towards zero with time constant ``tau_s``.

    This is a first-order Gauss–Markov (Ornstein–Uhlenbeck) velocity model. It lets
    moored or anchored vessels, whose GPS positions jitter by metres, be explained without
    inventing motion. Returns (x_pred, F) for the 5-D state; ω is forced to zero.
    """
    k = np.exp(-dt / tau_s)
    g = tau_s * (1 - k)  # ∫ velocity decay
    F = np.eye(5)
    F[0, 2] = F[1, 3] = g
    F[2, 2] = F[3, 3] = k
    F[4, 4] = 0.0
    return F @ x, F


def stationary_process_noise(dt: float, q_accel: float, tau_s: float) -> Array:
    """Process noise for the stationary mode.

    This is a CV noise structure scaled by the decay; a small, adequate approximation.
    """
    Q = np.zeros((5, 5))
    Q[:4, :4] = cv_process_noise(min(dt, tau_s), q_accel)
    Q[4, 4] = 1e-12
    return Q
