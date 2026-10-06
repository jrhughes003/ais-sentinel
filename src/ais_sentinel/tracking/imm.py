"""Interacting Multiple Model (IMM) filter for vessel tracking.

Why IMM? One Kalman filter has one motion model. Tuned for steady cruising (small process
noise), it lags badly in turns. Tuned for turns (large noise), it is jittery on straight
legs. An IMM runs a small *bank* of filters, one per motion mode, and tracks a
probability for each mode. Here the modes are:

0. **stationary:** moored or anchored; velocity decays towards zero.
1. **cruising:** nearly constant velocity, with turn rate forced to zero.
2. **turning:** coordinated turn; the turn rate ω is part of the state (EKF).

Mode switches follow a Markov chain. Over a step of dt seconds, mode i persists with
probability ``exp(-dt / τ_i)`` (τ_i is the mean time spent in that mode). Otherwise it
switches to mode j with weight ``W[i, j]``. Making the transition matrix depend on dt
matters for irregular AIS sampling: after a 10-minute gap, a change of mode is much
more likely than after 10 seconds.

Each IMM cycle (Bar-Shalom, Li & Kirubarajan 2001, §11.6):

1. **Mixing:** each mode-matched filter starts from a probability-weighted blend of all
   modes' previous estimates. The weights are the probability that the target *was* in
   mode i, given it is *now* in mode j.
2. **Mode-matched prediction and update:** an ordinary KF or EKF step per mode.
3. **Mode probability update:** Bayes' rule. Each mode's prior probability is multiplied
   by the likelihood of the new measurement under that mode's prediction.
4. **Combination:** the output is the probability-weighted mixture of the mode
   estimates, moment-matched to one Gaussian.

All modes share the 5-D state ``[e, n, ve, vn, ω]``. That makes mixing a plain weighted
average with no dimension bookkeeping.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from ais_sentinel.tracking.kf import FilterResult, gap_noise, post_gap_restart, two_point_velocity
from ais_sentinel.tracking.models import (
    ct_predict,
    ct_process_noise,
    cv_process_noise,
    cv_transition,
    stationary_predict,
    stationary_process_noise,
)

Array = NDArray[np.float64]
MODES = ("stationary", "cruising", "turning")
H5 = np.array([[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0]])
_LOG2PI = np.log(2 * np.pi)


def _default_switch() -> Array:
    # Row i: where mode i goes when it leaves. Stationary vessels start cruising; cruising
    # vessels mostly start turning; turning vessels mostly return to cruising.
    return np.array([[0.0, 0.9, 0.1], [0.2, 0.0, 0.8], [0.1, 0.9, 0.0]])


@dataclass(frozen=True)
class IMMParams:
    """Tuning parameters of the 3-mode IMM."""

    r_pos_m: float = 10.0
    q_stationary: float = 1e-4  # m²/s³, small jitter while moored
    tau_stationary_s: float = 30.0  # velocity decay time constant
    q_cruise: float = 0.002  # m²/s³
    q_turn: float = 0.01  # m²/s³
    q_omega: float = 2e-7  # rad²/s³, turn-rate random walk
    sojourn_s: tuple[float, float, float] = (1800.0, 900.0, 180.0)
    switch: Array = field(default_factory=_default_switch)
    init_probs: tuple[float, float, float] = (0.2, 0.6, 0.2)
    init_vel_std: float = 5.0
    sogcog_vel_std: float = 0.5
    init_omega_std: float = float(np.radians(1.0))
    clutter_density: float = 5e-11  # per m²: outlier prior / area, e.g. 0.005 / (10 km)²
    gap_q: float = 0.0  # extra acceleration PSD on steps longer than gap_s (see kf.gap_noise)
    gap_s: float = 180.0
    gap_reinit_s: float = 300.0  # restart if the first fix after a longer gap fails the test
    skip_repeats: bool = True  # a repeated identical coordinate is not a new measurement
    max_consecutive_rejects: int = 2


@dataclass
class IMMResult(FilterResult):
    """IMM outputs. Adds per-fix mode probabilities (N, 3) to :class:`FilterResult`."""

    mu: Array = field(default_factory=lambda: np.zeros((0, 3)))
    snap_idx: NDArray[np.int64] = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    snap_x: Array = field(default_factory=lambda: np.zeros((0, 3, 5)))  # per-mode states
    snap_P: Array = field(default_factory=lambda: np.zeros((0, 3, 5, 5)))


def transition_matrix(dt: float, p: IMMParams) -> Array:
    """Mode transition matrix Π(dt), with rows summing to one."""
    stay = np.exp(-dt / np.asarray(p.sojourn_s))
    pi = (1 - stay)[:, None] * p.switch
    pi[np.diag_indices(3)] = stay
    return pi


def predict_mode(j: int, x: Array, P: Array, dt: float, p: IMMParams) -> tuple[Array, Array, Array]:
    """Mode-matched prediction. Returns (x_pred, P_pred, F)."""
    if j == 0:
        xp, F = stationary_predict(x, dt, p.tau_stationary_s)
        Q = stationary_process_noise(dt, p.q_stationary, p.tau_stationary_s)
    elif j == 1:
        F = np.zeros((5, 5))
        F[:4, :4] = cv_transition(dt)  # F[4, 4] = 0: cruising means no turn rate
        xp = F @ x
        Q = np.zeros((5, 5))
        Q[:4, :4] = cv_process_noise(dt, p.q_cruise)
        Q[4, 4] = 1e-12
    else:
        xp, F = ct_predict(x, dt)
        Q = ct_process_noise(dt, p.q_turn, p.q_omega)
    extra = gap_noise(dt, p)
    if extra:
        Q = Q.copy()
        Q[:4, :4] += cv_process_noise(dt, extra)  # same gap allowance in every mode
    return xp, F @ P @ F.T + Q, F


def mixture_moments(w: Array, xs: Array, Ps: Array) -> tuple[Array, Array]:
    """Moment-match a Gaussian mixture: weights (M,), means (M, d), covariances (M, d, d)."""
    x = w @ xs
    d = xs - x
    return x, np.einsum("m,mij->ij", w, Ps) + np.einsum("m,mi,mj->ij", w, d, d)


def _init(z: Array, vel: Array | None, p: IMMParams) -> tuple[Array, Array, Array]:
    x = np.zeros(5)
    x[:2] = z
    P = np.diag(
        [p.r_pos_m**2, p.r_pos_m**2, p.init_vel_std**2, p.init_vel_std**2, p.init_omega_std**2]
    )
    mu = np.asarray(p.init_probs, dtype=float)
    if vel is not None and np.all(np.isfinite(vel)):
        x[2:4] = vel
        P[2, 2] = P[3, 3] = p.sogcog_vel_std**2
        if np.hypot(*vel) < 0.25:  # under 0.5 kn: probably moored
            mu = np.array([0.8, 0.15, 0.05])
    xs = np.tile(x, (3, 1))
    Ps = np.tile(P, (3, 1, 1))
    return xs, Ps, mu


def run_imm(
    t: Array,
    z: Array,
    params: IMMParams,
    vel_meas: Array | None = None,
    snapshot_idx: NDArray[np.int64] | None = None,
) -> IMMResult:
    """Run the IMM over one voyage. Arguments are as for :func:`run_cv_filter`.

    ``snapshot_idx`` lists fix indices at which the per-mode posterior states and
    covariances are recorded (``snap_x``, ``snap_P``). Forecasting needs them for an exact
    mode-mixture extrapolation.

    **Outlier test: likelihood ratio against a clutter model.** Each fix is either from the
    target, with likelihood L = Σⱼ cⱼ N(ν; 0, Sⱼ) (the mode mixture, weighted by the
    predicted mode probabilities cⱼ), or an outlier spread uniformly with density λ
    (``clutter_density``). The fix is rejected when L < λ, i.e. when "outlier" is the more
    probable explanation (as in probabilistic data association).

    Two simpler gates were tried first and failed in the simulation study:

    * Gating on the moment-matched *combined* prediction was dominated by the most likely
      mode, and rejected the first fixes of every turn.
    * Accepting a fix inside *any* mode's χ² gate let a wide mode with ~1% probability
      admit kilometre-scale outliers on stationary vessels.

    If the fix is rejected, every mode coasts and the probabilities follow the Markov chain.
    The reported ``nis`` is the combined-prediction NIS, used for consistency checks.
    """
    n = len(t)
    if n == 0:
        raise ValueError("empty track")
    if np.any(np.diff(t) <= 0):
        raise ValueError("times must be strictly increasing")
    log_clutter = float(np.log(params.clutter_density))
    R = np.eye(2) * params.r_pos_m**2
    out_x, out_P = np.zeros((n, 5)), np.zeros((n, 5, 5))
    out_xp, out_Pp = np.zeros((n, 5)), np.zeros((n, 5, 5))
    out_F = np.tile(np.eye(5), (n, 1, 1))
    out_mu = np.zeros((n, 3))
    nis = np.full(n, np.nan)
    lls = np.full(n, np.nan)
    accepted = np.zeros(n, dtype=bool)
    reinit = np.zeros(n, dtype=bool)

    def vel_at(i: int) -> Array | None:
        return None if vel_meas is None else vel_meas[i]

    snap = np.zeros(0, dtype=np.int64) if snapshot_idx is None else np.asarray(snapshot_idx)
    snap_pos = {int(k): j for j, k in enumerate(snap)}
    snap_x = np.zeros((len(snap), 3, 5))
    snap_P = np.zeros((len(snap), 3, 5, 5))
    xs, Ps, mu = _init(z[0], vel_at(0), params)
    if 0 in snap_pos:
        snap_x[snap_pos[0]], snap_P[snap_pos[0]] = xs, Ps
    out_x[0], out_P[0] = mixture_moments(mu, xs, Ps)
    out_xp[0], out_Pp[0], out_mu[0] = out_x[0], out_P[0], mu
    accepted[0] = reinit[0] = True
    rejects = 0
    for i in range(1, n):
        dt = float(t[i] - t[i - 1])
        pi = transition_matrix(dt, params)
        c = mu @ pi  # predicted mode probabilities
        c = np.maximum(c, 1e-300)
        mix = (pi * mu[:, None]) / c[None, :]  # mix[i, j] = P(was i | now j)
        xps, Pps, Fs = np.zeros((3, 5)), np.zeros((3, 5, 5)), np.zeros((3, 5, 5))
        for j in range(3):
            x0, P0 = mixture_moments(mix[:, j], xs, Ps)
            xps[j], Pps[j], Fs[j] = predict_mode(j, x0, P0, dt, params)
        out_xp[i], out_Pp[i] = mixture_moments(c, xps, Pps)
        out_F[i] = np.einsum("m,mij->ij", c, Fs)  # effective linearisation for smoothing
        if params.skip_repeats and np.array_equal(z[i], z[i - 1]):
            # Repeated coordinate: no new measurement, so coast (see CVParams.skip_repeats).
            xs, Ps, mu = xps, Pps, c
            accepted[i] = True
            out_x[i], out_P[i] = mixture_moments(mu, xs, Ps)
            out_mu[i] = mu
            if i in snap_pos:
                snap_x[snap_pos[i]], snap_P[snap_pos[i]] = xs, Ps
            continue
        nu = z[i] - out_xp[i][:2]
        S = out_Pp[i][:2, :2] + R
        nis[i] = float(nu @ np.linalg.solve(S, nu))
        nu_m = z[i][None, :] - xps[:, :2]  # (3, 2)
        S_m = Pps[:, :2, :2] + R  # (3, 2, 2)
        S_inv = np.linalg.inv(S_m)
        nis_m = np.einsum("mi,mij,mj->m", nu_m, S_inv, nu_m)
        loglik = -0.5 * (nis_m + np.log(np.linalg.det(S_m)) + 2 * _LOG2PI)
        log_mix = float(np.logaddexp.reduce(np.log(c) + loglik))
        lls[i] = log_mix
        if log_mix >= log_clutter:
            for j in range(3):
                K = Pps[j] @ H5.T @ S_inv[j]
                xs[j] = xps[j] + K @ nu_m[j]
                IKH = np.eye(5) - K @ H5
                Ps[j] = IKH @ Pps[j] @ IKH.T + K @ R @ K.T
            w = np.log(c) + loglik
            w = np.exp(w - w.max())
            mu = w / w.sum()
            accepted[i] = True
            rejects = 0
        elif post_gap_restart(dt, params):
            # First fix after a long silence fails the outlier test: restart here (D21).
            v0 = vel_at(i)
            if v0 is not None and np.all(np.isfinite(v0)):
                xs, Ps, mu = _init(z[i], v0, params)
            else:
                xs, Ps, mu = _init(z[i], None, params)
                xs[:, 2:4] = out_xp[i][2:4]  # rough velocity guess, wide uncertainty
            reinit[i] = accepted[i] = True
            rejects = 0
        else:
            xs, Ps, mu = xps, Pps, c
            rejects += 1
            if rejects >= params.max_consecutive_rejects:
                xs, Ps, mu = _init(z[i], two_point_velocity(t, z, i), params)
                dt_v = max(float(t[i] - t[i - 1]), 1.0)
                Ps[:, 2, 2] = Ps[:, 3, 3] = 2 * params.r_pos_m**2 / dt_v**2 + 0.05
                reinit[i] = accepted[i] = True
                nis[i] = np.nan
                rejects = 0
        out_x[i], out_P[i] = mixture_moments(mu, xs, Ps)
        out_mu[i] = mu
        if i in snap_pos:
            snap_x[snap_pos[i]], snap_P[snap_pos[i]] = xs, Ps
    lls[0] = np.nan  # only the first fix has no prediction; restarts are still scored
    return IMMResult(
        t,
        out_x,
        out_P,
        out_xp,
        out_Pp,
        out_F,
        nis,
        accepted,
        reinit,
        loglik=lls,
        mu=out_mu,
        snap_idx=snap,
        snap_x=snap_x,
        snap_P=snap_P,
    )
