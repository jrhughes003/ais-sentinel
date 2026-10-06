"""Simulate vessel trajectories with known ground truth and AIS-like measurements.

Real AIS has no ground truth, because the reported position *is* the measurement. To
validate a tracker we therefore simulate vessels whose true path we know, then degrade
it the way MarineCadastre data is degraded (PLAN §6.1):

* **Truth:** a sequence of manoeuvres (straight legs, coordinated turns, speed changes, and
  stops with dwell), integrated at 1 s. Every second carries a segment label, so errors
  can be broken down by "straight" vs "manoeuvre".
* **Reporting times:** a nominal 60 s interval (MarineCadastre's 1-minute downsampling)
  with ±5 s jitter; 180 s while stopped (the AIS 3-minute rate); random dropouts; and
  occasional multi-minute bursts of missing reports.
* **Noise:** GPS position noise of σ = 5 m per axis (≈ 10 m at 95%), plus rare gross
  outliers of 0.2–5 km, as when two vessels share an MMSI or a GPS glitch occurs.
* **SOG/COG:** reported with small noise, for filters that use velocity measurements.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from ais_sentinel.geo import KNOT_MS

Array = NDArray[np.float64]

SEGMENTS = ("straight", "turn", "speed", "stopped")


@dataclass(frozen=True)
class SimConfig:
    """Parameters of the trajectory and measurement simulator."""

    duration_s: float = 4 * 3600.0
    speed_kn: tuple[float, float] = (6.0, 16.0)
    straight_s: tuple[float, float] = (180.0, 1200.0)
    turn_rate_deg_s: tuple[float, float] = (0.05, 1.0)
    turn_angle_deg: tuple[float, float] = (20.0, 120.0)
    accel_ms2: tuple[float, float] = (0.01, 0.05)
    stop_dwell_s: tuple[float, float] = (600.0, 2400.0)
    p_turn: float = 0.45
    p_speed: float = 0.2
    p_stop: float = 0.07
    report_s: float = 60.0
    report_jitter_s: float = 5.0
    stopped_report_s: float = 180.0
    p_drop: float = 0.05
    p_burst: float = 0.01
    burst_s: tuple[float, float] = (120.0, 1200.0)
    pos_sigma_m: float = 5.0
    p_outlier: float = 0.005
    outlier_m: tuple[float, float] = (200.0, 5000.0)
    sog_sigma_kn: float = 0.1
    cog_sigma_deg: float = 2.0


@dataclass
class SimTrack:
    """A simulated voyage: truth at report times, measurements and labels."""

    t: Array  # (N,) report times, s
    z: Array  # (N, 2) measured ENU position, m
    truth: Array  # (N, 4) true [e, n, ve, vn] at report times
    omega: Array  # (N,) true turn rate, rad/s
    segment: NDArray[np.str_]  # (N,) segment label
    outlier: NDArray[np.bool_]  # (N,) gross outlier injected
    sog_kn: Array  # (N,) reported SOG
    cog_deg: Array  # (N,) reported COG
    meta: dict[str, float] = field(default_factory=dict)


def _truth_1hz(rng: np.random.Generator, c: SimConfig) -> tuple[Array, Array, NDArray[np.str_]]:
    """Integrate a random manoeuvre sequence at 1 s. Returns (state (T,4), omega (T,), labels)."""
    n = int(c.duration_s) + 1
    state = np.zeros((n, 4))
    omega = np.zeros(n)
    label = np.empty(n, dtype="<U8")
    speed = rng.uniform(*c.speed_kn) * KNOT_MS
    heading = rng.uniform(0, 2 * np.pi)  # math convention: radians CCW from east
    e = nn = 0.0
    i = 0
    cruise = speed

    def emit(seconds: int, w: float, accel: float, lab: str) -> None:
        nonlocal i, e, nn, speed, heading
        for _ in range(seconds):
            if i >= n:
                return
            state[i] = [e, nn, speed * np.cos(heading), speed * np.sin(heading)]
            omega[i] = w
            label[i] = lab
            e += speed * np.cos(heading)
            nn += speed * np.sin(heading)
            heading += w
            speed = max(0.0, speed + accel)
            i += 1

    while i < n:
        u = rng.uniform()
        if u < c.p_stop and speed > 0:
            # Decelerate, dwell, accelerate back to cruise.
            a = rng.uniform(*c.accel_ms2)
            emit(int(speed / a), 0.0, -a, "speed")
            speed = 0.0
            emit(int(rng.uniform(*c.stop_dwell_s)), 0.0, 0.0, "stopped")
            emit(int(cruise / a), 0.0, a, "speed")
            speed = cruise
        elif u < c.p_stop + c.p_turn:
            rate = np.radians(rng.uniform(*c.turn_rate_deg_s)) * rng.choice([-1.0, 1.0])
            angle = np.radians(rng.uniform(*c.turn_angle_deg))
            emit(max(1, int(angle / abs(rate))), rate, 0.0, "turn")
        elif u < c.p_stop + c.p_turn + c.p_speed:
            target = rng.uniform(*c.speed_kn) * KNOT_MS
            a = rng.uniform(*c.accel_ms2) * np.sign(target - speed)
            emit(max(1, int(abs(target - speed) / max(abs(a), 1e-9))), 0.0, a, "speed")
            cruise = speed
        emit(int(rng.uniform(*c.straight_s)), 0.0, 0.0, "straight" if speed > 0 else "stopped")
    return state, omega, label


def _report_times(rng: np.random.Generator, c: SimConfig, label: NDArray[np.str_]) -> Array:
    """Irregular AIS-like report times (integer seconds within the truth span)."""
    times: list[float] = []
    t = 0.0
    end = len(label) - 1
    while t <= end:
        stopped = label[int(t)] == "stopped"
        step = c.stopped_report_s if stopped else c.report_s
        if rng.uniform() < c.p_burst:
            t += rng.uniform(*c.burst_s)
            continue
        if rng.uniform() >= c.p_drop:
            times.append(t)
        t += step + rng.uniform(-c.report_jitter_s, c.report_jitter_s)
    return np.unique(np.round(np.asarray(times)))


def simulate_track(seed: int, c: SimConfig | None = None) -> SimTrack:
    """Simulate one voyage with a reproducible seed."""
    c = c or SimConfig()
    rng = np.random.default_rng(seed)
    state, omega, label = _truth_1hz(rng, c)
    t = _report_times(rng, c, label)
    idx = t.astype(int)
    truth = state[idx]
    z = truth[:, :2] + rng.normal(0, c.pos_sigma_m, (len(t), 2))
    outlier = rng.uniform(size=len(t)) < c.p_outlier
    outlier[0] = False  # keep initialisation clean; robustness to a bad first fix is separate
    k = int(outlier.sum())
    if k:
        ang = rng.uniform(0, 2 * np.pi, k)
        dist = rng.uniform(*c.outlier_m, k)
        z[outlier] += np.column_stack([dist * np.cos(ang), dist * np.sin(ang)])
    speed = np.hypot(truth[:, 2], truth[:, 3]) / KNOT_MS
    sog = np.maximum(0.0, speed + rng.normal(0, c.sog_sigma_kn, len(t)))
    cog = (
        np.degrees(np.arctan2(truth[:, 2], truth[:, 3])) + rng.normal(0, c.cog_sigma_deg, len(t))
    ) % 360
    return SimTrack(
        t=t,
        z=z,
        truth=truth,
        omega=omega[idx],
        segment=label[idx],
        outlier=outlier,
        sog_kn=sog,
        cog_deg=cog,
        meta={"seed": float(seed)},
    )
