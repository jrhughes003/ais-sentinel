"""Feature causality, loss correctness and training sanity for the sequence models."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import torch
from scipy.stats import multivariate_normal

from ais_sentinel.prediction.features import build_features
from ais_sentinel.prediction.ml import MLConfig, ensemble_forecast, mixture_nll, train_model
from ais_sentinel.prediction.samples import build_samples
from tests.test_knn import REF, l_route_voyage

H = [15, 30]


def _meta(ids: list[str]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "voyage_id": ids,
            "split": ["train"] * len(ids),
            "vessel_group": ["cargo"] * len(ids),
            "length_m": [150.0] * len(ids),
        }
    )


def test_features_are_causal() -> None:
    rng = np.random.default_rng(0)
    v = l_route_voyage("a", datetime(2023, 6, 1, tzinfo=UTC), 10.0, rng)
    s = build_samples(v, _meta(["a"]), H, 30, 10)
    f1 = build_features(v, s, H, 30, *REF)
    t_cut = s["t0"][0]
    v2 = v.with_columns(  # scramble everything after the first anchor
        lat=pl.when(pl.col("t") > t_cut).then(pl.col("lat") + 0.1).otherwise(pl.col("lat"))
    )
    f2 = build_features(v2, s, H, 30, *REF)
    np.testing.assert_array_equal(f1.seq[0], f2.seq[0])
    np.testing.assert_array_equal(f1.ctx[0], f2.ctx[0])


def test_gaussian_nll_matches_scipy() -> None:
    mu = torch.tensor([[[[0.3, -0.2]]]])
    log_s = torch.tensor([[[[np.log(0.5), np.log(1.5)]]]], dtype=torch.float32)
    rho = torch.tensor([[[0.4]]])
    p = {"logit": torch.zeros(1, 1, 1), "mu": mu, "log_s": log_s, "rho": rho}
    y = torch.tensor([[[1.0, 0.5]]])
    got = float(mixture_nll(p, y, torch.ones(1, 1, dtype=torch.bool)))
    cov = np.array([[0.25, 0.4 * 0.5 * 1.5], [0.4 * 0.5 * 1.5, 2.25]])
    want = -multivariate_normal(mean=[0.3, -0.2], cov=cov).logpdf([1.0, 0.5])
    assert abs(got - want) < 1e-4


def _route_features(n_voyages: int, seed: int):  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(seed)
    t0 = datetime(2023, 6, 1, tzinfo=UTC)
    vs = [
        l_route_voyage(f"v{seed}_{i}", t0 + timedelta(hours=3 * i), rng.uniform(8, 12), rng)
        for i in range(n_voyages)
    ]
    tracks = pl.concat(vs)
    s = build_samples(tracks, _meta([f"v{seed}_{i}" for i in range(n_voyages)]), H, 20, 2)
    return s, build_features(tracks, s, H, 20, *REF)


def test_training_is_seeded_and_learns_the_turn() -> None:
    _, tr = _route_features(10, 0)
    sv, va = _route_features(3, 1)
    cfg = MLConfig(hidden=16, mlp=32, max_epochs=15, patience=15, batch=64, threads=1)
    a = train_model(tr, va, cfg)
    b = train_model(tr, va, cfg)
    assert a.history[-1]["val_nll"] == b.history[-1]["val_nll"]  # deterministic
    assert a.history[-1]["train_nll"] < a.history[0]["train_nll"]
    fc = ensemble_forecast([a], va, sv["lat0"].to_numpy(), sv["lon0"].to_numpy(), H, "gru")
    true_e = (va.y + va.dr)[..., :] * 1000
    err_ml = np.linalg.norm(fc.mean_en - true_e, axis=-1)[va.ymask]
    err_dr = np.linalg.norm(va.dr * 1000 - true_e, axis=-1)[va.ymask]
    assert err_ml.mean() < 0.7 * err_dr.mean()
