"""Export the M1 GRU ensemble to ONNX so the website can run it in the browser.

Writes to ``<site_data>/../model/``:

* ``gru_<k>.onnx``: one file per ensemble member, with inputs ``seq`` (B, T, 6) and
  ``ctx`` (B, C) and outputs ``mu``, ``log_s``, ``rho`` and ``logit`` (see
  :class:`~ais_sentinel.prediction.ml.SeqModel`);
* ``model.json``: what the browser needs to rebuild features and decode outputs. That is
  horizons, history length, per-member target scales, the validation covariance-calibration
  factors, the regional reference point and the vessel-group order. It also holds **parity
  fixtures**: raw inputs plus Python features and outputs for a few real test samples, which
  the site's test suite recomputes in the browser and compares.

The browser feature code is ``site/src/ml/features.ts`` and mirrors
:func:`ais_sentinel.prediction.features.build_features`. Change both together.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch

from ais_sentinel.config import Config
from ais_sentinel.data.vessel_types import COMMERCIAL, GROUPS
from ais_sentinel.io import write_text
from ais_sentinel.prediction.features import SEQ_CHANNELS, build_features
from ais_sentinel.prediction.knn import region_centre
from ais_sentinel.prediction.ml import ensemble_forecast
from ais_sentinel.prediction.run import load_models

log = logging.getLogger(__name__)
PARITY_TOL = 1e-4


class _Wrapper(torch.nn.Module):
    """Expose the model's distribution parameters as plain tensor outputs."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, seq: torch.Tensor, ctx: torch.Tensor) -> tuple[torch.Tensor, ...]:
        p = self.model(seq, ctx)
        return p["mu"], p["log_s"], p["rho"], p["logit"]


def _check_parity(wrapper: _Wrapper, path: Path, seq: np.ndarray, ctx: np.ndarray) -> float:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path))
    got = sess.run(None, {"seq": seq, "ctx": ctx})
    with torch.no_grad():
        want = [t.numpy() for t in wrapper(torch.from_numpy(seq), torch.from_numpy(ctx))]
    return max(float(np.abs(a - b).max()) for a, b in zip(got, want, strict=True))


def export_onnx(cfg: Config, n_fixtures: int = 3) -> int:
    """Export the Gaussian GRU ensemble plus metadata. Returns bytes written (0 if no model)."""
    model_dir = Path(cfg.paths.get("models", "models"))
    if not (model_dir / "gru.pt").exists():
        log.info("onnx: no trained gru.pt; skipping")
        return 0
    base = Path(cfg.paths.processed)
    split = "test" if (base / "samples_test.parquet").exists() else "val"
    samples = pl.read_parquet(base / f"samples_{split}.parquet")
    tracks = pl.read_parquet(base / "tracks.parquet")
    H = list(cfg.prediction.horizons_min)
    hist = int(cfg.prediction.history_min)
    ref = region_centre(dict(cfg.region))

    pool = samples.filter(pl.col("vessel_group").is_in(sorted(COMMERCIAL)))
    rng = np.random.default_rng(int(cfg.seed))
    fx = pool[
        sorted(rng.choice(pool.height, size=min(n_fixtures, pool.height), replace=False).tolist())
    ]
    fs = build_features(tracks, fx, H, hist, *ref)
    members = load_models(cfg, "gru", fs.seq.shape[-1], fs.ctx.shape[-1], len(H))

    out_dir = Path(cfg.paths.site_data).parent / "model"
    out_dir.mkdir(parents=True, exist_ok=True)
    size = 0
    for k, m in enumerate(members):
        # eval() on the *wrapper*: export restores the caller's training flag recursively, so
        # a wrapper left in training mode would switch dropout back on for the parity check.
        wrapper = _Wrapper(m.model).eval()
        path = out_dir / f"gru_{k}.onnx"
        torch.onnx.export(
            wrapper,
            (torch.from_numpy(fs.seq[:1]), torch.from_numpy(fs.ctx[:1])),
            str(path),
            input_names=["seq", "ctx"],
            output_names=["mu", "log_s", "rho", "logit"],
            dynamic_axes={"seq": {0: "batch"}, "ctx": {0: "batch"}},
            opset_version=17,
            dynamo=False,
        )
        wrapper.eval()
        err = _check_parity(wrapper, path, fs.seq, fs.ctx)
        if err > PARITY_TOL:
            raise RuntimeError(f"ONNX/PyTorch mismatch {err:.2e} for member {k}")
        size += path.stat().st_size

    choices = json.loads((model_dir / "choices.json").read_text(encoding="utf-8"))
    cal = {int(h): float(v) for h, v in choices["cov_scales"]["gru"].items()}
    fc = ensemble_forecast(members, fs, fx["lat0"].to_numpy(), fx["lon0"].to_numpy(), H, "gru")
    fc = fc.with_cov_scale(cal)
    lat, lon = fc.latlon()

    by_voyage = {v["voyage_id"][0]: v.sort("t") for v in tracks.partition_by("voyage_id")}
    fixtures: list[dict[str, Any]] = []
    for i, s in enumerate(fx.iter_rows(named=True)):
        v = by_voyage[s["voyage_id"]]
        w = v.filter(
            # Voyages never contain gaps > 30 min, so hist + 35 min always includes the fix
            # that brackets the start of the history window (needed for identical features).
            (pl.col("t") <= s["t0"]) & (pl.col("t") >= s["t0"] - pl.duration(minutes=hist + 35))
        )
        fixtures.append(
            {
                "raw": {
                    "t": w["t"].dt.epoch("s").to_list(),
                    "lat": w["lat"].to_list(),
                    "lon": w["lon"].to_list(),
                    "sog": w["sog_kn"].cast(pl.Float64).to_list(),
                    "cog": w["cog_deg"].cast(pl.Float64).to_list(),
                },
                "t0": int(s["t0"].timestamp()),
                "lat0": float(s["lat0"]),
                "lon0": float(s["lon0"]),
                "sog0": None if s["sog_kn"] is None else float(s["sog_kn"]),
                "cog0": None if s["cog_deg"] is None else float(s["cog_deg"]),
                "length_m": None if s["length_m"] is None else float(s["length_m"]),
                "group": s["vessel_group"],
                "seq": fs.seq[i].round(6).tolist(),
                "ctx": fs.ctx[i].round(6).tolist(),
                "pred_lat": lat[i].tolist(),
                "pred_lon": lon[i].tolist(),
                "pred_cov": fc.cov[i].round(1).tolist(),
            }
        )
    meta = {
        "model": "M1 GRU (Gaussian ensemble)",
        "members": [f"gru_{k}.onnx" for k in range(len(members))],
        "horizons_min": H,
        "history_min": hist,
        "seq_channels": list(SEQ_CHANNELS),
        "groups": list(GROUPS),
        "ref_lat": ref[0],
        "ref_lon": ref[1],
        "scales": [m.scale.astype(float).tolist() for m in members],
        "cov_calibration": [cal.get(h, 1.0) for h in H],
        "fixtures": fixtures,
    }
    text = json.dumps(meta, separators=(",", ":"))
    size += write_text(out_dir / "model.json", text)
    log.info("onnx: %d members, %.0f KB", len(members), size / 1024)
    return size
