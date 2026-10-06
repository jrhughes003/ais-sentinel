"""ML sequence models: a GRU encoder with Gaussian (M1) or mixture-density (M2) heads.

Architecture:

* A GRU reads the 1-minute history sequence. Its final hidden state is concatenated with
  the static context vector.
* An MLP then outputs, for every horizon, the parameters of a 2-D distribution over the
  *correction to dead reckoning* (see :mod:`ais_sentinel.prediction.features`).
* **Gaussian head:** mean (2), log standard deviations (2), correlation ρ = 0.99·tanh(r).
* **Mixture density network (MDN) head:** K weighted Gaussians, so the model can say
  "either it turns into the channel or it continues straight" rather than averaging the
  two options.

The model is trained by minimising the **negative log-likelihood (NLL)** of the true
correction. Unlike plain squared error, NLL forces the predicted uncertainty to match the
errors the model actually makes. Targets are divided by a per-horizon scale (their training
standard deviation) so that all horizons contribute comparably to the loss.

Uncertainty from several random seeds (a **deep ensemble**) is combined by moment
matching. The ensemble covariance is the average predicted covariance (aleatoric) plus the
spread of the members' means (epistemic).
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from ais_sentinel.prediction.baselines import Forecast
from ais_sentinel.prediction.features import FeatureSet

log = logging.getLogger(__name__)
LOG2PI = math.log(2 * math.pi)


@dataclass(frozen=True)
class MLConfig:
    """Hyperparameters for the sequence models."""

    hidden: int = 64
    layers: int = 1
    mlp: int = 128
    components: int = 1  # 1 = Gaussian head (M1); >1 = MDN (M2)
    dropout: float = 0.1
    lr: float = 2e-3
    weight_decay: float = 1e-5
    batch: int = 512
    max_epochs: int = 30
    patience: int = 4
    seed: int = 0
    threads: int = 4


class SeqModel(nn.Module):
    """GRU encoder + context MLP → per-horizon (mixture of) bivariate Gaussians."""

    def __init__(self, n_seq: int, n_ctx: int, horizons: int, cfg: MLConfig) -> None:
        super().__init__()
        self.h, self.k = horizons, cfg.components
        self.gru = nn.GRU(n_seq, cfg.hidden, num_layers=cfg.layers, batch_first=True)
        per = 6 if self.k > 1 else 5  # (logit,) mu_e, mu_n, log_se, log_sn, rho
        self.head = nn.Sequential(
            nn.Linear(cfg.hidden + n_ctx, cfg.mlp),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.mlp, cfg.mlp),
            nn.GELU(),
            nn.Linear(cfg.mlp, horizons * self.k * per),
        )

    def forward(self, seq: torch.Tensor, ctx: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return distribution parameters, each shaped (B, H, K[, 2])."""
        _, hN = self.gru(seq)
        out = self.head(torch.cat([hN[-1], ctx], dim=-1))
        b = out.shape[0]
        per = 6 if self.k > 1 else 5
        out = out.view(b, self.h, self.k, per)
        if self.k > 1:
            logit, out = out[..., 0], out[..., 1:]
        else:
            logit = torch.zeros(b, self.h, 1, device=out.device)
        mu = out[..., 0:2]
        log_s = out[..., 2:4].clamp(-7.0, 5.0)
        rho = 0.99 * torch.tanh(out[..., 4])
        return {"logit": logit, "mu": mu, "log_s": log_s, "rho": rho}


def mixture_nll(p: dict[str, torch.Tensor], y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean NLL of y (B, H, 2) under the predicted mixture, over unmasked (B, H) entries."""
    z = (y.unsqueeze(2) - p["mu"]) / p["log_s"].exp()  # (B, H, K, 2)
    rho = p["rho"]
    one_m = (1 - rho**2).clamp_min(1e-4)
    q = (z[..., 0] ** 2 + z[..., 1] ** 2 - 2 * rho * z[..., 0] * z[..., 1]) / one_m
    log_n = -LOG2PI - p["log_s"].sum(-1) - 0.5 * torch.log(one_m) - 0.5 * q
    log_pi = torch.log_softmax(p["logit"], dim=-1)
    ll = torch.logsumexp(log_pi + log_n, dim=-1)  # (B, H)
    m = mask.float()
    return -(ll * m).sum() / m.sum().clamp_min(1.0)


def _moments(p: dict[str, torch.Tensor]) -> tuple[np.ndarray, np.ndarray]:
    """Moment-matched mean (B, H, 2) and covariance (B, H, 2, 2) of the mixture."""
    w = torch.softmax(p["logit"], dim=-1)  # (B, H, K)
    s = p["log_s"].exp()
    cov = torch.zeros(*s.shape[:-1], 2, 2)
    cov[..., 0, 0] = s[..., 0] ** 2
    cov[..., 1, 1] = s[..., 1] ** 2
    cov[..., 0, 1] = cov[..., 1, 0] = p["rho"] * s[..., 0] * s[..., 1]
    mean = (w.unsqueeze(-1) * p["mu"]).sum(2)
    d = p["mu"] - mean.unsqueeze(2)
    c = (w[..., None, None] * (cov + d.unsqueeze(-1) * d.unsqueeze(-2))).sum(2)
    return mean.numpy(), c.numpy()


@dataclass
class TrainedModel:
    """A fitted model plus the target scales needed to undo normalisation."""

    model: SeqModel
    scale: NDArray[np.float32]  # (H,) km
    cfg: MLConfig
    history: list[dict[str, float]]

    @torch.no_grad()
    def predict(self, fs: FeatureSet, batch: int = 4096) -> tuple[np.ndarray, np.ndarray]:
        """Predicted correction mean (M, H, 2) km and covariance (M, H, 2, 2) km²."""
        self.model.eval()
        means, covs = [], []
        for i in range(0, len(fs.sample_id), batch):
            p = self.model(
                torch.from_numpy(fs.seq[i : i + batch]), torch.from_numpy(fs.ctx[i : i + batch])
            )
            m, c = _moments(p)
            means.append(m)
            covs.append(c)
        s = self.scale[None, :, None]
        return np.concatenate(means) * s, np.concatenate(covs) * (s[..., None] ** 2)


def _tensors(fs: FeatureSet, scale: np.ndarray) -> tuple[torch.Tensor, ...]:
    y = fs.y / scale[None, :, None]
    return (
        torch.from_numpy(fs.seq),
        torch.from_numpy(fs.ctx),
        torch.from_numpy(y.astype(np.float32)),
        torch.from_numpy(fs.ymask),
    )


def train_model(train: FeatureSet, val: FeatureSet, cfg: MLConfig) -> TrainedModel:
    """Train with Adam and early stopping on validation NLL. Fully seeded."""
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(cfg.threads)
    scale = np.ones(train.y.shape[1], dtype=np.float32)
    for k in range(train.y.shape[1]):
        vals = train.y[train.ymask[:, k], k]
        if vals.size >= 2:  # horizons with no training truth keep scale 1 (and are masked)
            scale[k] = max(float(np.std(vals)), 1e-3)
    seq, ctx, y, mask = _tensors(train, scale)
    vseq, vctx, vy, vmask = _tensors(val, scale)
    model = SeqModel(seq.shape[-1], ctx.shape[-1], y.shape[1], cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=1)
    gen = torch.Generator().manual_seed(cfg.seed)
    best, best_state, bad = math.inf, None, 0
    history: list[dict[str, float]] = []
    n = seq.shape[0]
    for epoch in range(cfg.max_epochs):
        t0 = time.perf_counter()
        model.train()
        perm = torch.randperm(n, generator=gen)
        total = 0.0
        for i in range(0, n, cfg.batch):
            b = perm[i : i + cfg.batch]
            loss = mixture_nll(model(seq[b], ctx[b]), y[b], mask[b])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss.detach()) * len(b)
        model.eval()
        with torch.no_grad():
            vl = sum(
                float(
                    mixture_nll(
                        model(vseq[i : i + 4096], vctx[i : i + 4096]),
                        vy[i : i + 4096],
                        vmask[i : i + 4096],
                    )
                )
                * len(vseq[i : i + 4096])
                for i in range(0, vseq.shape[0], 4096)
            ) / max(vseq.shape[0], 1)
        sched.step(vl)
        history.append(
            {"epoch": epoch, "train_nll": total / n, "val_nll": vl, "sec": time.perf_counter() - t0}
        )
        log.info("epoch %d train %.4f val %.4f (%.0fs)", epoch, total / n, vl, history[-1]["sec"])
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return TrainedModel(model, scale, cfg, history)


def ensemble_forecast(
    models: list[TrainedModel],
    fs: FeatureSet,
    lat0: np.ndarray,
    lon0: np.ndarray,
    horizons_min: list[int],
    name: str,
) -> Forecast:
    """Combine ensemble members by moment matching into a :class:`Forecast` (metres)."""
    means, covs = zip(*(m.predict(fs) for m in models), strict=True)
    mu = np.mean(means, axis=0)
    d = np.stack(means) - mu
    cov = np.mean(covs, axis=0) + np.einsum("emhi,emhj->mhij", d, d) / len(models)
    mean_en = (fs.dr.astype(float) + mu) * 1000.0
    return Forecast(name, list(fs.sample_id), list(horizons_min), lat0, lon0, mean_en, cov * 1e6)


def n_params(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def state_summary(tm: TrainedModel) -> dict[str, Any]:
    """Small JSON-able summary of a trained model."""
    return {
        "params": n_params(tm.model),
        "epochs": len(tm.history),
        "best_val_nll": min(h["val_nll"] for h in tm.history),
        "cfg": tm.cfg.__dict__,
    }
