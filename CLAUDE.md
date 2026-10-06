# CLAUDE.md — ais-sentinel

Read **PROGRESS.md** first ("Current state" and "Next up"), then PLAN.md (§9 lists the
success criteria) and DECISIONS.md.

## Project
A maritime domain awareness system built on MarineCadastre AIS data (CC0) for the
Detroit–St. Clair corridor, 2023-05-01 to 2023-10-31. It has three pillars:
1. Kalman/IMM tracking.
2. Multi-horizon trajectory prediction with uncertainty: physics/kNN baselines versus a GRU.
3. Rule-based anomaly detection, evaluated by synthetic injection.

A static Vite + TypeScript + MapLibre site sits in `site/` and is deployed to GitHub
Pages. The repo is public: https://github.com/jrhughes003/ais-sentinel

## Owner
- Intermediate Python (pandas, functions). Engineering statistics plus predictive and
  intelligent control.
- Knows Kalman filters, state-space models and Bayesian reasoning. **Explain anything
  beyond that in the docs** (IMM mixing, MDNs, conformal calibration, bootstrap CIs, and so on).
- Windows 11, VS Code, PowerShell. Use `py -3.12` (bare `python` is 3.10).
- Works in short sittings, so keep PROGRESS.md resumable.
- Machine: **CPU only** (i7-1165G7, 16 GB RAM, no CUDA) and **about 32 GB free disk**.
  Never keep national AIS files; stream-filter them and delete.

## Commands (Windows PowerShell; on Linux/macOS use `.venv/bin/` instead of `.venv\Scripts\`)
```powershell
py -3.12 -m venv .venv; .venv\Scripts\python -m pip install -e ".[ml,dev]"
.venv\Scripts\ruff check . ; .venv\Scripts\ruff format .
.venv\Scripts\mypy
.venv\Scripts\pytest -m "not network and not slow"
.venv\Scripts\ais-sentinel run-all --config configs/default.yaml
cd site; npm ci; npm run dev     # npm run build; npx playwright test
```

## Conventions
- src layout: `src/ais_sentinel/{data,tracking,prediction,anomaly,evaluation,export}`.
- Every function has type hints and a docstring.
- **Units:** SI internally (m, s, m/s, rad). Knots and degrees appear only at I/O
  boundaries, and converted variables carry suffixes (`sog_kn`, `cog_deg`).
- **Time:** timezone-aware UTC everywhere (`datetime64[ns, UTC]` or int64 epoch seconds).
  Never use naive local times.
- **Coordinates:** WGS84 lat/lon, converted to a local ENU frame through
  `ais_sentinel.geo`, using exact formulas via ECEF. Never use degrees as distances.
  Errors are reported as haversine km.
- **Config:** values live in `configs/*.yaml`, never hard-coded thresholds. Seed from
  config.
- **Data:**
  - `data/raw` (transient), `data/interim`, `data/processed` — all gitignored.
  - Site extracts live in `site/public/data/` and must stay small (≤ 50 MB total), with
    attribution.
- Commit after every working step, and push (the remote exists). Update PROGRESS.md as
  you go.

## Guardrails
- Never modify anything outside this folder.
- With `gh`, touch only this repo. Never change visibility, force-push or rewrite history.
- **Evaluation rules:**
  - Splits are by time with buffers, and voyages are disjoint across splits.
  - The **locked test** (October 2023) is run once per completed model version and logged
    in `reports/holdout_runs.md`.
  - Tune only on validation.
- Every ML model is compared with all baselines on the same samples, horizons and metrics,
  with voyage-cluster bootstrap CIs.
- Treat results that look too good as a bug until disproved.
- Targets in PLAN.md §9 may only change with a logged reason in DECISIONS.md.
- **Privacy:** the site showcases commercial vessels (cargo, tanker, passenger, tug,
  fishing). Avoid featuring pleasure craft.
- No paid services, no accounts, no committed secrets (use a gitignored `.env`).
- Rate-limit downloads, one file at a time.
- Stuck three times on the same problem? Log it under "Blocked" in PROGRESS.md and move on.
