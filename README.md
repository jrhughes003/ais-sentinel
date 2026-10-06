# ais-sentinel

**Maritime domain awareness from public AIS data.** This project turns noisy vessel position
reports into smooth tracks with uncertainty, forecasts where each vessel will be in 15–120
minutes, and flags behaviour an analyst would want to review: going dark, impossible jumps,
loitering, route deviation and rendezvous.

> 🚧 Work in progress. See [PROGRESS.md](PROGRESS.md) for current status and [PLAN.md](PLAN.md) for
> the design, evaluation protocol and success criteria.

- **Live site:** https://jrhughes003.github.io/ais-sentinel/
- **Region:** Detroit–St. Clair corridor (Port Huron → western Lake Erie), May–Oct 2023.
- **Data:** MarineCadastre.gov AIS (USCG / NOAA / BOEM), CC0.

## Quick start (Windows PowerShell)

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install -e ".[ml,dev]"
.venv\Scripts\pytest
```

On Linux/macOS, use `python3.12 -m venv .venv` and `.venv/bin/...`.

## Data attribution

AIS data: U.S. Coast Guard Navigation Center, via NOAA Office for Coastal Management and BOEM,
[MarineCadastre.gov](https://marinecadastre.gov/) "Nationwide Automatic Identification System",
2023, released under CC0 1.0. The data are provided "as is" and are not for navigation.

## Licence

Code: [MIT](LICENSE).
