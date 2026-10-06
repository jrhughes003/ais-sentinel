# Overnight v2 run (DECISIONS D21). Run from the repo root:
#   powershell -ExecutionPolicy Bypass -File scripts\run_v2.ps1
# Expect roughly 4.5-7 hours on a 4-core laptop. Logs go to the logs folder (v2_*.log). Site
# export is done afterwards, so the published v1 site data is not overwritten.
$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
New-Item -ItemType Directory -Force logs | Out-Null
$exe = ".venv\Scripts\ais-sentinel.exe"

Write-Host "== attempt-4 real-data NIS (validation)"
& .venv\Scripts\python.exe scripts\nis_val.py *> logs\v2_nis.log
if ($LASTEXITCODE -ne 0) { throw "nis_val failed; see logs\v2_nis.log" }

foreach ($st in "build", "track", "evaluate", "anomaly", "holdout") {
    Write-Host "== $st  $(Get-Date -Format HH:mm:ss)"
    & $exe $st --config configs\v2.yaml *> "logs\v2_$st.log"
    if ($LASTEXITCODE -ne 0) { throw "stage $st failed; see logs\v2_$st.log" }
}
Write-Host "== done $(Get-Date -Format HH:mm:ss)"
