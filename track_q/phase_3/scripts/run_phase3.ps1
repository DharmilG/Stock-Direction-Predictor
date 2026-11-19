# Phase 3 (thinking-cap ablation) runner.
# Resumable: re-running continues from (case_id, thinking_budget) pairs already
# in the ledger. Fault tolerance: durable fsync'd ledger appends, quarantined
# corrupt lines, config fingerprint guard, and a live heartbeat.
# Run from the repo root:  powershell -NoProfile -ExecutionPolicy Bypass -File .\track_q\phase_3\scripts\run_phase3.ps1
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location $repo
$py = Join-Path $repo ".venv\Scripts\python.exe"
& $py -m stockml.qpipeline --phase phase3
exit $LASTEXITCODE
