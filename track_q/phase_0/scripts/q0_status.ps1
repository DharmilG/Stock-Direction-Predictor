# Q0-D1 status view (read-only, no administrator rights required)
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\q0_status.ps1
#
# Shows the scheduled task state, supervisor/runner heartbeats, ledger count,
# GPU temperature and clock, and any alert/done markers.

$script:RepoRoot = $PSScriptRoot
if (-not $script:RepoRoot) { $script:RepoRoot = (Get-Location).Path }
Set-Location $script:RepoRoot

$logDir = Join-Path $script:RepoRoot "track_q\phase_0\logs"
$supervisorHeartbeat = Join-Path $logDir "q0_supervisor.json"
$runnerHeartbeat = Join-Path $logDir "q0_heartbeat.json"
$alertPath = Join-Path $logDir "q0_ALERT.txt"
$donePath = Join-Path $logDir "q0_DONE.txt"
$setupResult = Join-Path $logDir "q0_SETUP_RESULT.txt"
$verdictPath = Join-Path $script:RepoRoot "track_q\phase_0\verdict\q0_d1.json"
$ledgerPath = Join-Path $script:RepoRoot "track_q\phase_0\state\qwen_rating_ledger.parquet"
$python = Join-Path $script:RepoRoot ".venv\Scripts\python.exe"
$taskName = "StockML_Q0_Supervisor"

function Get-HeartbeatAge {
    param([string]$Path)
    if (-not (Test-Path $Path)) { return $null }
    try {
        $payload = Get-Content $Path -Raw | ConvertFrom-Json
        if ($null -eq $payload.updated_at) { return $null }
        return [math]::Round(((Get-Date) - [datetime]$payload.updated_at).TotalSeconds, 0)
    } catch {
        return $null
    }
}

function Write-KeyValue {
    param([string]$Label, [object]$Value)
    if ($null -eq $Value) { $Value = "-" }
    Write-Host ("{0,-22} {1}" -f $Label, $Value)
}

Write-Host "=== Q0-D1 status ==="
Write-Host ("Time: " + (Get-Date).ToString("yyyy-MM-dd HH:mm:ss"))

Write-Host ""
Write-Host "== Scheduled task =="
try {
    $task = Get-ScheduledTask -TaskName $taskName -ErrorAction Stop
    $info = Get-ScheduledTaskInfo -TaskName $taskName -ErrorAction SilentlyContinue
    Write-KeyValue "Task state" $task.State
    if ($null -ne $info) {
        Write-KeyValue "Last run" $info.LastRunTime
        Write-KeyValue "Last result" $info.LastTaskResult
    }
} catch {
    Write-Host "Task '$taskName' not found (manual supervisor mode)."
}

Write-Host ""
Write-Host "== Heartbeats =="
$supervisorAge = Get-HeartbeatAge -Path $supervisorHeartbeat
$runnerAge = Get-HeartbeatAge -Path $runnerHeartbeat
if ($null -ne $supervisorAge) {
    $stale = ""
    if ($supervisorAge -gt 120) { $stale = "  [STALE]" }
    Write-KeyValue "Supervisor age" "$supervisorAge s$stale"
} else {
    Write-KeyValue "Supervisor age" "missing"
}
if ($null -ne $runnerAge) {
    $note = ""
    if ($runnerAge -gt 600) { $note = "  [old - likely resting; see supervisor state]" }
    Write-KeyValue "Runner age" "$runnerAge s$note"
} else {
    Write-KeyValue "Runner age" "missing"
}

if (Test-Path $supervisorHeartbeat) {
    try {
        $supervisor = Get-Content $supervisorHeartbeat -Raw | ConvertFrom-Json
        Write-KeyValue "Supervisor state" $supervisor.state
        Write-KeyValue "Supervisor pid" $supervisor.supervisor_pid
        $childText = "$($supervisor.child_pid) (alive: $($supervisor.child_alive))"
        Write-KeyValue "Child pid" $childText
        Write-KeyValue "Temperature" "$($supervisor.temperature_c) C"
        $clockText = "$($supervisor.clock_mhz) MHz (cap $($supervisor.clock_cap_mhz), applied $($supervisor.applied_clock_mhz))"
        Write-KeyValue "GPU clock" $clockText
        if ($null -ne $supervisor.runner) {
            $runner = $supervisor.runner
            Write-KeyValue "Runner phase" $runner.phase
            Write-KeyValue "Current case" $runner.current_case
            Write-KeyValue "Last case" $runner.last_case_id
            Write-KeyValue "Ledger" "$($runner.ledger_total) / $($runner.pool_total)"
            Write-KeyValue "Rated / judged" "$($runner.rated_total) / $($runner.judged_total)"
            Write-KeyValue "Harness fail rate" $runner.harness_fail_rate
            Write-KeyValue "Avg case seconds" $runner.avg_case_seconds
            Write-KeyValue "ETA hours" $runner.eta_hours
            Write-KeyValue "Work since break" "$($runner.work_seconds) s / $($runner.proactive_target_seconds) s"
            Write-KeyValue "Thermal state" $runner.thermal_state
            Write-KeyValue "Max temp" "$($runner.max_temperature_c) C"
            $thermalText = "rests=$($runner.rests) hard_trips=$($runner.hard_trips) cooldowns=$($runner.cooldowns) proactive=$($runner.proactive_breaks) aborts=$($runner.aborts)"
            Write-KeyValue "Thermal counters" $thermalText
            Write-KeyValue "Last checkpoint" $runner.last_checkpoint_at
        }
    } catch {
        Write-Host "Could not parse supervisor heartbeat: $_"
    }
}

Write-Host ""
Write-Host "== Ledger =="
if (Test-Path $ledgerPath) {
    try {
        $count = & $python -c "import pandas as pd; print(len(pd.read_parquet('track_q/phase_0/state/qwen_rating_ledger.parquet')))" 2>$null
        Write-KeyValue "Ledger rows" $count
    } catch {
        Write-Host "Ledger read failed (is the Python venv present?)."
    }
} else {
    Write-Host "Ledger not found."
}

Write-Host ""
Write-Host "== Verdict =="
if (Test-Path $verdictPath) {
    try {
        $verdict = Get-Content $verdictPath -Raw | ConvertFrom-Json
        Write-KeyValue "Complete" $verdict.complete
        Write-KeyValue "n (judged)" $verdict.n
        Write-KeyValue "Qwen agreement" $verdict.qwen_agreement
        Write-KeyValue "FinBERT agreement" $verdict.finbert_agreement
        Write-KeyValue "Delta" $verdict.delta
        Write-KeyValue "Gate pass" $verdict.pass
    } catch {
        Write-Host "Verdict parse failed."
    }
} else {
    Write-Host "No verdict yet."
}

Write-Host ""
Write-Host "== Markers =="
if (Test-Path $setupResult) {
    Write-Host "--- q0_SETUP_RESULT.txt ---"
    Get-Content $setupResult | ForEach-Object { Write-Host "  $_" }
}
if (Test-Path $alertPath) {
    Write-Host "--- q0_ALERT.txt (tail) ---"
    Get-Content $alertPath -Tail 10 | ForEach-Object { Write-Host "  $_" }
}
if (Test-Path $donePath) {
    Write-Host "--- q0_DONE.txt ---"
    Get-Content $donePath | ForEach-Object { Write-Host "  $_" }
}
if (-not (Test-Path $setupResult) -and -not (Test-Path $alertPath) -and -not (Test-Path $donePath)) {
    Write-Host "No setup result, alert, or done markers."
}
