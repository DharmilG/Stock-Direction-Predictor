# Q0-D1 overnight supervisor (Windows PowerShell)
#
# Zero-touch design (Option A, 2026-10-20):
# - One-time elevated setup registers a Scheduled Task; runtime never prompts.
# - Single-instance mutex + stale supervisor/worker cleanup.
# - Self-elevates only in manual mode; the scheduled task already has Highest.
# - Applies nvidia-smi -lgc 0,1400 (nearest supported <= cap) and resets -rgc.
# - Runs 24/7 (no night window) until all 1,500 Pool C cases are rated.
# - 89C emergency: kills the whole Python process tree, unloads, rests 30 min.
# - Ollama health check: restarts `ollama serve` if the local API stops.
# - Crash backoff: 60s -> 120s -> 5m -> 10m -> 15m -> 30m, never gives up.
# - Heartbeat: logs/q0_supervisor.json + logs/q0_heartbeat.json (from Python).
#
# Usage:
#   Manual:      powershell -NoProfile -ExecutionPolicy Bypass -File .\run_q0_overnight.ps1
#   Scheduled:   powershell -NoProfile -ExecutionPolicy Bypass -File .\run_q0_overnight.ps1 -NoElevate -Scheduled
#   Inspect:     powershell -NoProfile -ExecutionPolicy Bypass -File .\run_q0_overnight.ps1 -DryRun
#   Clock test:  powershell -NoProfile -ExecutionPolicy Bypass -File .\run_q0_overnight.ps1 -TestClocks
#
# If the supervisor is force-killed, the clock cap can remain at the last
# applied value. Reset manually with: nvidia-smi -rgc

param(
    [int]$ClockCapMHz = 1400,
    [int]$EmergencyC = 89,
    [int]$EmergencyMinutes = 30,
    [switch]$TestClocks,
    [switch]$NoElevate,
    [switch]$Scheduled,
    [switch]$DryRun,
    [string]$MutexName = "Global\StockML_Q0_Supervisor"
)

$ErrorActionPreference = "Stop"
$script:RepoRoot = $PSScriptRoot
if (-not $script:RepoRoot) { $script:RepoRoot = (Get-Location).Path }
Set-Location $script:RepoRoot

$script:Python = Join-Path $script:RepoRoot ".venv\Scripts\python.exe"
$script:LogDir = Join-Path $script:RepoRoot "track_q\phase_0\logs"
New-Item -ItemType Directory -Force -Path $script:LogDir | Out-Null

if ($Scheduled -or $NoElevate) {
    $script:Transcript = Join-Path $script:LogDir "q0_supervisor_task.log"
} else {
    $script:Transcript = Join-Path $script:LogDir "q0_supervisor.log"
}
$script:SupervisorHeartbeatPath = Join-Path $script:LogDir "q0_supervisor.json"
$script:RunnerHeartbeatPath = Join-Path $script:LogDir "q0_heartbeat.json"
$script:EventLogPath = Join-Path $script:LogDir "q0_supervisor_events.log"
$script:AlertPath = Join-Path $script:LogDir "q0_ALERT.txt"
$script:DonePath = Join-Path $script:LogDir "q0_DONE.txt"
$script:OllamaUrl = "http://127.0.0.1:11434/api/tags"

$script:ClockApplied = $false
$script:AppliedClockMHz = $null
$script:ActiveChild = $null
$script:MutexOwned = $false
$script:SupervisorMutex = $null
$script:Taskkill = Join-Path $env:SystemRoot "System32\taskkill.exe"

function Test-Admin {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Write-Log {
    param([string]$Message)
    $line = "{0} | {1}" -f (Get-Date -Format o), $Message
    Write-Host $line
    try {
        Add-Content -Path $script:EventLogPath -Value $line -Encoding UTF8
    } catch {
        # Logging must never be the reason the supervisor fails.
    }
}

function Write-Alert {
    param([string]$Message)
    $line = "{0} | {1}" -f (Get-Date -Format o), $Message
    Write-Log "ALERT: $Message"
    try {
        Add-Content -Path $script:AlertPath -Value $line -Encoding UTF8
    } catch {
    }
}

function Get-GpuTemp {
    try {
        $raw = & nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>$null
        if ($LASTEXITCODE -eq 0 -and $raw -match '^\s*(\d+)') {
            return [int]$Matches[1]
        }
    } catch {
        return $null
    }
    return $null
}

function Get-GpuClock {
    try {
        $raw = & nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits 2>$null
        if ($LASTEXITCODE -eq 0 -and $raw -match '^\s*(\d+)') {
            return [int]$Matches[1]
        }
    } catch {
        return $null
    }
    return $null
}

function Get-SupportedGraphicsClocks {
    try {
        $lines = & nvidia-smi -q -d SUPPORTED_CLOCKS 2>$null
        $values = foreach ($line in $lines) {
            if ($line -match '^\s+Graphics\s+:\s+(\d+)\s+MHz') {
                [int]$Matches[1]
            }
        }
        return @($values | Sort-Object -Unique)
    } catch {
        return @()
    }
}

function Set-ClockCap {
    if (-not (Test-Admin)) {
        Write-Log "Not elevated - skipping ${ClockCapMHz} MHz clock cap."
        return $false
    }
    try {
        $out = & nvidia-smi -lgc "0,$ClockCapMHz" 2>&1
        if ($LASTEXITCODE -eq 0) {
            $script:AppliedClockMHz = $ClockCapMHz
            Write-Log "GPU clock cap applied: 0..$ClockCapMHz MHz"
            return $true
        }

        $fallback = Get-SupportedGraphicsClocks |
            Where-Object { $_ -le $ClockCapMHz } |
            Sort-Object -Descending |
            Select-Object -First 1
        if ($null -ne $fallback) {
            $out2 = & nvidia-smi -lgc "0,$fallback" 2>&1
            if ($LASTEXITCODE -eq 0) {
                $script:AppliedClockMHz = [int]$fallback
                Write-Log "GPU clock cap applied: 0..$fallback MHz (nearest supported <= $ClockCapMHz)"
                return $true
            }
        }
        Write-Log "GPU clock cap not supported (continuing without it): $out"
    } catch {
        Write-Log "GPU clock cap failed (continuing without it): $_"
    }
    return $false
}

function Reset-Clocks {
    if (-not (Test-Admin)) {
        Write-Log "Not elevated - cannot reset GPU clocks. Run: nvidia-smi -rgc"
        return
    }
    try {
        & nvidia-smi -rgc 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) {
            $script:ClockApplied = $false
            Write-Log "GPU clock cap reset to driver default."
        } else {
            Write-Log "nvidia-smi -rgc returned $LASTEXITCODE"
        }
    } catch {
        Write-Log "GPU clock reset failed: $_"
    }
}

function Stop-Model {
    # `ollama stop` writes ANSI status to stderr even on success; suppress it.
    $previous = $ErrorActionPreference
    $ErrorActionPreference = "SilentlyContinue"
    try {
        & ollama stop qwen3:8b 2>$null | Out-Null
    } catch {
    } finally {
        $ErrorActionPreference = $previous
    }
    # HTTP fallback for installs where the CLI is not on PATH.
    try {
        $body = '{"model":"qwen3:8b","keep_alive":0}'
        Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/generate" -Method Post `
            -Body $body -ContentType "application/json" -TimeoutSec 15 | Out-Null
    } catch {
        # Server already stopped or not reachable; both are fine here.
    }
}

function Test-Ollama {
    try {
        $response = Invoke-WebRequest -Uri $script:OllamaUrl -UseBasicParsing -TimeoutSec 4
        return ($response.StatusCode -eq 200)
    } catch {
        return $false
    }
}

function Ensure-Ollama {
    if (Test-Ollama) { return $true }
    Write-Log "Ollama endpoint is not responding - attempting to start 'ollama serve'."
    try {
        $command = Get-Command ollama -ErrorAction Stop
        Start-Process -FilePath $command.Source -ArgumentList @("serve") -WindowStyle Hidden | Out-Null
    } catch {
        Write-Alert "Could not start Ollama: $_"
        return $false
    }
    for ($i = 0; $i -lt 30; $i++) {
        Start-Sleep -Seconds 2
        if (Test-Ollama) {
            Write-Log "Ollama is responding again."
            return $true
        }
    }
    Write-Alert "Ollama did not respond within 60s after restart."
    return $false
}

function Get-LedgerCount {
    try {
        $count = & $script:Python -c "import pandas as pd; print(len(pd.read_parquet('track_q/phase_0/state/qwen_rating_ledger.parquet')))" 2>$null
        if ($LASTEXITCODE -eq 0 -and $count -match '^\s*(\d+)') {
            return [int]$Matches[1]
        }
    } catch {
    }
    return $null
}

function Test-Complete {
    $verdictPath = Join-Path $script:RepoRoot "track_q\phase_0\verdict\q0_d1.json"
    if (Test-Path $verdictPath) {
        try {
            $verdict = Get-Content $verdictPath -Raw | ConvertFrom-Json
            if ($null -ne $verdict.complete -and [bool]$verdict.complete) {
                return $true
            }
        } catch {
        }
    }
    $count = Get-LedgerCount
    if ($null -ne $count) { return ($count -ge 1500) }
    return $false
}

function Get-RunnerHeartbeat {
    if (-not (Test-Path $script:RunnerHeartbeatPath)) { return $null }
    try {
        return Get-Content $script:RunnerHeartbeatPath -Raw | ConvertFrom-Json
    } catch {
        return $null
    }
}

function Write-SupervisorHeartbeat {
    param(
        [string]$State = "running",
        [string]$Message = "",
        [Nullable[int]]$ChildPid = $null,
        [Nullable[int]]$ExitCode = $null,
        [int]$RestartInSeconds = 0,
        [Nullable[int]]$TemperatureC = $null,
        [Nullable[int]]$ClockMHz = $null
    )
    try {
        $childAlive = $false
        if ($null -ne $ChildPid -and $ChildPid -gt 0) {
            try {
                $childAlive = -not (Get-Process -Id $ChildPid -ErrorAction Stop).HasExited
            } catch {
                $childAlive = $false
            }
        }
        $payload = [ordered]@{
            updated_at = (Get-Date).ToString("o")
            supervisor_pid = $PID
            scheduled = [bool]$Scheduled
            admin = (Test-Admin)
            state = $State
            message = $Message
            child_pid = $ChildPid
            child_alive = $childAlive
            last_exit_code = $ExitCode
            restart_in_seconds = $RestartInSeconds
            temperature_c = $TemperatureC
            clock_mhz = $ClockMHz
            clock_cap_mhz = $ClockCapMHz
            applied_clock_mhz = $script:AppliedClockMHz
            mutex_name = $MutexName
            runner = (Get-RunnerHeartbeat)
        }
        $json = $payload | ConvertTo-Json -Depth 8
        $tmp = "$($script:SupervisorHeartbeatPath).tmp"
        Set-Content -Path $tmp -Value $json -Encoding UTF8
        Move-Item -Force -Path $tmp -Destination $script:SupervisorHeartbeatPath
    } catch {
        # Heartbeat is diagnostic only; never fail the run because of it.
    }
}

function Stop-StaleSupervisors {
    param([int]$CurrentPid)
    try {
        $stale = Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" -ErrorAction SilentlyContinue |
            Where-Object {
                $_.ProcessId -ne $CurrentPid -and
                [string]$_.CommandLine -match 'run_q0_overnight\.ps1'
            }
        foreach ($process in @($stale)) {
            Write-Log "Stopping stale supervisor PowerShell PID $($process.ProcessId)."
            & $script:Taskkill "/PID" $process.ProcessId "/T" "/F" 2>$null | Out-Null
        }
    } catch {
        Write-Log "Stale-supervisor scan failed: $_"
    }
}

function Stop-StaleWorkers {
    $killed = 0
    try {
        $tasks = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue
        foreach ($process in @($tasks)) {
            $commandLine = [string]$process.CommandLine
            if ($commandLine -match 'stockml\.data\.q0_d1') {
                Write-Log "Killing stale Q0 worker PID $($process.ProcessId)."
                & $script:Taskkill "/PID" $process.ProcessId "/T" "/F" 2>$null | Out-Null
                $killed++
            }
        }
    } catch {
        Write-Log "Stale-worker scan failed: $_"
    }
    if ($killed -gt 0) { Start-Sleep -Seconds 2 }
    return $killed
}

function Set-PowerSettings {
    try {
        & cmd.exe /c "powercfg /change standby-timeout-ac 0 1>nul 2>&1" | Out-Null
        & cmd.exe /c "powercfg /change hibernate-timeout-ac 0 1>nul 2>&1" | Out-Null
        & cmd.exe /c "powercfg /change monitor-timeout-ac 15 1>nul 2>&1" | Out-Null
        & cmd.exe /c "powercfg /setacvalueindex SCHEME_CURRENT SUB_BUTTONS LIDACTION 0 1>nul 2>&1" | Out-Null
        & cmd.exe /c "powercfg /setactive SCHEME_CURRENT 1>nul 2>&1" | Out-Null
        Write-Log "Power settings applied: no sleep/hibernate on AC; lid close does nothing."
    } catch {
        Write-Log "Power settings update failed: $_"
    }
}

function Get-NextBackoffSeconds {
    param([int]$FailureCount)
    if ($FailureCount -le 1) { return 60 }
    $steps = @(60, 120, 300, 600, 900, 1800)
    $index = [Math]::Min($FailureCount - 1, $steps.Count - 1)
    return $steps[$index]
}

function Start-SleepWithHeartbeat {
    param(
        [int]$Seconds,
        [string]$State = "resting",
        [string]$Message = "",
        [Nullable[int]]$ExitCode = $null
    )
    # Sleep in 30-second chunks so the heartbeat never looks dead during a
    # 15/20/30-minute rest or a long crash backoff.
    $remaining = $Seconds
    while ($remaining -gt 0) {
        $chunk = [Math]::Min(30, $remaining)
        Start-Sleep -Seconds $chunk
        $remaining -= $chunk
        Write-SupervisorHeartbeat -State $State -Message "$Message ($remaining s left)" -ExitCode $ExitCode `
            -TemperatureC (Get-GpuTemp) -ClockMHz (Get-GpuClock)
    }
}

function Write-DoneMarker {
    $verdictText = ""
    $verdictPath = Join-Path $script:RepoRoot "track_q\phase_0\verdict\q0_d1.json"
    if (Test-Path $verdictPath) {
        try { $verdictText = Get-Content $verdictPath -Raw } catch { }
    }
    $ledgerCount = Get-LedgerCount
    $lines = @(
        "Q0-D1 supervisor complete at $((Get-Date).ToString('o'))",
        "Ledger: $ledgerCount / 1500",
        "Verdict:",
        $verdictText
    )
    try {
        Set-Content -Path $script:DonePath -Value ($lines -join "`r`n") -Encoding UTF8
    } catch {
        Write-Log "Could not write done marker: $_"
    }
}

# --- Dry run is always safe and never touches the GPU or the task -----------
if ($DryRun) {
    Write-Host "Q0-D1 supervisor dry run"
    Write-Host "Repo: $script:RepoRoot"
    Write-Host "Python: $script:Python (exists: $(Test-Path $script:Python))"
    Write-Host "Admin: $(Test-Admin)"
    Write-Host "Mode: Scheduled=$Scheduled NoElevate=$NoElevate Mutex=$MutexName"
    Write-Host "Action: $script:Python -m stockml.data.q0_d1 --full"
    Write-Host "Heartbeat: $script:SupervisorHeartbeatPath"
    Write-Host "Alert: $script:AlertPath"
    exit 0
}

# --- Self-elevation (manual mode only) --------------------------------------
if (-not (Test-Admin) -and -not $NoElevate -and -not $Scheduled) {
    $argList = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "`"$PSCommandPath`"")
    if ($TestClocks) { $argList += "-TestClocks" }
    if ($ClockCapMHz -ne 1400) { $argList += @("-ClockCapMHz", "$ClockCapMHz") }
    if ($EmergencyC -ne 89) { $argList += @("-EmergencyC", "$EmergencyC") }
    if ($EmergencyMinutes -ne 30) { $argList += @("-EmergencyMinutes", "$EmergencyMinutes") }
    if ($MutexName -ne "Global\StockML_Q0_Supervisor") { $argList += @("-MutexName", $MutexName) }
    Write-Host "Requesting administrator rights (UAC prompt)..."
    try {
        Start-Process -FilePath "powershell.exe" -Verb RunAs -ArgumentList $argList | Out-Null
    } catch {
        Write-Error "Elevation was cancelled or failed: $_"
    }
    exit
}

# --- Single instance guard --------------------------------------------------
try {
    $script:SupervisorMutex = New-Object System.Threading.Mutex($false, $MutexName)
} catch {
    $MutexName = "Local\StockML_Q0_Supervisor"
    Write-Warning "Global mutex unavailable; falling back to $MutexName"
    $script:SupervisorMutex = New-Object System.Threading.Mutex($false, $MutexName)
}
try {
    $script:MutexOwned = $script:SupervisorMutex.WaitOne(0)
} catch [System.Threading.AbandonedMutexException] {
    $script:MutexOwned = $true
} catch {
    $script:MutexOwned = $false
}
if (-not $script:MutexOwned) {
    Write-Host "Another Q0 supervisor is already running (mutex $MutexName); exiting."
    exit 0
}

try {
    Start-Transcript -Path $script:Transcript -Append | Out-Null
} catch {
    Write-Warning "Could not start transcript: $_"
}

try {
    Write-Log "=== Q0-D1 supervisor start ==="
    Write-Log "Repo: $script:RepoRoot"
    Write-Log "Admin: $(Test-Admin) | scheduled: $Scheduled | clock cap: $ClockCapMHz MHz | emergency: $EmergencyC C"

    Stop-StaleSupervisors -CurrentPid $PID
    $staleWorkers = Stop-StaleWorkers
    if ($staleWorkers -gt 0) {
        Write-Log "Removed $staleWorkers stale Q0 worker(s) before starting."
    }

    if ($TestClocks) {
        if (-not (Test-Admin)) {
            Write-Log "Clock test requires administrator rights."
            exit 1
        }
        $script:ClockApplied = Set-ClockCap
        Start-Sleep -Seconds 5
        Reset-Clocks
        exit 0
    }

    if (-not (Test-Path $script:Python)) {
        Write-Alert "Python not found at $script:Python"
        exit 1
    }

    Set-PowerSettings
    $script:ClockApplied = Set-ClockCap

    if (-not (Test-Ollama)) {
        Write-Log "Ollama not reachable at startup; attempting recovery."
        Ensure-Ollama | Out-Null
    }

    $failureCount = 0
    while ($true) {
        if (-not (Test-Ollama)) {
            Write-SupervisorHeartbeat -State "waiting-for-ollama" -Message "Ollama endpoint not responding" `
                -TemperatureC (Get-GpuTemp) -ClockMHz (Get-GpuClock)
            if (-not (Ensure-Ollama)) {
                Start-SleepWithHeartbeat -Seconds 60 -State "waiting-for-ollama" -Message "Ollama unavailable"
                continue
            }
        }

        $childStart = Get-Date
        $child = Start-Process -FilePath $script:Python `
            -ArgumentList @("-m", "stockml.data.q0_d1", "--full") `
            -WorkingDirectory $script:RepoRoot -PassThru -NoNewWindow
        $script:ActiveChild = $child
        Write-Log "Started Python child PID $($child.Id)."
        Write-SupervisorHeartbeat -State "running" -Message "child started" -ChildPid $child.Id `
            -TemperatureC (Get-GpuTemp) -ClockMHz (Get-GpuClock)

        $hotReadings = 0
        $thermalKill = $false
        $ollamaDown = $false
        $lastOllamaCheck = Get-Date
        $lastHeartbeat = Get-Date
        while (-not $child.HasExited) {
            Start-Sleep -Seconds 5
            $temperature = Get-GpuTemp
            $clock = Get-GpuClock

            if ($null -ne $temperature) {
                if ($temperature -ge $EmergencyC) {
                    $hotReadings++
                    Write-Log "EMERGENCY: GPU ${temperature}C >= ${EmergencyC}C ($hotReadings/2)"
                    if ($hotReadings -ge 2) {
                        Write-Log "Emergency thermal kill; stopping process tree, unloading, resting $EmergencyMinutes min."
                        try {
                            & $script:Taskkill "/PID" $child.Id "/T" "/F" 2>$null | Out-Null
                            $child.WaitForExit()
                        } catch {
                            Write-Log "Emergency taskkill failed: $_"
                        }
                        $thermalKill = $true
                        break
                    }
                } else {
                    $hotReadings = 0
                }
            }

            if (((Get-Date) - $lastOllamaCheck).TotalSeconds -ge 60) {
                $lastOllamaCheck = Get-Date
                if (-not (Test-Ollama)) {
                    Write-Log "Ollama stopped responding; stopping the worker for server recovery."
                    try {
                        & $script:Taskkill "/PID" $child.Id "/T" "/F" 2>$null | Out-Null
                        $child.WaitForExit()
                    } catch {
                        Write-Log "Ollama-recovery taskkill failed: $_"
                    }
                    $ollamaDown = $true
                    break
                }
            }

            if (((Get-Date) - $lastHeartbeat).TotalSeconds -ge 30) {
                $lastHeartbeat = Get-Date
                Write-SupervisorHeartbeat -State "running" -Message "monitoring" -ChildPid $child.Id `
                    -TemperatureC $temperature -ClockMHz $clock
            }
        }

        if ($thermalKill) {
            $script:ActiveChild = $null
            Stop-Model
            Write-SupervisorHeartbeat -State "thermal-rest" -Message "89C emergency; resting $EmergencyMinutes min" `
                -TemperatureC (Get-GpuTemp) -ClockMHz (Get-GpuClock)
            Start-SleepWithHeartbeat -Seconds ($EmergencyMinutes * 60) -State "thermal-rest" -Message "89C emergency rest"
            $failureCount = 0
            continue
        }

        if ($ollamaDown) {
            $script:ActiveChild = $null
            Stop-Model
            Write-SupervisorHeartbeat -State "ollama-recovery" -Message "restarting Ollama" `
                -TemperatureC (Get-GpuTemp) -ClockMHz (Get-GpuClock)
            Ensure-Ollama | Out-Null
            Start-Sleep -Seconds 15
            $failureCount = 0
            continue
        }

        try { $child.WaitForExit() } catch { }
        $exitCode = $child.ExitCode
        $script:ActiveChild = $null
        $elapsed = ((Get-Date) - $childStart).TotalSeconds
        Write-Log "Python exited code=$exitCode after $([int]$elapsed)s."
        Stop-Model

        if (Test-Complete) {
            Write-Log "All Pool C cases rated; supervisor complete."
            Write-DoneMarker
            break
        }

        if ($elapsed -lt 120) {
            $failureCount++
        } else {
            $failureCount = 0
        }
        if ($failureCount -ge 5) {
            Write-Alert "Python failed $failureCount times quickly; still retrying with backoff."
        }
        $backoffSeconds = Get-NextBackoffSeconds -FailureCount $failureCount
        Write-SupervisorHeartbeat -State "restarting" -Message "child exit $exitCode; backoff ${backoffSeconds}s" `
            -ExitCode $exitCode -RestartInSeconds $backoffSeconds `
            -TemperatureC (Get-GpuTemp) -ClockMHz (Get-GpuClock)
        Write-Log "Restarting in ${backoffSeconds}s (ledger resumes automatically)."
        Start-SleepWithHeartbeat -Seconds $backoffSeconds -State "backoff" -Message "child exit $exitCode" -ExitCode $exitCode
    }
} finally {
    try {
        if ($null -ne $script:ActiveChild -and -not $script:ActiveChild.HasExited) {
            & $script:Taskkill "/PID" $script:ActiveChild.Id "/T" "/F" 2>$null | Out-Null
        }
    } catch {
        Write-Log "Could not stop Python child tree in finally: $_"
    }
    Stop-Model
    if ($script:ClockApplied) { Reset-Clocks }
    Write-SupervisorHeartbeat -State "stopped" -Message "supervisor exiting" `
        -TemperatureC (Get-GpuTemp) -ClockMHz (Get-GpuClock)
    try { Stop-Transcript | Out-Null } catch { }
    if ($script:MutexOwned) {
        try { $script:SupervisorMutex.ReleaseMutex() } catch { }
    }
    try { $script:SupervisorMutex.Dispose() } catch { }
}
