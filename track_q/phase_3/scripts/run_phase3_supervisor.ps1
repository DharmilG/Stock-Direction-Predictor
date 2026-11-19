# Phase 3 (thinking-cap ablation) auto-restart supervisor (Windows PowerShell)
#
# Mirrors the proven Q0 supervisor. The Python phase resumes itself from its
# ledger (durable fsync'd appends + config fingerprint); this supervisor only
# keeps a worker alive and restarts it with backoff after a crash.
#
# - Single-instance mutex + stale supervisor/worker cleanup.
# - 89C emergency: kill the process tree, stop the model, rest 30 min.
# - Ollama health check: restart `ollama serve` if the API stops.
# - Crash backoff 60s -> 120s -> 5m -> 10m -> 15m -> 30m, never gives up.
# - Heartbeat: track_q/phase_3/logs/q3_supervisor.json (plus the Python
#   q3_heartbeat.json written by the phase itself).
# - Completion: q3_DONE.txt written by the phase, or results q3_thinking_cap.json
#   with a verdict and pass field.
#
# Usage (from repo root):
#   Manual:   powershell -NoProfile -ExecutionPolicy Bypass -File .\track_q\phase_3\scripts\run_phase3_supervisor.ps1
#   Inspect:  ... -DryRun
#   Pilot:    ... -SampleSize 60
#   Full:     ... (default: n=1000, budgets 256/512/1024)

param(
    [int]$ClockCapMHz = 1400,
    [int]$EmergencyC = 89,
    [int]$EmergencyMinutes = 30,
    [int]$SampleSize = 0,
    [switch]$TestClocks,
    [switch]$NoElevate,
    [switch]$Scheduled,
    [switch]$DryRun,
    [string]$MutexName = "Global\StockML_Q3_Supervisor"
)

$ErrorActionPreference = "Stop"
# Repo root is three levels above this script (scripts -> phase_3 -> track_q -> repo)
$script:RepoRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location $script:RepoRoot

$script:Python = Join-Path $script:RepoRoot ".venv\Scripts\python.exe"
$script:LogDir = Join-Path $script:RepoRoot "track_q\phase_3\logs"
New-Item -ItemType Directory -Force -Path $script:LogDir | Out-Null

if ($Scheduled -or $NoElevate) {
    $script:Transcript = Join-Path $script:LogDir "q3_supervisor_task.log"
} else {
    $script:Transcript = Join-Path $script:LogDir "q3_supervisor.log"
}
$script:SupervisorHeartbeatPath = Join-Path $script:LogDir "q3_supervisor.json"
$script:RunnerHeartbeatPath = Join-Path $script:LogDir "q3_heartbeat.json"
$script:EventLogPath = Join-Path $script:LogDir "q3_supervisor_events.log"
$script:AlertPath = Join-Path $script:LogDir "q3_ALERT.txt"
$script:DonePath = Join-Path $script:LogDir "q3_DONE.txt"
$script:ResultsPath = Join-Path $script:RepoRoot "track_q\phase_3\results\q3_thinking_cap.json"
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
    try { Add-Content -Path $script:EventLogPath -Value $line -Encoding UTF8 } catch { }
}

function Write-Alert {
    param([string]$Message)
    Write-Log "ALERT: $Message"
    try {
        Add-Content -Path $script:AlertPath -Value ("{0} | {1}" -f (Get-Date -Format o), $Message) -Encoding UTF8
    } catch { }
}

function Get-GpuTemp {
    try {
        $raw = & nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>$null
        if ($LASTEXITCODE -eq 0 -and $raw -match '^\s*(\d+)') { return [int]$Matches[1] }
    } catch { return $null }
    return $null
}

function Get-GpuClock {
    try {
        $raw = & nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits 2>$null
        if ($LASTEXITCODE -eq 0 -and $raw -match '^\s*(\d+)') { return [int]$Matches[1] }
    } catch { return $null }
    return $null
}

function Get-SupportedGraphicsClocks {
    try {
        $lines = & nvidia-smi -q -d SUPPORTED_CLOCKS 2>$null
        $values = foreach ($line in $lines) {
            if ($line -match '^\s+Graphics\s+:\s+(\d+)\s+MHz') { [int]$Matches[1] }
        }
        return @($values | Sort-Object -Unique)
    } catch { return @() }
}

function Set-ClockCap {
    if (-not (Test-Admin)) { Write-Log "Not elevated - skipping clock cap."; return $false }
    try {
        $out = & nvidia-smi -lgc "0,$ClockCapMHz" 2>&1
        if ($LASTEXITCODE -eq 0) {
            $script:AppliedClockMHz = $ClockCapMHz
            Write-Log "GPU clock cap applied: 0..$ClockCapMHz MHz"
            return $true
        }
        $fallback = Get-SupportedGraphicsClocks | Where-Object { $_ -le $ClockCapMHz } |
            Sort-Object -Descending | Select-Object -First 1
        if ($null -ne $fallback) {
            $out2 = & nvidia-smi -lgc "0,$fallback" 2>&1
            if ($LASTEXITCODE -eq 0) {
                $script:AppliedClockMHz = [int]$fallback
                Write-Log "GPU clock cap applied: 0..$fallback MHz (nearest supported <= $ClockCapMHz)"
                return $true
            }
        }
        Write-Log "GPU clock cap not supported (continuing without it): $out"
    } catch { Write-Log "GPU clock cap failed (continuing without it): $_" }
    return $false
}

function Reset-Clocks {
    if (-not (Test-Admin)) { Write-Log "Not elevated - cannot reset clocks. Run: nvidia-smi -rgc"; return }
    try {
        & nvidia-smi -rgc 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) { $script:ClockApplied = $false; Write-Log "GPU clock cap reset to driver default." }
    } catch { Write-Log "GPU clock reset failed: $_" }
}

function Stop-Model {
    $previous = $ErrorActionPreference
    $ErrorActionPreference = "SilentlyContinue"
    try { & ollama stop qwen3:8b 2>$null | Out-Null } catch { } finally { $ErrorActionPreference = $previous }
    try {
        $body = '{"model":"qwen3:8b","keep_alive":0}'
        Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/generate" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 15 | Out-Null
    } catch { }
}

function Test-Ollama {
    try { $r = Invoke-WebRequest -Uri $script:OllamaUrl -UseBasicParsing -TimeoutSec 4; return ($r.StatusCode -eq 200) }
    catch { return $false }
}

function Ensure-Ollama {
    if (Test-Ollama) { return $true }
    Write-Log "Ollama endpoint not responding - attempting to start 'ollama serve'."
    try {
        $command = Get-Command ollama -ErrorAction Stop
        Start-Process -FilePath $command.Source -ArgumentList @("serve") -WindowStyle Hidden | Out-Null
    } catch { Write-Alert "Could not start Ollama: $_"; return $false }
    for ($i = 0; $i -lt 30; $i++) {
        Start-Sleep -Seconds 2
        if (Test-Ollama) { Write-Log "Ollama is responding again."; return $true }
    }
    Write-Alert "Ollama did not respond within 60s after restart."
    return $false
}

function Get-RunnerHeartbeat {
    if (-not (Test-Path $script:RunnerHeartbeatPath)) { return $null }
    try { return Get-Content $script:RunnerHeartbeatPath -Raw | ConvertFrom-Json } catch { return $null }
}

function Write-SupervisorHeartbeat {
    param(
        [string]$State = "running", [string]$Message = "",
        [Nullable[int]]$ChildPid = $null, [Nullable[int]]$ExitCode = $null,
        [int]$RestartInSeconds = 0, [Nullable[int]]$TemperatureC = $null, [Nullable[int]]$ClockMHz = $null
    )
    try {
        $childAlive = $false
        if ($null -ne $ChildPid -and $ChildPid -gt 0) {
            try { $childAlive = -not (Get-Process -Id $ChildPid -ErrorAction Stop).HasExited } catch { $childAlive = $false }
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
            sample_size = $SampleSize
            mutex_name = $MutexName
            runner = (Get-RunnerHeartbeat)
        }
        $json = $payload | ConvertTo-Json -Depth 8
        $tmp = "$($script:SupervisorHeartbeatPath).tmp"
        Set-Content -Path $tmp -Value $json -Encoding UTF8
        Move-Item -Force -Path $tmp -Destination $script:SupervisorHeartbeatPath
    } catch { }
}

function Stop-StaleSupervisors {
    param([int]$CurrentPid)
    try {
        $stale = Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" -ErrorAction SilentlyContinue |
            Where-Object { $_.ProcessId -ne $CurrentPid -and [string]$_.CommandLine -match 'run_phase3_supervisor\.ps1' }
        foreach ($p in @($stale)) {
            Write-Log "Stopping stale supervisor PowerShell PID $($p.ProcessId)."
            & $script:Taskkill "/PID" $p.ProcessId "/T" "/F" 2>$null | Out-Null
        }
    } catch { Write-Log "Stale-supervisor scan failed: $_" }
}

function Stop-StaleWorkers {
    $killed = 0
    try {
        $tasks = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue
        foreach ($p in @($tasks)) {
            $cl = [string]$p.CommandLine
            if ($cl -match 'q3_thinking_cap' -or ($cl -match 'qpipeline' -and $cl -match 'phase3')) {
                Write-Log "Killing stale Phase 3 worker PID $($p.ProcessId)."
                & $script:Taskkill "/PID" $p.ProcessId "/T" "/F" 2>$null | Out-Null
                $killed++
            }
        }
    } catch { Write-Log "Stale-worker scan failed: $_" }
    if ($killed -gt 0) { Start-Sleep -Seconds 2 }
    return $killed
}

function Set-PowerSettings {
    try {
        & cmd.exe /c "powercfg /change standby-timeout-ac 0 1>nul 2>&1" | Out-Null
        & cmd.exe /c "powercfg /change hibernate-timeout-ac 0 1>nul 2>&1" | Out-Null
        & cmd.exe /c "powercfg /setacvalueindex SCHEME_CURRENT SUB_BUTTONS LIDACTION 0 1>nul 2>&1" | Out-Null
        & cmd.exe /c "powercfg /setactive SCHEME_CURRENT 1>nul 2>&1" | Out-Null
        Write-Log "Power settings applied."
    } catch { Write-Log "Power settings update failed: $_" }
}

function Get-NextBackoffSeconds {
    param([int]$FailureCount)
    if ($FailureCount -le 1) { return 60 }
    $steps = @(60, 120, 300, 600, 900, 1800)
    $index = [Math]::Min($FailureCount - 1, $steps.Count - 1)
    return $steps[$index]
}

function Start-SleepWithHeartbeat {
    param([int]$Seconds, [string]$State = "resting", [string]$Message = "", [Nullable[int]]$ExitCode = $null)
    $remaining = $Seconds
    while ($remaining -gt 0) {
        $chunk = [Math]::Min(30, $remaining)
        Start-Sleep -Seconds $chunk
        $remaining -= $chunk
        Write-SupervisorHeartbeat -State $State -Message "$Message ($remaining s left)" -ExitCode $ExitCode `
            -TemperatureC (Get-GpuTemp) -ClockMHz (Get-GpuClock)
    }
}

function Test-Complete {
    if (Test-Path $script:DonePath) { return $true }
    if (Test-Path $script:ResultsPath) {
        try {
            $r = Get-Content $script:ResultsPath -Raw | ConvertFrom-Json
            if ($null -ne $r.n_target -and $null -ne $r.pass -and $null -ne $r.updated_at) { return $true }
        } catch { }
    }
    return $false
}

function Write-DoneMarker {
    $resultText = ""
    if (Test-Path $script:ResultsPath) {
        try { $resultText = Get-Content $script:ResultsPath -Raw } catch { }
    }
    $lines = @("Phase 3 supervisor complete at $((Get-Date).ToString('o'))", "Results:", $resultText)
    try { Set-Content -Path $script:DonePath -Value ($lines -join "`r`n") -Encoding UTF8 } catch { }
}

# --- Dry run ---------------------------------------------------------------
if ($DryRun) {
    Write-Host "Phase 3 supervisor dry run"
    Write-Host "Repo: $script:RepoRoot"
    Write-Host "Python: $script:Python (exists: $(Test-Path $script:Python))"
    Write-Host "Admin: $(Test-Admin) | Mutex: $MutexName | SampleSize: $SampleSize"
    Write-Host "Action: $script:Python -m stockml.evaluation.q3_thinking_cap" + $(if ($SampleSize -gt 0) { " --sample-size $SampleSize" } else { "" })
    Write-Host "Heartbeat: $script:SupervisorHeartbeatPath"
    exit 0
}

# --- Single instance guard -------------------------------------------------
try { $script:SupervisorMutex = New-Object System.Threading.Mutex($false, $MutexName) }
catch {
    $MutexName = "Local\StockML_Q3_Supervisor"
    Write-Warning "Global mutex unavailable; falling back to $MutexName"
    $script:SupervisorMutex = New-Object System.Threading.Mutex($false, $MutexName)
}
try { $script:MutexOwned = $script:SupervisorMutex.WaitOne(0) }
catch [System.Threading.AbandonedMutexException] { $script:MutexOwned = $true }
catch { $script:MutexOwned = $false }
if (-not $script:MutexOwned) {
    Write-Host "Another Phase 3 supervisor is running (mutex $MutexName); exiting."
    exit 0
}

try { Start-Transcript -Path $script:Transcript -Append | Out-Null } catch { Write-Warning "Could not start transcript: $_" }

try {
    Write-Log "=== Phase 3 supervisor start ==="
    Write-Log "Repo: $script:RepoRoot"
    Write-Log "Admin: $(Test-Admin) | scheduled: $Scheduled | clock cap: $ClockCapMHz MHz | sample: $SampleSize"

    Stop-StaleSupervisors -CurrentPid $PID
    $staleWorkers = Stop-StaleWorkers
    if ($staleWorkers -gt 0) { Write-Log "Removed $staleWorkers stale Phase 3 worker(s)." }

    if ($TestClocks) {
        if (-not (Test-Admin)) { Write-Log "Clock test requires admin."; exit 1 }
        $script:ClockApplied = Set-ClockCap
        Start-Sleep -Seconds 5
        Reset-Clocks
        exit 0
    }

    if (-not (Test-Path $script:Python)) { Write-Alert "Python not found at $script:Python"; exit 1 }

    Set-PowerSettings
    $script:ClockApplied = Set-ClockCap

    if (-not (Test-Ollama)) { Write-Log "Ollama not reachable at startup; attempting recovery."; Ensure-Ollama | Out-Null }

    if (Test-Complete) {
        Write-Log "Phase 3 already complete (done marker / verdict present); supervisor exiting."
        exit 0
    }

    $failureCount = 0
    while ($true) {
        if (-not (Test-Ollama)) {
            Write-SupervisorHeartbeat -State "waiting-for-ollama" -Message "Ollama not responding" -TemperatureC (Get-GpuTemp) -ClockMHz (Get-GpuClock)
            if (-not (Ensure-Ollama)) {
                Start-SleepWithHeartbeat -Seconds 60 -State "waiting-for-ollama" -Message "Ollama unavailable"
                continue
            }
        }

        $argList = @("-m", "stockml.evaluation.q3_thinking_cap")
        if ($SampleSize -gt 0) { $argList += @("--sample-size", "$SampleSize") }
        $childStart = Get-Date
        $child = Start-Process -FilePath $script:Python -ArgumentList $argList `
            -WorkingDirectory $script:RepoRoot -PassThru -NoNewWindow
        $script:ActiveChild = $child
        Write-Log "Started Python child PID $($child.Id). Args: $($argList -join ' ')"
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

            if ($null -ne $temperature -and $temperature -ge $EmergencyC) {
                $hotReadings++
                Write-Log "EMERGENCY: GPU ${temperature}C >= ${EmergencyC}C ($hotReadings/2)"
                if ($hotReadings -ge 2) {
                    try { & $script:Taskkill "/PID" $child.Id "/T" "/F" 2>$null | Out-Null; $child.WaitForExit() } catch { }
                    $thermalKill = $true
                    break
                }
            } else { $hotReadings = 0 }

            if (((Get-Date) - $lastOllamaCheck).TotalSeconds -ge 60) {
                $lastOllamaCheck = Get-Date
                if (-not (Test-Ollama)) {
                    Write-Log "Ollama stopped responding; stopping worker for recovery."
                    try { & $script:Taskkill "/PID" $child.Id "/T" "/F" 2>$null | Out-Null; $child.WaitForExit() } catch { }
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
            Write-SupervisorHeartbeat -State "thermal-rest" -Message "${EmergencyC}C emergency; resting $EmergencyMinutes min" -TemperatureC (Get-GpuTemp) -ClockMHz (Get-GpuClock)
            Start-SleepWithHeartbeat -Seconds ($EmergencyMinutes * 60) -State "thermal-rest" -Message "emergency rest"
            $failureCount = 0
            continue
        }

        if ($ollamaDown) {
            $script:ActiveChild = $null
            Stop-Model
            Write-SupervisorHeartbeat -State "ollama-recovery" -Message "restarting Ollama" -TemperatureC (Get-GpuTemp) -ClockMHz (Get-GpuClock)
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
            Write-Log "Phase 3 complete; supervisor exiting."
            Write-DoneMarker
            break
        }

        if ($elapsed -lt 120) { $failureCount++ } else { $failureCount = 0 }
        if ($failureCount -ge 5) { Write-Alert "Python failed $failureCount times quickly; retrying with backoff." }
        $backoff = Get-NextBackoffSeconds -FailureCount $failureCount
        Write-SupervisorHeartbeat -State "restarting" -Message "child exit $exitCode; backoff ${backoff}s" `
            -ExitCode $exitCode -RestartInSeconds $backoff -TemperatureC (Get-GpuTemp) -ClockMHz (Get-GpuClock)
        Write-Log "Restarting in ${backoff}s (ledger resumes automatically)."
        Start-SleepWithHeartbeat -Seconds $backoff -State "backoff" -Message "child exit $exitCode" -ExitCode $exitCode
    }
} finally {
    try {
        if ($null -ne $script:ActiveChild -and -not $script:ActiveChild.HasExited) {
            & $script:Taskkill "/PID" $script:ActiveChild.Id "/T" "/F" 2>$null | Out-Null
        }
    } catch { Write-Log "Could not stop child tree in finally: $_" }
    Stop-Model
    if ($script:ClockApplied) { Reset-Clocks }
    Write-SupervisorHeartbeat -State "stopped" -Message "supervisor exiting" -TemperatureC (Get-GpuTemp) -ClockMHz (Get-GpuClock)
    try { Stop-Transcript | Out-Null } catch { }
    if ($script:MutexOwned) { try { $script:SupervisorMutex.ReleaseMutex() } catch { } }
    try { $script:SupervisorMutex.Dispose() } catch { }
}
