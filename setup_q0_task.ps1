# Q0-D1 one-time scheduled-task setup (Windows PowerShell)
#
# Run this ONCE. It requests administrator rights (one UAC prompt), registers
# a Highest-privileges Scheduled Task, stops the current manual supervisor,
# starts the task, and verifies that the heartbeat and ledger resume.
#
# After this, the run is zero-touch: at logon the task starts elevated and
# every future start uses the stored task principal without a UAC prompt.
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\setup_q0_task.ps1
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\setup_q0_task.ps1 -DryRun
#
# This does NOT start Phase 3. It only keeps Q0-D1 running to completion.

param(
    [string]$TaskName = "StockML_Q0_Supervisor",
    [switch]$NoStart,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$script:RepoRoot = $PSScriptRoot
if (-not $script:RepoRoot) { $script:RepoRoot = (Get-Location).Path }
Set-Location $script:RepoRoot

$script:SupervisorPath = Join-Path $script:RepoRoot "run_q0_overnight.ps1"
$script:Python = Join-Path $script:RepoRoot ".venv\Scripts\python.exe"
$script:LogDir = Join-Path $script:RepoRoot "track_q\phase_0\logs"
New-Item -ItemType Directory -Force -Path $script:LogDir | Out-Null
$script:SetupLog = Join-Path $script:LogDir "q0_setup.log"
$script:ResultPath = Join-Path $script:LogDir "q0_SETUP_RESULT.txt"
$script:AlertPath = Join-Path $script:LogDir "q0_ALERT.txt"
$script:Taskkill = Join-Path $env:SystemRoot "System32\taskkill.exe"
$script:PowershellExe = Join-Path $env:SystemRoot "System32\WindowsPowerShell\v1.0\powershell.exe"

function Test-Admin {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Write-SetupLog {
    param([string]$Message)
    $line = "{0} | {1}" -f (Get-Date -Format o), $Message
    Write-Host $line
    try { Add-Content -Path $script:SetupLog -Value $line -Encoding UTF8 } catch { }
}

function Write-SetupResult {
    param([string]$Status, [string]$Detail = "")
    $text = @(
        "STATUS: $Status",
        "AT: $((Get-Date).ToString('o'))",
        "TASK: $TaskName",
        "REPO: $script:RepoRoot",
        $Detail
    ) -join "`r`n"
    try { Set-Content -Path $script:ResultPath -Value $text -Encoding UTF8 } catch { }
}

function Write-Alert {
    param([string]$Message)
    Write-SetupLog "ALERT: $Message"
    try {
        Add-Content -Path $script:AlertPath -Value ("{0} | {1}" -f (Get-Date -Format o), $Message) -Encoding UTF8
    } catch { }
}

function Invoke-TreeKill {
    param([int]$ProcessId)
    if ($ProcessId -le 0) { return }
    & cmd.exe /c "taskkill /PID $ProcessId /T /F 1>nul 2>&1" | Out-Null
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
            Write-SetupLog "Stopping stale supervisor PowerShell PID $($process.ProcessId)."
            Invoke-TreeKill -ProcessId $process.ProcessId
        }
    } catch {
        Write-SetupLog "Stale-supervisor scan failed: $_"
    }
}

function Stop-StaleWorkers {
    try {
        $tasks = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue
        foreach ($process in @($tasks)) {
            if ([string]$process.CommandLine -match 'stockml\.data\.q0_d1') {
                Write-SetupLog "Stopping stale Q0 worker PID $($process.ProcessId)."
                Invoke-TreeKill -ProcessId $process.ProcessId
            }
        }
    } catch {
        Write-SetupLog "Stale-worker scan failed: $_"
    }
}

function Stop-OllamaModel {
    try {
        & cmd.exe /c "ollama stop qwen3:8b 1>nul 2>&1" | Out-Null
    } catch { }
    try {
        $body = '{"model":"qwen3:8b","keep_alive":0}'
        Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/generate" -Method Post `
            -Body $body -ContentType "application/json" -TimeoutSec 15 | Out-Null
    } catch { }
}

function Set-PowerSettings {
    try {
        & cmd.exe /c "powercfg /change standby-timeout-ac 0 1>nul 2>&1" | Out-Null
        & cmd.exe /c "powercfg /change hibernate-timeout-ac 0 1>nul 2>&1" | Out-Null
        & cmd.exe /c "powercfg /change monitor-timeout-ac 15 1>nul 2>&1" | Out-Null
        & cmd.exe /c "powercfg /setacvalueindex SCHEME_CURRENT SUB_BUTTONS LIDACTION 0 1>nul 2>&1" | Out-Null
        & cmd.exe /c "powercfg /setactive SCHEME_CURRENT 1>nul 2>&1" | Out-Null
        Write-SetupLog "Power settings applied: no sleep/hibernate on AC; lid close does nothing."
    } catch {
        Write-SetupLog "Power settings update failed: $_"
    }
}

function Get-HeartbeatAgeSeconds {
    param([string]$Path)
    if (-not (Test-Path $Path)) { return $null }
    try {
        $payload = Get-Content $Path -Raw | ConvertFrom-Json
        if ($null -eq $payload.updated_at) { return $null }
        $updated = [datetime]$payload.updated_at
        return ((Get-Date) - $updated).TotalSeconds
    } catch {
        return $null
    }
}

# --- Dry run: print the plan without admin, registration, or process changes -
if ($DryRun) {
    Write-Host "Q0-D1 scheduled-task setup dry run"
    Write-Host "Repo: $script:RepoRoot"
    Write-Host "Task: $TaskName"
    Write-Host "Admin now: $(Test-Admin)"
    Write-Host "Python: $script:Python (exists: $(Test-Path $script:Python))"
    Write-Host "Supervisor: $script:SupervisorPath (exists: $(Test-Path $script:SupervisorPath))"
    Write-Host "Action: $script:PowershellExe -NoProfile -ExecutionPolicy Bypass -File `"$script:SupervisorPath`" -NoElevate -Scheduled"
    Write-Host "Trigger: At logon of $env:USERDOMAIN\$env:USERNAME"
    Write-Host "Principal: Interactive, RunLevel Highest"
    Write-Host "Settings: no execution limit, restart on failure, ignore duplicate starts"
    Write-Host "Also: AC power never sleeps, lid close does nothing, current manual supervisor is stopped"
    exit 0
}

# --- One-time elevation ------------------------------------------------------
if (-not (Test-Admin)) {
    $argList = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "`"$PSCommandPath`"")
    if ($NoStart) { $argList += "-NoStart" }
    if ($TaskName -ne "StockML_Q0_Supervisor") { $argList += @("-TaskName", "`"$TaskName`"") }
    Write-Host "Requesting administrator rights for one-time task setup (UAC prompt)..."
    try {
        Start-Process -FilePath "powershell.exe" -Verb RunAs -ArgumentList $argList | Out-Null
    } catch {
        Write-Error "Elevation was cancelled or failed: $_"
    }
    exit
}

Write-SetupLog "=== Q0-D1 scheduled-task setup start ==="
Write-SetupLog "Admin: True | task: $TaskName | no-start: $NoStart"

if (-not (Test-Path $script:SupervisorPath)) {
    Write-Alert "Supervisor not found at $script:SupervisorPath"
    Write-SetupResult -Status "FAILED" -Detail "Supervisor script missing."
    exit 1
}
if (-not (Test-Path $script:Python)) {
    Write-Alert "Python not found at $script:Python"
    Write-SetupResult -Status "FAILED" -Detail "Python missing."
    exit 1
}

$taskAction = New-ScheduledTaskAction `
    -Execute $script:PowershellExe `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$script:SupervisorPath`" -NoElevate -Scheduled" `
    -WorkingDirectory $script:RepoRoot
$taskTrigger = New-ScheduledTaskTrigger -AtLogOn -User "$env:USERDOMAIN\$env:USERNAME"
$taskSettings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -RestartCount 10 `
    -RestartInterval (New-TimeSpan -Minutes 1)
$taskPrincipal = New-ScheduledTaskPrincipal `
    -UserId "$env:USERDOMAIN\$env:USERNAME" `
    -LogonType Interactive `
    -RunLevel Highest

# Stop any existing instance before re-registering.
try { Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue } catch { }

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $taskAction `
    -Trigger $taskTrigger `
    -Settings $taskSettings `
    -Principal $taskPrincipal `
    -Description "Q0-D1 Qwen rater supervisor: keeps the Pool C run alive, capped at 1400 MHz, with 87C/89C thermal guards." `
    -Force | Out-Null
Write-SetupLog "Scheduled task registered: $TaskName"

Set-PowerSettings

# Hand over from the current manual supervisor to the scheduled task.
Write-SetupLog "Stopping any current manual supervisor and worker."
Stop-StaleSupervisors -CurrentPid $PID
Stop-StaleWorkers
Stop-OllamaModel
Start-Sleep -Seconds 3

if ($NoStart) {
    Write-SetupLog "Task registered but not started (-NoStart)."
    Write-SetupResult -Status "REGISTERED-NOT-STARTED" -Detail "Run: Start-ScheduledTask -TaskName $TaskName"
    exit 0
}

Start-ScheduledTask -TaskName $TaskName
Write-SetupLog "Scheduled task start requested."

$supervisorHeartbeat = Join-Path $script:LogDir "q0_supervisor.json"
$runnerHeartbeat = Join-Path $script:LogDir "q0_heartbeat.json"
$deadline = (Get-Date).AddSeconds(120)
$verified = $false
$lastState = "unknown"

while ((Get-Date) -lt $deadline) {
    Start-Sleep -Seconds 5
    try {
        $lastState = (Get-ScheduledTask -TaskName $TaskName -ErrorAction Stop).State
    } catch {
        $lastState = "query-failed"
    }
    $supervisorAge = Get-HeartbeatAgeSeconds -Path $supervisorHeartbeat
    $runnerAge = Get-HeartbeatAgeSeconds -Path $runnerHeartbeat
    if ($lastState -eq "Running" -and $null -ne $supervisorAge -and $supervisorAge -lt 120) {
        $verified = $true
        $runnerAgeText = "n/a"
        if ($null -ne $runnerAge) { $runnerAgeText = "$([int]$runnerAge)s" }
        Write-SetupLog "Verified: task Running; supervisor heartbeat age $([int]$supervisorAge)s; runner heartbeat age $runnerAgeText"
        break
    }
}

if ($verified) {
    $info = $null
    try { $info = Get-ScheduledTaskInfo -TaskName $TaskName -ErrorAction SilentlyContinue } catch { }
    $detail = @(
        "Task state: $lastState",
        "Supervisor heartbeat: $supervisorHeartbeat",
        "Runner heartbeat: $runnerHeartbeat",
        "Status command: powershell -NoProfile -ExecutionPolicy Bypass -File .\q0_status.ps1",
        "No further UAC prompts are needed."
    ) -join "`r`n"
    Write-SetupLog "SETUP SUCCESS."
    Write-SetupResult -Status "SUCCESS" -Detail $detail
    exit 0
}

Write-Alert "Scheduled task did not produce a fresh heartbeat within 120s (state=$lastState)."
Write-SetupLog "Falling back to a manual elevated supervisor."
Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
Start-Process -FilePath $script:PowershellExe `
    -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-WindowStyle", "Hidden", "-File", "`"$script:SupervisorPath`"", "-NoElevate") `
    -WorkingDirectory $script:RepoRoot
$fallbackDetail = @(
    "Scheduled task did not verify (state=$lastState).",
    "A manual elevated supervisor is starting as fallback.",
    "Check logs/q0_ALERT.txt and logs/q0_supervisor_events.log."
) -join "`r`n"
Write-SetupResult -Status "FAILED-FALLBACK" -Detail $fallbackDetail
exit 1
