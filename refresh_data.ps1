<#
.SYNOPSIS
    Refresh market data, rebuild features, and regenerate predictions.

.DESCRIPTION
    Wraps `python -m stockml.main refresh`. Data-only by default: the model and
    its holdout window are left untouched. Pass -Retrain to also retrain, which
    moves the holdout end date forward and logs a holdout touch.

    Progress is written to artifacts/state/refresh.json, so the dashboard shows a
    live step counter while this runs.

.EXAMPLE
    .\refresh_data.ps1 -DryRun
    .\refresh_data.ps1
    .\refresh_data.ps1 -SkipNews -SkipAnnouncements
    .\refresh_data.ps1 -Retrain
#>
[CmdletBinding()]
param(
    [switch]$Retrain,
    [switch]$SkipNews,
    [switch]$SkipAnnouncements,
    [switch]$AnnScore,
    [switch]$IncludeGdelt,
    [switch]$NoHeal,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$script:RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $script:RepoRoot

$python = Join-Path $script:RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Host "Virtual environment not found. Run .\start_run.bat once to create it." -ForegroundColor Red
    exit 2
}

$arguments = @("-m", "stockml.main")
$arguments += $(if ($Retrain) { "train-fresh" } else { "refresh" })
if ($SkipNews) { $arguments += "--skip-news" }
if ($SkipAnnouncements) { $arguments += "--skip-announcements" }
if ($AnnScore) { $arguments += "--ann-score" }
if ($IncludeGdelt) { $arguments += "--include-gdelt" }
if ($NoHeal) { $arguments += "--no-heal" }
if ($DryRun) { $arguments += "--dry-run" }

$mode = if ($Retrain) { "RETRAIN (holdout moves forward)" } else { "data + predictions only" }
$annScoreMode = if ($Retrain -or $AnnScore) { "included (FinBERT)" } else { "skipped" }
Write-Host ""
Write-Host "Stock direction platform - data refresh" -ForegroundColor Cyan
Write-Host "  mode     : $mode"
Write-Host "  news     : $(if ($SkipNews) { 'skipped' } else { 'included (AlphaVantage quota applies)' })"
Write-Host "  announce : $(if ($SkipAnnouncements) { 'skipped' } else { 'fetched' })"
Write-Host "  ann-score: $annScoreMode"
Write-Host "  gdelt-bq : $(if ($IncludeGdelt) { 'INCLUDED (~164 GB BigQuery scan)' } else { 'skipped' })"
Write-Host "  heal     : $(if ($NoHeal) { 'skipped' } else { 'repair internal price holes' })"
Write-Host "  dry run  : $($DryRun.IsPresent)"
Write-Host ""

$started = Get-Date
& $python @arguments
$code = $LASTEXITCODE
$elapsed = (Get-Date) - $started

Write-Host ""
if ($code -eq 0) {
    Write-Host ("Refresh finished in {0:hh\:mm\:ss}." -f $elapsed) -ForegroundColor Green
    $statePath = Join-Path $script:RepoRoot "artifacts\state\refresh.json"
    # A dry run leaves the previous run's state file untouched; reporting it here
    # would look like this invocation had just completed that run.
    if (-not $DryRun -and (Test-Path $statePath)) {
        $state = Get-Content $statePath -Raw | ConvertFrom-Json
        Write-Host ("  status : {0}" -f $state.status)
        if ($state.failures) {
            foreach ($failure in $state.failures) {
                Write-Host ("  warning: {0} - {1}" -f $failure.step, $failure.error) -ForegroundColor Yellow
            }
        }
    }
    Write-Host "  next   : check /dashboard/data-health in the UI, or run .\q0_status.ps1 for the LLM run."
} else {
    Write-Host ("Refresh failed with exit code {0} after {1:hh\:mm\:ss}. See logs\refresh.log." -f $code, $elapsed) -ForegroundColor Red
}
exit $code
