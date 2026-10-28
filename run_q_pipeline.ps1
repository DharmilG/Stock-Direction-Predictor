# Track Q phase runner wrapper. CODE READY — UNTESTED.
[CmdletBinding()]
param(
    [Parameter(Mandatory = $false)]
    [ValidateSet('phase3','phase6_probe','phase4','phase5','phase6_train','phase7','phase8')]
    [string]$Phase,

    [switch]$Status,

    [int]$Limit = 0,

    [int]$SampleSize = 0,

    [int]$ThinkingBudget = 0,

    [switch]$Probe
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

if ($Status) {
    & (Join-Path $root 'q_pipeline_status.ps1')
    return
}

if ($Probe) {
    python -m stockml.qpipeline --phase phase6_probe
    return
}

if (-not $Phase) {
    Write-Host 'Usage: .\run_q_pipeline.ps1 -Phase phase3'
    Write-Host '       .\run_q_pipeline.ps1 -Status'
    Write-Host '       .\run_q_pipeline.ps1 -Probe'
    return
}

$pyArgs = @('-m','stockml.qpipeline','--phase', $Phase)
if ($Limit -gt 0) { $pyArgs += @('--limit', $Limit) }
if ($SampleSize -gt 0) { $pyArgs += @('--sample-size', $SampleSize) }
if ($ThinkingBudget -gt 0) { $pyArgs += @('--thinking-budget', $ThinkingBudget) }
python @pyArgs
