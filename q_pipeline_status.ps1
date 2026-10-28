# Track Q pipeline status (read-only). CODE READY — UNTESTED.
[CmdletBinding()]
param(
    [switch]$Json
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$statePath = Join-Path $root 'artifacts\state\q_track_state.json'
$q0Path = Join-Path $root 'results\q0_d1.json'

function Read-JsonFile($path) {
    if (-not (Test-Path $path)) { return $null }
    try { return Get-Content -Raw -Path $path | ConvertFrom-Json } catch { return $null }
}

$q0 = Read-JsonFile $q0Path
$state = Read-JsonFile $statePath

if ($Json) {
    [pscustomobject]@{
        q0 = $q0
        state = $state
    } | ConvertTo-Json -Depth 6
    return
}

Write-Host '== Track Q status =='
if ($q0) {
    Write-Host ("Q0 complete={0} pass={1} rated={2} judged={3}" -f $q0.complete, $q0.pass, $q0.rated_total, $q0.judged_total)
} else {
    Write-Host 'Q0 results missing.'
}
if ($state -and $state.phases) {
    foreach ($p in $state.phases.PSObject.Properties) {
        Write-Host ("{0}: {1}" -f $p.Name, $p.Value.status)
    }
} else {
    Write-Host 'No phase state yet.'
}
