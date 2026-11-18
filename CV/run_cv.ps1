param(
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$CvArgs
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) { $python = 'python' }

$env:PYTHONPATH = $PSScriptRoot
Push-Location $PSScriptRoot
try {
  & $python -m chartsense.cli @CvArgs
  exit $LASTEXITCODE
}
finally {
  Pop-Location
}
