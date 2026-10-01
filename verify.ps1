<#
    One-click verification for Windows.

        powershell -ExecutionPolicy Bypass -File verify.ps1

    Needs no GPU, no PTB-XL, no network and no LaTeX. It runs the
    statistics unit tests, regenerates the derived quantities, and then
    checks that every table, macro and figure-data file published in
    paper/generated/ is exactly what scripts/aggregate_results.py
    produces -- that is, that no published number was typed by hand.

    Exit code 0 means all three steps passed.
#>

$ErrorActionPreference = 'Stop'
$code   = Join-Path $PSScriptRoot 'code'
$failed = @()

function Step($n, $text) {
    Write-Host ""
    Write-Host ("== {0}  {1} {2}" -f $n, $text,
                ('=' * [Math]::Max(0, 58 - $text.Length))) -ForegroundColor Cyan
}

$py = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $py) { $py = (Get-Command python3 -ErrorAction SilentlyContinue).Source }
if (-not $py) { Write-Host "python not found on PATH" -ForegroundColor Red; exit 1 }

Push-Location $code
try {
    Step '1/3' 'statistics unit tests'
    & $py tests/test_stats.py
    if ($LASTEXITCODE -ne 0) { $failed += 'test_stats' }

    Step '2/3' 'regenerating derived quantities'
    & $py scripts/link_budget.py --paper-dir ../paper
    if ($LASTEXITCODE -ne 0) { $failed += 'link_budget' }
    & $py scripts/emit_config_table.py --paper-dir ../paper
    if ($LASTEXITCODE -ne 0) { $failed += 'emit_config_table' }

    Step '3/3' 'published numbers are the generator''s output'
    & $py tests/test_reproduce_tables.py --require-published-match
    if ($LASTEXITCODE -ne 0) { $failed += 'test_reproduce_tables' }
}
finally { Pop-Location }

Write-Host ""
if ($failed.Count) {
    Write-Host ("FAILED: " + ($failed -join ', ')) -ForegroundColor Red
    exit 1
}
Write-Host "VERIFY COMPLETE - every published number is generator output." -ForegroundColor Green
exit 0
