# Starts the DFIT Interpretation Tool with no file loaded.
# Run:  right-click start-app.cmd > open,  OR  from a shell:  .\start-app.ps1

$python = 'C:\Users\LucasChristman\.venvs\dfit\Scripts\python.exe'
$projectRoot = $PSScriptRoot

try {
    if (-not (Test-Path $python)) {
        throw "Python venv not found at $python. See dfit_tool\README.md for setup."
    }

    Set-Location $projectRoot
    Write-Host "Starting DFIT tool (no file loaded)..."
    & $python -m dfit_tool.app
    $code = $LASTEXITCODE

    if ($code -ne 0) {
        throw "App exited with code $code."
    }
}
catch {
    Write-Host ""
    Write-Host "ERROR: $_" -ForegroundColor Red
    Read-Host "Press Enter to close"
    exit 1
}
