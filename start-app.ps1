# Starts the DFIT Interpretation Tool with no file loaded.
# Run:  right-click start-app.cmd > open,  OR  from a shell:  .\start-app.ps1

$venvDir = 'C:\Users\LucasChristman\.venvs\dfit'
$python = "$venvDir\Scripts\python.exe"
$projectRoot = $PSScriptRoot

try {
    if (-not (Test-Path $python)) {
        Write-Host "Venv not found at $venvDir. Creating it..."
        if (Get-Command py -ErrorAction SilentlyContinue) {
            & py -3.14 -m venv $venvDir
        } else {
            & C:\Python314\python.exe -m venv $venvDir
        }

        if (-not (Test-Path $python)) {
            throw "Failed to create venv at $venvDir (python.exe not found after venv creation)."
        }
    }

    & $python -c "import numpy, pandas, matplotlib, scipy, openpyxl" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Installing required packages..."
        & $python -m pip install --upgrade pip
        if ($LASTEXITCODE -ne 0) {
            throw "pip upgrade failed with exit code $LASTEXITCODE."
        }
        & $python -m pip install -r "$projectRoot\requirements.txt"
        if ($LASTEXITCODE -ne 0) {
            throw "pip install -r requirements.txt failed with exit code $LASTEXITCODE."
        }
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
