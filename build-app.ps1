# Builds the FracClosure distribution: a PyInstaller one-dir folder plus a zip of it.
# Run:  double-click build-app.cmd,  OR  from a shell:  .\build-app.ps1
#
# Build output goes outside the project so OneDrive does not sync ~1,000 build files:
#   C:\Users\LucasChristman\.builds\FracClosure\dist\FracClosure\     the app folder
#   C:\Users\LucasChristman\.builds\FracClosure\FracClosure-<version>.zip

$venvDir = 'C:\Users\LucasChristman\.venvs\dfit'
$python = "$venvDir\Scripts\python.exe"
$projectRoot = $PSScriptRoot
$outRoot = 'C:\Users\LucasChristman\.builds\FracClosure'
$name = 'FracClosure'

try {
    if (-not (Test-Path $python)) {
        throw "Venv not found at $venvDir. Run start-app.cmd once to create it."
    }

    & $python -c "import PyInstaller" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Installing PyInstaller..."
        & $python -m pip install -r "$projectRoot\requirements-build.txt"
        if ($LASTEXITCODE -ne 0) {
            throw "pip install -r requirements-build.txt failed with exit code $LASTEXITCODE."
        }
    }

    Set-Location $projectRoot
    $version = (& $python -c "import dfit_tool; print(dfit_tool.__version__)").Trim()
    if ($LASTEXITCODE -ne 0 -or -not $version) {
        throw "Could not read dfit_tool.__version__."
    }

    $assets = Join-Path $projectRoot 'dfit_tool\assets'
    $icon = Join-Path $assets 'app_icon.ico'
    $distDir = Join-Path $outRoot 'dist'
    $appDir = Join-Path $distDir $name
    $zipPath = Join-Path $outRoot "$name-$version.zip"

    Write-Host "Building $name $version..."
    # Paths are absolute because --specpath makes relative --add-data paths resolve
    # against the spec directory, not the project.
    & $python -m PyInstaller `
        --noconfirm --clean `
        --onedir --windowed `
        --name $name `
        --icon $icon `
        --add-data "$assets;dfit_tool\assets" `
        --paths $projectRoot `
        --exclude-module scipy `
        --exclude-module pytest `
        --distpath $distDir `
        --workpath (Join-Path $outRoot 'build') `
        --specpath $outRoot `
        (Join-Path $projectRoot 'packaging\launcher.py')
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller failed with exit code $LASTEXITCODE."
    }

    Copy-Item (Join-Path $projectRoot 'packaging\README.txt') $appDir -Force

    Write-Host "Zipping..."
    if (Test-Path $zipPath) { Remove-Item $zipPath -Force }
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    # includeBaseDirectory = $true, so the zip extracts to a FracClosure\ folder.
    [System.IO.Compression.ZipFile]::CreateFromDirectory(
        $appDir, $zipPath, [System.IO.Compression.CompressionLevel]::Optimal, $true)

    $sizeMb = [math]::Round((Get-Item $zipPath).Length / 1MB, 1)
    Write-Host ""
    Write-Host "App folder: $appDir"
    Write-Host "Zip:        $zipPath ($sizeMb MB)"
}
catch {
    Write-Host ""
    Write-Host "ERROR: $_" -ForegroundColor Red
    exit 1
}
