$ErrorActionPreference = "Stop"

param(
    [string]$Name = "VisualizerPioneerUSB",
    [switch]$OneFile = $true
)

Write-Host "==> Building $Name (PyInstaller)" -ForegroundColor Cyan

if (-not (Get-Command pyinstaller -ErrorAction SilentlyContinue)) {
    Write-Host "Installing PyInstaller..." -ForegroundColor Yellow
    pip install pyinstaller | Out-Null
}

$root = Split-Path -Parent $MyInvocation.MyCommand.Path | Split-Path -Parent
Push-Location $root

Remove-Item -Recurse -Force -ErrorAction SilentlyContinue build, dist | Out-Null

$oneFileFlag = if ($OneFile) { "--onefile" } else { "" }

pyinstaller --noconfirm --clean `
  --name "$Name" `
  $oneFileFlag `
  --add-data "configs\audio.yaml;configs" `
  --add-data "configs\analysis.yaml;configs" `
  --add-data "configs\mapping.yaml;configs" `
  --add-data "configs\visuals.yaml;configs" `
  scripts\live_fractal.py

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "==> Build output: dist\$Name.exe" -ForegroundColor Green

Pop-Location


