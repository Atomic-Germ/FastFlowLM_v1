# Setup script for Windows using venv
# This script sets up the Python environment and installs dependencies for FLM testing
#
# Behavior:
# - Creates a virtual environment using Python's built-in venv
# - If requirements.txt exists, installs packages with pip
# - Supports -Force to recreate the environment

param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"

Write-Host "Setting up FLM test environment for Windows..." -ForegroundColor Green

function Fail($Message) {
    Write-Host "Error: $Message" -ForegroundColor Red
    exit 1
}

function Warn($Message) {
    Write-Host "Warning: $Message" -ForegroundColor Yellow
}

# Check if Python is available
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Fail "Python is not installed or not in PATH. Install Python 3.x from https://www.python.org/downloads/"
}

try {
    $pythonVersion = python --version
    Write-Host "Using $pythonVersion"
} catch {
    Fail "Python was found, but 'python --version' failed."
}

$requirementsFile = "requirements.txt"
$venvPath = "venv"

if ($Force -and (Test-Path $venvPath)) {
    Write-Host "Removing existing virtual environment at $venvPath..."
    Remove-Item -Recurse -Force $venvPath
}

if (-not (Test-Path $venvPath)) {
    Write-Host "Creating virtual environment..."
    python -m venv $venvPath
    if ($LASTEXITCODE -ne 0) {
        Fail "Failed to create virtual environment."
    }
} else {
    Write-Host "Virtual environment already exists at $venvPath."
}

Write-Host "Activating virtual environment..."
& ".\$venvPath\Scripts\Activate.ps1"

if (Test-Path $requirementsFile) {
    Write-Host "Installing Python packages from $requirementsFile..."
    pip install -r $requirementsFile
    if ($LASTEXITCODE -ne 0) {
        Fail "Failed to install requirements from $requirementsFile."
    }
} else {
    Warn "$requirementsFile not found. Skipping package installation."
}

Write-Host "Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "To activate the virtual environment in future sessions, run:" -ForegroundColor Cyan
Write-Host ".\$venvPath\Scripts\Activate.ps1"
Write-Host ""
Write-Host "To run tests, use:" -ForegroundColor Cyan
Write-Host "python main.py --help"
