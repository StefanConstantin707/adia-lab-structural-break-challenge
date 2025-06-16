# make.ps1 - PowerShell equivalent of Makefile for Windows
# Usage: .\make.ps1 <command>

param(
    [Parameter(Position=0)]
    [string]$Command = "help",

    [Parameter()]
    [string]$MSG = ""
)

# Define Python executable
$PYTHON = "python"
$PIP = "pip"
$PYTEST = "pytest"
$JUPYTER = "jupyter"
$CRUNCH = "crunch"

function Show-Help {
    Write-Host "ADIA Lab Structural Break Challenge - Available Commands:" -ForegroundColor Cyan
    Write-Host "  .\make.ps1 help          - Show this help message"
    Write-Host "  .\make.ps1 setup         - Set up the complete environment"
    Write-Host "  .\make.ps1 install       - Install Python dependencies"
    Write-Host "  .\make.ps1 clean         - Clean up temporary files and caches"
    Write-Host "  .\make.ps1 test          - Run unit tests"
    Write-Host "  .\make.ps1 train         - Train the model"
    Write-Host "  .\make.ps1 evaluate      - Evaluate model performance"
    Write-Host "  .\make.ps1 notebook      - Start Jupyter notebook server"
    Write-Host "  .\make.ps1 submit -MSG 'message' - Submit to CrunchDAO"
    Write-Host "  .\make.ps1 lint          - Run code linting"
    Write-Host "  .\make.ps1 format        - Format code with black"
}

function Install-Dependencies {
    Write-Host "Installing Python dependencies..." -ForegroundColor Green
    & $PIP install --upgrade pip
    & $PIP install -r requirements.txt
    Write-Host "Dependencies installed successfully!" -ForegroundColor Green
}

function Clean-Project {
    Write-Host "Cleaning up..." -ForegroundColor Yellow

    # Remove __pycache__ directories
    Get-ChildItem -Path . -Directory -Recurse -Filter "__pycache__" | Remove-Item -Recurse -Force

    # Remove .pyc and .pyo files
    Get-ChildItem -Path . -File -Recurse -Include "*.pyc", "*.pyo" | Remove-Item -Force

    # Remove other temporary files
    @(".pytest_cache", ".coverage", "htmlcov", ".mypy_cache") | ForEach-Object {
        if (Test-Path $_) {
            Remove-Item $_ -Recurse -Force
        }
    }

    Write-Host "Cleanup complete!" -ForegroundColor Green
}

function Run-Tests {
    Write-Host "Running unit tests..." -ForegroundColor Green
    & $PYTEST tests/ -v --tb=short
}

function Run-CrunchTest {
    Write-Host "Running CrunchDAO local test..." -ForegroundColor Green
    & $CRUNCH test
}

function Train-Model {
    Write-Host "Training model..." -ForegroundColor Green
    & $PYTHON scripts/train_model.py
}

function Evaluate-Model {
    Write-Host "Evaluating model performance..." -ForegroundColor Green
    & $PYTHON scripts/evaluate_model.py
}

function Start-Notebook {
    Write-Host "Starting Jupyter notebook server..." -ForegroundColor Green
    Set-Location notebooks
    & $JUPYTER notebook
    Set-Location ..
}

function Submit-ToCrunchDAO {
    if ([string]::IsNullOrEmpty($MSG)) {
        Write-Host "Error: MSG is not set. Usage: .\make.ps1 submit -MSG 'your message'" -ForegroundColor Red
        return
    }

    Write-Host "Testing submission locally first..." -ForegroundColor Yellow
    & $CRUNCH test

    if ($LASTEXITCODE -eq 0) {
        Write-Host "Submitting to CrunchDAO..." -ForegroundColor Green
        & $CRUNCH push --message $MSG
    } else {
        Write-Host "Local test failed. Fix errors before submitting." -ForegroundColor Red
    }
}

function Run-Lint {
    Write-Host "Running pylint..." -ForegroundColor Yellow
    & pylint src/ --disable=C0111,R0903,R0913,W0613

    Write-Host "Running flake8..." -ForegroundColor Yellow
    & flake8 src/ --max-line-length=100 --ignore=E203,W503
}

function Format-Code {
    Write-Host "Formatting code with black..." -ForegroundColor Green
    & black src/ scripts/ tests/ --line-length=100
}

function Create-Directories {
    Write-Host "Creating directory structure..." -ForegroundColor Green

    $dirs = @(
        "data",
        "notebooks",
        "src/data",
        "src/features",
        "src/models",
        "src/utils",
        "tests",
        "experiments/configs",
        "experiments/results",
        "resources",
        "submissions_archive",
        "scripts"
    )

    foreach ($dir in $dirs) {
        if (!(Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
            Write-Host "  Created: $dir" -ForegroundColor Gray
        }
    }
}

function Setup-Environment {
    Write-Host "Setting up complete environment..." -ForegroundColor Cyan
    Install-Dependencies
    Create-Directories
    Write-Host ""
    Write-Host "Setup complete! Now run:" -ForegroundColor Green
    Write-Host "  crunch setup structural-break model-v1 --token YOUR_TOKEN" -ForegroundColor Yellow
    Write-Host "Get your token from: https://www.crunchdao.com/competitions/structural-break/submit" -ForegroundColor Gray
}

function Test-Pipeline {
    Write-Host "Testing complete pipeline..." -ForegroundColor Cyan
    & $PYTHON scripts/test_pipeline.py
}

# Main switch statement
switch ($Command.ToLower()) {
    "help"      { Show-Help }
    "setup"     { Setup-Environment }
    "install"   { Install-Dependencies }
    "clean"     { Clean-Project }
    "test"      { Run-Tests }
    "test-crunch" { Run-CrunchTest }
    "train"     { Train-Model }
    "evaluate"  { Evaluate-Model }
    "notebook"  { Start-Notebook }
    "submit"    { Submit-ToCrunchDAO }
    "lint"      { Run-Lint }
    "format"    { Format-Code }
    "dirs"      { Create-Directories }
    "pipeline"  { Test-Pipeline }
    default     {
        Write-Host "Unknown command: $Command" -ForegroundColor Red
        Write-Host ""
        Show-Help
    }
}