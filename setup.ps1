# One-time setup on Windows: virtualenv, dependencies, and a local .env from the template.
# Run from PowerShell:  .\setup.ps1
# If scripts are blocked: powershell -ExecutionPolicy Bypass -File .\setup.ps1
$ErrorActionPreference = 'Stop'
Set-Location -Path $PSScriptRoot

function Invoke-Checked {
    # Native commands don't throw on failure, so check the exit code ourselves.
    param([string]$Exe, [string[]]$Arguments)
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed ($LASTEXITCODE): $Exe $($Arguments -join ' ')"
    }
}

function Find-Python {
    # The Microsoft Store 'python' stub exists but fails, so probe each candidate.
    $candidates = @()
    if ($env:PYTHON) { $candidates += $env:PYTHON }
    $candidates += 'python', 'py', 'python3'
    foreach ($candidate in $candidates) {
        if (-not (Get-Command $candidate -ErrorAction SilentlyContinue)) { continue }
        try {
            & $candidate --version *> $null
            if ($LASTEXITCODE -eq 0) { return $candidate }
        } catch { }
    }
    throw 'Python 3.11+ not found. Install it from https://www.python.org/downloads/ or set $env:PYTHON.'
}

if (-not (Test-Path .venv)) {
    $python = Find-Python
    Write-Host 'Creating virtualenv in .venv'
    Invoke-Checked $python @('-m', 'venv', '.venv')
}

$venvPy = '.venv\Scripts\python.exe'
Invoke-Checked $venvPy @('-m', 'pip', 'install', '--upgrade', 'pip')
Invoke-Checked $venvPy @('-m', 'pip', 'install', '-r', 'requirements.txt')

if (-not (Test-Path .env)) {
    Copy-Item .env.example .env
    Write-Host 'Created .env from .env.example. Fill in your league details.'
} else {
    Write-Host '.env already exists, leaving it alone.'
}

Write-Host ''
Write-Host 'Done. Next:'
Write-Host '  .venv\Scripts\Activate.ps1'
Write-Host '  python -m ninecat.ingest     # pull and cache league data'
Write-Host '  streamlit run app.py'
