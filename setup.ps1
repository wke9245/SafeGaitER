param([ValidateSet('cpu', 'cu124')][string]$TorchIndex = 'cpu')
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    & python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Environment creation failed.' }
}
& .\.venv\Scripts\python.exe -m pip install torch==2.6.0 --index-url "https://download.pytorch.org/whl/$TorchIndex"
if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed.' }
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
