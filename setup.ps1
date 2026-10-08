param(
    [ValidateSet('3.12', '3.13', '3.14', '3.15')]
    [string]$PythonVersion = '3.14'
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py "-$PythonVersion" -m venv .venv
    } else {
        & python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)"
        if ($LASTEXITCODE -ne 0) { throw 'Install Python 3.12 or newer; Python 3.14 is recommended.' }
        & python -m venv .venv
    }
    if ($LASTEXITCODE -ne 0) { throw 'Environment creation failed. Install the selected Python version.' }
}

$gpuPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
& $gpuPython -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)"
if ($LASTEXITCODE -ne 0) { throw 'The existing environment requires Python 3.12 or newer.' }

& $gpuPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'Installer update failed.' }
& $gpuPython -m pip install --upgrade -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'CUDA dependency installation failed.' }

$gpuCheck = @'
import sys
import torch
if torch.version.cuda is None or not torch.cuda.is_available():
    sys.exit("A CUDA-enabled PyTorch installation and an accessible NVIDIA GPU are required. Update the NVIDIA driver.")
print("PyTorch: " + torch.__version__)
print("CUDA runtime: " + torch.version.cuda)
print("GPU: " + torch.cuda.get_device_name(0))
'@
& $gpuPython -c $gpuCheck
if ($LASTEXITCODE -ne 0) { throw 'CUDA GPU initialization failed.' }
