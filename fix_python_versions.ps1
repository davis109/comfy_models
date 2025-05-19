Write-Host "=== StableVITON Python Environment Fixer ===" -ForegroundColor Cyan
Write-Host "This script will set up StableVITON to run with Python 3.10" -ForegroundColor Cyan
Write-Host ""

# Create a Python 3.10 specific virtual environment
Write-Host "Creating virtual environment with Python 3.10..." -ForegroundColor Yellow
py -3.10 -m venv stableviton_env

# Activate the virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
. .\stableviton_env\Scripts\Activate.ps1

# Install compatible versions of dependencies
Write-Host "Installing compatible dependencies..." -ForegroundColor Yellow
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy==1.24.3
pip install opencv-python==4.8.0.76
pip install diffusers transformers accelerate
pip install matplotlib pillow scikit-image
pip install tokenizers --no-binary tokenizers

Write-Host ""
Write-Host "=== Setup Complete ===" -ForegroundColor Green
Write-Host ""
Write-Host "To use this environment:" -ForegroundColor White
Write-Host "1. Activate it with: . .\stableviton_env\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "2. Run your StableVITON scripts with: python run_stableviton.py --person zz.png --cloth shirt.png" -ForegroundColor White
Write-Host ""
Write-Host "Environment activated. You can now run StableVITON scripts." -ForegroundColor Green

# Keep the PowerShell window open
$null = Read-Host "Press Enter to exit..." 