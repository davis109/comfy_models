# StableVITON Dependencies Installer
# This script installs all required dependencies for StableVITON

Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "         STABLEVITON DEPENDENCIES INSTALLER" -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python is installed
try {
    $pythonVersion = python --version
    Write-Host "Found Python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "Python not found! Please install Python 3.10 first." -ForegroundColor Red
    exit 1
}

# Create requirements.txt with all dependencies
$requirementsContent = @"
pillow
numpy
matplotlib
opencv-python
torch==1.13.1
torchvision==0.14.1
omegaconf
tqdm
einops
transformers
kornia
scikit-image
pytorch-lightning==1.5.0
diffusers==0.14.0
"@

# Write to requirements.txt
Set-Content -Path "stableviton_requirements.txt" -Value $requirementsContent
Write-Host "Created requirements file with all dependencies" -ForegroundColor Green

# Install CUDA Toolkit 11.7 if not installed (optional step)
# Uncomment if needed:
# Write-Host "Note: This script doesn't install CUDA. If needed, please install CUDA 11.7 manually." -ForegroundColor Yellow

# Install all requirements at once
Write-Host ""
Write-Host "Installing all dependencies (this may take a while)..." -ForegroundColor Yellow
python -m pip install --no-cache-dir -r stableviton_requirements.txt

# Check if installation was successful
if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "All dependencies installed successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "You can now run StableVITON with:" -ForegroundColor Cyan
    Write-Host ".\run_stableviton.bat person.jpg clothing.jpg" -ForegroundColor Cyan
} else {
    Write-Host ""
    Write-Host "Error installing dependencies. Please check the error messages above." -ForegroundColor Red
}

# Wait for user input before closing
Write-Host ""
Write-Host "Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") 