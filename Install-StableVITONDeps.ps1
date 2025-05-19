# StableVITON Dependencies Installer PowerShell Script
# This script installs all required dependencies for StableVITON in one go

# Display header
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "         STABLEVITON DEPENDENCIES INSTALLER" -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python is installed
try {
    $pythonVersion = python --version
    Write-Host "Found Python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "Error: Python not found! Please install Python 3.10 first." -ForegroundColor Red
    Write-Host "Press any key to exit..."
    $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit 1
}

# Install main dependencies first (everything except PyTorch)
$mainPackages = @(
    "pillow",
    "numpy",
    "matplotlib",
    "opencv-python",
    "omegaconf",
    "tqdm",
    "einops",
    "transformers",
    "kornia",
    "scikit-image",
    "diffusers==0.14.0"
)

Write-Host "Installing main dependencies..." -ForegroundColor Yellow
$pipCommand = "python -m pip install --no-cache-dir " + ($mainPackages -join " ")
Write-Host "Running: $pipCommand" -ForegroundColor Gray
Invoke-Expression $pipCommand

# Install PyTorch with CUDA support using direct URLs
Write-Host ""
Write-Host "Installing PyTorch 1.13.1 with CUDA 11.7 support..." -ForegroundColor Yellow
$torchCommand = "python -m pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117"
Write-Host "Running: $torchCommand" -ForegroundColor Gray
Invoke-Expression $torchCommand

# Install PyTorch Lightning with specific version
Write-Host ""
Write-Host "Installing PyTorch Lightning..." -ForegroundColor Yellow
$lightningCommand = "python -m pip install pytorch-lightning==1.5.0"
Write-Host "Running: $lightningCommand" -ForegroundColor Gray
Invoke-Expression $lightningCommand

# Check if installation was successful
if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "All dependencies installed successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "You can now run StableVITON with:" -ForegroundColor Cyan
    Write-Host "python run_stableviton.py --person zz.png --cloth shirt.png" -ForegroundColor Cyan
} else {
    Write-Host ""
    Write-Host "Error installing some dependencies. Please check the error messages above." -ForegroundColor Red
    Write-Host "You may need to install the problematic packages manually." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Press any key to exit..."
$Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") 