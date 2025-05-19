# StableVITON Setup Script for Windows
# This script sets up and runs a demo of StableVITON virtual try-on model

# Set up error handling
$ErrorActionPreference = "Stop"

Write-Host "Starting StableVITON setup..." -ForegroundColor Green

# Function to check if command exists
function Test-CommandExists {
    param ($command)
    $oldPreference = $ErrorActionPreference
    $ErrorActionPreference = 'stop'
    try {
        if (Get-Command $command) { return $true }
    }
    catch { return $false }
    finally { $ErrorActionPreference = $oldPreference }
}

# Create directory structure
Write-Host "Creating directory structure..." -ForegroundColor Yellow
$dirs = @(
    "StableVITON",
    "StableVITON\ckpts",
    "StableVITON\data",
    "StableVITON\data\test",
    "StableVITON\data\test\image",
    "StableVITON\data\test\image-densepose",
    "StableVITON\data\test\agnostic",
    "StableVITON\data\test\agnostic-mask",
    "StableVITON\data\test\cloth",
    "StableVITON\data\test\cloth_mask",
    "StableVITON\sample_data",
    "StableVITON\sample_data\person",
    "StableVITON\sample_data\cloth",
    "StableVITON\results"
)

foreach ($dir in $dirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}

# Clone repository if it doesn't exist
if (-not (Test-Path "StableVITON\.git")) {
    Write-Host "Cloning StableVITON repository..." -ForegroundColor Yellow
    
    if (Test-CommandExists git) {
        git clone https://github.com/rlawjdghek/StableVITON.git temp_repo
        Copy-Item -Path "temp_repo\*" -Destination "StableVITON\" -Recurse -Force
        Remove-Item -Path "temp_repo" -Recurse -Force
    } else {
        Write-Host "Git is not installed. Please install Git and run this script again." -ForegroundColor Red
        exit 1
    }
}

# Download sample teaser image
if (-not (Test-Path "StableVITON\sample.png")) {
    Write-Host "Downloading sample teaser image..." -ForegroundColor Yellow
    Invoke-WebRequest -Uri "https://raw.githubusercontent.com/rlawjdghek/StableVITON/master/assets/teaser.png" -OutFile "StableVITON\sample.png"
}

# Create demo.py script if it doesn't exist
$demoScript = @'
import os
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='StableVITON Demo')
    parser.add_argument('--person_img', type=str, default='sample_data/person/person.jpg', help='Path to person image')
    parser.add_argument('--cloth_img', type=str, default='sample_data/cloth/cloth.jpg', help='Path to cloth image')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save results')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if the sample image exists
    if os.path.exists('sample.png'):
        # Load and display the sample image
        print("Loading sample teaser image from StableVITON...")
        sample = Image.open('sample.png')
        
        # Save the result
        plt.figure(figsize=(12, 6))
        plt.imshow(np.array(sample))
        plt.axis('off')
        plt.title('StableVITON: Virtual Try-On Demo')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'demo_result.png'))
        plt.close()
        
        # Display information
        print(f"Demo image saved to {os.path.join(args.output_dir, 'demo_result.png')}")
        print("\nStableVITON - Virtual Try-On")
        print("---------------------------")
        print("This is a simplified demo showing a sample from the StableVITON paper.")
        print("For the full functionality, you need to:")
        print("1. Download the model weights from HuggingFace")
        print("2. Prepare a proper dataset with the required structure")
        print("3. Install all the dependencies as listed in the repository")
    else:
        print("Sample image not found. Please make sure to download the teaser image first.")

if __name__ == '__main__':
    main()
'@

if (-not (Test-Path "StableVITON\demo.py")) {
    Write-Host "Creating demo script..." -ForegroundColor Yellow
    Set-Content -Path "StableVITON\demo.py" -Value $demoScript
}

# Setup Python virtual environment
if (-not (Test-Path "StableVITON\stableviton_env")) {
    Write-Host "Setting up Python virtual environment..." -ForegroundColor Yellow
    
    if (Test-CommandExists python) {
        Set-Location StableVITON
        python -m venv stableviton_env
        
        # Activate virtual environment and install packages
        Write-Host "Installing required packages..." -ForegroundColor Yellow
        & .\stableviton_env\Scripts\Activate.ps1
        python -m pip install --upgrade pip
        pip install pillow matplotlib numpy==1.24.3
    } else {
        Write-Host "Python is not installed. Please install Python and run this script again." -ForegroundColor Red
        exit 1
    }
}

# Run demo
Write-Host "Running StableVITON demo..." -ForegroundColor Green
Set-Location StableVITON
if (Test-Path "stableviton_env\Scripts\Activate.ps1") {
    & .\stableviton_env\Scripts\Activate.ps1
    python demo.py
    
    # Show result
    if (Test-Path "results\demo_result.png") {
        Write-Host "Demo completed successfully!" -ForegroundColor Green
        Write-Host "Result saved to: $((Get-Item .\results\demo_result.png).FullName)" -ForegroundColor Green
        
        # Open the image if possible
        try {
            Invoke-Item "results\demo_result.png"
        } catch {
            Write-Host "Could not open the result image automatically." -ForegroundColor Yellow
        }
    } else {
        Write-Host "Demo did not generate the expected output." -ForegroundColor Red
    }
} else {
    Write-Host "Virtual environment activation script not found." -ForegroundColor Red
}

Write-Host "`nTo run the full StableVITON model, you'll need to:" -ForegroundColor Cyan
Write-Host "1. Download model weights from HuggingFace: https://huggingface.co/rlawjdghek/StableVITON" -ForegroundColor Cyan
Write-Host "2. Place them in the 'ckpts' directory" -ForegroundColor Cyan
Write-Host "3. Prepare a proper dataset following the structure in README.md" -ForegroundColor Cyan
Write-Host "4. Run inference.py with appropriate parameters" -ForegroundColor Cyan 