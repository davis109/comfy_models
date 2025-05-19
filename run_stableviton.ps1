# StableVITON Inference Script
# This script automates the process of running StableVITON inference

$ErrorActionPreference = "Stop"

Write-Host "StableVITON Inference Script" -ForegroundColor Green
Write-Host "==============================" -ForegroundColor Green

# Function to create directory if it doesn't exist
function EnsureDirectory {
    param([string]$path)
    if (-not (Test-Path $path)) {
        New-Item -Path $path -ItemType Directory -Force | Out-Null
        Write-Host "Created directory: $path" -ForegroundColor Yellow
    }
}

# 1. Check if checkpoint files exist
Write-Host "Checking for checkpoint files..." -ForegroundColor Cyan
$poseModelExists = Test-Path "VITONHD_PBE_POSE.ckpt"
$vaeModelExists = Test-Path "VITONHD_VAE_finetuning.ckpt"

if (-not $poseModelExists) {
    Write-Host "Error: VITONHD_PBE_POSE.ckpt not found in current directory" -ForegroundColor Red
    exit 1
}

if (-not $vaeModelExists) {
    Write-Host "Warning: VITONHD_VAE_finetuning.ckpt not found. VAE fine-tuning will be skipped." -ForegroundColor Yellow
}

# 2. Ensure StableVITON directories exist
Write-Host "Creating required directories..." -ForegroundColor Cyan
EnsureDirectory "StableVITON/ckpts"
EnsureDirectory "StableVITON/data/test/image"
EnsureDirectory "StableVITON/data/test/cloth"
EnsureDirectory "StableVITON/data/test/cloth_mask"
EnsureDirectory "StableVITON/data/test/image-densepose"
EnsureDirectory "StableVITON/data/test/agnostic"
EnsureDirectory "StableVITON/data/test/agnostic-mask"
EnsureDirectory "results_inference"

# 3. Copy checkpoint files
Write-Host "Copying checkpoint files to StableVITON/ckpts..." -ForegroundColor Cyan
Copy-Item -Path "VITONHD_PBE_POSE.ckpt" -Destination "StableVITON/ckpts/VITONHD_PBE_pose.ckpt" -Force
Write-Host "Copied VITONHD_PBE_POSE.ckpt to StableVITON/ckpts/VITONHD_PBE_pose.ckpt" -ForegroundColor Green

if ($vaeModelExists) {
    Copy-Item -Path "VITONHD_VAE_finetuning.ckpt" -Destination "StableVITON/ckpts/VITONHD_VAE_finetuning.ckpt" -Force
    Write-Host "Copied VITONHD_VAE_finetuning.ckpt to StableVITON/ckpts/VITONHD_VAE_finetuning.ckpt" -ForegroundColor Green
}

# 4. Process test pairs
Write-Host "Processing test pairs..." -ForegroundColor Cyan
$pairsFile = "test_pairs.txt"
if (-not (Test-Path $pairsFile)) {
    Write-Host "Warning: $pairsFile not found. Skipping test pairs processing." -ForegroundColor Yellow
} else {
    $pairs = Get-Content $pairsFile | Select-Object -First 5
    $totalPairs = $pairs.Count
    $currentPair = 0
    
    foreach ($pair in $pairs) {
        $currentPair++
        $items = $pair -split '\s+'
        if ($items.Count -ge 2) {
            $image = $items[0]
            $cloth = $items[1]
            
            Write-Host "Processing pair $currentPair/$totalPairs: $image + $cloth" -ForegroundColor Cyan
            
            # Copy image files
            if (Test-Path "test/image/$image") {
                Copy-Item -Path "test/image/$image" -Destination "StableVITON/data/test/image/$image" -Force
            } else {
                Write-Host "Warning: File test/image/$image not found" -ForegroundColor Yellow
            }
            
            if (Test-Path "test/cloth/$cloth") {
                Copy-Item -Path "test/cloth/$cloth" -Destination "StableVITON/data/test/cloth/$cloth" -Force
            } else {
                Write-Host "Warning: File test/cloth/$cloth not found" -ForegroundColor Yellow
            }
            
            # Copy other necessary files if they exist
            if (Test-Path "test/cloth-mask/$cloth") {
                Copy-Item -Path "test/cloth-mask/$cloth" -Destination "StableVITON/data/test/cloth_mask/$cloth" -Force
            } else {
                Write-Host "Warning: File test/cloth-mask/$cloth not found" -ForegroundColor Yellow
            }
            
            if (Test-Path "test/image-densepose/$image") {
                Copy-Item -Path "test/image-densepose/$image" -Destination "StableVITON/data/test/image-densepose/$image" -Force
            } else {
                Write-Host "Warning: File test/image-densepose/$image not found" -ForegroundColor Yellow
            }
            
            if (Test-Path "test/agnostic-v3.2/$image") {
                Copy-Item -Path "test/agnostic-v3.2/$image" -Destination "StableVITON/data/test/agnostic/$image" -Force
            } else {
                Write-Host "Warning: File test/agnostic-v3.2/$image not found" -ForegroundColor Yellow
            }
            
            if (Test-Path "test/agnostic-mask/$image") {
                Copy-Item -Path "test/agnostic-mask/$image" -Destination "StableVITON/data/test/agnostic-mask/$image" -Force
            } else {
                Write-Host "Warning: File test/agnostic-mask/$image not found" -ForegroundColor Yellow
            }
        }
    }
}

# 5. Configure for VAE fine-tuning if available
if ($vaeModelExists) {
    Write-Host "Configuring StableVITON to use VAE fine-tuning..." -ForegroundColor Cyan
    try {
        python StableVITON/use_vae.py --config_file StableVITON/configs/VITONHD.yaml --vae_ckpt StableVITON/ckpts/VITONHD_VAE_finetuning.ckpt
        Write-Host "VAE configuration completed" -ForegroundColor Green
    } catch {
        Write-Host "Error configuring VAE: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "Continuing without VAE..." -ForegroundColor Yellow
    }
}

# 6. Run inference
Write-Host "Running StableVITON inference..." -ForegroundColor Cyan
try {
    $cmd = "python StableVITON/inference.py --config_path StableVITON/configs/VITONHD.yaml --model_load_path StableVITON/ckpts/VITONHD_PBE_pose.ckpt --batch_size 1 --data_root_dir StableVITON/data --save_dir results_inference --denoise_steps 50 --img_H 512 --img_W 384"
    Write-Host "Executing: $cmd" -ForegroundColor Yellow
    Invoke-Expression $cmd
    Write-Host "Inference completed successfully!" -ForegroundColor Green
    Write-Host "Results saved to: results_inference" -ForegroundColor Green
} catch {
    Write-Host "Error running inference: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host "StableVITON Inference Script Completed" -ForegroundColor Green 