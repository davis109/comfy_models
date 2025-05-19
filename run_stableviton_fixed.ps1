Write-Host "=== Running StableVITON with Python 3.10 ===" -ForegroundColor Cyan

# Check if environment exists
if (-not (Test-Path -Path "stableviton_env")) {
    Write-Host "Virtual environment not found!" -ForegroundColor Red
    Write-Host "Please run fix_python_versions.ps1 first to set up the environment." -ForegroundColor Yellow
    Read-Host "Press Enter to exit..."
    exit
}

# Activate the virtual environment
Write-Host "Activating Python 3.10 environment..." -ForegroundColor Yellow
. .\stableviton_env\Scripts\Activate.ps1

# Get person and cloth image paths from command line arguments
$PERSON_IMG = if ($args[0]) { $args[0] } else { "zz.png" }
$CLOTH_IMG = if ($args[1]) { $args[1] } else { "shirt.png" }

Write-Host "Running StableVITON with:" -ForegroundColor Green
Write-Host "- Person image: $PERSON_IMG" -ForegroundColor White
Write-Host "- Cloth image: $CLOTH_IMG" -ForegroundColor White
Write-Host ""

# Run the StableVITON script
python run_stableviton.py --person $PERSON_IMG --cloth $CLOTH_IMG

Write-Host ""
Write-Host "StableVITON completed." -ForegroundColor Green
Read-Host "Press Enter to exit..." 