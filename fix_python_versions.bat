@echo off
echo === StableVITON Python Environment Fixer ===
echo This script will set up StableVITON to run with Python 3.10

REM Create a Python 3.10 specific virtual environment
echo Creating virtual environment with Python 3.10...
py -3.10 -m venv stableviton_env

REM Activate the virtual environment
echo Activating virtual environment...
call stableviton_env\Scripts\activate.bat

REM Install compatible versions of dependencies
echo Installing compatible dependencies...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy==1.24.3
pip install opencv-python==4.8.0.76
pip install diffusers transformers accelerate
pip install matplotlib pillow scikit-image
pip install tokenizers --no-binary tokenizers

echo.
echo === Setup Complete ===
echo.
echo To use this environment:
echo 1. Activate it with: call stableviton_env\Scripts\activate.bat
echo 2. Run your StableVITON scripts with: python run_stableviton.py --person zz.png --cloth shirt.png
echo.
echo Press any key to activate the environment now...
pause > nul

call stableviton_env\Scripts\activate.bat
echo Environment activated. You can now run StableVITON scripts.
cmd /k 