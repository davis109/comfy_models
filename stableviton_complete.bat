@echo off
echo.
echo ===================================================
echo        STABLEVITON COMPLETE SOLUTION
echo ===================================================
echo.

REM Check if environment exists, if not create it
if not exist stableviton_env (
    echo Virtual environment not found, creating one...
    
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
    pip install omegaconf einops kornia pytorch-lightning
    
    echo.
    echo Environment setup complete.
    echo.
) else (
    echo Using existing virtual environment...
    call stableviton_env\Scripts\activate.bat
)

REM Get person and cloth image paths from command line arguments
set PERSON_IMG=%1
set CLOTH_IMG=%2

REM Check if arguments were provided
if "%PERSON_IMG%"=="" set PERSON_IMG=zz.png
if "%CLOTH_IMG%"=="" set CLOTH_IMG=shirt.png

echo Running StableVITON with:
echo - Person image: %PERSON_IMG%
echo - Cloth image: %CLOTH_IMG%
echo.

REM First check if StableVITON folder structure exists
if not exist StableVITON\data (
    echo Creating data directory structure...
    mkdir StableVITON\data\test\image
    mkdir StableVITON\data\test\cloth
    mkdir StableVITON\data\test\cloth-mask
    mkdir StableVITON\data\test\agnostic
    mkdir StableVITON\data\test\agnostic-mask
    mkdir StableVITON\data\test\image-densepose
)

REM Run the dataset preparation
echo Preparing dataset...
python stableviton_dataset_prep.py %PERSON_IMG% %CLOTH_IMG%

REM Run the visualization
echo Creating visualization...
python visualize_tryons.py %PERSON_IMG% %CLOTH_IMG%

echo.
echo ===================================================
echo            STABLEVITON COMPLETE
echo ===================================================
echo.
echo Dataset preparation complete and visualization created.
echo.
echo To generate the final try-on results, you would need to run:
echo   python StableVITON\inference.py --config_path StableVITON\configs\VITONHD.yaml 
echo      --model_load_path StableVITON\ckpts\VITONHD_PBE_pose.ckpt --batch_size 1 
echo      --data_root_dir StableVITON\data --save_dir results_custom 
echo      --denoise_steps 50 --img_H 512 --img_W 384
echo.
echo Opening visualization...
start stableviton_visualization_prep.png

pause 