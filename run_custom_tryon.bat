@echo off
echo === Running StableVITON Custom Try-on ===

REM Activate the virtual environment
call stableviton_env\Scripts\activate.bat

REM Get person and cloth image paths from command line arguments
set PERSON_IMG=zz.png
set CLOTH_IMG=shirt.png

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
python stableviton_dataset_prep.py %PERSON_IMG% %CLOTH_IMG%

REM Run the StableVITON inference
python StableVITON\inference.py --config_path StableVITON\configs\VITONHD.yaml --model_load_path StableVITON\ckpts\VITONHD_PBE_pose.ckpt --batch_size 1 --data_root_dir StableVITON\data --save_dir results_custom --denoise_steps 50 --img_H 512 --img_W 384

echo.
echo StableVITON completed. Check the results_custom folder.
pause 