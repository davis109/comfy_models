@echo off
echo === Running StableVITON Visualization ===

REM Activate the virtual environment
call stableviton_env\Scripts\activate.bat

REM Get person and cloth image paths from command line arguments
set PERSON_IMG=zz.png
set CLOTH_IMG=shirt.png

echo Visualizing StableVITON prep/results with:
echo - Person image: %PERSON_IMG%
echo - Cloth image: %CLOTH_IMG%
echo.

REM Run the visualization script
python visualize_tryons.py %PERSON_IMG% %CLOTH_IMG%

echo.
echo Visualization completed.
start stableviton_visualization_prep.png
pause 