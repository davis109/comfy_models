@echo off
echo === Running StableVITON with Python 3.10 ===

REM Check if environment exists
if not exist stableviton_env (
    echo Virtual environment not found!
    echo Please run fix_python_versions.bat first to set up the environment.
    pause
    exit /b
)

REM Activate the virtual environment
call stableviton_env\Scripts\activate.bat

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

REM Run the StableVITON script
python run_stableviton.py --person %PERSON_IMG% --cloth %CLOTH_IMG%

echo.
echo StableVITON completed.
pause 