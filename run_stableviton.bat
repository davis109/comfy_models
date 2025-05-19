@echo off
echo ======================================================
echo            STABLEVITON VIRTUAL TRY-ON SYSTEM
echo ======================================================
echo.

if "%~1"=="" goto :no_args
if "%~2"=="" goto :no_args

echo Running StableVITON with:
echo Person image: %1
echo Clothing image: %2
echo.

python run_stableviton.py --person %1 --cloth %2
goto :end

:no_args
echo Running StableVITON with default images (zz.png and shirt.png)
echo.
python run_stableviton.py

:end
echo.
echo Press any key to exit...
pause > nul 