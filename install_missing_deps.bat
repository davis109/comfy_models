@echo off
echo === Installing Missing StableVITON Dependencies ===

REM Activate the virtual environment
call stableviton_env\Scripts\activate.bat

pip install omegaconf einops kornia pytorch-lightning wandb

echo === Dependencies Installed ===
pause 