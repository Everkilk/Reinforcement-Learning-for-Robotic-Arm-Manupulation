REM Optimize CPU threading for multi-environment training
set OMP_NUM_THREADS=6
set MKL_NUM_THREADS=6
set NUMEXPR_NUM_THREADS=6

REM Cleanup old experiments automatically (keep last 3)
echo [INFO] Cleaning up old experiment folders...
for /f "skip=10 delims=" %%D in ('dir /AD /B /O-D runs\exp* 2^>nul') do (
    echo [INFO] Removing old experiment: %%D
    rd /s /q "runs\%%D" 2>nul
)
echo.

REM Check if IsaacLab is available
if not exist "C:\Users\PC\IsaacLab\isaaclab.bat" (
    echo [ERROR] IsaacLab not found at C:\Users\PC\IsaacLab
    echo Please ensure Isaac Sim and IsaacLab are properly installed.
    pause
    exit /b 1
)

REM Run the training script using Isaac Sim's Python
C:\Users\PC\IsaacLab\isaaclab.bat -p franka_train.py --task lift --num-envs 8 --algo sac --seed 42 --num_cycles 200 --num_updates 256 --headless

echo.
echo ========================================
echo Training Complete
echo ========================================
pause
