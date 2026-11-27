@echo off
setlocal

cd /d "%~dp0.."

echo ========================================
echo HelixRT Full Training Pipeline
echo ========================================

python -c "import torch; assert torch.cuda.is_available(), 'CUDA is not available'"

if errorlevel 1 exit /b 1

for %%M in (standard augmented invariant) do (

    for %%S in (42 123 456) do (

        if "%%M_%%S"=="standard_42" (

            echo Skipping completed standard seed 42 run.

        ) else (

            echo.
            echo Training model: %%M
            echo Random seed: %%S
            echo.

            python -m src.train ^
                --model %%M ^
                --epochs 100 ^
                --batch-size 128 ^
                --learning-rate 0.001 ^
                --patience 15 ^
                --seed %%S ^
                --num-workers 0

            if errorlevel 1 (
                echo Training failed.
                exit /b 1
            )
        )
    )
)

echo.
echo ========================================
echo All remaining experiments completed.
echo ========================================

endlocal