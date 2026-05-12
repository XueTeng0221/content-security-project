@REM @echo off
@REM echo === Step 1: Tests ===
@REM pytest tests/ -v
@REM if errorlevel 1 (echo Tests failed & exit /b 1)

echo === Step 1: Train ===
python -m src.train --config configs/train.yaml

echo === Step 2: Ablation ===
python -m src.ablation --config configs/ablation.yaml
