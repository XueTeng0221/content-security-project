@REM @echo off
@REM echo === Step 1: Tests ===
@REM pytest tests/ -v
@REM if errorlevel 1 (echo Tests failed & exit /b 1)

echo === Step 1: Train ===

echo === Training on FFPlusPlus dataset ===
python -m src.train --config configs/ffpp.yaml

echo === Training on FakeAVCeleb dataset ===
python -m src.train --config configs/favc.yaml

echo === Step 2: Ablation ===

echo == Ablation on FFPlusPlus dataset ==
python -m src.ablation --config configs/ffpp.yaml

echo == Ablation on FakeAVCeleb dataset ==
python -m src.ablation --config configs/favc.yaml
