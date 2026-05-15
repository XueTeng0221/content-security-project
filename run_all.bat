@echo off

echo === Stage 1: Pretrain AV emotion-consistency on MELD ===
python -m src.train --config configs/meld.yaml
if errorlevel 1 (echo Pretrain failed & exit /b 1)

echo === Stage 2: Finetune on FakeAVCeleb ===
python -m src.train --config configs/favc.yaml
if errorlevel 1 (echo FakeAVCeleb finetune failed & exit /b 1)

echo === Stage 2: Finetune on FF++ ===
python -m src.train --config configs/ffpp.yaml
if errorlevel 1 (echo FF++ finetune failed & exit /b 1)

echo === Ablation on FakeAVCeleb ===
python -m src.ablation --config configs/favc.yaml --out_dir outputs/ablation/fakeavceleb
if errorlevel 1 (echo Ablation FakeAVCeleb failed & exit /b 1)

echo === Ablation on FF++ ===
python -m src.ablation --config configs/ffpp.yaml --out_dir outputs/ablation/ffplusplus
if errorlevel 1 (echo Ablation FF++ failed & exit /b 1)

echo === All done ===
