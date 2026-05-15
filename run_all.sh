#!/usr/bin/env bash
set -e

echo "=== Stage 1: Pretrain AV emotion-consistency on MELD ==="
python -m src.train --config configs/meld.yaml

echo "=== Stage 2: Finetune on FakeAVCeleb ==="
python -m src.train --config configs/favc.yaml

echo "=== Stage 2: Finetune on FF++ ==="
python -m src.train --config configs/ffpp.yaml

echo "=== Ablation on FakeAVCeleb ==="
python -m src.ablation --config configs/favc.yaml --out_dir outputs/ablation/fakeavceleb

echo "=== Ablation on FF++ ==="
python -m src.ablation --config configs/ffpp.yaml --out_dir outputs/ablation/ffplusplus

echo "=== All done ==="
