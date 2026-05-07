# Audio-Visual Deepfake Detection via Cross-Modal Spectral Alignment

Detects diffusion-based deepfakes by analyzing spectral alignment between audio and video.

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Usage

```bash
# Train model
python src/training/train.py --config configs/base_config.yaml

# Evaluate model
python src/evaluation/evaluate.py --checkpoint path/to/model.pth
```
