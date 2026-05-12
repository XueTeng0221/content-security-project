import torch
import pytest
from src.data.transforms import extract_mel, video_transform
from PIL import Image
import numpy as np


def test_extract_mel_shape():
    waveform = torch.randn(1, 16000)  # 1s mono
    mel = extract_mel(waveform)
    assert mel.shape[0] == 1
    assert mel.shape[1] == 80


def test_extract_mel_stereo_to_mono():
    waveform = torch.randn(2, 16000)
    mel = extract_mel(waveform)
    assert mel.shape[0] == 1


def test_video_transform_output():
    img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
    t = video_transform(img)
    assert t.shape == (3, 224, 224)
    assert t.dtype == torch.float32
