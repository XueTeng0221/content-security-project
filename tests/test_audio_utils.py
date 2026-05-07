# tests/test_audio_utils.py
import pytest
import torch
import numpy as np
from src.utils.audio_utils import extract_mel_spectrogram

def test_extract_mel_spectrogram_shape():
    # Create dummy audio: 3 seconds at 16kHz
    audio = torch.randn(1, 48000)
    sample_rate = 16000

    mel_spec = extract_mel_spectrogram(
        audio,
        sample_rate=sample_rate,
        n_fft=2048,
        hop_length=512,
        n_mels=80
    )

    # Expected time frames: ~300 for 3 seconds
    assert mel_spec.shape[0] == 1
    assert mel_spec.shape[1] == 80  # n_mels
    assert mel_spec.shape[2] > 0  # time frames

def test_extract_mel_spectrogram_log_scale():
    # Scale audio to realistic [-1, 1] range; log of small mel values is negative
    audio = torch.randn(1, 48000) * 0.01
    mel_spec = extract_mel_spectrogram(audio, sample_rate=16000)

    # Log scale should produce negative values for quiet audio
    assert torch.any(mel_spec < 0)
