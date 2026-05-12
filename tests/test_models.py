import torch
import pytest
from src.models.visual_encoder import VisualEncoder
from src.models.audio_encoder import AudioEncoder
from src.models.cross_attention import CrossModalAttention
from src.models.detector import AVDeepfakeDetector


B, T, D = 2, 4, 64


@pytest.fixture
def visual_enc():
    return VisualEncoder(embed_dim=D, freeze_backbone=False)


@pytest.fixture
def audio_enc():
    return AudioEncoder(n_mels=80, embed_dim=D, num_layers=1, num_heads=4, num_tokens=T)


def test_visual_encoder_shape(visual_enc):
    frames = torch.randn(B, T, 3, 224, 224)
    out = visual_enc(frames)
    assert out.shape == (B, T, D)


def test_audio_encoder_shape(audio_enc):
    mel = torch.randn(B, 1, 80, 200)
    out = audio_enc(mel)
    assert out.shape == (B, T, D)


def test_cross_attention_shape():
    ca = CrossModalAttention(embed_dim=D, num_heads=4, num_layers=1)
    v = torch.randn(B, T, D)
    a = torch.randn(B, T, D)
    out = ca(v, a)
    assert out.shape == (B, T, D)


def test_detector_output_shape():
    model = AVDeepfakeDetector(embed_dim=D, freeze_visual=False)
    frames = torch.randn(B, T, 3, 224, 224)
    mel = torch.randn(B, 1, 80, 200)
    logits = model(frames, mel)
    assert logits.shape == (B, 2)


def test_detector_no_nan():
    model = AVDeepfakeDetector(embed_dim=D, freeze_visual=False)
    frames = torch.randn(B, T, 3, 224, 224)
    mel = torch.randn(B, 1, 80, 200)
    logits = model(frames, mel)
    assert not torch.isnan(logits).any()
