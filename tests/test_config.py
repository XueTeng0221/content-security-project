# tests/test_config.py
import pytest
from pathlib import Path
from src.utils.config import Config

def test_config_load_from_yaml():
    config = Config.from_yaml("configs/base_config.yaml")
    assert config.audio.sample_rate == 16000
    assert config.audio.n_mels == 80
    assert config.model.embed_dim == 256
