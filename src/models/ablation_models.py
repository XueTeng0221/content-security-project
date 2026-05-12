import torch
import torch.nn as nn
from .visual_encoder import VisualEncoder
from .audio_encoder import AudioEncoder
from .cross_attention import CrossModalAttention


class VisualOnlyDetector(nn.Module):
    def __init__(self, embed_dim: int = 256, num_classes: int = 2):
        super().__init__()
        self.visual_enc = VisualEncoder(embed_dim=embed_dim)
        self.classifier = nn.Sequential(nn.Linear(embed_dim, 64), nn.GELU(), nn.Linear(64, num_classes))

    def forward(self, frames, mel=None):
        v = self.visual_enc(frames).mean(dim=1)
        return self.classifier(v)


class AudioOnlyDetector(nn.Module):
    def __init__(self, embed_dim: int = 256, num_classes: int = 2):
        super().__init__()
        self.audio_enc = AudioEncoder(embed_dim=embed_dim)
        self.classifier = nn.Sequential(nn.Linear(embed_dim, 64), nn.GELU(), nn.Linear(64, num_classes))

    def forward(self, frames=None, mel=None):
        a = self.audio_enc(mel).mean(dim=1)
        return self.classifier(a)


class ConcatFusionDetector(nn.Module):
    """Dual-stream with simple concat fusion — no cross-attention."""
    def __init__(self, embed_dim: int = 256, num_classes: int = 2):
        super().__init__()
        self.visual_enc = VisualEncoder(embed_dim=embed_dim)
        self.audio_enc  = AudioEncoder(embed_dim=embed_dim)
        self.classifier = nn.Sequential(nn.Linear(embed_dim * 2, 64), nn.GELU(), nn.Linear(64, num_classes))

    def forward(self, frames, mel):
        v = self.visual_enc(frames).mean(dim=1)
        a = self.audio_enc(mel).mean(dim=1)
        return self.classifier(torch.cat([v, a], dim=-1))
