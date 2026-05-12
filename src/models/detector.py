import torch
import torch.nn as nn
from .visual_encoder import VisualEncoder
from .audio_encoder import AudioEncoder
from .cross_attention import CrossModalAttention


class AVDeepfakeDetector(nn.Module):
    def __init__(self, embed_dim: int = 256, freeze_visual: bool = True, num_classes: int = 2):
        super().__init__()
        self.visual_enc = VisualEncoder(embed_dim=embed_dim, freeze_backbone=freeze_visual)
        self.audio_enc = AudioEncoder(embed_dim=embed_dim)
        self.cross_attn = CrossModalAttention(embed_dim=embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, frames: torch.Tensor, mel: torch.Tensor):
        """
        frames: (B, T, 3, 224, 224)
        mel:    (B, 1, 80, T_mel)
        returns: logits (B, num_classes)
        """
        v = self.visual_enc(frames)          # (B, T, D)
        a = self.audio_enc(mel)              # (B, T_a, D)
        fused = self.cross_attn(v, a)        # (B, T, D)
        pooled = fused.mean(dim=1)           # (B, D)
        return self.classifier(pooled)
