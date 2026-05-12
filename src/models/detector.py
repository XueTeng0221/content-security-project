import torch
import torch.nn as nn
import torch.nn.functional as F
from .visual_encoder import VisualEncoder
from .audio_encoder import AudioEncoder
from .cross_attention import CrossModalAttention


class AVDeepfakeDetector(nn.Module):
    def __init__(self, embed_dim: int = 256, freeze_visual: bool = True,
                 num_classes: int = 2, infonce_temp: float = 0.07):
        super().__init__()
        self.visual_enc = VisualEncoder(embed_dim=embed_dim, freeze_backbone=freeze_visual)
        self.audio_enc = AudioEncoder(embed_dim=embed_dim)
        self.cross_attn = CrossModalAttention(embed_dim=embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, num_classes),
        )
        self.temp = infonce_temp

    def _infonce(self, v: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Symmetric InfoNCE on mean-pooled visual/audio embeddings."""
        v = F.normalize(v, dim=-1)   # (B, D)
        a = F.normalize(a, dim=-1)   # (B, D)
        logits = v @ a.T / self.temp  # (B, B)
        labels = torch.arange(v.size(0), device=v.device)
        return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

    def forward(self, frames: torch.Tensor, mel: torch.Tensor):
        """
        frames: (B, T, 3, 224, 224)
        mel:    (B, 1, 80, T_mel)
        returns: (logits (B, num_classes), align_loss scalar)
        """
        v = self.visual_enc(frames)          # (B, T, D)
        a = self.audio_enc(mel)              # (B, T_a, D)
        fused = self.cross_attn(v, a)        # (B, T, D)
        v_pool = v.mean(dim=1)               # (B, D)
        a_pool = a.mean(dim=1)               # (B, D)
        align_loss = self._infonce(v_pool, a_pool)
        pooled = fused.mean(dim=1)           # (B, D)
        return self.classifier(pooled), align_loss
