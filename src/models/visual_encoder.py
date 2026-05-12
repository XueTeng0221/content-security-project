import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights


class VisualEncoder(nn.Module):
    """ViT-B/16 frame encoder. Returns per-frame patch tokens."""

    def __init__(self, embed_dim: int = 256, freeze_backbone: bool = True):
        super().__init__()
        vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        # Remove classification head; keep patch embedding + transformer
        self.patch_embed = vit.conv_proj
        self.class_token = vit.class_token
        self.pos_embed = vit.encoder.pos_embedding
        self.encoder = vit.encoder.layers
        self.ln = vit.encoder.ln
        if freeze_backbone:
            for p in list(self.patch_embed.parameters()) + list(self.encoder.parameters()):
                p.requires_grad_(False)
        self.proj = nn.Linear(768, embed_dim)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        frames: (B, T, 3, 224, 224)
        returns: (B, T, embed_dim)
        """
        B, T, C, H, W = frames.shape
        x = frames.view(B * T, C, H, W)

        x = self.patch_embed(x)                          # (BT, 768, 14, 14)
        x = x.flatten(2).transpose(1, 2)                 # (BT, 196, 768)
        cls = self.class_token.expand(B * T, -1, -1)     # (BT, 1, 768)
        x = torch.cat([cls, x], dim=1)                   # (BT, 197, 768)
        x = x + self.pos_embed
        for layer in self.encoder:
            x = layer(x)
        x = self.ln(x)
        cls_out = x[:, 0]                                 # (BT, 768)
        return self.proj(cls_out).view(B, T, -1)          # (B, T, embed_dim)
