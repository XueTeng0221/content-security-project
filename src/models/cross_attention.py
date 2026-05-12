import torch
import torch.nn as nn


class CrossModalAttention(nn.Module):
    """
    Cross-Attention: visual tokens as Query, audio tokens as Key/Value.
    Models emotion alignment between face expressions and speech.
    """

    def __init__(self, embed_dim: int = 256, num_heads: int = 8, num_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "cross_attn": nn.MultiheadAttention(embed_dim, num_heads, batch_first=True),
                "norm1": nn.LayerNorm(embed_dim),
                "ffn": nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * 4),
                    nn.GELU(),
                    nn.Linear(embed_dim * 4, embed_dim),
                ),
                "norm2": nn.LayerNorm(embed_dim),
            })
            for _ in range(num_layers)
        ])

    def forward(self, visual: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        """
        visual: (B, T_v, embed_dim)  — Query
        audio:  (B, T_a, embed_dim)  — Key / Value
        returns: (B, T_v, embed_dim)
        """
        x = visual
        for layer in self.layers:
            attn_out, _ = layer["cross_attn"](x, audio, audio)
            x = layer["norm1"](x + attn_out)
            x = layer["norm2"](x + layer["ffn"](x))
        return x
