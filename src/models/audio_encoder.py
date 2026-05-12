import torch
import torch.nn as nn
import math


class AudioEncoder(nn.Module):
    """Mel spectrogram -> temporal tokens via small Transformer."""

    def __init__(self, n_mels: int = 80, embed_dim: int = 256,
                 num_layers: int = 4, num_heads: int = 4, num_tokens: int = 16):
        super().__init__()
        self.num_tokens = num_tokens
        self.input_proj = nn.Conv1d(n_mels, embed_dim, kernel_size=3, padding=1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(num_tokens)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        mel: (B, 1, n_mels, T)
        returns: (B, num_tokens, embed_dim)
        """
        B = mel.shape[0]
        x = mel.squeeze(1)                    # (B, n_mels, T)
        x = self.input_proj(x)                # (B, embed_dim, T)
        x = self.pool(x)                      # (B, embed_dim, num_tokens)
        x = x.transpose(1, 2)                 # (B, num_tokens, embed_dim)
        return self.transformer(x)            # (B, num_tokens, embed_dim)
