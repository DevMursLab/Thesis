"""
Modality-Specific Encoders
===========================
Each encoder maps raw modality features to a shared 256-dim embedding space.
Projecting all three modalities into the same space is what enables
cross-modal attention to compare and fuse them meaningfully.

Face  : AU time-series  (N, T, 20)   → Bi-LSTM → (N, 256)
Audio : MFCC time-series(N, T, 120)  → Bi-LSTM → (N, 256) [reuses AudioLSTM]
Text  : TF-IDF vector   (N, 10000)   → FC      → (N, 256)
"""

import torch
import torch.nn as nn


class FaceAUEncoder(nn.Module):
    """
    Encodes CLNF Action Unit sequences with a Bidirectional LSTM.

    Input : (B, T, n_au)  — time-series of 20 AU channels
    Output: (B, embed_dim) — fixed-size face representation
    """

    def __init__(self, n_au: int = 20, hidden: int = 128,
                 embed_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_au,
            hidden_size=hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        # mean-pool over time (more stable than last-step for variable-length)
        feat = out.mean(dim=1)
        return self.proj(feat)


class TextEncoder(nn.Module):
    """
    Encodes TF-IDF bigram vectors with two FC layers + residual connection.

    Input : (B, vocab_size)
    Output: (B, embed_dim)
    """

    def __init__(self, vocab_size: int = 10000, hidden: int = 512,
                 embed_dim: int = 256, dropout: float = 0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vocab_size, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class AudioEncoder(nn.Module):
    """
    Bi-LSTM encoder for MFCC+delta+delta2 sequences.
    Same architecture as FaceAUEncoder but tuned for audio.

    Input : (B, T, n_mfcc)
    Output: (B, embed_dim)
    """

    def __init__(self, n_mfcc: int = 120, hidden: int = 128,
                 embed_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_mfcc,
            hidden_size=hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        feat = out.mean(dim=1)
        return self.proj(feat)


if __name__ == "__main__":
    B = 4
    face  = FaceAUEncoder(n_au=20)
    audio = AudioEncoder(n_mfcc=120)
    text  = TextEncoder(vocab_size=10000)

    f_out = face(torch.randn(B, 200, 20))
    a_out = audio(torch.randn(B, 300, 120))
    t_out = text(torch.randn(B, 10000))

    print("Face  encoder:", f_out.shape)   # (4, 256)
    print("Audio encoder:", a_out.shape)   # (4, 256)
    print("Text  encoder:", t_out.shape)   # (4, 256)
