"""
Configurable Fusion Model — for rigorous ablation
==================================================
A single model class that can be instantiated with ANY subset of the three
modalities {face, audio, text}. This is what makes the Phase 7 ablation study
*fair*: every configuration shares identical encoder architectures, the same
cross-modal attention mechanism, and the same classifier head — so any
performance difference is attributable to the modalities present, not to
incidental architectural changes.

  - 1 modality  : encoder -> classifier            (no attention; nothing to attend to)
  - >= 2 modalities: encoder per modality -> each queries the others via
                      CrossModalAttention -> concat -> classifier

This is the controlled-variable design an IEEE reviewer expects from an
ablation table.
"""

import torch
import torch.nn as nn

from src.models.encoders import FaceAUEncoder, AudioEncoder, TextEncoder
from src.models.fusion_attention import CrossModalAttention

VALID_MODALITIES = ("face", "audio", "text")


class ConfigurableFusionModel(nn.Module):
    def __init__(
        self,
        modalities,
        n_au: int = 20,
        n_mfcc: int = 120,
        vocab_size: int = 1000,
        embed_dim: int = 64,
        num_classes: int = 2,
        dropout: float = 0.6,
    ):
        super().__init__()
        modalities = tuple(m for m in VALID_MODALITIES if m in modalities)
        if not modalities:
            raise ValueError("at least one modality required")
        self.modalities = modalities

        if "face" in modalities:
            self.face_enc = FaceAUEncoder(n_au=n_au, hidden=embed_dim,
                                          embed_dim=embed_dim, dropout=dropout/2)
        if "audio" in modalities:
            self.audio_enc = AudioEncoder(n_mfcc=n_mfcc, hidden=embed_dim,
                                          embed_dim=embed_dim, dropout=dropout/2)
        if "text" in modalities:
            self.text_enc = TextEncoder(vocab_size=vocab_size, hidden=embed_dim,
                                        embed_dim=embed_dim, dropout=dropout)

        # cross-modal attention only makes sense with >= 2 modalities
        self.use_attention = len(modalities) >= 2
        if self.use_attention:
            self.attn = nn.ModuleDict(
                {m: CrossModalAttention(embed_dim) for m in modalities})

        cls_in = embed_dim * len(modalities)
        self.classifier = nn.Sequential(
            nn.Linear(cls_in, max(cls_in // 2, embed_dim)),
            nn.LayerNorm(max(cls_in // 2, embed_dim)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(cls_in // 2, embed_dim), num_classes),
        )

    def _encode(self, face, audio, text):
        feats = {}
        if "face" in self.modalities:
            feats["face"] = self.face_enc(face)
        if "audio" in self.modalities:
            feats["audio"] = self.audio_enc(audio)
        if "text" in self.modalities:
            feats["text"] = self.text_enc(text)
        return feats

    def forward(self, face, audio, text):
        feats = self._encode(face, audio, text)

        if not self.use_attention:
            fused = next(iter(feats.values()))
        else:
            attended = []
            for m in self.modalities:
                others = torch.stack(
                    [feats[o] for o in self.modalities if o != m], dim=1)
                att, _ = self.attn[m](feats[m], others)
                attended.append(att)
            fused = torch.cat(attended, dim=1)

        return self.classifier(fused)


if __name__ == "__main__":
    B = 4
    face  = torch.randn(B, 200, 20)
    audio = torch.randn(B, 300, 120)
    text  = torch.randn(B, 1000)
    for combo in [("face",), ("audio",), ("text",),
                  ("face", "audio"), ("audio", "text"),
                  ("face", "audio", "text")]:
        m = ConfigurableFusionModel(combo)
        out = m(face, audio, text)
        n = sum(p.numel() for p in m.parameters())
        print(f"{'+'.join(combo):20s} -> logits {tuple(out.shape)}  params={n:,}")
