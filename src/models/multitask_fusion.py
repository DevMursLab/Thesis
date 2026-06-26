"""
Multi-Task Fusion Model with Modality Dropout
=============================================
Three novel additions over the Phase 4/7 tri-modal model:

  1. CLINICAL AUDIO (COVAREP+FORMANT):
     AudioEncoder now accepts 199-dim input (MFCC+COVAREP+FORMANT) instead of 120.

  2. MULTI-TASK LEARNING:
     Three prediction heads trained jointly:
       (a) binary_head  : depressed / not   (CrossEntropy, primary task)
       (b) score_head   : PHQ-8 score 0-24  (MSE regression, auxiliary)
       (c) symptom_head : 8 symptoms × {0,1,2,3}  (CE per symptom, auxiliary)
     L_total = L_binary + λ_s·L_score + λ_y·L_symptoms + λ_f·L_fairness

     Multi-task learning acts as implicit regularization on small-N data:
     predicting PHQ-8 sub-scores forces the shared representation to encode
     *which* symptoms are present, not just the binary label.

  3. MODALITY DROPOUT:
     During training, each modality is independently zeroed with probability p.
     This forces the model to be robust to missing modalities at inference —
     a practical clinical constraint (not every patient has all three channels).
     During evaluation, all modalities are used (no dropout).

Architecture:
    FaceAUEncoder(200,20 → 64)  ─┐
    AudioEncoder(300,199 → 64)  ─┤→ CrossModalAttention×3 → concat(192)
    TextEncoder(1000 → 64)      ─┘         │
                                    ┌───────┴──────────┐
                               binary_head(2)  score_head(1)
                                symptom_head(8×4)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.encoders import FaceAUEncoder, AudioEncoder, TextEncoder
from src.models.fusion_attention import CrossModalAttention

VALID_MODALITIES = ("face", "audio", "text")


class MultiTaskFusionModel(nn.Module):
    def __init__(
        self,
        n_au: int       = 20,
        n_audio_feat: int = 199,      # MFCC(120)+COVAREP(74)+FORMANT(5)
        vocab_size: int = 1000,
        embed_dim: int  = 64,
        dropout: float  = 0.6,
        n_symptoms: int = 8,
        modality_dropout_p: float = 0.15,
    ):
        super().__init__()
        self.embed_dim          = embed_dim
        self.modality_dropout_p = modality_dropout_p
        self.n_symptoms         = n_symptoms

        # --- Encoders ---
        self.face_enc  = FaceAUEncoder(n_au=n_au, hidden=embed_dim,
                                       embed_dim=embed_dim, dropout=dropout / 2)
        self.audio_enc = AudioEncoder(n_mfcc=n_audio_feat, hidden=embed_dim,
                                      embed_dim=embed_dim, dropout=dropout / 2)
        self.text_enc  = TextEncoder(vocab_size=vocab_size, hidden=embed_dim,
                                     embed_dim=embed_dim, dropout=dropout)

        # --- Cross-modal attention (each modality queries the other two) ---
        self.attn = nn.ModuleDict({m: CrossModalAttention(embed_dim)
                                   for m in VALID_MODALITIES})

        fused_dim = embed_dim * 3  # 192

        # --- Shared representation layer ---
        self.shared = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # --- Head 1: Binary depression classification ---
        self.binary_head = nn.Linear(fused_dim, 2)

        # --- Head 2: PHQ-8 total score regression (0-24) ---
        self.score_head = nn.Sequential(
            nn.Linear(fused_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid(),            # output in [0,1], scale × 24 in loss
        )

        # --- Head 3: 8-symptom multi-class classification (0/1/2/3 each) ---
        self.symptom_head = nn.Sequential(
            nn.Linear(fused_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(embed_dim * 2, n_symptoms * 4),  # 8×4 logits
        )

    def _apply_modality_dropout(self, feats: dict) -> dict:
        """Zero out each modality with probability p during training."""
        if not self.training:
            return feats
        out = {}
        for m, f in feats.items():
            if torch.rand(1).item() < self.modality_dropout_p:
                out[m] = torch.zeros_like(f)
            else:
                out[m] = f
        return out

    def forward(self, face, audio, text):
        # Encode each modality
        feats = {
            "face":  self.face_enc(face),
            "audio": self.audio_enc(audio),
            "text":  self.text_enc(text),
        }

        # Modality dropout (training only)
        feats = self._apply_modality_dropout(feats)

        # Cross-modal attention
        attended = []
        for m in VALID_MODALITIES:
            others  = torch.stack([feats[o] for o in VALID_MODALITIES if o != m], dim=1)
            att, _  = self.attn[m](feats[m], others)
            attended.append(att)

        fused  = torch.cat(attended, dim=1)   # (B, 192)
        shared = self.shared(fused)           # (B, 192)

        # Head outputs
        logits_binary   = self.binary_head(shared)                          # (B, 2)
        score_raw       = self.score_head(shared).squeeze(-1)               # (B,)
        symptom_logits  = self.symptom_head(shared).view(-1, self.n_symptoms, 4)  # (B,8,4)

        return logits_binary, score_raw, symptom_logits


def multitask_loss(
    logits_binary, score_raw, symptom_logits,
    y_binary, y_score, y_symptoms,
    class_weights=None,
    lambda_score=0.3,
    lambda_symptom=0.2,
):
    """
    L_total = L_CE(binary) + λ_s·L_MSE(score) + λ_y·L_CE(symptoms)

    y_binary   : (B,) int64  {0,1}
    y_score    : (B,) float  [0,24]
    y_symptoms : (B,8) int64 [0,3]
    score_raw  : (B,) float  [0,1] — multiply by 24 for scale
    """
    # Primary: binary cross-entropy
    l_binary = F.cross_entropy(logits_binary, y_binary, weight=class_weights)

    # Auxiliary 1: PHQ-8 score regression — normalize score to [0,1]
    score_pred   = score_raw                         # already in [0,1] (Sigmoid)
    score_target = y_score / 24.0                    # normalize to [0,1]
    l_score      = F.mse_loss(score_pred, score_target)

    # Auxiliary 2: 8-symptom classification — flatten to (B×8,4) vs (B×8,)
    B = symptom_logits.size(0)
    l_symptom = F.cross_entropy(
        symptom_logits.view(B * 8, 4),
        y_symptoms.view(B * 8).long(),
    )

    return l_binary + lambda_score * l_score + lambda_symptom * l_symptom


if __name__ == "__main__":
    import torch
    B = 4
    face  = torch.randn(B, 200, 20)
    audio = torch.randn(B, 300, 199)   # 199-dim clinical audio
    text  = torch.randn(B, 1000)

    model = MultiTaskFusionModel()
    model.train()

    lb, sr, sl = model(face, audio, text)
    print(f"binary logits : {tuple(lb.shape)}")
    print(f"score output  : {tuple(sr.shape)}")
    print(f"symptom logits: {tuple(sl.shape)}")

    n = sum(p.numel() for p in model.parameters())
    print(f"Total params  : {n:,}")

    y_b   = torch.randint(0, 2, (B,))
    y_s   = torch.rand(B) * 24
    y_sym = torch.randint(0, 4, (B, 8))
    loss  = multitask_loss(lb, sr, sl, y_b, y_s, y_sym)
    print(f"Loss          : {loss.item():.4f}")
