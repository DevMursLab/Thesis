"""
Cross-Modal Attention Fusion — Core Novel Architecture
=======================================================
IEEE-level novelty: instead of naively concatenating face+audio+text features,
this module lets each modality *attend* to the other two.

Intuition:
  When a person looks sad (face) AND sounds monotone (audio), depression is
  more likely than if only one signal is present. Cross-modal attention
  captures this inter-modal dependency explicitly.

Architecture (per modality):
  For each modality m ∈ {face, audio, text}:
    query  = embed_m
    keys   = stack of the OTHER two modality embeddings
    values = same keys
    → scaled dot-product attention → context vector
    → residual add + layer-norm → attended_m

  Final: concat([attended_face, attended_audio, attended_text])
       → FC classifier head
       → logits (depressed / not depressed)

References:
  Tsai et al. "Multimodal Transformer for Unaligned Multimodal Language
  Sequences." ACL 2019.  (inspirational, not copied)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.encoders import FaceAUEncoder, AudioEncoder, TextEncoder


class CrossModalAttention(nn.Module):
    """
    One modality queries the other two.
    Scaled dot-product attention (single-head for interpretability).

    query : (B, D)
    keys  : (B, 2, D)  ← the other two modalities stacked
    → output: (B, D)   ← attended representation
    """

    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.scale = embed_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, query, context):
        """
        query  : (B, D)
        context: (B, K, D)  K = number of other modalities (2)
        """
        Q = self.q_proj(query).unsqueeze(1)          # (B, 1, D)
        K = self.k_proj(context)                      # (B, K, D)
        V = self.v_proj(context)                      # (B, K, D)

        attn = torch.bmm(Q, K.transpose(1, 2)) * self.scale   # (B, 1, K)
        attn = F.softmax(attn, dim=-1)                          # (B, 1, K)
        out  = torch.bmm(attn, V).squeeze(1)                    # (B, D)
        out  = self.out_proj(out)

        # residual + layer-norm
        return self.norm(query + out), attn.squeeze(1)          # (B, D), (B, K)


class TriModalFusionModel(nn.Module):
    """
    Full tri-modal depression detection model.

    Inputs:
      face_seq  : (B, T_f, n_au)      CLNF Action Unit sequence
      audio_seq : (B, T_a, n_mfcc)    MFCC + delta + delta2 sequence
      text_vec  : (B, vocab_size)      TF-IDF bigram vector

    Output (training):
      logits    : (B, 2)              depressed / not-depressed
      attn_weights: dict with per-modality attention scores (for XAI)

    Output (feature extraction):
      fused     : (B, embed_dim*3)    concatenated attended embeddings
    """

    def __init__(
        self,
        n_au: int = 20,
        n_mfcc: int = 120,
        vocab_size: int = 1000,
        embed_dim: int = 64,
        num_classes: int = 2,
        dropout: float = 0.6,
    ):
        super().__init__()

        # --- Modality encoders ---
        # small-N regime: encoder hidden widths scale with embed_dim, and
        # the text path is kept deliberately narrow (it is the easiest to overfit)
        self.face_enc  = FaceAUEncoder(n_au=n_au,   hidden=embed_dim,
                                       embed_dim=embed_dim, dropout=dropout/2)
        self.audio_enc = AudioEncoder(n_mfcc=n_mfcc, hidden=embed_dim,
                                       embed_dim=embed_dim, dropout=dropout/2)
        self.text_enc  = TextEncoder(vocab_size=vocab_size, hidden=embed_dim,
                                      embed_dim=embed_dim, dropout=dropout)

        # --- Cross-modal attention (each modality attends to the other two) ---
        self.face_attn  = CrossModalAttention(embed_dim)
        self.audio_attn = CrossModalAttention(embed_dim)
        self.text_attn  = CrossModalAttention(embed_dim)

        # --- Classifier head ---
        fused_dim = embed_dim * 3
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes),
        )

        self.embed_dim = embed_dim

    def forward(self, face_seq, audio_seq, text_vec, return_attention=False):
        # --- Encode each modality ---
        f = self.face_enc(face_seq)     # (B, D)
        a = self.audio_enc(audio_seq)   # (B, D)
        t = self.text_enc(text_vec)     # (B, D)

        # --- Cross-modal attention ---
        # face attends to audio + text
        f_ctx = torch.stack([a, t], dim=1)          # (B, 2, D)
        f_att, f_w = self.face_attn(f, f_ctx)       # (B, D), (B, 2)

        # audio attends to face + text
        a_ctx = torch.stack([f, t], dim=1)
        a_att, a_w = self.audio_attn(a, a_ctx)

        # text attends to face + audio
        t_ctx = torch.stack([f, a], dim=1)
        t_att, t_w = self.text_attn(t, t_ctx)

        # --- Fuse and classify ---
        fused  = torch.cat([f_att, a_att, t_att], dim=1)   # (B, D*3)
        logits = self.classifier(fused)

        if return_attention:
            return logits, {
                "face_attends_to":  {"audio": f_w[:, 0], "text": f_w[:, 1]},
                "audio_attends_to": {"face":  a_w[:, 0], "text": a_w[:, 1]},
                "text_attends_to":  {"face":  t_w[:, 0], "audio": t_w[:, 1]},
            }
        return logits

    def get_embeddings(self, face_seq, audio_seq, text_vec):
        """Returns fused embedding (no classifier) — for downstream tasks."""
        with torch.no_grad():
            f = self.face_enc(face_seq)
            a = self.audio_enc(audio_seq)
            t = self.text_enc(text_vec)
            return torch.cat([f, a, t], dim=1)


if __name__ == "__main__":
    B = 4
    model = TriModalFusionModel()

    face  = torch.randn(B, 200, 20)
    audio = torch.randn(B, 300, 120)
    text  = torch.randn(B, 10000)

    logits, attn = model(face, audio, text, return_attention=True)
    print("Output logits:", logits.shape)           # (4, 2)
    print("Face  → audio attn:", attn["face_attends_to"]["audio"].shape)
    print("Total params:", sum(p.numel() for p in model.parameters()))
