"""
Fusion Model — Face + Audio
===========================
দুই branch এর feature যোগ করে একসাথে decision নেয়।

এটা Phase 4 এর কাজ। Face CNN আর Audio LSTM আলাদা train হওয়ার পর
এদের feature concatenate করে এই fusion head দিয়ে final prediction হয়।

এটাই তোমার project এর "multimodal" অংশ — defence এ এটাই USP।
"""

import torch
import torch.nn as nn


class FusionModel(nn.Module):
    """
    Feature-level fusion (early fusion):
      face_feat (256)  +  audio_feat (256)  ->  concat (512)  ->  classifier
    """

    def __init__(self, face_dim: int = 256, audio_dim: int = 256,
                 num_classes: int = 2):
        super().__init__()
        fused_dim = face_dim + audio_dim

        self.fusion_head = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, face_feat, audio_feat):
        fused = torch.cat([face_feat, audio_feat], dim=1)  # (B, 512)
        return self.fusion_head(fused)


if __name__ == "__main__":
    model = FusionModel()
    face = torch.randn(4, 256)
    audio = torch.randn(4, 256)
    y = model(face, audio)
    print("Fusion output shape:", y.shape)   # (4, 2)
