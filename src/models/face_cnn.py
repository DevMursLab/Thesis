"""
Face Branch — CNN
=================
মুখের ছবি (face image) থেকে emotion/depression-related feature বের করে।

এটা Phase 1 এর মূল model। FER2013 দিয়ে train হবে।
পরে এই CNN এর শেষ feature layer টাই fusion এ audio এর সাথে যোগ হবে।
"""

import torch
import torch.nn as nn


class FaceCNN(nn.Module):
    """
    সহজ কিন্তু কাজের CNN। 48x48 grayscale image input নেয়।

    Architecture:
      Conv -> BN -> ReLU -> Pool  (×3 blocks)
      -> Flatten -> FC -> feature vector -> classifier
    """

    def __init__(self, num_classes: int = 7, return_features: bool = False):
        super().__init__()
        self.return_features = return_features

        self.features = nn.Sequential(
            # Block 1: 1 -> 32 channels
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # 48 -> 24

            # Block 2: 32 -> 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # 24 -> 12

            # Block 3: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # 12 -> 6
        )

        self.feature_dim = 128 * 6 * 6  # flattened size

        self.fc = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)              # (B, 128, 6, 6)
        x = torch.flatten(x, 1)           # (B, 4608)
        feat = self.fc(x)                 # (B, 256)  <- এই feature টাই fusion এ যাবে
        out = self.classifier(feat)       # (B, num_classes)
        if self.return_features:
            return out, feat
        return out


if __name__ == "__main__":
    # quick sanity check
    model = FaceCNN(num_classes=7)
    dummy = torch.randn(4, 1, 48, 48)   # batch of 4 fake images
    y = model(dummy)
    print("Output shape:", y.shape)      # should be (4, 7)
    print("Total params:", sum(p.numel() for p in model.parameters()))
