"""
Audio Branch — MFCC + LSTM
==========================
ভয়েস (audio) থেকে depression-related feature বের করে।

কণ্ঠের pattern: ধীর কথা, কম energy, কম pitch variation — এগুলো MFCC তে ধরা পড়ে।
এটা Phase 3 এর কাজ। এখন skeleton, পরে DAIC-WOZ audio দিয়ে ভরবে।
"""

import torch
import torch.nn as nn


class AudioLSTM(nn.Module):
    """
    Input: MFCC sequence (B, T, n_mfcc)  — T = time steps
    Output: class logits + feature vector (fusion এর জন্য)
    """

    def __init__(self, n_mfcc: int = 40, hidden: int = 128,
                 num_classes: int = 2, return_features: bool = False):
        super().__init__()
        self.return_features = return_features

        self.lstm = nn.LSTM(
            input_size=n_mfcc,
            hidden_size=hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )
        self.feature_dim = hidden * 2  # bidirectional
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        # x: (B, T, n_mfcc)
        out, (h_n, c_n) = self.lstm(x)
        # শেষ time-step এর দুই দিকের hidden নিই
        feat = out[:, -1, :]              # (B, hidden*2)
        logits = self.classifier(feat)
        if self.return_features:
            return logits, feat
        return logits


if __name__ == "__main__":
    model = AudioLSTM(n_mfcc=40, num_classes=2)
    dummy = torch.randn(4, 100, 40)   # 4 samples, 100 time-steps, 40 MFCC
    y = model(dummy)
    print("Output shape:", y.shape)    # (4, 2)
