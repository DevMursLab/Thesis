"""
Phase 3 Training — Audio Bi-LSTM (DAIC-WOZ)
=============================================
MFCC sequence দিয়ে Bi-LSTM train করে depression detect করে।

কণ্ঠের feature (MFCC + delta + delta2) time-series হিসেবে LSTM এ যায়।
Bidirectional LSTM কারণ speech pattern এর context দুই দিক থেকে আসে।

চালানোর নিয়ম:
    python src/training/train_audio.py

আগে চালাতে হবে:
    python src/preprocessing/audio_preprocess.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, roc_auc_score, classification_report

from configs.config import (DATA_PROCESSED, BATCH_SIZE, LEARNING_RATE,
                             EPOCHS, SEED, N_MFCC)
from src.models.audio_lstm import AudioLSTM

AUDIO_OUT    = DATA_PROCESSED / "daic_audio"
SAVE_PATH    = Path("results") / "audio_lstm_phase3.pth"
METRICS_PATH = Path("results") / "metrics" / "phase3_metrics.json"

N_FEAT = N_MFCC * 3   # MFCC + delta + delta2


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_class_weights(y: np.ndarray) -> torch.Tensor:
    counts  = np.bincount(y, minlength=2)
    weights = len(y) / (2 * counts.astype(float))
    return torch.tensor(weights, dtype=torch.float32)


def evaluate(model, dl, device) -> dict:
    model.eval()
    all_preds, all_probs, all_true = [], [], []
    with torch.no_grad():
        for xb, yb in dl:
            logits = model(xb.to(device))
            probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds  = logits.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_probs.extend(probs)
            all_true.extend(yb.numpy())

    all_true  = np.array(all_true)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    acc = (all_preds == all_true).mean() * 100
    f1  = f1_score(all_true, all_preds, average="binary", zero_division=0)
    auc = roc_auc_score(all_true, all_probs) if len(np.unique(all_true)) > 1 else 0.0
    return {"accuracy": acc, "f1": f1, "auc": auc,
            "preds": all_preds, "true": all_true}


def main():
    set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if not AUDIO_OUT.exists():
        print("Audio processed data না পাওয়া গেছে।")
        print("আগে চালাও: python src/preprocessing/audio_preprocess.py")
        return

    X_train = np.load(AUDIO_OUT / "X_train.npy")
    y_train = np.load(AUDIO_OUT / "y_train.npy")
    X_dev   = np.load(AUDIO_OUT / "X_dev.npy")
    y_dev   = np.load(AUDIO_OUT / "y_dev.npy")

    print(f"Train: {X_train.shape}  Dev: {X_dev.shape}")
    print(f"Feature dim per step: {X_train.shape[2]}  (MFCC+delta+delta2)")
    print(f"Train — not-dep: {(y_train==0).sum()}  dep: {(y_train==1).sum()}")

    cw = compute_class_weights(y_train).to(device)
    print(f"Class weights: {cw.cpu().numpy()}")

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    dev_ds   = TensorDataset(torch.tensor(X_dev),   torch.tensor(y_dev))
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    dev_dl   = DataLoader(dev_ds,   batch_size=BATCH_SIZE)

    model = AudioLSTM(
        n_mfcc=N_FEAT,
        hidden=128,
        num_classes=2,
        return_features=False,
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=cw)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    epochs   = 3 if len(X_train) <= 200 else EPOCHS
    best_f1  = 0.0
    history  = {"train_loss": [], "dev_f1": [], "dev_auc": [], "dev_acc": []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        metrics = evaluate(model, dev_dl, device)
        history["train_loss"].append(total_loss / len(train_dl))
        history["dev_f1"].append(metrics["f1"])
        history["dev_auc"].append(metrics["auc"])
        history["dev_acc"].append(metrics["accuracy"])

        print(f"Epoch {epoch+1:02d}/{epochs} — "
              f"loss: {history['train_loss'][-1]:.4f}  "
              f"F1: {metrics['f1']:.4f}  "
              f"AUC: {metrics['auc']:.4f}  "
              f"Acc: {metrics['accuracy']:.2f}%")

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), SAVE_PATH)

    print(f"\nBest Dev F1: {best_f1:.4f}")
    final = evaluate(model, dev_dl, device)
    print(classification_report(final["true"], final["preds"],
                                 target_names=["Not Depressed", "Depressed"],
                                 zero_division=0))

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "phase": 3,
        "dataset": "DAIC-WOZ (audio MFCC)",
        "feature": f"MFCC+delta+delta2 ({N_FEAT} dims, {X_train.shape[1]} time steps)",
        "model": "Bi-LSTM 2-layer hidden=128",
        "epochs_run": epochs,
        "best_dev_f1": best_f1,
        "final_dev_auc": final["auc"],
        "final_dev_acc": final["accuracy"],
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nMetrics saved: {METRICS_PATH}")
    print(f"Best model saved: {SAVE_PATH}")


if __name__ == "__main__":
    main()
