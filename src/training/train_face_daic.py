"""
Phase 2 Training — Fine-tune Face CNN on DAIC-WOZ
===================================================
Phase 1 এ FER2013 দিয়ে emotion শেখা CNN টাকে এখন
DAIC-WOZ face frames দিয়ে depression detection এ fine-tune করা হয়।

কেন fine-tune?
  FER2013 CNN emotion feature শিখেছে (sad, fear, angry = depression-linked)।
  সেই knowledge টা transfer করে DAIC-WOZ এ binary classification করলে
  অনেক কম data দিয়েও ভালো result পাওয়া যায়।

চালানোর নিয়ম:
    python src/training/train_face_daic.py

আগে চালাতে হবে:
    1. python src/preprocessing/daic_preprocess.py
    2. results/face_cnn_phase1.pth থাকতে হবে (Phase 1 এর saved weights)
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
                             EPOCHS, SEED, IMG_SIZE)
from src.models.face_cnn import FaceCNN

DAIC_OUT   = DATA_PROCESSED / "daic_faces"
WEIGHTS_P1 = Path("results") / "face_cnn_phase1.pth"
SAVE_PATH  = Path("results") / "face_cnn_phase2_daic.pth"
METRICS_PATH = Path("results") / "metrics" / "phase2_metrics.json"

DEPRESSION_CLASSES = 2   # 0=not depressed, 1=depressed
# fine-tune with lower LR so pretrained features aren't destroyed
FINETUNE_LR = LEARNING_RATE * 0.1


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_npy_split(split: str):
    """data/processed/daic_faces/ থেকে X, y load করে।"""
    X = np.load(DAIC_OUT / f"X_{split}.npy")
    y = np.load(DAIC_OUT / f"y_{split}.npy")
    return X, y


def build_model(device: str) -> FaceCNN:
    """
    FaceCNN বানায় — classifier head টা replace করে 2-class (depression) এ।
    Phase 1 weights থাকলে feature layers load করে (transfer learning)।
    """
    model = FaceCNN(num_classes=DEPRESSION_CLASSES, return_features=False)

    if WEIGHTS_P1.exists():
        print(f"Loading Phase 1 weights: {WEIGHTS_P1}")
        state = torch.load(WEIGHTS_P1, map_location="cpu")

        # শুধু feature + fc layer load করি, classifier skip করি (shape mismatch)
        own_state = model.state_dict()
        pretrained = {k: v for k, v in state.items()
                      if k in own_state and "classifier" not in k
                      and own_state[k].shape == v.shape}
        own_state.update(pretrained)
        model.load_state_dict(own_state)
        print(f"  Loaded {len(pretrained)}/{len(own_state)} layers from Phase 1.")
    else:
        print(f"Phase 1 weights না পাওয়া গেছে ({WEIGHTS_P1}). Scratch থেকে train হবে.")

    return model.to(device)


def compute_class_weights(y: np.ndarray) -> torch.Tensor:
    """
    DAIC-WOZ এ depressed class কম থাকে (~30%)।
    Class weight দিয়ে imbalance handle করা হয়।
    """
    counts = np.bincount(y, minlength=DEPRESSION_CLASSES)
    weights = len(y) / (DEPRESSION_CLASSES * counts.astype(float))
    return torch.tensor(weights, dtype=torch.float32)


def evaluate(model, dl, device) -> dict:
    model.eval()
    all_preds, all_probs, all_true = [], [], []
    with torch.no_grad():
        for xb, yb in dl:
            logits = model(xb.to(device))
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = logits.argmax(1).cpu().numpy()
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

    # ---- Data ----
    if not DAIC_OUT.exists():
        print("DAIC processed data না পাওয়া গেছে।")
        print("আগে চালাও: python src/preprocessing/daic_preprocess.py")
        return

    X_train, y_train = load_npy_split("train")
    X_dev,   y_dev   = load_npy_split("dev")
    print(f"Train: {X_train.shape}  Dev: {X_dev.shape}")
    print(f"Train balance — not-dep: {(y_train==0).sum()}  dep: {(y_train==1).sum()}")

    cw = compute_class_weights(y_train).to(device)
    print(f"Class weights: {cw.cpu().numpy()}")

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    dev_ds   = TensorDataset(torch.tensor(X_dev),   torch.tensor(y_dev))
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    dev_dl   = DataLoader(dev_ds,   batch_size=BATCH_SIZE)

    # ---- Model ----
    model     = build_model(device)
    criterion = nn.CrossEntropyLoss(weight=cw)
    optimizer = torch.optim.Adam(model.parameters(), lr=FINETUNE_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # synthetic data হলে 3 epoch, আসল হলে full EPOCHS
    epochs = 3 if len(X_train) <= 200 else EPOCHS

    history = {"train_loss": [], "dev_f1": [], "dev_auc": [], "dev_acc": []}
    best_f1 = 0.0

    # ---- Train ----
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
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

    # ---- Final report ----
    print(f"\nBest Dev F1: {best_f1:.4f}")
    final = evaluate(model, dev_dl, device)
    print(classification_report(final["true"], final["preds"],
                                 target_names=["Not Depressed", "Depressed"],
                                 zero_division=0))

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "phase": 2,
        "dataset": "DAIC-WOZ (face frames)",
        "transfer_from": "FER2013 (Phase 1)" if WEIGHTS_P1.exists() else "scratch",
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
