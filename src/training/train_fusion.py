"""
Phase 4 Training — Tri-Modal Cross-Attention Fusion
=====================================================
Trains the full TriModalFusionModel on DAIC-WOZ using:
  - Face : CLNF Action Units  (data/processed/daic_faces/)
  - Audio: MFCC+delta+delta2  (data/processed/daic_audio/)
  - Text : TF-IDF bigrams     (data/processed/daic_text/)

Novel training objectives:
  1. Task loss       — CrossEntropy (class-weighted for imbalance)
  2. Fairness loss   — equalize F1 across gender groups (adversarial)

Run:
    python src/training/train_fusion.py
"""

import sys, json
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, roc_auc_score, classification_report

from configs.config import (DATA_PROCESSED, BATCH_SIZE, LEARNING_RATE,
                             EPOCHS, SEED, N_MFCC, VOCAB_SIZE,
                             N_AU_FEATURES, AU_TIME_STEPS, AUDIO_TIME_STEPS)
from src.models.fusion_attention import TriModalFusionModel

FACE_OUT  = DATA_PROCESSED / "daic_faces"
AUDIO_OUT = DATA_PROCESSED / "daic_audio"
TEXT_OUT  = DATA_PROCESSED / "daic_text"
SAVE_PATH = Path("results") / "fusion_trimodal_best.pth"
METRICS_P = Path("results") / "metrics" / "phase4_metrics.json"

FAIRNESS_LAMBDA = 0.3    # weight of fairness penalty in total loss


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_split(split: str):
    """Load all three modalities + labels + gender for one split."""
    X_face  = np.load(FACE_OUT  / f"X_{split}.npy")
    y       = np.load(FACE_OUT  / f"y_{split}.npy")
    gender  = np.load(FACE_OUT  / f"gender_{split}.npy")

    X_audio = np.load(AUDIO_OUT / f"X_{split}.npy")
    X_text  = np.load(TEXT_OUT  / f"X_{split}.npy")

    # align sizes — keep only participants present in ALL three modalities
    n = min(len(X_face), len(X_audio), len(X_text))
    return (X_face[:n].astype(np.float32),
            X_audio[:n].astype(np.float32),
            X_text[:n].astype(np.float32),
            y[:n].astype(np.int64),
            gender[:n].astype(np.int64))


def fairness_loss(logits, labels, gender):
    """
    Equalized Opportunity fairness penalty.
    Penalizes difference in True Positive Rate between male (0) and female (1).
    TPR = recall for the depressed class.
    """
    probs = torch.softmax(logits, dim=1)[:, 1]
    loss  = torch.tensor(0.0, requires_grad=True, device=logits.device)

    tprs = []
    for g in [0, 1]:
        mask = (gender == g) & (labels == 1)
        if mask.sum() < 2:
            continue
        tpr = probs[mask].mean()
        tprs.append(tpr)

    if len(tprs) == 2:
        loss = (tprs[0] - tprs[1]) ** 2

    return loss


def compute_class_weights(y):
    counts  = np.bincount(y, minlength=2)
    weights = len(y) / (2 * counts.astype(float))
    return torch.tensor(weights, dtype=torch.float32)


def evaluate(model, dl, device):
    model.eval()
    preds_all, probs_all, true_all, gender_all = [], [], [], []
    with torch.no_grad():
        for xf, xa, xt, yb, gb in dl:
            logits = model(xf.to(device), xa.to(device), xt.to(device))
            probs  = torch.softmax(logits, 1)[:, 1].cpu().numpy()
            preds  = logits.argmax(1).cpu().numpy()
            preds_all.extend(preds);  probs_all.extend(probs)
            true_all.extend(yb.numpy()); gender_all.extend(gb.numpy())

    t = np.array(true_all); p = np.array(preds_all)
    pr = np.array(probs_all); g = np.array(gender_all)

    acc = (p == t).mean() * 100
    f1  = f1_score(t, p, average="binary", zero_division=0)
    auc = roc_auc_score(t, pr) if len(np.unique(t)) > 1 else 0.0

    # per-gender F1 (fairness metric)
    f1_m = f1_score(t[g==0], p[g==0], average="binary", zero_division=0) if (g==0).sum() > 0 else 0.0
    f1_f = f1_score(t[g==1], p[g==1], average="binary", zero_division=0) if (g==1).sum() > 0 else 0.0

    return {"acc": acc, "f1": f1, "auc": auc,
            "f1_male": f1_m, "f1_female": f1_f,
            "fairness_gap": abs(f1_m - f1_f),
            "preds": p, "true": t}


def main():
    set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ---- Load data ----
    missing = [p for p in [FACE_OUT, AUDIO_OUT, TEXT_OUT] if not p.exists()]
    if missing:
        print("Missing processed data:", missing)
        print("Run all three preprocessing scripts first.")
        return

    Xf_tr, Xa_tr, Xt_tr, y_tr, g_tr = load_split("train")
    Xf_dv, Xa_dv, Xt_dv, y_dv, g_dv = load_split("dev")

    print(f"Train: {len(y_tr)} | Dev: {len(y_dv)}")
    print(f"Train depressed: {y_tr.sum()} ({100*y_tr.mean():.1f}%)")

    cw = compute_class_weights(y_tr).to(device)

    train_ds = TensorDataset(
        torch.tensor(Xf_tr), torch.tensor(Xa_tr),
        torch.tensor(Xt_tr), torch.tensor(y_tr), torch.tensor(g_tr))
    dev_ds = TensorDataset(
        torch.tensor(Xf_dv), torch.tensor(Xa_dv),
        torch.tensor(Xt_dv), torch.tensor(y_dv), torch.tensor(g_dv))

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
    dev_dl   = DataLoader(dev_ds,   batch_size=BATCH_SIZE, shuffle=False)

    # ---- Model ----
    model = TriModalFusionModel(
        n_au=N_AU_FEATURES, n_mfcc=N_MFCC*3,
        vocab_size=VOCAB_SIZE, embed_dim=256,
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=cw)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                   weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    epochs   = 5 if len(y_tr) <= 200 else EPOCHS
    best_f1  = 0.0
    history  = []

    for epoch in range(epochs):
        model.train()
        total_loss = total_task = total_fair = 0

        for xf, xa, xt, yb, gb in train_dl:
            xf, xa, xt = xf.to(device), xa.to(device), xt.to(device)
            yb, gb     = yb.to(device), gb.to(device)

            optimizer.zero_grad()
            logits = model(xf, xa, xt)

            task_l = criterion(logits, yb)
            fair_l = fairness_loss(logits, yb, gb)
            loss   = task_l + FAIRNESS_LAMBDA * fair_l

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_task += task_l.item()
            total_fair += fair_l.item() if isinstance(fair_l, torch.Tensor) else 0

        scheduler.step()
        m = evaluate(model, dev_dl, device)
        history.append(m)

        print(f"Epoch {epoch+1:02d}/{epochs} | "
              f"loss={total_loss/len(train_dl):.4f} "
              f"(task={total_task/len(train_dl):.3f} "
              f"fair={total_fair/len(train_dl):.3f}) | "
              f"F1={m['f1']:.4f} AUC={m['auc']:.4f} "
              f"Fair-gap={m['fairness_gap']:.4f} "
              f"[M:{m['f1_male']:.3f} F:{m['f1_female']:.3f}]")

        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), SAVE_PATH)

    # ---- Final report ----
    print(f"\n{'='*60}")
    print(f"Best Dev F1: {best_f1:.4f}")
    model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
    final = evaluate(model, dev_dl, device)
    print(classification_report(final["true"], final["preds"],
                                 target_names=["Not Depressed", "Depressed"],
                                 zero_division=0))
    print(f"Fairness gap (|F1_male - F1_female|): {final['fairness_gap']:.4f}")

    METRICS_P.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "phase": 4,
        "model": "TriModalFusionModel (Cross-Modal Attention)",
        "modalities": ["CLNF_AUs", "MFCC+delta+delta2", "TF-IDF_bigrams"],
        "dataset": "DAIC-WOZ (USC)",
        "fairness_loss": True,
        "fairness_lambda": FAIRNESS_LAMBDA,
        "best_dev_f1": best_f1,
        "final_dev_auc": final["auc"],
        "final_dev_acc": final["acc"],
        "f1_male": final["f1_male"],
        "f1_female": final["f1_female"],
        "fairness_gap": final["fairness_gap"],
    }
    with open(METRICS_P, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nMetrics: {METRICS_P}")
    print(f"Model:   {SAVE_PATH}")


if __name__ == "__main__":
    main()
