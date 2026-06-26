"""
Phase 8 Training — Multi-Task Fusion with COVAREP + Modality Dropout
======================================================================
Extends Phase 4 with three upgrades:

  1. CLINICAL AUDIO: COVAREP + FORMANT features (199-dim) instead of MFCC-only (120-dim)
  2. MULTI-TASK: joint training on binary + PHQ-8 score + 8-symptom heads
  3. MODALITY DROPOUT: each modality masked with p=0.15 during training

Loss:
    L = L_CE(binary, class-weighted)
      + 0.3 · L_MSE(PHQ-8 score)
      + 0.2 · L_CE(8 symptoms)
      + λ_f · [(TPR_M-TPR_F)² + (FPR_M-FPR_F)²]  (Equalized Odds, warmup 8 ep)

Run:
    python -m src.training.train_multitask
"""

import sys, json, warnings
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

from configs.config import (
    DATA_PROCESSED, SEED, VOCAB_SIZE,
    N_AU_FEATURES, AU_TIME_STEPS, AUDIO_TIME_STEPS, AUDIO_FEAT_DIM,
    EMBED_DIM, FUSION_DROPOUT, WEIGHT_DECAY,
    LAMBDA_SCORE, LAMBDA_SYMPTOM, LAMBDA_FAIRNESS, FAIRNESS_WARMUP,
    MODALITY_DROPOUT_P, N_PHQ_SYMPTOMS,
)
from src.models.multitask_fusion import MultiTaskFusionModel, multitask_loss

FACE_OUT     = DATA_PROCESSED / "daic_faces"
AUDIO_OUT    = DATA_PROCESSED / "daic_audio_covarep"   # NEW: COVAREP features
TEXT_OUT     = DATA_PROCESSED / "daic_text"
SAVE_PATH    = Path("results") / "multitask_best.pth"
METRICS_PATH = Path("results") / "metrics" / "phase8_multitask.json"

BATCH_SIZE   = 16
LR           = 5e-4
EPOCHS       = 50
PATIENCE     = 15
MAX_CW_RATIO = 1.8


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_split(split: str):
    """Load face + clinical audio (COVAREP) + text + all labels."""
    X_face     = np.load(FACE_OUT  / f"X_{split}.npy").astype(np.float32)
    y          = np.load(FACE_OUT  / f"y_{split}.npy").astype(np.int64)
    gender_raw = np.load(FACE_OUT  / f"gender_{split}.npy")
    gender     = gender_raw.astype(np.int64)

    X_audio    = np.load(AUDIO_OUT / f"X_{split}.npy").astype(np.float32)
    y_phq8     = np.load(AUDIO_OUT / f"phq8_{split}.npy").astype(np.float32)
    sym_raw    = np.load(AUDIO_OUT / f"symptoms_{split}.npy").astype(np.float32)
    y_symptoms = np.nan_to_num(sym_raw, nan=0.0).clip(0, 3).astype(np.int64)
    X_text     = np.load(TEXT_OUT  / f"X_{split}.npy").astype(np.float32)

    n = min(len(X_face), len(X_audio), len(X_text))
    return (X_face[:n], X_audio[:n], X_text[:n],
            y[:n], gender[:n], y_phq8[:n], y_symptoms[:n])


def compute_class_weights_capped(y, max_ratio=MAX_CW_RATIO):
    cw = compute_class_weight("balanced", classes=np.unique(y), y=y)
    if cw[1] / cw[0] > max_ratio:
        cw[1] = cw[0] * max_ratio
    return torch.tensor(cw, dtype=torch.float32)


def fairness_loss(logits, labels, gender):
    """Equalized Odds: penalize both TPR gap and FPR gap."""
    probs = torch.softmax(logits, dim=1)[:, 1]
    loss  = torch.tensor(0.0, requires_grad=True, device=logits.device)

    for g in [0, 1]:
        mask = gender == g
        if mask.sum() < 2:
            return loss

    def _rate(mask, pos_label):
        sub_m = mask & (labels == pos_label)
        if sub_m.sum() == 0:
            return torch.tensor(0.5, device=logits.device)
        return probs[sub_m].mean()

    tpr_m = _rate(gender == 0, 1)
    tpr_f = _rate(gender == 1, 1)
    fpr_m = _rate(gender == 0, 0)
    fpr_f = _rate(gender == 1, 0)

    return (tpr_m - tpr_f) ** 2 + (fpr_m - fpr_f) ** 2


def best_threshold(y_true, probs):
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.2, 0.8, 0.02):
        f1 = f1_score(y_true, (probs >= t).astype(int), average="macro",
                      zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t


@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    all_probs, all_labels = [], []
    total_loss = 0.0

    for Xf, Xa, Xt, yb, _, yp, ys in loader:
        Xf, Xa, Xt = Xf.to(device), Xa.to(device), Xt.to(device)
        yb, yp, ys = yb.to(device), yp.to(device), ys.to(device)

        lb, sr, sl = model(Xf, Xa, Xt)
        loss = multitask_loss(lb, sr, sl, yb, yp.float(), ys,
                              lambda_score=LAMBDA_SCORE,
                              lambda_symptom=LAMBDA_SYMPTOM)
        total_loss += loss.item()
        all_probs.extend(torch.softmax(lb, 1)[:, 1].cpu().tolist())
        all_labels.extend(yb.cpu().tolist())

    probs  = np.array(all_probs)
    labels = np.array(all_labels)
    preds  = (probs >= threshold).astype(int)
    auc    = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.5
    f1     = f1_score(labels, preds, average="macro", zero_division=0)
    acc    = (preds == labels).mean()
    return total_loss / max(len(loader), 1), auc, f1, acc, probs, labels


def train():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading data...")
    tr = load_split("train")
    dv = load_split("dev")
    Xf_tr, Xa_tr, Xt_tr, y_tr, g_tr, yp_tr, ys_tr = tr
    Xf_dv, Xa_dv, Xt_dv, y_dv, g_dv, yp_dv, ys_dv = dv
    print(f"Train: {len(y_tr)} | Dev: {len(y_dv)}")
    print(f"Audio feat dim: {Xa_tr.shape[2]} (expected {AUDIO_FEAT_DIM})")

    # Tensors
    def to_t(*args, dtype=torch.float32):
        return [torch.tensor(a, dtype=dtype) for a in args]

    tr_ds = TensorDataset(
        torch.tensor(Xf_tr), torch.tensor(Xa_tr), torch.tensor(Xt_tr),
        torch.tensor(y_tr), torch.tensor(g_tr),
        torch.tensor(yp_tr), torch.tensor(ys_tr),
    )
    dv_ds = TensorDataset(
        torch.tensor(Xf_dv), torch.tensor(Xa_dv), torch.tensor(Xt_dv),
        torch.tensor(y_dv), torch.tensor(g_dv),
        torch.tensor(yp_dv), torch.tensor(ys_dv),
    )
    tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=False)
    dv_loader = DataLoader(dv_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    class_weights = compute_class_weights_capped(y_tr).to(device)
    print(f"Class weights: {class_weights.tolist()}")

    model = MultiTaskFusionModel(
        n_au=N_AU_FEATURES,
        n_audio_feat=Xa_tr.shape[2],   # actual feature dim from data
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        dropout=FUSION_DROPOUT,
        n_symptoms=N_PHQ_SYMPTOMS,
        modality_dropout_p=MODALITY_DROPOUT_P,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_f1, best_auc_at_best_f1, best_ep, patience_cnt = 0.0, 0.0, 0, 0
    threshold = 0.5
    history = []

    # auxiliary loss warmup: start aux tasks after epoch 10
    AUX_WARMUP = 10

    print(f"\n{'Ep':>3} {'Train Loss':>10} {'Dev Loss':>9} {'AUC':>6} "
          f"{'F1':>6} {'Acc':>6} {'Fair':>6}")
    print("─" * 60)

    for ep in range(1, EPOCHS + 1):
        model.train()
        ep_loss = 0.0

        # scale down auxiliary losses; ramp up after warmup
        aux_scale = min(1.0, (ep - AUX_WARMUP) / 5.0) if ep > AUX_WARMUP else 0.0

        for Xf, Xa, Xt, yb, g, yp, ys in tr_loader:
            Xf, Xa, Xt = Xf.to(device), Xa.to(device), Xt.to(device)
            yb, g      = yb.to(device), g.to(device)
            yp, ys     = yp.to(device), ys.to(device)

            optimizer.zero_grad()
            lb, sr, sl = model(Xf, Xa, Xt)

            loss = multitask_loss(lb, sr, sl, yb, yp.float(), ys,
                                  class_weights=class_weights,
                                  lambda_score=aux_scale * LAMBDA_SCORE,
                                  lambda_symptom=aux_scale * LAMBDA_SYMPTOM)

            if ep > FAIRNESS_WARMUP:
                f_loss = fairness_loss(lb, yb, g)
                loss   = loss + LAMBDA_FAIRNESS * f_loss
                fair_v = f_loss.item()
            else:
                fair_v = 0.0

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item()

        scheduler.step()
        tr_loss = ep_loss / max(len(tr_loader), 1)

        # Evaluate — find best threshold on train probs
        _, _, _, _, tr_probs, tr_labels = evaluate(model, tr_loader, device)
        threshold = best_threshold(tr_labels, tr_probs)
        dv_loss, auc, f1, acc, probs_dv, labels_dv = evaluate(
            model, dv_loader, device, threshold)

        history.append({"epoch": ep, "auc": auc, "f1": f1, "acc": float(acc)})
        print(f"{ep:>3} {tr_loss:>10.4f} {dv_loss:>9.4f} {auc:>6.3f} "
              f"{f1:>6.3f} {acc:>6.3f} {fair_v:>6.3f}")

        # Save by F1 (primary clinical metric)
        if f1 > best_f1:
            best_f1, best_auc_at_best_f1, best_ep, patience_cnt = f1, auc, ep, 0
            torch.save(model.state_dict(), SAVE_PATH)
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"\nEarly stop — best F1 {best_f1:.3f} (AUC {best_auc_at_best_f1:.3f}) at epoch {best_ep}")
                break

    # Final evaluation with best F1 model — recompute threshold from train set
    model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
    _, _, _, _, tr_probs_f, tr_labels_f = evaluate(model, tr_loader, device, 0.5)
    threshold = best_threshold(tr_labels_f, tr_probs_f)
    _, auc, f1, acc, probs, labels = evaluate(model, dv_loader, device, threshold)

    print(f"\n{'='*60}")
    print(f"  PHASE 8 RESULTS (Multi-Task + COVAREP + Modality Dropout)")
    print(f"{'='*60}")
    print(f"  Dev AUC       : {auc:.4f}")
    print(f"  Dev F1 (macro): {f1:.4f}")
    print(f"  Dev Accuracy  : {acc:.4f}")
    print(f"  Threshold     : {threshold:.2f}")
    print(f"  Best epoch    : {best_ep}")
    print(f"  Audio feat dim: {Xa_tr.shape[2]}")
    print(f"  Model params  : {n_params:,}")

    metrics = {
        "phase": 8,
        "model": "MultiTaskFusionModel",
        "audio_feat_dim": int(Xa_tr.shape[2]),
        "n_params": n_params,
        "dev_auc": round(auc, 4),
        "dev_f1": round(f1, 4),
        "dev_accuracy": round(float(acc), 4),
        "decision_threshold": round(threshold, 3),
        "best_epoch": best_ep,
        "save_criterion": "best_F1",
        "upgrades": {
            "covarep_formant": True,
            "multi_task_score": True,
            "multi_task_symptoms": True,
            "modality_dropout": True,
            "equalized_odds_fairness": True,
        },
        "training": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "dropout": FUSION_DROPOUT,
            "lambda_score": LAMBDA_SCORE,
            "lambda_symptom": LAMBDA_SYMPTOM,
            "lambda_fairness": LAMBDA_FAIRNESS,
            "fairness_warmup": FAIRNESS_WARMUP,
            "modality_dropout_p": MODALITY_DROPOUT_P,
        },
        "history": history[-10:],
    }

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Metrics saved: {METRICS_PATH}")


if __name__ == "__main__":
    train()
