"""
Reviewer-Proof Experiments — Phase 9
=====================================
Closes the two strongest reviewer attacks identified in DEFENSE_ANALYSIS.md:

  EXPERIMENT A — Fairness Loss Effectiveness (before/after)
    Train the multi-task model with lambda_f=0 (no fairness) vs lambda_f=0.1
    (Equalized Odds), across 3 seeds. Measure the gender TPR gap in both
    conditions. A reviewer asks: "Does your fairness loss actually reduce
    the gap, or is it cosmetic?" This experiment answers it directly.

  EXPERIMENT B — Attention vs. Concatenation (ablation of the core novelty)
    Train two variants with identical encoders + classifier:
      (1) cross-modal attention fusion  (our method)
      (2) plain concatenation fusion    (no attention)
    A reviewer asks: "Is the improvement from attention, or just from having
    three modalities?" This isolates the attention contribution.

Run:
    python -m src.experiments.reviewer_proof
"""

import sys, json, warnings
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

from configs.config import (
    DATA_PROCESSED, VOCAB_SIZE, N_AU_FEATURES, EMBED_DIM,
    FUSION_DROPOUT, WEIGHT_DECAY, N_PHQ_SYMPTOMS,
)
from src.models.encoders import FaceAUEncoder, AudioEncoder, TextEncoder
from src.models.fusion_attention import CrossModalAttention
from src.models.multitask_fusion import MultiTaskFusionModel, multitask_loss

FACE_OUT  = DATA_PROCESSED / "daic_faces"
AUDIO_OUT = DATA_PROCESSED / "daic_audio_covarep"
TEXT_OUT  = DATA_PROCESSED / "daic_text"
METRICS_P = Path("results") / "metrics" / "phase9_reviewer_proof.json"

SEEDS    = [42, 1, 7]
EPOCHS   = 45
PATIENCE = 14
BATCH    = 16
LR       = 5e-4
AUX_WARMUP = 10
FAIR_WARMUP = 8


# ============================================================
# Concat-only variant (no cross-modal attention)
# ============================================================
class ConcatFusionModel(nn.Module):
    """Identical encoders + classifier to MultiTaskFusionModel, but fuses
    by plain concatenation instead of cross-modal attention. Multi-task
    heads retained for fair comparison."""

    def __init__(self, n_audio_feat=199, embed_dim=64, dropout=0.6,
                 n_symptoms=8):
        super().__init__()
        self.face_enc  = FaceAUEncoder(n_au=N_AU_FEATURES, hidden=embed_dim,
                                       embed_dim=embed_dim, dropout=dropout/2)
        self.audio_enc = AudioEncoder(n_mfcc=n_audio_feat, hidden=embed_dim,
                                      embed_dim=embed_dim, dropout=dropout/2)
        self.text_enc  = TextEncoder(vocab_size=VOCAB_SIZE, hidden=embed_dim,
                                     embed_dim=embed_dim, dropout=dropout)
        fused = embed_dim * 3
        self.shared = nn.Sequential(
            nn.Linear(fused, fused), nn.LayerNorm(fused),
            nn.GELU(), nn.Dropout(dropout))
        self.binary_head  = nn.Linear(fused, 2)
        self.score_head   = nn.Sequential(
            nn.Linear(fused, embed_dim), nn.GELU(),
            nn.Dropout(dropout/2), nn.Linear(embed_dim, 1), nn.Sigmoid())
        self.symptom_head = nn.Sequential(
            nn.Linear(fused, embed_dim*2), nn.GELU(),
            nn.Dropout(dropout/2), nn.Linear(embed_dim*2, n_symptoms*4))
        self.n_symptoms = n_symptoms

    def forward(self, face, audio, text):
        f = self.face_enc(face); a = self.audio_enc(audio); t = self.text_enc(text)
        fused  = torch.cat([f, a, t], dim=1)        # plain concat — no attention
        shared = self.shared(fused)
        return (self.binary_head(shared),
                self.score_head(shared).squeeze(-1),
                self.symptom_head(shared).view(-1, self.n_symptoms, 4))


# ============================================================
def set_seed(s):
    np.random.seed(s); torch.manual_seed(s)


def load_data():
    def g(p): return np.load(p)
    Xf = g(FACE_OUT/"X_train.npy").astype(np.float32)
    y  = g(FACE_OUT/"y_train.npy").astype(np.int64)
    gd = g(FACE_OUT/"gender_train.npy").astype(np.int64)
    Xa = g(AUDIO_OUT/"X_train.npy").astype(np.float32)
    yp = g(AUDIO_OUT/"phq8_train.npy").astype(np.float32)
    ys = np.nan_to_num(g(AUDIO_OUT/"symptoms_train.npy").astype(np.float32),
                       nan=0.0).clip(0,3).astype(np.int64)
    Xt = g(TEXT_OUT/"X_train.npy").astype(np.float32)

    Xfd = g(FACE_OUT/"X_dev.npy").astype(np.float32)
    yd  = g(FACE_OUT/"y_dev.npy").astype(np.int64)
    gdd = g(FACE_OUT/"gender_dev.npy").astype(np.int64)
    Xad = g(AUDIO_OUT/"X_dev.npy").astype(np.float32)
    ypd = g(AUDIO_OUT/"phq8_dev.npy").astype(np.float32)
    ysd = np.nan_to_num(g(AUDIO_OUT/"symptoms_dev.npy").astype(np.float32),
                        nan=0.0).clip(0,3).astype(np.int64)
    Xtd = g(TEXT_OUT/"X_dev.npy").astype(np.float32)

    n  = min(len(y), len(Xa), len(Xt))
    nd = min(len(yd), len(Xad), len(Xtd))
    return ((Xf[:n], Xa[:n], Xt[:n], y[:n], gd[:n], yp[:n], ys[:n]),
            (Xfd[:nd], Xad[:nd], Xtd[:nd], yd[:nd], gdd[:nd], ypd[:nd], ysd[:nd]))


def fairness_loss_fn(logits, labels, gender):
    probs = torch.softmax(logits, 1)[:, 1]
    loss = torch.tensor(0.0, requires_grad=True, device=logits.device)
    for gg in [0, 1]:
        if (gender == gg).sum() < 2:
            return loss
    def rate(mask, pos):
        s = mask & (labels == pos)
        return probs[s].mean() if s.sum() > 0 else torch.tensor(0.5, device=logits.device)
    tm, tf = rate(gender==0,1), rate(gender==1,1)
    fm, ff = rate(gender==0,0), rate(gender==1,0)
    return (tm-tf)**2 + (fm-ff)**2


def best_threshold(y, p):
    bt, bf = 0.5, 0.0
    for t in np.arange(0.2, 0.8, 0.02):
        f = f1_score(y, (p>=t).astype(int), average="macro", zero_division=0)
        if f > bf: bf, bt = f, t
    return bt


@torch.no_grad()
def predict(model, loader, device):
    model.eval(); P, Y, G = [], [], []
    for Xf, Xa, Xt, yb, g, yp, ys in loader:
        lb, _, _ = model(Xf.to(device), Xa.to(device), Xt.to(device))
        P.extend(torch.softmax(lb,1)[:,1].cpu().tolist())
        Y.extend(yb.tolist()); G.extend(g.tolist())
    return np.array(P), np.array(Y), np.array(G)


def tpr_gap(y, p, g, thr):
    pred = (p >= thr).astype(int)
    def tpr(mask):
        s = mask & (y == 1)
        return pred[s].mean() if s.sum() > 0 else 0.0
    return abs(tpr(g==0) - tpr(g==1))


def make_loaders(train, dev, seed):
    set_seed(seed)
    tds = TensorDataset(*[torch.tensor(a) for a in train])
    dds = TensorDataset(*[torch.tensor(a) for a in dev])
    return (DataLoader(tds, batch_size=BATCH, shuffle=True),
            DataLoader(dds, batch_size=BATCH, shuffle=False))


def train_model(model, tr_loader, dv_loader, device, y_tr, lambda_f, seed):
    set_seed(seed)
    cw = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
    if cw[1]/cw[0] > 1.8: cw[1] = cw[0]*1.8
    cw_t = torch.tensor(cw, dtype=torch.float32).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    best_f1, best_state, pc = 0.0, None, 0
    for ep in range(1, EPOCHS+1):
        model.train()
        aux = min(1.0, (ep-AUX_WARMUP)/5.0) if ep > AUX_WARMUP else 0.0
        for Xf, Xa, Xt, yb, g, yp, ys in tr_loader:
            Xf,Xa,Xt = Xf.to(device),Xa.to(device),Xt.to(device)
            yb,g,yp,ys = yb.to(device),g.to(device),yp.to(device),ys.to(device)
            opt.zero_grad()
            lb, sr, sl = model(Xf, Xa, Xt)
            loss = multitask_loss(lb, sr, sl, yb, yp.float(), ys,
                                  class_weights=cw_t,
                                  lambda_score=aux*0.3, lambda_symptom=aux*0.2)
            if lambda_f > 0 and ep > FAIR_WARMUP:
                loss = loss + lambda_f * fairness_loss_fn(lb, yb, g)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        P, Y, _ = predict(model, dv_loader, device)
        Pt, Yt, _ = predict(model, tr_loader, device)
        thr = best_threshold(Yt, Pt)
        f1 = f1_score(Y, (P>=thr).astype(int), average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_state, pc = f1, {k:v.clone() for k,v in model.state_dict().items()}, 0
        else:
            pc += 1
            if pc >= PATIENCE: break
    model.load_state_dict(best_state)
    return model


# ============================================================
def experiment_A(train, dev, device):
    """Fairness before/after."""
    print("\n" + "="*60)
    print("  EXPERIMENT A — Fairness Loss Effectiveness")
    print("="*60)
    y_tr = train[3]
    results = {"no_fairness": [], "with_fairness": []}

    for cond, lf in [("no_fairness", 0.0), ("with_fairness", 0.1)]:
        for seed in SEEDS:
            tr_l, dv_l = make_loaders(train, dev, seed)
            model = MultiTaskFusionModel(
                n_audio_feat=train[1].shape[2], embed_dim=EMBED_DIM,
                dropout=FUSION_DROPOUT, n_symptoms=N_PHQ_SYMPTOMS,
                modality_dropout_p=0.15).to(device)
            model = train_model(model, tr_l, dv_l, device, y_tr, lf, seed)
            P, Y, G = predict(model, dv_l, device)
            Pt, Yt, _ = predict(model, tr_l, device)
            thr = best_threshold(Yt, Pt)
            gap = tpr_gap(Y, P, G, thr)
            f1  = f1_score(Y, (P>=thr).astype(int), average="macro", zero_division=0)
            results[cond].append({"seed": seed, "tpr_gap": round(float(gap),4),
                                  "f1": round(float(f1),4)})
            print(f"  {cond:14s} seed={seed:4d}  TPR_gap={gap:.3f}  F1={f1:.3f}")

    no_gap   = np.mean([r["tpr_gap"] for r in results["no_fairness"]])
    with_gap = np.mean([r["tpr_gap"] for r in results["with_fairness"]])
    no_f1    = np.mean([r["f1"] for r in results["no_fairness"]])
    with_f1  = np.mean([r["f1"] for r in results["with_fairness"]])
    print(f"\n  Mean TPR gap  WITHOUT fairness: {no_gap:.3f}")
    print(f"  Mean TPR gap  WITH    fairness: {with_gap:.3f}")
    print(f"  Gap reduction: {no_gap-with_gap:+.3f}  "
          f"(F1 cost: {with_f1-no_f1:+.3f})")
    return {
        "no_fairness_runs": results["no_fairness"],
        "with_fairness_runs": results["with_fairness"],
        "mean_tpr_gap_no_fairness": round(float(no_gap),4),
        "mean_tpr_gap_with_fairness": round(float(with_gap),4),
        "gap_reduction": round(float(no_gap-with_gap),4),
        "mean_f1_no_fairness": round(float(no_f1),4),
        "mean_f1_with_fairness": round(float(with_f1),4),
        "f1_cost": round(float(with_f1-no_f1),4),
    }


def experiment_B(train, dev, device):
    """Attention vs concat."""
    print("\n" + "="*60)
    print("  EXPERIMENT B — Cross-Modal Attention vs. Concatenation")
    print("="*60)
    y_tr = train[3]
    results = {"attention": [], "concat": []}

    for cond in ["attention", "concat"]:
        for seed in SEEDS:
            tr_l, dv_l = make_loaders(train, dev, seed)
            if cond == "attention":
                model = MultiTaskFusionModel(
                    n_audio_feat=train[1].shape[2], embed_dim=EMBED_DIM,
                    dropout=FUSION_DROPOUT, n_symptoms=N_PHQ_SYMPTOMS,
                    modality_dropout_p=0.15).to(device)
            else:
                model = ConcatFusionModel(
                    n_audio_feat=train[1].shape[2], embed_dim=EMBED_DIM,
                    dropout=FUSION_DROPOUT, n_symptoms=N_PHQ_SYMPTOMS).to(device)
            model = train_model(model, tr_l, dv_l, device, y_tr, 0.0, seed)
            P, Y, _ = predict(model, dv_l, device)
            Pt, Yt, _ = predict(model, tr_l, device)
            thr = best_threshold(Yt, Pt)
            auc = roc_auc_score(Y, P) if len(np.unique(Y))>1 else 0.5
            f1  = f1_score(Y, (P>=thr).astype(int), average="macro", zero_division=0)
            results[cond].append({"seed": seed, "auc": round(float(auc),4),
                                  "f1": round(float(f1),4)})
            print(f"  {cond:10s} seed={seed:4d}  AUC={auc:.3f}  F1={f1:.3f}")

    att_f1  = np.mean([r["f1"] for r in results["attention"]])
    con_f1  = np.mean([r["f1"] for r in results["concat"]])
    att_auc = np.mean([r["auc"] for r in results["attention"]])
    con_auc = np.mean([r["auc"] for r in results["concat"]])
    print(f"\n  Attention   F1={att_f1:.3f}  AUC={att_auc:.3f}")
    print(f"  Concat      F1={con_f1:.3f}  AUC={con_auc:.3f}")
    print(f"  Attention gain: F1 {att_f1-con_f1:+.3f}  AUC {att_auc-con_auc:+.3f}")
    return {
        "attention_runs": results["attention"],
        "concat_runs": results["concat"],
        "attention_f1": round(float(att_f1),4),
        "concat_f1": round(float(con_f1),4),
        "attention_auc": round(float(att_auc),4),
        "concat_auc": round(float(con_auc),4),
        "attention_gain_f1": round(float(att_f1-con_f1),4),
        "attention_gain_auc": round(float(att_auc-con_auc),4),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Seeds: {SEEDS}")
    train, dev = load_data()
    print(f"Train N={len(train[3])}  Dev N={len(dev[3])}")

    A = experiment_A(train, dev, device)
    B = experiment_B(train, dev, device)

    out = {"phase": 9, "experiments": {"fairness_before_after": A,
                                       "attention_vs_concat": B},
           "seeds": SEEDS}
    METRICS_P.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_P, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {METRICS_P}")


if __name__ == "__main__":
    main()
