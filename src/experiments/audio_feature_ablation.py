"""
Internal Audio Feature Ablation: MFCC vs. COVAREP vs. FORMANT
=================================================================
The paper combines three acoustic feature families (MFCC 120-dim +
COVAREP 74-dim + FORMANT 5-dim = 199-dim) but never isolates which
of the three actually drives the audio-only performance advantage
reported in the ablation study. This experiment trains the same
tri-modal architecture (face+audio+text, single-task) with different
audio feature subsets, holding face/text/model/training protocol fixed,
to attribute the audio signal to specific acoustic feature families.

Feature layout in the saved 199-dim array (see covarep_preprocess.py):
  [0:120]   MFCC + delta + delta-delta
  [120:194] COVAREP (F0, NAQ, QOQ, H1H2, Peak Slope, Rd, 24 MCEP)
  [194:199] FORMANT (F1-F5)

Run:
    python -m src.experiments.audio_feature_ablation
"""

import sys, json
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, roc_auc_score

from configs.config import (DATA_PROCESSED, METRICS,
                            VOCAB_SIZE, N_AU_FEATURES, EMBED_DIM,
                            FUSION_DROPOUT, WEIGHT_DECAY)
from src.models.configurable_fusion import ConfigurableFusionModel

FACE_DIR  = DATA_PROCESSED / "daic_faces"
AUDIO_DIR = DATA_PROCESSED / "daic_audio_covarep"   # 199-dim combined array
TEXT_DIR  = DATA_PROCESSED / "daic_text"
OUT_JSON  = METRICS / "audio_feature_ablation.json"

SEEDS = [42, 1, 7]   # reduced from 5 to keep runtime bounded; still multi-seed

SLICES = {
    "MFCC-only":            (0, 120),
    "COVAREP-only":         (120, 194),
    "FORMANT-only":         (194, 199),
    "MFCC+COVAREP":         (0, 194),
    "COVAREP+FORMANT":      (120, 199),
    "MFCC+COVAREP+FORMANT (full)": (0, 199),
}


def set_seed(seed):
    np.random.seed(seed); torch.manual_seed(seed)


def load_split(split):
    Xf = np.load(FACE_DIR / f"X_{split}.npy").astype(np.float32)
    Xa = np.load(AUDIO_DIR / f"X_{split}.npy").astype(np.float32)
    Xt = np.load(TEXT_DIR / f"X_{split}.npy").astype(np.float32)
    y  = np.load(FACE_DIR / f"y_{split}.npy").astype(np.int64)
    n = min(len(Xf), len(Xa), len(Xt), len(y))
    return Xf[:n], Xa[:n], Xt[:n], y[:n]


def class_weights(y):
    counts = np.bincount(y, minlength=2)
    w = len(y) / (2 * counts.astype(float)); w = w / w.min()
    return torch.tensor(np.minimum(w, 1.8), dtype=torch.float32)


def best_threshold(true, probs):
    if len(np.unique(true)) < 2:
        return 0.5
    best_t, best_f1 = 0.5, -1
    for thr in np.linspace(0.1, 0.9, 33):
        f1 = f1_score(true, (probs >= thr).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, thr
    return float(best_t)


def train_one(data, audio_dim, device, seed, epochs=40, patience=12):
    set_seed(seed)
    Xf_tr, Xa_tr, Xt_tr, y_tr = data["train"]
    Xf_dv, Xa_dv, Xt_dv, y_dv = data["dev"]

    model = ConfigurableFusionModel(
        ("face", "audio", "text"), n_au=N_AU_FEATURES, n_mfcc=audio_dim,
        vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, dropout=FUSION_DROPOUT).to(device)

    cw = class_weights(y_tr).to(device)
    crit = nn.CrossEntropyLoss(weight=cw)
    opt  = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=WEIGHT_DECAY)
    sch  = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2)

    tr_dl = DataLoader(TensorDataset(
        torch.tensor(Xf_tr), torch.tensor(Xa_tr),
        torch.tensor(Xt_tr), torch.tensor(y_tr)),
        batch_size=16, shuffle=True)

    dev_t = (torch.tensor(Xf_dv).to(device), torch.tensor(Xa_dv).to(device),
             torch.tensor(Xt_dv).to(device))

    best_auc, best_probs, wait = 0.0, None, 0
    for ep in range(epochs):
        model.train()
        for xf, xa, xt, yb in tr_dl:
            opt.zero_grad()
            out = model(xf.to(device), xa.to(device), xt.to(device))
            loss = crit(out, yb.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()

        model.eval()
        with torch.no_grad():
            pr = torch.softmax(model(*dev_t), 1)[:, 1].cpu().numpy()
        auc = roc_auc_score(y_dv, pr) if len(np.unique(y_dv)) > 1 else 0.0
        if auc > best_auc + 1e-4:
            best_auc, best_probs, wait = auc, pr, 0
        else:
            wait += 1
            if wait >= patience:
                break

    thr = best_threshold(y_dv, best_probs)
    preds = (best_probs >= thr).astype(int)
    f1 = f1_score(y_dv, preds, zero_division=0)
    acc = (preds == y_dv).mean()
    return {"auc": float(best_auc), "f1": float(f1), "acc": float(acc)}


def main():
    device = "cpu"
    Xf, Xa_full, Xt, y = load_split("train")
    Xf_d, Xa_full_d, Xt_d, y_d = load_split("dev")

    print(f"Train N={len(y)}  Dev N={len(y_d)}  audio_full_dim={Xa_full.shape[-1]}")
    print(f"Testing {len(SLICES)} audio feature subsets x {len(SEEDS)} seeds "
          f"= {len(SLICES) * len(SEEDS)} runs\n")

    results = {}
    for name, (lo, hi) in SLICES.items():
        Xa_tr_sub = Xa_full[:, :, lo:hi]
        Xa_dv_sub = Xa_full_d[:, :, lo:hi]
        data = {"train": (Xf, Xa_tr_sub, Xt, y), "dev": (Xf_d, Xa_dv_sub, Xt_d, y_d)}
        dim = hi - lo

        runs = [train_one(data, dim, device, seed=s) for s in SEEDS]
        aucs = np.array([r["auc"] for r in runs])
        f1s  = np.array([r["f1"] for r in runs])
        accs = np.array([r["acc"] for r in runs])
        results[name] = {
            "dim": dim,
            "auc_mean": round(float(aucs.mean()), 4), "auc_std": round(float(aucs.std()), 4),
            "f1_mean": round(float(f1s.mean()), 4), "f1_std": round(float(f1s.std()), 4),
            "acc_mean": round(float(accs.mean()), 4),
            "per_seed_auc": [round(float(a), 4) for a in aucs],
            "per_seed_f1": [round(float(f), 4) for f in f1s],
        }
        r = results[name]
        print(f"  {name:32s} (dim={dim:3d})  AUC={r['auc_mean']:.3f}+/-{r['auc_std']:.3f}  "
              f"F1={r['f1_mean']:.3f}+/-{r['f1_std']:.3f}")

    best_name = max(results, key=lambda k: results[k]["auc_mean"])
    print(f"\nBest audio subset by dev AUC: {best_name} "
          f"(AUC={results[best_name]['auc_mean']:.3f})")

    report = {
        "analysis": "internal audio-feature ablation (MFCC vs COVAREP vs FORMANT) "
                    "within tri-modal single-task fusion model",
        "seeds": SEEDS,
        "results": results,
        "best_subset_by_auc": best_name,
    }
    METRICS.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved: {OUT_JSON}")


if __name__ == "__main__":
    main()
