"""
Paired Significance Test: Multi-Task vs. Single-Task Tri-Modal Baseline
==========================================================================
The paper currently argues the multi-task improvement (F1 0.607 -> 0.629)
is real based only on "consistency of the gain across all five seeds"
without a formal paired significance test. This script recovers the
missing per-seed single-task tri-modal F1 values (only the aggregate was
previously saved) by re-running that one configuration for the same five
seeds, then runs a paired Wilcoxon signed-rank test against the already-
recorded per-seed multi-task F1 values (results/metrics/phase8_validation.json).

CAVEAT (reported honestly, not hidden): the single-task tri-modal baseline
uses the original 120-dim MFCC-only audio features (as in the Phase-7
ablation), while the multi-task model uses the extended 199-dim
MFCC+COVAREP+FORMANT audio features. This comparison therefore bundles
two changes (audio feature set + multi-task heads), not multi-task learning
in isolation. This is the same comparison the paper already reports as
"0.607 -> 0.629"; this script only adds a formal significance test to it,
it does not change what is being compared.

Run:
    python -m src.experiments.multitask_significance
"""

import sys, json
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, roc_auc_score
from scipy.stats import wilcoxon

from configs.config import (DATA_PROCESSED, RESULTS, METRICS,
                            N_MFCC, VOCAB_SIZE, N_AU_FEATURES, EMBED_DIM,
                            FUSION_DROPOUT, WEIGHT_DECAY)
from src.models.configurable_fusion import ConfigurableFusionModel

FACE_OUT  = DATA_PROCESSED / "daic_faces"
AUDIO_OUT = DATA_PROCESSED / "daic_audio"       # 120-dim, matches Phase-7 ablation
TEXT_OUT  = DATA_PROCESSED / "daic_text"
VALIDATION_JSON = METRICS / "phase8_validation.json"
OUT_JSON  = METRICS / "multitask_significance.json"

SEEDS = [42, 1, 7, 123, 2024]


def set_seed(seed):
    np.random.seed(seed); torch.manual_seed(seed)


def load_split(split):
    n = min(len(np.load(FACE_OUT / f"X_{split}.npy")),
            len(np.load(AUDIO_OUT / f"X_{split}.npy")),
            len(np.load(TEXT_OUT / f"X_{split}.npy")))
    Xf = np.load(FACE_OUT  / f"X_{split}.npy")[:n].astype(np.float32)
    Xa = np.load(AUDIO_OUT / f"X_{split}.npy")[:n].astype(np.float32)
    Xt = np.load(TEXT_OUT  / f"X_{split}.npy")[:n].astype(np.float32)
    y  = np.load(FACE_OUT  / f"y_{split}.npy")[:n].astype(np.int64)
    return Xf, Xa, Xt, y


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


def train_tri_single_task(data, device, seed, epochs=40, patience=12):
    set_seed(seed)
    Xf_tr, Xa_tr, Xt_tr, y_tr = data["train"]
    Xf_dv, Xa_dv, Xt_dv, y_dv = data["dev"]

    model = ConfigurableFusionModel(
        ("face", "audio", "text"), n_au=N_AU_FEATURES, n_mfcc=N_MFCC * 3,
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
    return {"seed": seed, "auc": round(float(best_auc), 4), "f1": round(float(f1), 4)}


def main():
    device = "cpu"
    data = {"train": load_split("train"), "dev": load_split("dev")}
    print(f"Train: {len(data['train'][3])} | Dev: {len(data['dev'][3])}")

    print(f"\nRe-training single-task tri-modal baseline for {len(SEEDS)} seeds "
          f"(recovering per-seed F1 not previously saved)...")
    single_task_runs = []
    for s in SEEDS:
        r = train_tri_single_task(data, device, seed=s)
        single_task_runs.append(r)
        print(f"  seed={s:5d}  F1={r['f1']:.4f}  AUC={r['auc']:.4f}")

    single_f1 = np.array([r["f1"] for r in single_task_runs])
    print(f"\nSingle-task tri-modal: F1 = {single_f1.mean():.4f} +/- {single_f1.std():.4f}")

    with open(VALIDATION_JSON) as f:
        val = json.load(f)
    multi_per_seed = {r["seed"]: r["f1"] for r in val["per_seed"]}
    multi_f1 = np.array([multi_per_seed[s] for s in SEEDS])
    print(f"Multi-task (from phase8_validation.json): "
          f"F1 = {multi_f1.mean():.4f} +/- {multi_f1.std():.4f}")

    diffs = multi_f1 - single_f1
    print(f"\nPer-seed paired differences (multi - single): "
          f"{[round(float(d), 4) for d in diffs]}")
    wins = int((diffs > 0).sum())

    try:
        stat, p_value = wilcoxon(multi_f1, single_f1)
    except ValueError as e:
        stat, p_value = None, None
        print(f"Wilcoxon could not be computed: {e}")

    print(f"\nWilcoxon signed-rank test (multi-task vs single-task tri-modal, "
          f"paired by seed, n={len(SEEDS)}):")
    print(f"  statistic = {stat}")
    print(f"  p-value   = {p_value}")
    print(f"  wins for multi-task: {wins}/{len(SEEDS)}")

    report = {
        "analysis": "paired significance test: multi-task vs single-task tri-modal F1",
        "caveat": "Single-task baseline uses 120-dim MFCC-only audio (Phase-7 "
                  "ablation setup); multi-task model uses 199-dim "
                  "MFCC+COVAREP+FORMANT audio. This test bundles the audio "
                  "feature-set change together with the multi-task-head change, "
                  "consistent with how the paper already reports 0.607->0.629.",
        "seeds": SEEDS,
        "single_task_per_seed": single_task_runs,
        "multi_task_per_seed": [{"seed": s, "f1": float(multi_per_seed[s])} for s in SEEDS],
        "single_task_f1_mean": round(float(single_f1.mean()), 4),
        "single_task_f1_std": round(float(single_f1.std()), 4),
        "multi_task_f1_mean": round(float(multi_f1.mean()), 4),
        "multi_task_f1_std": round(float(multi_f1.std()), 4),
        "per_seed_diff_mean": round(float(diffs.mean()), 4),
        "wins_multitask": f"{wins}/{len(SEEDS)}",
        "wilcoxon_statistic": float(stat) if stat is not None else None,
        "wilcoxon_p_value": float(p_value) if p_value is not None else None,
        "significant_at_0.05": bool(p_value is not None and p_value < 0.05),
    }
    METRICS.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved: {OUT_JSON}")


if __name__ == "__main__":
    main()
