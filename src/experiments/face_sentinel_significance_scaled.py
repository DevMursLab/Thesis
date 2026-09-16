"""
Scaled Paired Significance Test: Face-Only Sentinel Fix (Original vs. Cleaned AUs)
====================================================================================
The paper's original 5-seed sentinel-fix comparison (face_sentinel_ablation.py)
reported "Wilcoxon p=0.031" for original vs. cleaned face-only AUC. Recomputing
that exact test from its own results file (results/metrics/face_sentinel_ablation.json)
gives the exact two-sided Wilcoxon p-value as 0.0625, not 0.031 -- the reported
value does not match any reproducible computation on the stored per-seed data,
and 0.0625 exceeds the exact two-sided Wilcoxon's theoretical minimum at n=5
(2 * 1/2^5 = 0.0625), which the paper itself states elsewhere. This script
corrects the error properly rather than patching the number: it re-trains both
face-only variants (original AU features with the -100 sentinel; cleaned AU
features with sentinel frames dropped) from scratch across a much larger,
pre-registered seed count so the resulting exact Wilcoxon test is both
correctly computed and adequately powered.

Run:
    python -m src.experiments.face_sentinel_significance_scaled
"""

import sys, json, time
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, f1_score
from scipy.stats import wilcoxon

from configs.config import (DATA_PROCESSED, N_AU_FEATURES, EMBED_DIM,
                            FUSION_DROPOUT, WEIGHT_DECAY)
from src.models.configurable_fusion import ConfigurableFusionModel

N_SEEDS = 30
SEEDS = list(range(1, N_SEEDS + 1))
OUT_JSON = Path("results") / "metrics" / "face_sentinel_significance_scaled.json"
OUT_PARTIAL = Path("results") / "metrics" / "face_sentinel_significance_scaled_partial.json"

VARIANTS = {
    "original": DATA_PROCESSED / "daic_faces",
    "cleaned": DATA_PROCESSED / "daic_faces_clean",
}


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def class_weights(y):
    counts = np.bincount(y, minlength=2)
    w = len(y) / (2 * counts.astype(float))
    w = w / w.min()
    return torch.tensor(np.minimum(w, 1.8), dtype=torch.float32)


def best_threshold(true, probs):
    if len(np.unique(true)) < 2:
        return 0.5
    best_t, best_f1 = 0.5, -1.0
    for thr in np.linspace(0.1, 0.9, 33):
        f1 = f1_score(true, (probs >= thr).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, thr
    return float(best_t)


def train_one(Xtr, ytr, Xdv, ydv, seed, epochs=40):
    set_seed(seed)
    model = ConfigurableFusionModel(
        ("face",), n_au=N_AU_FEATURES, embed_dim=EMBED_DIM, dropout=FUSION_DROPOUT)

    crit = nn.CrossEntropyLoss(weight=class_weights(ytr))
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=WEIGHT_DECAY)
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2)

    dummy_a = torch.zeros(1, 300, 120)
    dummy_t = torch.zeros(1, 1000)

    dl = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(ytr)),
                    batch_size=16, shuffle=True)

    def fwd(X):
        n = len(X)
        return model(torch.tensor(X), dummy_a.expand(n, -1, -1), dummy_t.expand(n, -1))

    best_auc, best_probs, wait, patience = 0.0, None, 0, 12
    for _ in range(epochs):
        model.train()
        for xb, yb in dl:
            n = xb.size(0)
            opt.zero_grad()
            out = model(xb, dummy_a.expand(n, -1, -1), dummy_t.expand(n, -1))
            loss = crit(out, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()

        model.eval()
        with torch.no_grad():
            pr = torch.softmax(fwd(Xdv), 1)[:, 1].numpy()
        auc = roc_auc_score(ydv, pr) if len(np.unique(ydv)) > 1 else 0.0
        if auc > best_auc + 1e-4:
            best_auc, best_probs, wait = auc, pr, 0
        else:
            wait += 1
            if wait >= patience:
                break

    thr = best_threshold(ydv, best_probs)
    f1 = f1_score(ydv, (best_probs >= thr).astype(int), zero_division=0)
    return float(best_auc), float(f1)


def _save_partial(per_seed):
    OUT_PARTIAL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PARTIAL, "w") as f:
        json.dump({"status": "partial_checkpoint", "n_completed": len(per_seed),
                    "per_seed": per_seed}, f, indent=2)


def main():
    t0 = time.time()
    data = {}
    for name, path in VARIANTS.items():
        data[name] = {
            "Xtr": np.load(path / "X_train.npy").astype(np.float32),
            "ytr": np.load(path / "y_train.npy").astype(np.int64),
            "Xdv": np.load(path / "X_dev.npy").astype(np.float32),
            "ydv": np.load(path / "y_dev.npy").astype(np.int64),
        }
    print(f"Running {N_SEEDS} paired seeds for face-only original vs. cleaned "
          f"(correcting an unreproducible p-value in the original 5-seed report)...\n")

    orig_auc, clean_auc, per_seed = [], [], []
    for i, s in enumerate(SEEDS):
        ts = time.time()
        oa, of1 = train_one(data["original"]["Xtr"], data["original"]["ytr"],
                             data["original"]["Xdv"], data["original"]["ydv"], s)
        ca, cf1 = train_one(data["cleaned"]["Xtr"], data["cleaned"]["ytr"],
                             data["cleaned"]["Xdv"], data["cleaned"]["ydv"], s)
        orig_auc.append(oa); clean_auc.append(ca)
        per_seed.append({"seed": s, "original_auc": oa, "original_f1": of1,
                          "cleaned_auc": ca, "cleaned_f1": cf1})
        elapsed = time.time() - ts
        print(f"[{i+1:>2}/{N_SEEDS}] seed={s:<3d} original_AUC={oa:.4f} "
              f"cleaned_AUC={ca:.4f} diff={ca-oa:+.4f}  ({elapsed:.1f}s)")
        if (i + 1) % 5 == 0:
            _save_partial(per_seed)

    orig_auc = np.array(orig_auc)
    clean_auc = np.array(clean_auc)
    diffs = clean_auc - orig_auc
    wins = int((diffs > 0).sum())

    stat, p_value = wilcoxon(clean_auc, orig_auc)
    d = float(diffs.mean() / diffs.std(ddof=1)) if diffs.std(ddof=1) > 0 else 0.0

    report = {
        "analysis": "Scaled paired significance test correcting an unreproducible "
                     "p=0.031 in the original 5-seed face-only sentinel-fix report "
                     "(recomputing the exact test on that report's own data gives "
                     "p=0.0625, not 0.031); this re-trains both arms from scratch "
                     f"for {N_SEEDS} seeds to obtain a correctly computed, "
                     "adequately powered result.",
        "n_seeds": N_SEEDS,
        "seeds": SEEDS,
        "original_auc_mean": round(float(orig_auc.mean()), 4),
        "original_auc_std": round(float(orig_auc.std()), 4),
        "cleaned_auc_mean": round(float(clean_auc.mean()), 4),
        "cleaned_auc_std": round(float(clean_auc.std()), 4),
        "per_seed_diff_mean": round(float(diffs.mean()), 4),
        "wins_cleaned": f"{wins}/{N_SEEDS}",
        "wilcoxon_statistic": float(stat),
        "wilcoxon_p_value": float(p_value),
        "significant_at_0.05": bool(p_value < 0.05),
        "cohens_d": round(d, 4),
        "per_seed": per_seed,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Original: AUC = {orig_auc.mean():.4f} +/- {orig_auc.std():.4f}")
    print(f"Cleaned:  AUC = {clean_auc.mean():.4f} +/- {clean_auc.std():.4f}")
    print(f"Wilcoxon p = {p_value:.4f}  (n={N_SEEDS}, wins={wins}/{N_SEEDS}, d={d:.3f})")
    print(f"Significant at 0.05: {p_value < 0.05}")
    print(f"Total runtime: {(time.time()-t0)/60:.1f} min")
    print(f"Saved: {OUT_JSON}")


if __name__ == "__main__":
    main()
