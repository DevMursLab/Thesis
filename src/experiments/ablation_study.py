"""
Phase 7 — Ablation Study, SOTA Comparison & Statistical Validation
===================================================================
The empirical core of the thesis. We answer three reviewer questions:

  Q1 (ablation)    : Does tri-modal fusion actually beat each unimodal and
                     bimodal configuration, or is one modality doing all the work?
  Q2 (significance): Is the tri-modal improvement statistically real, or within
                     the noise band of a 34-participant dev set?
  Q3 (positioning) : How does this sit against published DAIC-WOZ results?

Method
------
* Seven configurations are trained under IDENTICAL conditions (same encoders,
  attention, classifier head, optimizer, schedule, seed) via
  ConfigurableFusionModel — so differences are causal w.r.t. modality content.
* Each configuration's dev AUC is reported with a 95% bootstrap confidence
  interval (2,000 resamples of the dev set).
* The tri-modal vs. best-unimodal AUC gap is tested with a paired bootstrap
  test; we report the one-sided p-value (H0: tri-modal <= best unimodal).
* Error analysis flags participants misclassified by every configuration
  (intrinsically hard cases) vs. those rescued by adding modalities.

Run:
    python -m src.experiments.ablation_study
"""

import sys, json
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, f1_score

from configs.config import (DATA_PROCESSED, RESULTS, FIGURES, METRICS,
                            N_MFCC, VOCAB_SIZE, N_AU_FEATURES, EMBED_DIM,
                            FUSION_DROPOUT, WEIGHT_DECAY, SEED)
from src.models.configurable_fusion import ConfigurableFusionModel

FACE_OUT  = DATA_PROCESSED / "daic_faces"
AUDIO_OUT = DATA_PROCESSED / "daic_audio"
TEXT_OUT  = DATA_PROCESSED / "daic_text"
REPORT_P  = METRICS / "phase7_ablation.json"

CONFIGS = [
    ("face",),
    ("audio",),
    ("text",),
    ("face", "audio"),
    ("face", "text"),
    ("audio", "text"),
    ("face", "audio", "text"),
]

# Published DAIC-WOZ depression-detection results for honest positioning.
# (Development-set binary depression F1, as reported in the cited works.)
SOTA = [
    {"method": "AVEC-2017 audio baseline (Ringeval+ 2017)", "modalities": "Audio",       "f1": 0.50},
    {"method": "AVEC-2017 text baseline (Ringeval+ 2017)",  "modalities": "Text",        "f1": 0.49},
    {"method": "Gong & Poellabauer (2017), topic-modeling", "modalities": "A+V+T",       "f1": 0.70},
    {"method": "Williamson+ (2016), multimodal",            "modalities": "A+V",         "f1": 0.57},
]

BOOTSTRAP_N = 2000
SEEDS = [42, 1, 7, 123, 2024]   # multi-seed: single-run results on N=34 are noise


def set_seed(seed=SEED):
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
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


def train_one(modalities, data, device, epochs=40, seed=SEED):
    set_seed(seed)
    Xf_tr, Xa_tr, Xt_tr, y_tr = data["train"]
    Xf_dv, Xa_dv, Xt_dv, y_dv = data["dev"]

    model = ConfigurableFusionModel(
        modalities, n_au=N_AU_FEATURES, n_mfcc=N_MFCC*3,
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

    best_auc, best_probs, wait, patience = 0.0, None, 0, 12
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
    f1  = f1_score(y_dv, preds, zero_division=0)
    acc = (preds == y_dv).mean()
    return {"auc": best_auc, "f1": f1, "acc": acc,
            "probs": best_probs, "preds": preds, "threshold": thr}


# --------------------------------------------------------------------------- #
#  Statistics
# --------------------------------------------------------------------------- #
def bootstrap_auc_ci(y, probs, n=BOOTSTRAP_N, seed=SEED):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y)); aucs = []
    for _ in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(y[s])) < 2:
            continue
        aucs.append(roc_auc_score(y[s], probs[s]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return round(float(lo), 4), round(float(hi), 4)


def paired_bootstrap_test(y, probs_a, probs_b, n=BOOTSTRAP_N, seed=SEED):
    """One-sided test H0: AUC_a <= AUC_b. Returns (mean_diff, p_value)."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y)); diffs = []
    for _ in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(y[s])) < 2:
            continue
        diffs.append(roc_auc_score(y[s], probs_a[s]) - roc_auc_score(y[s], probs_b[s]))
    diffs = np.array(diffs)
    p = float((diffs <= 0).mean())          # P(no improvement)
    return round(float(diffs.mean()), 4), round(p, 4)


# --------------------------------------------------------------------------- #
def plot_ablation(results, path):
    names = ["+".join(c) for c in CONFIGS]
    aucs  = [results[n]["auc_mean"] for n in names]
    err   = [results[n]["auc_std"] for n in names]
    colors = ["#3B82F6"]*3 + ["#8B5CF6"]*3 + ["#EF4444"]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(names, aucs, yerr=err, capsize=4, color=colors)
    ax.axhline(0.5, ls="--", color="gray", label="chance")
    ax.set_ylabel(f"Dev AUC  (mean ± std over {len(SEEDS)} seeds)")
    ax.set_title("Ablation: Modality Configurations (red = full tri-modal)",
                 fontweight="bold")
    ax.set_ylim(0, 1); plt.xticks(rotation=20, ha="right"); ax.legend()
    for i, a in enumerate(aucs):
        ax.text(i, a + err[i] + 0.02, f"{a:.3f}", ha="center", fontsize=8)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


# --------------------------------------------------------------------------- #
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    data = {"train": load_split("train"), "dev": load_split("dev")}
    y_dv = data["dev"][3]
    print(f"Train: {len(data['train'][3])} | Dev: {len(y_dv)}\n")

    results = {}
    print(f"Training {len(CONFIGS)} configurations x {len(SEEDS)} seeds "
          f"= {len(CONFIGS)*len(SEEDS)} runs (this takes a few minutes)\n")
    print("=" * 78)
    print(f"  {'Configuration':18s} {'AUC (mean±std)':>18s} {'95% CI':>16s} "
          f"{'F1':>12s} {'Acc':>7s}")
    print("=" * 78)
    for combo in CONFIGS:
        name = "+".join(combo)
        runs = [train_one(combo, data, device, seed=s) for s in SEEDS]
        aucs = np.array([r["auc"] for r in runs])
        f1s  = np.array([r["f1"]  for r in runs])
        accs = np.array([r["acc"] for r in runs])
        probs_mean = np.mean([r["probs"] for r in runs], axis=0)   # seed-ensemble
        thr   = best_threshold(y_dv, probs_mean)
        preds = (probs_mean >= thr).astype(int)
        ci    = bootstrap_auc_ci(y_dv, probs_mean)
        results[name] = {
            "aucs": aucs, "auc_mean": float(aucs.mean()), "auc_std": float(aucs.std()),
            "f1_mean": float(f1s.mean()), "f1_std": float(f1s.std()),
            "acc_mean": float(accs.mean()),
            "probs": probs_mean, "preds": preds, "threshold": thr, "ci": ci,
        }
        r = results[name]
        print(f"  {name:18s} {r['auc_mean']:.3f}±{r['auc_std']:.3f}       "
              f"[{ci[0]:.3f},{ci[1]:.3f}]  {r['f1_mean']:.3f}±{r['f1_std']:.3f} "
              f"{r['acc_mean']:7.3f}")

    # ---- statistical comparison: tri-modal vs best unimodal (paired across seeds) ----
    tri = results["face+audio+text"]
    unimodals = {k: results[k] for k in ["face", "audio", "text"]}
    best_uni_name = max(unimodals, key=lambda k: unimodals[k]["auc_mean"])
    best_uni = unimodals[best_uni_name]

    per_seed_gain = tri["aucs"] - best_uni["aucs"]          # paired by seed
    wins = int((per_seed_gain > 0).sum())
    # bootstrap p-value on the seed-ensemble probabilities
    diff_bs, pval_bs = paired_bootstrap_test(y_dv, tri["probs"], best_uni["probs"])

    print("\n" + "=" * 78)
    print("  STATISTICAL VALIDATION")
    print("=" * 78)
    print(f"  Tri-modal AUC (mean±std)        : {tri['auc_mean']:.3f}±{tri['auc_std']:.3f}")
    print(f"  Best unimodal ({best_uni_name}) AUC      : {best_uni['auc_mean']:.3f}±{best_uni['auc_std']:.3f}")
    print(f"  Per-seed AUC gain (mean)        : {per_seed_gain.mean():+.4f}")
    print(f"  Seeds where tri-modal wins      : {wins}/{len(SEEDS)}")
    print(f"  Bootstrap p (H0: tri<=best uni) : {pval_bs:.4f}")
    print(f"  -> {'SIGNIFICANT (p<0.05)' if pval_bs < 0.05 else 'NOT significant at 0.05 (expected on N=34 dev)'}")

    # ---- error analysis ----
    all_wrong = np.ones(len(y_dv), dtype=bool)
    for r in results.values():
        all_wrong &= (r["preds"] != y_dv)
    uni_wrong = (results[best_uni_name]["preds"] != y_dv)
    rescued = uni_wrong & (tri["preds"] == y_dv)
    print("\n" + "=" * 70)
    print("  ERROR ANALYSIS")
    print("=" * 70)
    print(f"  Hard cases wrong in ALL configs : {int(all_wrong.sum())}/{len(y_dv)}")
    print(f"  Rescued by tri-modal vs best uni : {int(rescued.sum())}")

    # ---- SOTA table ----
    print("\n" + "=" * 70)
    print("  POSITIONING vs PUBLISHED DAIC-WOZ RESULTS (dev F1)")
    print("=" * 70)
    for s in SOTA:
        print(f"  {s['method']:42s} [{s['modalities']:5s}] F1={s['f1']:.2f}")
    print(f"  {'THIS WORK (tri-modal fusion)':42s} [A+V+T] F1={tri['f1_mean']:.2f}")

    # ---- figure + report ----
    FIGURES.mkdir(parents=True, exist_ok=True)
    plot_ablation(results, FIGURES / "phase7_ablation.png")

    METRICS.mkdir(parents=True, exist_ok=True)
    report = {
        "phase": 7,
        "analysis": "Ablation + statistical validation + SOTA positioning",
        "protocol": {"seeds": SEEDS, "bootstrap_resamples": BOOTSTRAP_N,
                     "dev_n": int(len(y_dv))},
        "configurations": {
            n: {"auc_mean": round(r["auc_mean"], 4), "auc_std": round(r["auc_std"], 4),
                "auc_ci95": r["ci"],
                "f1_mean": round(r["f1_mean"], 4), "f1_std": round(r["f1_std"], 4),
                "acc_mean": round(r["acc_mean"], 4), "threshold": r["threshold"]}
            for n, r in results.items()
        },
        "statistical_test": {
            "method": "multi-seed paired comparison + bootstrap (2000 resamples)",
            "tri_modal_auc_mean": round(tri["auc_mean"], 4),
            "best_unimodal": best_uni_name,
            "best_unimodal_auc_mean": round(best_uni["auc_mean"], 4),
            "per_seed_mean_gain": round(float(per_seed_gain.mean()), 4),
            "seeds_trimodal_wins": f"{wins}/{len(SEEDS)}",
            "bootstrap_p_value": pval_bs,
            "significant_at_0.05": bool(pval_bs < 0.05),
        },
        "error_analysis": {
            "hard_cases_wrong_in_all": int(all_wrong.sum()),
            "rescued_by_trimodal": int(rescued.sum()),
            "dev_n": int(len(y_dv)),
        },
        "sota_positioning": SOTA + [
            {"method": "THIS WORK (tri-modal cross-attention)",
             "modalities": "A+V+T", "f1": round(tri["f1_mean"], 4)}],
        "references": [
            "Ringeval et al., AVEC 2017 Workshop",
            "Gong & Poellabauer, Topic Modeling DAIC-WOZ, AVEC 2017",
            "Williamson et al., Multimodal Depression Detection, AVEC 2016",
        ],
    }
    with open(REPORT_P, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Figure: {FIGURES / 'phase7_ablation.png'}")
    print(f"  Report: {REPORT_P}")


if __name__ == "__main__":
    main()
