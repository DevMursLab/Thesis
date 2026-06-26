"""
Phase 5 — Algorithmic Fairness Audit
=====================================
Clinical AI demands more than aggregate accuracy. A depression screening model
that is 80% accurate overall but systematically misses depression in women is
unsafe to deploy. This module quantifies *group fairness* across the protected
attribute (gender) using the standard fairness criteria from the algorithmic
fairness literature.

Fairness criteria computed (Barocas, Hardt & Narayanan, 2019):
  - Demographic Parity        : P(Ŷ=1 | M)  vs  P(Ŷ=1 | F)
  - Equal Opportunity         : TPR_M       vs  TPR_F        (recall on depressed)
  - Equalized Odds            : (TPR, FPR)_M vs (TPR, FPR)_F
  - Predictive Parity         : PPV_M        vs  PPV_F        (precision)
  - Per-group AUC / F1 / Acc

Each criterion yields a *gap* (absolute difference between groups). A gap below
0.10 is conventionally treated as "fair"; below 0.05 as "strongly fair".

Run:
    python src/fairness/fairness_analysis.py
"""

import sys, json
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc as sk_auc,
    f1_score, precision_score, recall_score, accuracy_score,
)

from configs.config import (DATA_PROCESSED, RESULTS, FIGURES, METRICS,
                            N_MFCC, VOCAB_SIZE, N_AU_FEATURES, BATCH_SIZE)
from src.models.fusion_attention import TriModalFusionModel

FACE_OUT  = DATA_PROCESSED / "daic_faces"
AUDIO_OUT = DATA_PROCESSED / "daic_audio"
TEXT_OUT  = DATA_PROCESSED / "daic_text"
MODEL_P   = RESULTS / "fusion_trimodal_best.pth"
REPORT_P  = METRICS / "phase5_fairness.json"

GROUP_NAMES = {0: "Male", 1: "Female"}
FAIR_THRESHOLD = 0.10   # gap below this = fair


# --------------------------------------------------------------------------- #
#  Data + inference
# --------------------------------------------------------------------------- #
def load_dev():
    n = min(
        len(np.load(FACE_OUT  / "X_dev.npy")),
        len(np.load(AUDIO_OUT / "X_dev.npy")),
        len(np.load(TEXT_OUT  / "X_dev.npy")),
    )
    Xf = np.load(FACE_OUT  / "X_dev.npy")[:n].astype(np.float32)
    Xa = np.load(AUDIO_OUT / "X_dev.npy")[:n].astype(np.float32)
    Xt = np.load(TEXT_OUT  / "X_dev.npy")[:n].astype(np.float32)
    y  = np.load(FACE_OUT  / "y_dev.npy")[:n].astype(np.int64)
    g  = np.load(FACE_OUT  / "gender_dev.npy")[:n].astype(np.int64)
    return Xf, Xa, Xt, y, g


def run_inference(model, Xf, Xa, Xt, device):
    model.eval()
    preds, probs = [], []
    with torch.no_grad():
        for i in range(0, len(Xf), BATCH_SIZE):
            sl = slice(i, i + BATCH_SIZE)
            logits = model(
                torch.tensor(Xf[sl]).to(device),
                torch.tensor(Xa[sl]).to(device),
                torch.tensor(Xt[sl]).to(device),
            )
            p = torch.softmax(logits, 1)[:, 1].cpu().numpy()
            probs.extend(p)
            preds.extend((p >= 0.5).astype(int))
    return np.array(preds), np.array(probs)


# --------------------------------------------------------------------------- #
#  Fairness metric primitives
# --------------------------------------------------------------------------- #
def rates(y_true, y_pred):
    """Return TPR, FPR, PPV, selection-rate for one group."""
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn) if (tp + fn) else 0.0          # recall / sensitivity
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    ppv = tp / (tp + fp) if (tp + fp) else 0.0          # precision
    sel = (tp + fp) / len(y_true) if len(y_true) else 0.0
    return dict(TPR=tpr, FPR=fpr, PPV=ppv, selection_rate=sel,
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def per_group_metrics(y, p, pr, g):
    out = {}
    for grp in [0, 1]:
        m = g == grp
        if m.sum() == 0:
            continue
        yt, yp, ys = y[m], p[m], pr[m]
        r = rates(yt, yp)
        try:
            grp_auc = sk_auc(*roc_curve(yt, ys)[:2]) if len(np.unique(yt)) > 1 else float("nan")
        except Exception:
            grp_auc = float("nan")
        out[GROUP_NAMES[grp]] = {
            "n": int(m.sum()),
            "n_depressed": int(yt.sum()),
            "accuracy":  round(accuracy_score(yt, yp), 4),
            "f1":        round(f1_score(yt, yp, zero_division=0), 4),
            "precision": round(precision_score(yt, yp, zero_division=0), 4),
            "recall":    round(recall_score(yt, yp, zero_division=0), 4),
            "auc":       round(grp_auc, 4) if grp_auc == grp_auc else None,
            **{k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()},
        }
    return out


def fairness_gaps(pg):
    """Compute absolute between-group gaps for each fairness criterion."""
    if "Male" not in pg or "Female" not in pg:
        return {}
    M, F = pg["Male"], pg["Female"]
    gaps = {
        "demographic_parity_gap": abs(M["selection_rate"] - F["selection_rate"]),
        "equal_opportunity_gap":  abs(M["TPR"] - F["TPR"]),
        "equalized_odds_gap":     max(abs(M["TPR"] - F["TPR"]),
                                      abs(M["FPR"] - F["FPR"])),
        "predictive_parity_gap":  abs(M["PPV"] - F["PPV"]),
        "f1_gap":                 abs(M["f1"] - F["f1"]),
        "accuracy_gap":           abs(M["accuracy"] - F["accuracy"]),
    }
    gaps = {k: round(v, 4) for k, v in gaps.items()}
    gaps["verdict"] = {
        k: ("FAIR" if v <= FAIR_THRESHOLD else "BIASED")
        for k, v in gaps.items()
    }
    return gaps


# --------------------------------------------------------------------------- #
#  Visualisations
# --------------------------------------------------------------------------- #
def plot_group_metrics(pg, path):
    metrics = ["accuracy", "f1", "precision", "recall", "auc"]
    male   = [pg["Male"].get(m)   or 0 for m in metrics]
    female = [pg["Female"].get(m) or 0 for m in metrics]

    x = np.arange(len(metrics)); w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - w/2, male,   w, label="Male",   color="#3B82F6")
    b2 = ax.bar(x + w/2, female, w, label="Female", color="#EC4899")
    ax.set_xticks(x); ax.set_xticklabels([m.upper() for m in metrics])
    ax.set_ylim(0, 1.05); ax.set_ylabel("Score")
    ax.set_title("Per-Gender Performance — Tri-Modal Fusion", fontweight="bold")
    ax.legend()
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.02,
                    f"{h:.2f}", ha="center", fontsize=8)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def plot_fairness_gaps(gaps, path):
    keys = ["demographic_parity_gap", "equal_opportunity_gap",
            "equalized_odds_gap", "predictive_parity_gap", "f1_gap"]
    vals = [gaps[k] for k in keys]
    labels = ["Demographic\nParity", "Equal\nOpportunity",
              "Equalized\nOdds", "Predictive\nParity", "F1"]
    colors = ["#10B981" if v <= FAIR_THRESHOLD else "#EF4444" for v in vals]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, vals, color=colors)
    ax.axhline(FAIR_THRESHOLD, ls="--", color="gray",
               label=f"Fairness threshold ({FAIR_THRESHOLD})")
    ax.set_ylabel("Absolute Gap  |Male − Female|")
    ax.set_title("Fairness Criterion Gaps  (lower = fairer)", fontweight="bold")
    ax.legend()
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def plot_roc_per_group(y, pr, g, path):
    fig, ax = plt.subplots(figsize=(7, 6))
    for grp, color in [(0, "#3B82F6"), (1, "#EC4899")]:
        m = g == grp
        if m.sum() == 0 or len(np.unique(y[m])) < 2:
            continue
        fpr, tpr, _ = roc_curve(y[m], pr[m])
        a = sk_auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, color=color,
                label=f"{GROUP_NAMES[grp]} (AUC={a:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC by Gender Group", fontweight="bold"); ax.legend()
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if not MODEL_P.exists():
        print(f"Trained model not found at {MODEL_P}. Run train_fusion.py first.")
        return

    Xf, Xa, Xt, y, g = load_dev()
    print(f"Dev set: {len(y)} participants | "
          f"Male: {(g==0).sum()}  Female: {(g==1).sum()}")

    model = TriModalFusionModel(
        n_au=N_AU_FEATURES, n_mfcc=N_MFCC*3,
        vocab_size=VOCAB_SIZE, embed_dim=256).to(device)
    model.load_state_dict(torch.load(MODEL_P, map_location=device))

    preds, probs = run_inference(model, Xf, Xa, Xt, device)

    pg   = per_group_metrics(y, preds, probs, g)
    gaps = fairness_gaps(pg)

    # ---- console report ----
    print("\n" + "=" * 64)
    print("  PER-GROUP PERFORMANCE")
    print("=" * 64)
    for grp, m in pg.items():
        print(f"\n  [{grp}]  n={m['n']}  depressed={m['n_depressed']}")
        print(f"     Acc={m['accuracy']:.3f}  F1={m['f1']:.3f}  "
              f"Prec={m['precision']:.3f}  Recall={m['recall']:.3f}  "
              f"AUC={m['auc']}")
        print(f"     TPR={m['TPR']:.3f}  FPR={m['FPR']:.3f}  "
              f"PPV={m['PPV']:.3f}  SelRate={m['selection_rate']:.3f}")

    print("\n" + "=" * 64)
    print("  FAIRNESS CRITERION GAPS")
    print("=" * 64)
    for k, v in gaps.items():
        if k == "verdict":
            continue
        verdict = gaps["verdict"][k]
        mark = "[OK]  " if verdict == "FAIR" else "[BIAS]"
        print(f"  {mark} {k:28s} = {v:.4f}   ({verdict})")

    n_fair  = sum(1 for x in gaps["verdict"].values() if x == "FAIR")
    n_total = len(gaps["verdict"])
    print(f"\n  Overall: {n_fair}/{n_total} criteria within fairness threshold "
          f"({FAIR_THRESHOLD})")

    # ---- figures ----
    FIGURES.mkdir(parents=True, exist_ok=True)
    plot_group_metrics(pg,        FIGURES / "phase5_group_metrics.png")
    plot_fairness_gaps(gaps,      FIGURES / "phase5_fairness_gaps.png")
    plot_roc_per_group(y, probs, g, FIGURES / "phase5_roc_per_group.png")
    print(f"\n  Figures saved to {FIGURES}")

    # ---- JSON report ----
    METRICS.mkdir(parents=True, exist_ok=True)
    report = {
        "phase": 5,
        "analysis": "Algorithmic Fairness Audit (gender)",
        "protected_attribute": "gender",
        "fairness_threshold": FAIR_THRESHOLD,
        "per_group": pg,
        "gaps": {k: v for k, v in gaps.items() if k != "verdict"},
        "verdict": gaps["verdict"],
        "criteria_passed": f"{n_fair}/{n_total}",
        "references": [
            "Barocas, Hardt & Narayanan, Fairness and Machine Learning, 2019",
            "Hardt et al., Equality of Opportunity in Supervised Learning, NeurIPS 2016",
        ],
    }
    with open(REPORT_P, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report saved to {REPORT_P}")


if __name__ == "__main__":
    main()
