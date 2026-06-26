"""
Phase 6 — Explainability for the Tri-Modal Fusion Model
========================================================
A clinically deployable depression screener must be able to answer *why*.
This module produces three complementary, mutually-validating explanations,
each grounded in an established XAI technique:

  1. CROSS-MODAL ATTENTION ROLLOUT  (model-intrinsic)
     The fusion layer already computes, per sample, how much each modality
     attends to the other two. We aggregate these weights over the dev set to
     reveal the model's learned inter-modal reliance.

  2. GRADIENT x INPUT SALIENCY  (Shrikumar et al., 2017)
     The gradient of P(depressed) w.r.t. each input feature, multiplied by the
     feature value, attributes the prediction to individual inputs. For the
     face stream we map this back to named Facial Action Units (FACS), so the
     output reads as clinical evidence ("AU15 Lip-Corner-Depressor drove this").

  3. LEAVE-ONE-MODALITY-OUT ABLATION  (occlusion-based importance)
     Zero each modality in turn and measure the drop in dev AUC. A large drop
     means the model genuinely depends on that channel — a causal sanity check
     that the attention weights are not merely cosmetic.

Run:
    python src/explainability/explain_fusion.py
"""

import sys, json
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from configs.config import (DATA_PROCESSED, RESULTS, FIGURES, METRICS,
                            N_MFCC, VOCAB_SIZE, N_AU_FEATURES, EMBED_DIM, AU_COLS)
from src.models.fusion_attention import TriModalFusionModel

FACE_OUT  = DATA_PROCESSED / "daic_faces"
AUDIO_OUT = DATA_PROCESSED / "daic_audio"
TEXT_OUT  = DATA_PROCESSED / "daic_text"
MODEL_P   = RESULTS / "fusion_trimodal_best.pth"
REPORT_P  = METRICS / "phase6_explainability.json"

# FACS clinical descriptions — the language a clinician understands.
AU_CLINICAL = {
    "AU01": "Inner Brow Raiser",    "AU02": "Outer Brow Raiser",
    "AU04": "Brow Lowerer*",        "AU05": "Upper Lid Raiser",
    "AU06": "Cheek Raiser",         "AU09": "Nose Wrinkler",
    "AU10": "Upper Lip Raiser",     "AU12": "Lip Corner Puller (smile)",
    "AU14": "Dimpler",              "AU15": "Lip Corner Depressor*",
    "AU17": "Chin Raiser*",         "AU20": "Lip Stretcher",
    "AU23": "Lip Tightener",        "AU25": "Lips Part",
    "AU26": "Jaw Drop",             "AU28": "Lip Suck",
    "AU45": "Blink",
}  # * = FACS units clinically associated with sadness / depressed affect


# --------------------------------------------------------------------------- #
def load_dev():
    n = min(len(np.load(FACE_OUT / "X_dev.npy")),
            len(np.load(AUDIO_OUT / "X_dev.npy")),
            len(np.load(TEXT_OUT / "X_dev.npy")))
    Xf = np.load(FACE_OUT  / "X_dev.npy")[:n].astype(np.float32)
    Xa = np.load(AUDIO_OUT / "X_dev.npy")[:n].astype(np.float32)
    Xt = np.load(TEXT_OUT  / "X_dev.npy")[:n].astype(np.float32)
    y  = np.load(FACE_OUT  / "y_dev.npy")[:n].astype(np.int64)
    return Xf, Xa, Xt, y


def load_model(device):
    model = TriModalFusionModel(
        n_au=N_AU_FEATURES, n_mfcc=N_MFCC*3,
        vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM).to(device)
    model.load_state_dict(torch.load(MODEL_P, map_location=device))
    model.eval()
    return model


# --------------------------------------------------------------------------- #
#  1. Cross-modal attention rollout
# --------------------------------------------------------------------------- #
def attention_rollout(model, Xf, Xa, Xt, device):
    with torch.no_grad():
        _, attn = model(torch.tensor(Xf).to(device),
                        torch.tensor(Xa).to(device),
                        torch.tensor(Xt).to(device),
                        return_attention=True)
    out = {
        "face_attends_to":  {k: float(v.mean()) for k, v in attn["face_attends_to"].items()},
        "audio_attends_to": {k: float(v.mean()) for k, v in attn["audio_attends_to"].items()},
        "text_attends_to":  {k: float(v.mean()) for k, v in attn["text_attends_to"].items()},
    }
    # "incoming attention" each modality receives = how much others rely on it
    received = {"face": 0.0, "audio": 0.0, "text": 0.0}
    for src, tgts in out.items():
        for tgt, w in tgts.items():
            received[tgt] += w
    total = sum(received.values()) or 1.0
    out["modality_importance"] = {k: round(v / total, 4) for k, v in received.items()}
    return out


# --------------------------------------------------------------------------- #
#  2. Gradient x Input saliency on face AU channels
# --------------------------------------------------------------------------- #
def au_saliency(model, Xf, Xa, Xt, device):
    """Mean |grad x input| attribution per AU channel for the depressed logit."""
    xf = torch.tensor(Xf, requires_grad=True, device=device)
    xa = torch.tensor(Xa, device=device)
    xt = torch.tensor(Xt, device=device)

    logits = model(xf, xa, xt)
    dep_score = logits[:, 1].sum()         # depressed-class logit
    model.zero_grad()
    dep_score.backward()

    # grad x input, |.|, mean over batch and time -> per-channel importance
    attr = (xf.grad * xf).abs().mean(dim=(0, 1)).detach().cpu().numpy()  # (n_au,)
    attr = attr / (attr.sum() + 1e-12)
    return attr


# --------------------------------------------------------------------------- #
#  3. Leave-one-modality-out ablation
# --------------------------------------------------------------------------- #
def auc_of(model, Xf, Xa, Xt, y, device):
    with torch.no_grad():
        logits = model(torch.tensor(Xf).to(device),
                       torch.tensor(Xa).to(device),
                       torch.tensor(Xt).to(device))
        pr = torch.softmax(logits, 1)[:, 1].cpu().numpy()
    return roc_auc_score(y, pr) if len(np.unique(y)) > 1 else float("nan")


def ablation_importance(model, Xf, Xa, Xt, y, device):
    base = auc_of(model, Xf, Xa, Xt, y, device)
    zf, za, zt = np.zeros_like(Xf), np.zeros_like(Xa), np.zeros_like(Xt)
    drops = {
        "without_face":  base - auc_of(model, zf, Xa, Xt, y, device),
        "without_audio": base - auc_of(model, Xf, za, Xt, y, device),
        "without_text":  base - auc_of(model, Xf, Xa, zt, y, device),
    }
    return base, {k: round(v, 4) for k, v in drops.items()}


# --------------------------------------------------------------------------- #
#  Figures
# --------------------------------------------------------------------------- #
def plot_au_saliency(attr, path):
    names  = [c.replace("_r", "").replace("_c", "") for c in AU_COLS]
    labels = [f"{n} — {AU_CLINICAL.get(n, n)}" for n in names]
    order  = np.argsort(attr)
    colors = ["#EF4444" if "*" in labels[i] else "#3B82F6" for i in order]

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(np.array(labels)[order], attr[order], color=colors)
    ax.set_xlabel("Normalized |gradient × input| attribution")
    ax.set_title("Facial Action Unit Importance for Depression Prediction\n"
                 "(red* = FACS units clinically linked to depressed affect)",
                 fontweight="bold", fontsize=11)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def plot_modality_importance(attn_imp, ablation, path):
    mods = ["face", "audio", "text"]
    attn_vals = [attn_imp[m] for m in mods]
    abl_vals  = [max(ablation[f"without_{m}"], 0) for m in mods]
    abl_norm  = [v / (sum(abl_vals) + 1e-12) for v in abl_vals]

    x = np.arange(len(mods)); w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w/2, attn_vals, w, label="Attention rollout", color="#8B5CF6")
    ax.bar(x + w/2, abl_norm,  w, label="Ablation AUC-drop", color="#F59E0B")
    ax.set_xticks(x); ax.set_xticklabels([m.capitalize() for m in mods])
    ax.set_ylabel("Normalized importance")
    ax.set_title("Per-Modality Importance — Two Independent Estimates",
                 fontweight="bold")
    ax.legend()
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


# --------------------------------------------------------------------------- #
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if not MODEL_P.exists():
        print(f"Trained model not found at {MODEL_P}. Run train_fusion.py first.")
        return

    Xf, Xa, Xt, y = load_dev()
    model = load_model(device)
    print(f"Dev set: {len(y)} participants\n")

    # 1. attention rollout
    attn = attention_rollout(model, Xf, Xa, Xt, device)
    print("=" * 60)
    print("  1. CROSS-MODAL ATTENTION ROLLOUT")
    print("=" * 60)
    for src, tgts in attn.items():
        if src == "modality_importance":
            continue
        parts = "  ".join(f"{k}={v:.3f}" for k, v in tgts.items())
        print(f"  {src:20s} {parts}")
    print(f"\n  Modality importance (attention): {attn['modality_importance']}")

    # 2. AU saliency
    attr = au_saliency(model, Xf, Xa, Xt, device)
    names = [c.replace("_r", "").replace("_c", "") for c in AU_COLS]
    top = np.argsort(attr)[::-1][:5]
    print("\n" + "=" * 60)
    print("  2. TOP-5 FACIAL ACTION UNITS (gradient x input)")
    print("=" * 60)
    for i in top:
        print(f"  {names[i]:6s} {AU_CLINICAL.get(names[i], ''):28s} {attr[i]:.4f}")

    # 3. ablation
    base, drops = ablation_importance(model, Xf, Xa, Xt, y, device)
    print("\n" + "=" * 60)
    print("  3. LEAVE-ONE-MODALITY-OUT ABLATION")
    print("=" * 60)
    print(f"  Full model dev AUC: {base:.4f}")
    for k, v in drops.items():
        print(f"  {k:16s} AUC drop = {v:+.4f}")

    # ---- figures ----
    FIGURES.mkdir(parents=True, exist_ok=True)
    plot_au_saliency(attr, FIGURES / "phase6_au_saliency.png")
    plot_modality_importance(attn["modality_importance"], drops,
                             FIGURES / "phase6_modality_importance.png")
    print(f"\n  Figures saved to {FIGURES}")

    # ---- JSON report ----
    METRICS.mkdir(parents=True, exist_ok=True)
    report = {
        "phase": 6,
        "analysis": "Explainability — attention rollout + gradient saliency + ablation",
        "full_model_auc": round(base, 4),
        "attention_rollout": attn,
        "au_saliency": {names[i]: round(float(attr[i]), 4)
                        for i in np.argsort(attr)[::-1]},
        "top_action_units": [
            {"au": names[i], "clinical": AU_CLINICAL.get(names[i], ""),
             "importance": round(float(attr[i]), 4)} for i in top
        ],
        "ablation_auc_drop": drops,
        "references": [
            "Shrikumar et al., Learning Important Features Through Propagating "
            "Activation Differences, ICML 2017",
            "Abnar & Zuidema, Quantifying Attention Flow in Transformers, ACL 2020",
        ],
    }
    with open(REPORT_P, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report saved to {REPORT_P}")


if __name__ == "__main__":
    main()
