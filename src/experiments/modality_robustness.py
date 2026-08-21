"""
Missing-Modality Robustness (Test-Time Ablation)
==================================================
Contribution #4 in the paper claims modality-dropout training is a
regularizer motivated by the clinical constraint that a modality may be
unavailable at inference time, but the paper explicitly notes this was
never validated with an actual inference-time-masking experiment.

This script closes that gap: it zeros out each modality (one at a time)
at TEST time on the trained multitask model and measures the resulting
F1/AUC drop on both dev and the held-out test split, compared to the
all-modalities-present baseline. This produces a real, reportable
missing-modality robustness result (or shows the claim does not hold).

Run:
    python -m src.experiments.modality_robustness
"""

import sys, json
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score

from configs.config import DATA_PROCESSED, RESULTS, METRICS, N_PHQ_SYMPTOMS

FACE_DIR   = DATA_PROCESSED / "daic_faces"
AUDIO_DIR  = DATA_PROCESSED / "daic_audio_covarep"
TEXT_DIR   = DATA_PROCESSED / "daic_text"
TEST_CACHE = DATA_PROCESSED / "daic_test_raw_cache"
CKPT       = RESULTS / "multitask_best.pth"
OUT_JSON   = METRICS / "modality_robustness.json"

from src.models.multitask_fusion import MultiTaskFusionModel


def load_dev():
    Xf = np.load(FACE_DIR / "X_dev.npy").astype(np.float32)
    Xa = np.load(AUDIO_DIR / "X_dev.npy").astype(np.float32)
    Xt = np.load(TEXT_DIR / "X_dev.npy").astype(np.float32)
    y  = np.load(FACE_DIR / "y_dev.npy").astype(np.int64)
    n = min(len(Xf), len(Xa), len(Xt), len(y))
    return Xf[:n], Xa[:n], Xt[:n], y[:n]


def load_test():
    Xf = np.load(TEST_CACHE / "X_face.npy").astype(np.float32)
    Xa = np.load(TEST_CACHE / "X_audio.npy").astype(np.float32)
    Xt = np.load(TEST_CACHE / "X_text.npy").astype(np.float32)
    y  = np.load(TEST_CACHE / "y.npy").astype(np.int64)
    return Xf, Xa, Xt, y


def load_model(device):
    model = MultiTaskFusionModel(
        n_au=20, n_audio_feat=199, vocab_size=1000, embed_dim=64,
        dropout=0.6, n_symptoms=N_PHQ_SYMPTOMS, modality_dropout_p=0.15,
    ).to(device)
    state = torch.load(CKPT, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def best_threshold(y_true, probs):
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.2, 0.8, 0.02):
        f1 = f1_score(y_true, (probs >= t).astype(int), average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t


@torch.no_grad()
def eval_with_mask(model, Xf, Xa, Xt, y, device, mask_modality=None):
    """mask_modality: None (no masking), or one of 'face','audio','text'."""
    Xf_, Xa_, Xt_ = Xf.copy(), Xa.copy(), Xt.copy()
    if mask_modality == "face":
        Xf_[:] = 0.0
    elif mask_modality == "audio":
        Xa_[:] = 0.0
    elif mask_modality == "text":
        Xt_[:] = 0.0

    Xf_t = torch.tensor(Xf_).to(device)
    Xa_t = torch.tensor(Xa_).to(device)
    Xt_t = torch.tensor(Xt_).to(device)
    logits, _, _ = model(Xf_t, Xa_t, Xt_t)
    probs = torch.softmax(logits, 1)[:, 1].cpu().numpy()

    auc = roc_auc_score(y, probs) if len(np.unique(y)) > 1 else float("nan")
    thr = best_threshold(y, probs)
    preds = (probs >= thr).astype(int)
    f1 = f1_score(y, preds, average="macro", zero_division=0)
    acc = (preds == y).mean()
    return {"auc": round(float(auc), 4), "f1": round(float(f1), 4),
            "acc": round(float(acc), 4), "threshold": round(float(thr), 4)}


def run_split(name, Xf, Xa, Xt, y, model, device):
    print(f"\n--- {name} split (N={len(y)}) ---")
    results = {}
    baseline = eval_with_mask(model, Xf, Xa, Xt, y, device, mask_modality=None)
    results["all_modalities"] = baseline
    print(f"  {'all modalities':16s}  F1={baseline['f1']:.3f}  AUC={baseline['auc']:.3f}")
    for m in ["face", "audio", "text"]:
        r = eval_with_mask(model, Xf, Xa, Xt, y, device, mask_modality=m)
        r["f1_drop"]  = round(baseline["f1"] - r["f1"], 4)
        r["auc_drop"] = round(baseline["auc"] - r["auc"], 4)
        results[f"missing_{m}"] = r
        print(f"  missing {m:8s}      F1={r['f1']:.3f} (d={r['f1_drop']:+.3f})  "
              f"AUC={r['auc']:.3f} (d={r['auc_drop']:+.3f})")
    return results


def main():
    device = "cpu"
    model = load_model(device)

    Xf_dv, Xa_dv, Xt_dv, y_dv = load_dev()
    Xf_te, Xa_te, Xt_te, y_te = load_test()

    dev_results  = run_split("Dev",  Xf_dv, Xa_dv, Xt_dv, y_dv, model, device)
    test_results = run_split("Test", Xf_te, Xa_te, Xt_te, y_te, model, device)

    report = {
        "analysis": "test-time missing-modality robustness "
                    "(zero-masking one modality at inference)",
        "note": "Model trained WITH modality-dropout(p=0.15) as a regularizer; "
                "this experiment validates whether that regularizer actually "
                "confers inference-time robustness to a missing modality.",
        "dev": dev_results,
        "test": test_results,
    }
    METRICS.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved: {OUT_JSON}")


if __name__ == "__main__":
    main()
