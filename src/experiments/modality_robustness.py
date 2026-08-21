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
all-modalities-present baseline.

Threshold protocol (IMPORTANT): for every masking condition, the decision
threshold is selected on the DEV split's probabilities only, then applied
FIXED to the TEST split -- matching the paper's stated protocol ("model
selection and threshold tuning are performed on the dev split only; the
test split is used exclusively for final held-out evaluation") and the
convention already used by src/experiments/test_evaluation.py. An earlier
version of this script incorrectly re-searched the threshold on the test
labels themselves for the test-split numbers; that is data leakage and has
been fixed here.

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
def get_probs(model, Xf, Xa, Xt, device, mask_modality=None):
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
    return torch.softmax(logits, 1)[:, 1].cpu().numpy()


def score(y, probs, thr):
    auc = roc_auc_score(y, probs) if len(np.unique(y)) > 1 else float("nan")
    preds = (probs >= thr).astype(int)
    f1 = f1_score(y, preds, average="macro", zero_division=0)
    acc = (preds == y).mean()
    return {"auc": round(float(auc), 4), "f1": round(float(f1), 4),
            "acc": round(float(acc), 4), "threshold": round(float(thr), 4)}


def run_condition(mask_name, model, device,
                   Xf_dv, Xa_dv, Xt_dv, y_dv,
                   Xf_te, Xa_te, Xt_te, y_te):
    """Threshold is selected on DEV only, then applied fixed to TEST."""
    dev_probs = get_probs(model, Xf_dv, Xa_dv, Xt_dv, device, mask_modality=mask_name)
    thr = best_threshold(y_dv, dev_probs)
    dev_result = score(y_dv, dev_probs, thr)

    test_probs = get_probs(model, Xf_te, Xa_te, Xt_te, device, mask_modality=mask_name)
    test_result = score(y_te, test_probs, thr)   # same dev-derived threshold, no leakage
    return dev_result, test_result


def main():
    device = "cpu"
    model = load_model(device)

    Xf_dv, Xa_dv, Xt_dv, y_dv = load_dev()
    Xf_te, Xa_te, Xt_te, y_te = load_test()

    dev_results, test_results = {}, {}
    conditions = [("all_modalities", None), ("missing_face", "face"),
                  ("missing_audio", "audio"), ("missing_text", "text")]

    print(f"\n--- Dev (N={len(y_dv)}) / Test (N={len(y_te)}) "
          f"[threshold always dev-derived, applied fixed to test] ---")
    for label, mask in conditions:
        dev_r, test_r = run_condition(mask, model, device,
                                       Xf_dv, Xa_dv, Xt_dv, y_dv,
                                       Xf_te, Xa_te, Xt_te, y_te)
        dev_results[label] = dev_r
        test_results[label] = test_r
        print(f"  {label:16s}  dev F1={dev_r['f1']:.3f} AUC={dev_r['auc']:.3f}  |  "
              f"test F1={test_r['f1']:.3f} AUC={test_r['auc']:.3f} "
              f"(thr={dev_r['threshold']:.2f})")

    baseline_dev, baseline_test = dev_results["all_modalities"], test_results["all_modalities"]
    for label in ["missing_face", "missing_audio", "missing_text"]:
        dev_results[label]["f1_drop"]  = round(baseline_dev["f1"] - dev_results[label]["f1"], 4)
        dev_results[label]["auc_drop"] = round(baseline_dev["auc"] - dev_results[label]["auc"], 4)
        test_results[label]["f1_drop"]  = round(baseline_test["f1"] - test_results[label]["f1"], 4)
        test_results[label]["auc_drop"] = round(baseline_test["auc"] - test_results[label]["auc"], 4)

    report = {
        "analysis": "test-time missing-modality robustness "
                    "(zero-masking one modality at inference)",
        "note": "Model trained WITH modality-dropout(p=0.15) as a regularizer; "
                "this experiment validates whether that regularizer actually "
                "confers inference-time robustness to a missing modality. "
                "Threshold is selected on dev only (per masking condition) and "
                "applied fixed to test -- no test-label leakage.",
        "dev": dev_results,
        "test": test_results,
    }
    METRICS.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved: {OUT_JSON}")


if __name__ == "__main__":
    main()
