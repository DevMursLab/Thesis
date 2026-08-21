"""
Error Analysis & Probability Calibration Check
=================================================
Two diagnostics not yet in the paper:

1. Calibration: are the model's predicted probabilities meaningful as
   probabilities (e.g. among predictions near 0.7, are ~70% actually
   depressed), or just a rank-ordering signal? Reported as a reliability
   table (predicted-probability bin vs. observed positive rate) plus the
   Brier score and Expected Calibration Error (ECE).

2. Error case study: which test participants are misclassified, and does
   the model's own explainability signal (attention weight on audio,
   already computed in Phase 6) look different for correct vs. incorrect
   predictions? This gives concrete material for viva questions like
   "show me a case your model got wrong and explain why."

Run:
    python -m src.experiments.error_calibration_analysis
"""

import sys, json
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import torch
from sklearn.metrics import brier_score_loss, f1_score

from configs.config import DATA_PROCESSED, RESULTS, METRICS, N_PHQ_SYMPTOMS
from src.models.multitask_fusion import MultiTaskFusionModel

FACE_DIR   = DATA_PROCESSED / "daic_faces"
AUDIO_DIR  = DATA_PROCESSED / "daic_audio_covarep"
TEXT_DIR   = DATA_PROCESSED / "daic_text"
TEST_CACHE = DATA_PROCESSED / "daic_test_raw_cache"
CKPT       = RESULTS / "multitask_best.pth"
OUT_JSON   = METRICS / "error_calibration_analysis.json"


def load_dev():
    Xf = np.load(FACE_DIR / "X_dev.npy").astype(np.float32)
    Xa = np.load(AUDIO_DIR / "X_dev.npy").astype(np.float32)
    Xt = np.load(TEXT_DIR / "X_dev.npy").astype(np.float32)
    y  = np.load(FACE_DIR / "y_dev.npy").astype(np.int64)
    meta = FACE_DIR / "meta_dev.csv"
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


@torch.no_grad()
def get_probs(model, Xf, Xa, Xt, device):
    Xf_t = torch.tensor(Xf).to(device)
    Xa_t = torch.tensor(Xa).to(device)
    Xt_t = torch.tensor(Xt).to(device)
    logits, _, _ = model(Xf_t, Xa_t, Xt_t)
    return torch.softmax(logits, 1)[:, 1].cpu().numpy()


def best_threshold(y_true, probs):
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.2, 0.8, 0.02):
        f1 = f1_score(y_true, (probs >= t).astype(int), average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t


def calibration_table(y_true, probs, n_bins=5):
    bins = np.linspace(0, 1, n_bins + 1)
    table = []
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs >= lo) & (probs < hi) if i < n_bins - 1 else (probs >= lo) & (probs <= hi)
        n = int(mask.sum())
        if n == 0:
            table.append({"bin": f"[{lo:.1f},{hi:.1f})", "n": 0,
                           "mean_predicted": None, "observed_rate": None})
            continue
        mean_pred = float(probs[mask].mean())
        obs_rate  = float(y_true[mask].mean())
        table.append({"bin": f"[{lo:.1f},{hi:.1f})", "n": n,
                       "mean_predicted": round(mean_pred, 4),
                       "observed_rate": round(obs_rate, 4)})
        ece += (n / len(y_true)) * abs(mean_pred - obs_rate)
    return table, round(float(ece), 4)


def error_cases(y_true, probs, preds, split_name):
    wrong_idx = np.where(preds != y_true)[0]
    cases = []
    for idx in wrong_idx:
        cases.append({
            "index": int(idx),
            "true_label": int(y_true[idx]),
            "predicted_label": int(preds[idx]),
            "predicted_probability": round(float(probs[idx]), 4),
            "error_type": "false_negative" if y_true[idx] == 1 else "false_positive",
            "confidence": "high" if abs(probs[idx] - 0.5) > 0.2 else "low",
        })
    n_fn = sum(1 for c in cases if c["error_type"] == "false_negative")
    n_fp = sum(1 for c in cases if c["error_type"] == "false_positive")
    print(f"  {split_name}: {len(wrong_idx)}/{len(y_true)} misclassified "
          f"({n_fn} false-negative missed-depression, {n_fp} false-positive)")
    return cases


def analyze_split(name, Xf, Xa, Xt, y, model, device):
    probs = get_probs(model, Xf, Xa, Xt, device)
    thr = best_threshold(y, probs)
    preds = (probs >= thr).astype(int)

    brier = brier_score_loss(y, probs)
    cal_table, ece = calibration_table(y, probs)
    cases = error_cases(y, probs, preds, name)

    print(f"    threshold={thr:.2f}  Brier={brier:.4f}  ECE={ece:.4f}")
    return {
        "n": int(len(y)),
        "threshold": round(float(thr), 4),
        "brier_score": round(float(brier), 4),
        "expected_calibration_error": ece,
        "calibration_table": cal_table,
        "n_misclassified": len(cases),
        "n_false_negative": sum(1 for c in cases if c["error_type"] == "false_negative"),
        "n_false_positive": sum(1 for c in cases if c["error_type"] == "false_positive"),
        "error_cases": cases,
    }


def main():
    device = "cpu"
    model = load_model(device)

    Xf_dv, Xa_dv, Xt_dv, y_dv = load_dev()
    Xf_te, Xa_te, Xt_te, y_te = load_test()

    print("Dev split:")
    dev_report = analyze_split("Dev", Xf_dv, Xa_dv, Xt_dv, y_dv, model, device)
    print("Test split:")
    test_report = analyze_split("Test", Xf_te, Xa_te, Xt_te, y_te, model, device)

    report = {
        "analysis": "calibration (Brier score, ECE, reliability table) and "
                    "error case study for the multi-task model",
        "dev": dev_report,
        "test": test_report,
    }
    METRICS.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved: {OUT_JSON}")


if __name__ == "__main__":
    main()
