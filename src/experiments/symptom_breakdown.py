"""
Symptom-Level Breakdown Analysis
=================================
The multi-task model predicts 8 individual PHQ-8 symptom items in addition
to the binary label. This script checks WHICH symptoms the model actually
learns well vs. poorly -- direct evidence for or against the ICMI-2025
reproducibility critique that DAIC-WOZ classifiers learn only disorder-
general cues (flat affect, low energy) rather than disorder-specific
structure (e.g. sleep, appetite, suicidal ideation-adjacent items).

If per-symptom accuracy is roughly uniform across all 8 items, that is
consistent with the model learning one generic "distress" signal. If
accuracy varies meaningfully across items, that is evidence the model
captures symptom-specific structure.

Run:
    python -m src.experiments.symptom_breakdown
"""

import sys, json
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score

from configs.config import DATA_PROCESSED, RESULTS, METRICS, N_PHQ_SYMPTOMS
from src.models.multitask_fusion import MultiTaskFusionModel

FACE_DIR  = DATA_PROCESSED / "daic_faces"
AUDIO_DIR = DATA_PROCESSED / "daic_audio_covarep"
TEXT_DIR  = DATA_PROCESSED / "daic_text"
CKPT      = RESULTS / "multitask_best.pth"
OUT_JSON  = METRICS / "symptom_breakdown.json"

PHQ_SYMPTOM_NAMES = [
    "No Interest/Pleasure", "Feeling Down/Depressed", "Sleep Problems",
    "Tired/Low Energy", "Appetite Problems", "Feeling Bad About Self",
    "Concentration Problems", "Psychomotor Change",
]


def load_dev():
    Xf = np.load(FACE_DIR / "X_dev.npy").astype(np.float32)
    Xa = np.load(AUDIO_DIR / "X_dev.npy").astype(np.float32)
    Xt = np.load(TEXT_DIR / "X_dev.npy").astype(np.float32)
    sym_raw = np.load(AUDIO_DIR / "symptoms_dev.npy").astype(np.float32)
    symptoms = np.nan_to_num(sym_raw, nan=0.0).clip(0, 3).astype(np.int64)
    n = min(len(Xf), len(Xa), len(Xt), len(symptoms))
    return Xf[:n], Xa[:n], Xt[:n], symptoms[:n]


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
def predict_symptoms(model, Xf, Xa, Xt, device):
    Xf_t = torch.tensor(Xf).to(device)
    Xa_t = torch.tensor(Xa).to(device)
    Xt_t = torch.tensor(Xt).to(device)
    _, _, symptom_logits = model(Xf_t, Xa_t, Xt_t)
    preds = symptom_logits.argmax(dim=-1).cpu().numpy()  # (N, 8)
    return preds


def main():
    device = "cpu"
    Xf, Xa, Xt, y_sym = load_dev()
    print(f"Dev set: N={len(Xf)}  symptoms shape={y_sym.shape}")

    model = load_model(device)
    preds = predict_symptoms(model, Xf, Xa, Xt, device)

    per_symptom = {}
    for i, name in enumerate(PHQ_SYMPTOM_NAMES):
        true_i = y_sym[:, i]
        pred_i = preds[:, i]
        acc = accuracy_score(true_i, pred_i)
        f1m = f1_score(true_i, pred_i, average="macro",
                        labels=[0, 1, 2, 3], zero_division=0)
        # exact-severity accuracy vs "within +-1 level" (clinically near-miss)
        near = (np.abs(true_i - pred_i) <= 1).mean()
        # binary-presence agreement (0 = absent, 1-3 = present) -- coarser,
        # clinically relevant framing (symptom present vs absent)
        true_bin = (true_i > 0).astype(int)
        pred_bin = (pred_i > 0).astype(int)
        bin_acc = accuracy_score(true_bin, pred_bin)
        bin_f1  = f1_score(true_bin, pred_bin, average="macro", zero_division=0)
        per_symptom[name] = {
            "exact_accuracy": round(float(acc), 4),
            "macro_f1_4class": round(float(f1m), 4),
            "within_1_level_accuracy": round(float(near), 4),
            "presence_absence_accuracy": round(float(bin_acc), 4),
            "presence_absence_macro_f1": round(float(bin_f1), 4),
            "true_class_distribution": {
                str(c): int((true_i == c).sum()) for c in range(4)
            },
        }

    exact_accs = [v["exact_accuracy"] for v in per_symptom.values()]
    bin_f1s    = [v["presence_absence_macro_f1"] for v in per_symptom.values()]

    print("\n" + "=" * 78)
    print(f"  {'Symptom':28s} {'Exact Acc':>10s} {'±1-lvl Acc':>11s} "
          f"{'Presence F1':>12s}")
    print("=" * 78)
    for name, v in per_symptom.items():
        print(f"  {name:28s} {v['exact_accuracy']:10.3f} "
              f"{v['within_1_level_accuracy']:11.3f} "
              f"{v['presence_absence_macro_f1']:12.3f}")
    print("=" * 78)
    print(f"  Range across symptoms (presence-absence macro-F1): "
          f"{min(bin_f1s):.3f} -- {max(bin_f1s):.3f}  "
          f"(spread={max(bin_f1s)-min(bin_f1s):.3f})")
    print(f"  Range across symptoms (exact accuracy): "
          f"{min(exact_accs):.3f} -- {max(exact_accs):.3f}  "
          f"(spread={max(exact_accs)-min(exact_accs):.3f})")

    report = {
        "analysis": "per-PHQ8-symptom breakdown of multi-task model, dev split",
        "n_dev": int(len(Xf)),
        "per_symptom": per_symptom,
        "summary": {
            "presence_f1_min": round(float(min(bin_f1s)), 4),
            "presence_f1_max": round(float(max(bin_f1s)), 4),
            "presence_f1_spread": round(float(max(bin_f1s) - min(bin_f1s)), 4),
            "exact_acc_min": round(float(min(exact_accs)), 4),
            "exact_acc_max": round(float(max(exact_accs)), 4),
            "exact_acc_spread": round(float(max(exact_accs) - min(exact_accs)), 4),
        },
    }
    METRICS.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved: {OUT_JSON}")


if __name__ == "__main__":
    main()
