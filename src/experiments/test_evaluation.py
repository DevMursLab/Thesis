"""
AVEC 2017 Test Set Evaluation  (N=47)
======================================
full_test_split.csv-এর labels ব্যবহার করে Phase 8 model-কে
official test split-এ evaluate করে।

Run:
    python -m src.experiments.test_evaluation
"""

import sys, json, pickle, warnings
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

from configs.config import (
    DAIC_RAW, DATA_PROCESSED,
    AU_COLS, AU_TIME_STEPS,
    N_MFCC, N_COVAREP, N_FORMANT, AUDIO_FEAT_DIM, AUDIO_TIME_STEPS,
    VOCAB_SIZE, EMBED_DIM, FUSION_DROPOUT,
    N_PHQ_SYMPTOMS,
)
from src.models.multitask_fusion import MultiTaskFusionModel
from src.preprocessing.covarep_preprocess import (
    extract_mfcc_stream, extract_covarep_stream, extract_formant_stream,
)

ROOT          = Path(__file__).resolve().parent.parent.parent
TEST_CSV      = ROOT / "full_test_split.csv"
DEV_FACE      = DATA_PROCESSED / "daic_faces"
DEV_AUDIO     = DATA_PROCESSED / "daic_audio_covarep"
DEV_TEXT      = DATA_PROCESSED / "daic_text"
VECTORIZER    = DEV_TEXT / "vectorizer.pkl"
CHECKPOINT    = ROOT / "results" / "multitask_best.pth"
OUT_JSON      = ROOT / "results" / "metrics" / "test_evaluation.json"


# ── Feature extraction ──────────────────────────────────────────────────────

def extract_au(pid: int) -> np.ndarray | None:
    au_file = DAIC_RAW / f"{pid}_CLNF_AUs.txt"
    if not au_file.exists():
        return None
    df = pd.read_csv(au_file, sep=",", skipinitialspace=True)
    df.columns = df.columns.str.strip()
    df = df[df["confidence"] >= 0.5].reset_index(drop=True)
    if len(df) < 10 or any(c not in df.columns for c in AU_COLS):
        return None
    feat = df[AU_COLS].values.astype(np.float32)
    mean = feat.mean(axis=0, keepdims=True)
    std  = feat.std(axis=0, keepdims=True) + 1e-8
    feat = (feat - mean) / std
    T = feat.shape[0]
    if T >= AU_TIME_STEPS:
        idx  = np.linspace(0, T - 1, AU_TIME_STEPS, dtype=int)
        feat = feat[idx]
    else:
        pad  = np.zeros((AU_TIME_STEPS - T, feat.shape[1]), dtype=np.float32)
        feat = np.concatenate([feat, pad], axis=0)
    return feat


def extract_audio(pid: int) -> np.ndarray:
    audio_path = DAIC_RAW / f"{pid}_AUDIO.wav"
    mfcc = extract_mfcc_stream(audio_path) if audio_path.exists() \
           else np.zeros((AUDIO_TIME_STEPS, N_MFCC * 3), dtype=np.float32)
    cov  = extract_covarep_stream(pid)
    fmt  = extract_formant_stream(pid)
    return np.concatenate([mfcc, cov, fmt], axis=1)  # (300, 199)


def extract_text(pid: int, vectorizer) -> np.ndarray:
    transcript = DAIC_RAW / f"{pid}_TRANSCRIPT.csv"
    if not transcript.exists():
        return np.zeros(VOCAB_SIZE, dtype=np.float32)
    df = pd.read_csv(transcript, sep="\t")
    df.columns = df.columns.str.strip()
    rows = df[df["speaker"].str.strip() == "Participant"]
    if len(rows) == 0:
        return np.zeros(VOCAB_SIZE, dtype=np.float32)
    text = " ".join(rows["value"].dropna().astype(str).tolist()).lower().strip()
    if len(text) < 10:
        return np.zeros(VOCAB_SIZE, dtype=np.float32)
    return vectorizer.transform([text]).toarray().astype(np.float32)[0]


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  AVEC 2017 Test Set Evaluation  (N=47)")
    print("=" * 60)

    # Load test labels
    df_test = pd.read_csv(TEST_CSV)
    df_test.columns = df_test.columns.str.strip()
    # Normalize column names (full_test_split uses PHQ_Binary vs PHQ8_Binary)
    df_test = df_test.rename(columns={
        "PHQ_Binary": "label",
        "PHQ8_Binary": "label",
        "PHQ_Score": "phq_score",
        "PHQ8_Score": "phq_score",
    })
    print(f"Test participants: {len(df_test)}")
    print(f"Depressed: {(df_test['label']==1).sum()}  |  Not: {(df_test['label']==0).sum()}")

    # Load TF-IDF vectorizer (fit on training data)
    with open(VECTORIZER, "rb") as f:
        vectorizer = pickle.load(f)

    # Build test features
    X_face_list, X_audio_list, X_text_list = [], [], []
    labels, genders, pids_kept = [], [], []

    for _, row in df_test.iterrows():
        pid    = int(row["Participant_ID"])
        label  = int(row["label"])
        gender = int(row["Gender"])

        au = extract_au(pid)
        if au is None:
            print(f"  [SKIP] {pid} — AU file missing/low quality")
            continue

        audio = extract_audio(pid)
        text  = extract_text(pid, vectorizer)

        X_face_list.append(au)
        X_audio_list.append(audio)
        X_text_list.append(text)
        labels.append(label)
        genders.append(gender)
        pids_kept.append(pid)
        print(f"  {pid}  label={label}  gender={'F' if gender else 'M'}")

    N = len(labels)
    print(f"\nProcessed: {N}/{len(df_test)} participants")

    X_face  = torch.tensor(np.stack(X_face_list),  dtype=torch.float32)
    X_audio = torch.tensor(np.stack(X_audio_list), dtype=torch.float32)
    X_text  = torch.tensor(np.stack(X_text_list),  dtype=torch.float32)
    y       = np.array(labels, dtype=np.int64)
    gender  = np.array(genders, dtype=np.int64)

    # Load model
    model = MultiTaskFusionModel(
        n_au=len(AU_COLS),
        n_audio_feat=AUDIO_FEAT_DIM,
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        dropout=FUSION_DROPOUT,
        n_symptoms=N_PHQ_SYMPTOMS,
    )
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model.eval()
    print(f"\nModel loaded: {CHECKPOINT.name}")

    # Inference
    with torch.no_grad():
        logits, _, _ = model(X_face, X_audio, X_text)
        probs = F.softmax(logits, dim=1)[:, 1].numpy()

    # Find best threshold on dev set (same approach as training)
    dev_probs, dev_y = _load_dev_probs(model)
    tau = _best_threshold(dev_probs, dev_y)
    print(f"Best threshold (from dev): {tau:.3f}")

    preds = (probs >= tau).astype(int)

    # Metrics
    f1  = f1_score(y, preds, average="macro", zero_division=0)
    auc = roc_auc_score(y, probs)
    acc = accuracy_score(y, preds)

    # Gender fairness
    male_mask   = gender == 0
    female_mask = gender == 1
    tpr_m = _tpr(y[male_mask],   preds[male_mask])
    tpr_f = _tpr(y[female_mask], preds[female_mask])
    tpr_gap = abs(tpr_m - tpr_f)

    print(f"\n{'='*60}")
    print(f"  TEST SET RESULTS  (N={N})")
    print(f"{'='*60}")
    print(f"  Macro-F1 : {f1:.4f}")
    print(f"  AUC-ROC  : {auc:.4f}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  TPR Male : {tpr_m:.4f}  ({male_mask.sum()} participants)")
    print(f"  TPR Fem  : {tpr_f:.4f}  ({female_mask.sum()} participants)")
    print(f"  TPR Gap  : {tpr_gap:.4f}")
    print(f"{'='*60}")

    # Compare with dev
    print(f"\n  Dev (Phase 8, 5-seed mean) : F1=0.629±0.021, AUC=0.658±0.029")
    print(f"  Test (this run)            : F1={f1:.3f},       AUC={auc:.3f}")

    results = {
        "n_test": N,
        "n_depressed": int((y == 1).sum()),
        "threshold": float(tau),
        "macro_f1": float(f1),
        "auc_roc": float(auc),
        "accuracy": float(acc),
        "tpr_male": float(tpr_m),
        "tpr_female": float(tpr_f),
        "tpr_gap": float(tpr_gap),
        "n_male": int(male_mask.sum()),
        "n_female": int(female_mask.sum()),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {OUT_JSON}")
    return results


def _load_dev_probs(model):
    X_face  = torch.tensor(np.load(DEV_FACE  / "X_dev.npy"),  dtype=torch.float32)
    X_audio = torch.tensor(np.load(DEV_AUDIO / "X_dev.npy"),  dtype=torch.float32)
    X_text  = torch.tensor(np.load(DEV_TEXT  / "X_dev.npy"),  dtype=torch.float32)
    y       = np.load(DEV_FACE / "y_dev.npy")
    n = min(len(X_face), len(X_audio), len(X_text))
    model.eval()
    with torch.no_grad():
        logits, _, _ = model(X_face[:n], X_audio[:n], X_text[:n])
        probs = F.softmax(logits, dim=1)[:, 1].numpy()
    return probs, y[:n]


def _best_threshold(probs, y):
    best_tau, best_f1 = 0.5, 0.0
    for tau in np.arange(0.2, 0.8, 0.01):
        preds = (probs >= tau).astype(int)
        f = f1_score(y, preds, average="macro", zero_division=0)
        if f > best_f1:
            best_f1, best_tau = f, tau
    return best_tau


def _tpr(y_true, y_pred):
    dep = (y_true == 1)
    if dep.sum() == 0:
        return float("nan")
    return float((y_pred[dep] == 1).sum() / dep.sum())


if __name__ == "__main__":
    main()
