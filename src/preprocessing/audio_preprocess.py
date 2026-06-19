"""
Phase 3 Preprocessing — DAIC-WOZ Audio (MFCC)
===============================================
DAIC-WOZ এর .wav audio থেকে MFCC feature বের করে।

কণ্ঠের যে pattern depression এ দেখা যায়:
  - কথা ধীর হয়, pause বেশি হয়
  - pitch variation কমে যায়
  - energy কমে যায়
এগুলো MFCC sequence তে ধরা পড়ে।

চালানোর নিয়ম:
    python src/preprocessing/audio_preprocess.py

Output:
    data/processed/daic_audio/X_train.npy  (N, T, n_mfcc)  float32
    data/processed/daic_audio/y_train.npy  (N,)             int64
    data/processed/daic_audio/X_dev.npy
    data/processed/daic_audio/y_dev.npy
    data/processed/daic_audio/meta_train.csv
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
from configs.config import (DATA_RAW, DATA_PROCESSED, N_MFCC,
                             SAMPLE_RATE, AUDIO_WINDOW_SEC, SEED)

PHQ8_THRESHOLD   = 10
DAIC_RAW         = DATA_RAW / "daic"
AUDIO_OUT        = DATA_PROCESSED / "daic_audio"

# fixed sequence length (time steps) — pad/trim to this
MAX_TIME_STEPS   = 300   # ~9 seconds at hop_length=512, sr=16000
HOP_LENGTH       = 512


def load_labels(split_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(split_csv)
    df.columns = df.columns.str.strip()
    id_col    = next(c for c in df.columns if "participant" in c.lower())
    score_col = next(c for c in df.columns if "phq" in c.lower() and "score" in c.lower())
    df = df[[id_col, score_col]].copy()
    df.columns = ["participant_id", "phq8"]
    df["label"] = (df["phq8"] >= PHQ8_THRESHOLD).astype(int)
    return df


def extract_mfcc(audio_path: Path) -> np.ndarray:
    """
    .wav file থেকে MFCC sequence বের করে।
    Output shape: (MAX_TIME_STEPS, N_MFCC)  — pad বা trim করা হয়।
    """
    try:
        import librosa
    except ImportError:
        raise ImportError("librosa install koro: pip install librosa")

    y, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)

    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=N_MFCC,
        hop_length=HOP_LENGTH,
        n_fft=1024,
    )  # shape: (N_MFCC, T)

    # delta features যোগ করি — speech rate এর জন্য গুরুত্বপূর্ণ
    delta  = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    feat   = np.concatenate([mfcc, delta, delta2], axis=0)  # (N_MFCC*3, T)

    feat = feat.T  # (T, N_MFCC*3)

    # normalize per feature
    mean = feat.mean(axis=0, keepdims=True)
    std  = feat.std(axis=0, keepdims=True) + 1e-8
    feat = (feat - mean) / std

    # pad or trim to MAX_TIME_STEPS
    T = feat.shape[0]
    if T >= MAX_TIME_STEPS:
        feat = feat[:MAX_TIME_STEPS]
    else:
        pad = np.zeros((MAX_TIME_STEPS - T, feat.shape[1]), dtype=np.float32)
        feat = np.concatenate([feat, pad], axis=0)

    return feat.astype(np.float32)


def process_split(label_df: pd.DataFrame, split_name: str):
    all_X, all_y = [], []
    meta_rows = []

    for _, row in label_df.iterrows():
        pid   = int(row["participant_id"])
        label = int(row["label"])
        phq8  = float(row["phq8"])

        audio_dir   = DAIC_RAW / f"{pid}_P"
        audio_files = list(audio_dir.glob(f"{pid}_AUDIO.wav")) if audio_dir.exists() else []
        if not audio_files:
            # some DAIC versions use _P.wav
            audio_files = list(audio_dir.glob(f"{pid}_P.wav")) if audio_dir.exists() else []

        if not audio_files:
            print(f"  [SKIP] {pid}_P — audio not found")
            meta_rows.append({"participant_id": pid, "phq8": phq8,
                               "label": label, "status": "missing"})
            continue

        feat = extract_mfcc(audio_files[0])
        all_X.append(feat)
        all_y.append(label)
        meta_rows.append({"participant_id": pid, "phq8": phq8,
                           "label": label, "status": "ok"})
        print(f"  {pid}_P — shape {feat.shape}  label={label}  PHQ-8={phq8:.0f}")

    if not all_X:
        return

    X = np.stack(all_X)           # (N, MAX_TIME_STEPS, N_MFCC*3)
    y = np.array(all_y, dtype=np.int64)

    AUDIO_OUT.mkdir(parents=True, exist_ok=True)
    np.save(AUDIO_OUT / f"X_{split_name}.npy", X)
    np.save(AUDIO_OUT / f"y_{split_name}.npy", y)
    pd.DataFrame(meta_rows).to_csv(AUDIO_OUT / f"meta_{split_name}.csv", index=False)

    dep = (y == 1).sum()
    print(f"\n  [{split_name}] {X.shape[0]} samples | "
          f"depressed: {dep}  not-depressed: {len(y)-dep}")
    print(f"  Feature shape per sample: {X.shape[1:]}")


def preprocess():
    if not DAIC_RAW.exists():
        print(f"DAIC-WOZ data না পাওয়া গেছে: {DAIC_RAW}")
        print("Access: https://dcapswoz.isi.edu/")
        print("\nSynthetic fallback চালাচ্ছি...")
        _synthetic_fallback()
        return

    for split, csv_name in [("train", "train_split_Depression_AVEC2017.csv"),
                              ("dev",   "dev_split_Depression_AVEC2017.csv")]:
        csv_path = DAIC_RAW / csv_name
        if not csv_path.exists():
            print(f"[SKIP] {csv_path} not found")
            continue
        print(f"\nProcessing {split} split...")
        labels = load_labels(csv_path)
        process_split(labels, split)

    print("\nDone. Output at:", AUDIO_OUT)


def _synthetic_fallback():
    """DAIC-WOZ audio না থাকলে synthetic MFCC দিয়ে pipeline verify করে।"""
    rng    = np.random.default_rng(SEED)
    n_feat = N_MFCC * 3   # MFCC + delta + delta2
    n      = 200

    X = rng.standard_normal((n, MAX_TIME_STEPS, n_feat)).astype(np.float32)
    y = np.array([0] * (n // 2) + [1] * (n // 2), dtype=np.int64)
    idx = rng.permutation(n)
    X, y = X[idx], y[idx]

    AUDIO_OUT.mkdir(parents=True, exist_ok=True)
    np.save(AUDIO_OUT / "X_train.npy", X[:160])
    np.save(AUDIO_OUT / "y_train.npy", y[:160])
    np.save(AUDIO_OUT / "X_dev.npy",   X[160:])
    np.save(AUDIO_OUT / "y_dev.npy",   y[160:])
    print(f"Synthetic MFCC saved — shape per sample: ({MAX_TIME_STEPS}, {n_feat})")


if __name__ == "__main__":
    preprocess()
