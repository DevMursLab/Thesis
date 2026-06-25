"""
Phase 2 Preprocessing — DAIC-WOZ Face (CLNF Action Units)
==========================================================
DAIC-WOZ এর pre-extracted CLNF Action Unit features থেকে
face time-series বানায়।

Action Units (AU) হলো মুখের muscle movement এর measure।
AU04 = brow lowerer (চিন্তিত হলে হয়), AU15 = lip corner depressor (দুঃখে হয়)
এগুলো depression এর সাথে strongly correlated।

Run:
    python src/preprocessing/daic_preprocess.py

Output:
    data/processed/daic_faces/X_train.npy  (N, AU_TIME_STEPS, N_AU_FEATURES)
    data/processed/daic_faces/y_train.npy  (N,) int64
    data/processed/daic_faces/X_dev.npy
    data/processed/daic_faces/y_dev.npy
    data/processed/daic_faces/meta_train.csv
    data/processed/daic_faces/meta_dev.csv
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
from configs.config import (DAIC_RAW, DAIC_LABELS, DATA_PROCESSED,
                             AU_COLS, AU_TIME_STEPS, SEED)

FACE_OUT = DATA_PROCESSED / "daic_faces"


def load_labels(split: str) -> pd.DataFrame:
    csv_path = DAIC_LABELS / f"{split}_split_Depression_AVEC2017.csv"
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    return df[["Participant_ID", "PHQ8_Binary", "PHQ8_Score", "Gender"]].copy()


def load_au_features(participant_id: int) -> np.ndarray | None:
    """
    CLNF_AUs.txt পড়ে AU time-series বানায়।
    Output: (AU_TIME_STEPS, N_AU_FEATURES) float32
    confidence < 0.5 হলে সেই frame বাদ দেওয়া হয়।
    """
    au_file = DAIC_RAW / f"{participant_id}_CLNF_AUs.txt"
    if not au_file.exists():
        return None

    df = pd.read_csv(au_file, sep=",", skipinitialspace=True)
    df.columns = df.columns.str.strip()

    # low confidence frames বাদ
    df = df[df["confidence"] >= 0.5].reset_index(drop=True)
    if len(df) < 10:
        return None

    # শুধু AU columns নাও
    missing = [c for c in AU_COLS if c not in df.columns]
    if missing:
        return None

    feat = df[AU_COLS].values.astype(np.float32)

    # normalize
    mean = feat.mean(axis=0, keepdims=True)
    std  = feat.std(axis=0, keepdims=True) + 1e-8
    feat = (feat - mean) / std

    # pad or trim to AU_TIME_STEPS
    T = feat.shape[0]
    if T >= AU_TIME_STEPS:
        # evenly sample AU_TIME_STEPS frames
        idx  = np.linspace(0, T - 1, AU_TIME_STEPS, dtype=int)
        feat = feat[idx]
    else:
        pad  = np.zeros((AU_TIME_STEPS - T, feat.shape[1]), dtype=np.float32)
        feat = np.concatenate([feat, pad], axis=0)

    return feat


def process_split(label_df: pd.DataFrame, split_name: str):
    all_X, all_y, all_gender = [], [], []
    meta_rows = []

    for _, row in label_df.iterrows():
        pid    = int(row["Participant_ID"])
        label  = int(row["PHQ8_Binary"])
        score  = float(row["PHQ8_Score"])
        gender = int(row["Gender"])   # 0=male, 1=female

        feat = load_au_features(pid)

        if feat is None:
            print(f"  [SKIP] {pid} — AU file missing or low quality")
            meta_rows.append({"pid": pid, "phq8": score,
                               "label": label, "gender": gender, "status": "skip"})
            continue

        all_X.append(feat)
        all_y.append(label)
        all_gender.append(gender)
        meta_rows.append({"pid": pid, "phq8": score,
                           "label": label, "gender": gender, "status": "ok"})
        print(f"  {pid} — AU shape {feat.shape}  label={label}  PHQ8={score:.0f}  gender={gender}")

    if not all_X:
        print(f"[{split_name}] No data processed!")
        return

    X = np.stack(all_X)
    y = np.array(all_y, dtype=np.int64)

    FACE_OUT.mkdir(parents=True, exist_ok=True)
    np.save(FACE_OUT / f"X_{split_name}.npy", X)
    np.save(FACE_OUT / f"y_{split_name}.npy", y)
    np.save(FACE_OUT / f"gender_{split_name}.npy", np.array(all_gender, dtype=np.int64))
    pd.DataFrame(meta_rows).to_csv(FACE_OUT / f"meta_{split_name}.csv", index=False)

    dep = (y == 1).sum()
    print(f"\n  [{split_name}] {len(y)} participants | "
          f"depressed: {dep} ({100*dep/len(y):.1f}%)  "
          f"not-depressed: {len(y)-dep}")
    print(f"  Feature shape: {X.shape}  (participants, time_steps, AU_features)")


def preprocess():
    if not DAIC_RAW.exists():
        print(f"DAIC-WOZ not found: {DAIC_RAW}")
        return

    print(f"DAIC-WOZ found: {DAIC_RAW}")

    for split in ["train", "dev"]:
        csv_path = DAIC_LABELS / f"{split}_split_Depression_AVEC2017.csv"
        if not csv_path.exists():
            print(f"[SKIP] Label file not found: {csv_path}")
            continue
        print(f"\nProcessing {split} split...")
        labels = load_labels(split)
        print(f"  Participants: {len(labels)} | "
              f"Depressed: {labels['PHQ8_Binary'].sum()} | "
              f"Not depressed: {(labels['PHQ8_Binary']==0).sum()}")
        process_split(labels, split)

    print(f"\nDone. Saved to: {FACE_OUT}")


if __name__ == "__main__":
    preprocess()
