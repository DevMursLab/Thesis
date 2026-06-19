"""
Phase 2 Preprocessing — DAIC-WOZ
==================================
DAIC-WOZ dataset থেকে:
  1. PHQ-8 score পড়ে binary depression label বানায় (score >= 10 → depressed)
  2. প্রতিটা participant এর video থেকে face frame extract করে 48×48 grayscale এ resize করে
  3. data/processed/daic_faces/ এ .npy হিসেবে save করে

DAIC-WOZ access: https://dcapswoz.isi.edu/  (request required)

Expected folder layout after download:
    data/raw/daic/
    ├── train_split_Depression_AVEC2017.csv   ← labels
    ├── dev_split_Depression_AVEC2017.csv
    ├── 300_P/
    │   └── 300_P.mp4                         ← video
    ├── 301_P/
    │   └── 301_P.mp4
    └── ...

চালানোর নিয়ম:
    python src/preprocessing/daic_preprocess.py

Output:
    data/processed/daic_faces/X_train.npy   (N, 1, 48, 48)  float32
    data/processed/daic_faces/y_train.npy   (N,)             int64  {0=not depressed, 1=depressed}
    data/processed/daic_faces/X_dev.npy
    data/processed/daic_faces/y_dev.npy
    data/processed/daic_faces/meta_train.csv  (participant_id, phq8, label, n_frames)
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
from configs.config import DATA_RAW, DATA_PROCESSED, IMG_SIZE, SEED

# PHQ-8 cutoff for depression (standard clinical threshold)
PHQ8_THRESHOLD = 10

# Frames to sample per video (evenly spaced — captures full interview arc)
FRAMES_PER_VIDEO = 30

DAIC_RAW = DATA_RAW / "daic"
DAIC_OUT = DATA_PROCESSED / "daic_faces"


def load_labels(split_csv: Path) -> pd.DataFrame:
    """
    PHQ-8 label CSV পড়ে।
    Expected columns: Participant_ID, PHQ8_Score  (AVEC2017 format)
    Returns DataFrame with columns: participant_id, phq8, label (0/1)
    """
    df = pd.read_csv(split_csv)
    df.columns = df.columns.str.strip()

    # column names vary slightly between AVEC versions
    id_col    = next(c for c in df.columns if "participant" in c.lower())
    score_col = next(c for c in df.columns if "phq" in c.lower() and "score" in c.lower())

    df = df[[id_col, score_col]].copy()
    df.columns = ["participant_id", "phq8"]
    df["label"] = (df["phq8"] >= PHQ8_THRESHOLD).astype(int)
    return df


def extract_faces_from_video(video_path: Path, n_frames: int = FRAMES_PER_VIDEO):
    """
    Video থেকে evenly-spaced frame নিয়ে face detect করে (48×48 grayscale)।
    OpenCV Haar cascade দিয়ে face crop করা হয়।
    Face না পেলে পুরো frame resize করা হয় (fallback)।

    Returns: np.ndarray  (n_valid_frames, 1, 48, 48)  float32  [0,1]
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("opencv-python install koro: pip install opencv-python")

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames < 1:
        cap.release()
        return np.empty((0, 1, IMG_SIZE, IMG_SIZE), dtype=np.float32)

    # evenly spaced frame indices
    frame_indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)

    faces = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                                  minSize=(30, 30))

        if len(detected) > 0:
            # largest face (가장 큰 얼굴)
            x, y, w, h = max(detected, key=lambda r: r[2] * r[3])
            crop = gray[y:y+h, x:x+w]
        else:
            crop = gray  # fallback: whole frame

        resized = cv2.resize(crop, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
        faces.append(resized)

    cap.release()

    if not faces:
        return np.empty((0, 1, IMG_SIZE, IMG_SIZE), dtype=np.float32)

    arr = np.stack(faces)[:, np.newaxis, :, :]  # (N, 1, 48, 48)
    return arr


def process_split(label_df: pd.DataFrame, split_name: str):
    """
    label_df এর প্রতিটা participant এর video process করে
    X (frames) ও y (labels) বানায়।
    """
    all_X, all_y = [], []
    meta_rows = []

    for _, row in label_df.iterrows():
        pid = int(row["participant_id"])
        label = int(row["label"])
        phq8 = float(row["phq8"])

        video_dir = DAIC_RAW / f"{pid}_P"
        video_files = list(video_dir.glob(f"{pid}_P.mp4")) if video_dir.exists() else []

        if not video_files:
            print(f"  [SKIP] {pid}_P — video not found")
            meta_rows.append({"participant_id": pid, "phq8": phq8,
                               "label": label, "n_frames": 0})
            continue

        frames = extract_faces_from_video(video_files[0])
        n = len(frames)

        if n == 0:
            print(f"  [SKIP] {pid}_P — no frames extracted")
            meta_rows.append({"participant_id": pid, "phq8": phq8,
                               "label": label, "n_frames": 0})
            continue

        all_X.append(frames)
        all_y.extend([label] * n)
        meta_rows.append({"participant_id": pid, "phq8": phq8,
                           "label": label, "n_frames": n})
        print(f"  {pid}_P — {n} frames  label={label}  PHQ-8={phq8:.0f}")

    if not all_X:
        return np.empty((0, 1, IMG_SIZE, IMG_SIZE), dtype=np.float32), np.array([], dtype=np.int64)

    X = np.concatenate(all_X, axis=0)
    y = np.array(all_y, dtype=np.int64)

    DAIC_OUT.mkdir(parents=True, exist_ok=True)
    np.save(DAIC_OUT / f"X_{split_name}.npy", X)
    np.save(DAIC_OUT / f"y_{split_name}.npy", y)
    pd.DataFrame(meta_rows).to_csv(DAIC_OUT / f"meta_{split_name}.csv", index=False)

    dep = (y == 1).sum()
    print(f"\n  [{split_name}] saved — {X.shape[0]} frames | "
          f"depressed: {dep} ({100*dep/len(y):.1f}%)  "
          f"not-depressed: {len(y)-dep}")
    return X, y


def preprocess():
    if not DAIC_RAW.exists():
        print(f"DAIC-WOZ data না পাওয়া গেছে: {DAIC_RAW}")
        print("\nAccess request: https://dcapswoz.isi.edu/")
        print("Data পেলে data/raw/daic/ এ রাখো।")
        print("\nডেমো হিসেবে synthetic fallback চালাচ্ছি...")
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
        print(f"  Participants: {len(labels)}  |  "
              f"Depressed: {labels['label'].sum()}  |  "
              f"Not depressed: {(labels['label']==0).sum()}")
        process_split(labels, split)

    print("\nDone. Output at:", DAIC_OUT)


def _synthetic_fallback():
    """
    DAIC-WOZ না থাকলে synthetic data দিয়ে pipeline verify করে।
    label=0 (not depressed) ও label=1 (depressed) সমানভাবে বানানো হয়।
    """
    rng = np.random.default_rng(SEED)
    n_per_class = 100
    n = n_per_class * 2

    X = rng.random((n, 1, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    y = np.array([0]*n_per_class + [1]*n_per_class, dtype=np.int64)
    idx = rng.permutation(n)
    X, y = X[idx], y[idx]

    DAIC_OUT.mkdir(parents=True, exist_ok=True)
    np.save(DAIC_OUT / "X_train.npy", X[:160])
    np.save(DAIC_OUT / "y_train.npy", y[:160])
    np.save(DAIC_OUT / "X_dev.npy",   X[160:])
    np.save(DAIC_OUT / "y_dev.npy",   y[160:])
    print(f"Synthetic fallback saved to {DAIC_OUT}")
    print(f"  Train: {X[:160].shape}  Dev: {X[160:].shape}")


if __name__ == "__main__":
    preprocess()
