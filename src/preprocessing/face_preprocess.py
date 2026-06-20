"""
Phase 1 Preprocessing — FER2013
================================
FER2013 CSV থেকে image array বানায়, augmentation দেয়, আর
data/processed/ এ .npy হিসেবে save করে।

চালানোর নিয়ম:
    python src/preprocessing/face_preprocess.py

Output:
    data/processed/X_train.npy   (N, 1, 48, 48)  float32  0-1 range
    data/processed/y_train.npy   (N,)             int64
    data/processed/X_test.npy
    data/processed/y_test.npy
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
from configs.config import DATA_RAW, DATA_PROCESSED, IMG_SIZE, NUM_EMOTION_CLASSES, SEED

# FER2013 emotion label map (for reference)
EMOTION_LABELS = {
    0: "Angry", 1: "Disgust", 2: "Fear",
    3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"
}


def _pixel_str_to_array(pixel_str: str) -> np.ndarray:
    """'70 80 90 ...' -> (1, 48, 48) float32 in [0, 1]"""
    arr = np.array(pixel_str.split(), dtype=np.float32)
    return arr.reshape(1, IMG_SIZE, IMG_SIZE) / 255.0


def augment(images: np.ndarray) -> np.ndarray:
    """
    Simple augmentation on (N, 1, H, W) arrays.
    Returns augmented copies stacked with the originals.

    Augmentations:
      - Horizontal flip
      - Brightness jitter ±15%
    """
    rng = np.random.default_rng(SEED)

    flipped = images[:, :, :, ::-1].copy()

    brightness = rng.uniform(0.85, 1.15, size=(len(images), 1, 1, 1)).astype(np.float32)
    brightened = np.clip(images * brightness, 0.0, 1.0)

    return np.concatenate([images, flipped, brightened], axis=0)


def load_and_split(csv_path: Path):
    """
    FER2013 CSV load করে train/test split করে।
    CSV কলাম: emotion, pixels, Usage  (standard FER2013 format)
    Usage column না থাকলে 80/20 split করে।
    """
    print(f"Loading: {csv_path}")
    df = pd.read_csv(csv_path)

    X = np.stack(df["pixels"].apply(_pixel_str_to_array).values)  # (N,1,48,48)
    y = df["emotion"].values.astype(np.int64)

    if "Usage" in df.columns:
        train_mask = df["Usage"].str.strip() == "Training"
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[~train_mask], y[~train_mask]
    else:
        n = len(X)
        split = int(n * 0.8)
        X_train, y_train = X[:split], y[:split]
        X_test, y_test = X[split:], y[split:]

    return X_train, y_train, X_test, y_test


def _synthetic_fallback():
    """FER2013 না থাকলে synthetic data দিয়ে pipeline verify করে।"""
    rng = np.random.default_rng(SEED)
    n = 500
    X = rng.random((n, 1, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    y = rng.integers(0, NUM_EMOTION_CLASSES, size=n, dtype=np.int64)

    split = int(n * 0.8)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    np.save(DATA_PROCESSED / "X_train.npy", X[:split])
    np.save(DATA_PROCESSED / "y_train.npy", y[:split])
    np.save(DATA_PROCESSED / "X_test.npy",  X[split:])
    np.save(DATA_PROCESSED / "y_test.npy",  y[split:])
    print(f"Synthetic fallback saved — train: {X[:split].shape}  test: {X[split:].shape}")


def preprocess(apply_augmentation: bool = True):
    csv_path = DATA_RAW / "fer2013.csv"
    if not csv_path.exists():
        print(f"FER2013 not found: {csv_path}")
        print("Download from: https://www.kaggle.com/datasets/msambare/fer2013")
        print("Place fer2013.csv in data/raw/")
        print("\nRunning synthetic fallback to verify pipeline...")
        _synthetic_fallback()
        return

    X_train, y_train, X_test, y_test = load_and_split(csv_path)
    print(f"Raw  — train: {X_train.shape}, test: {X_test.shape}")

    if apply_augmentation:
        X_train_aug = augment(X_train)
        y_train_aug = np.tile(y_train, 3)
        idx = np.random.default_rng(SEED).permutation(len(X_train_aug))
        X_train, y_train = X_train_aug[idx], y_train_aug[idx]
        print(f"After augment — train: {X_train.shape}")

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    np.save(DATA_PROCESSED / "X_train.npy", X_train)
    np.save(DATA_PROCESSED / "y_train.npy", y_train)
    np.save(DATA_PROCESSED / "X_test.npy", X_test)
    np.save(DATA_PROCESSED / "y_test.npy", y_test)

    print(f"\nSaved to {DATA_PROCESSED}/")
    print(f"  X_train: {X_train.shape}  y_train: {y_train.shape}")
    print(f"  X_test:  {X_test.shape}   y_test:  {y_test.shape}")

    class_counts = np.bincount(y_train, minlength=NUM_EMOTION_CLASSES)
    print("\nClass distribution (train):")
    for i, label in EMOTION_LABELS.items():
        print(f"  {label:10s}: {class_counts[i]:6d}")


if __name__ == "__main__":
    preprocess(apply_augmentation=True)
