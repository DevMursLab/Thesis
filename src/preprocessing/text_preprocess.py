"""
Phase 3+ Preprocessing — DAIC-WOZ Transcript (Text)
=====================================================
DAIC-WOZ transcript থেকে participant এর কথা বের করে
Bag-of-Words বা TF-IDF feature বানায়।

Depressed মানুষের speech pattern:
  - বেশি first-person singular ("I", "me", "my") — self-focus
  - বেশি negative words ("hopeless", "worthless", "tired")
  - কম positive words
  - ছোট ছোট উত্তর, কম word count

Run:
    python src/preprocessing/text_preprocess.py

Output:
    data/processed/daic_text/X_train.npy  (N, VOCAB_SIZE)  float32 (TF-IDF)
    data/processed/daic_text/y_train.npy  (N,)             int64
    data/processed/daic_text/X_dev.npy
    data/processed/daic_text/y_dev.npy
    data/processed/daic_text/vectorizer.pkl  (TF-IDF vectorizer)
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import pickle
import numpy as np
import pandas as pd
from configs.config import (DAIC_RAW, DAIC_LABELS, DATA_PROCESSED,
                             VOCAB_SIZE, MAX_TEXT_LEN, SEED)

TEXT_OUT = DATA_PROCESSED / "daic_text"


def load_labels(split: str) -> pd.DataFrame:
    csv_path = DAIC_LABELS / f"{split}_split_Depression_AVEC2017.csv"
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    return df[["Participant_ID", "PHQ8_Binary", "PHQ8_Score"]].copy()


def extract_participant_text(participant_id: int) -> str | None:
    """
    Transcript থেকে শুধু participant এর কথা বের করে।
    Ellie (interviewer robot) এর কথা বাদ দেওয়া হয়।
    """
    transcript_file = DAIC_RAW / f"{participant_id}_TRANSCRIPT.csv"
    if not transcript_file.exists():
        return None

    df = pd.read_csv(transcript_file, sep="\t")
    df.columns = df.columns.str.strip()

    # participant এর কথা শুধু (Ellie বাদ)
    participant_rows = df[df["speaker"].str.strip() == "Participant"]
    if len(participant_rows) == 0:
        return None

    text = " ".join(participant_rows["value"].dropna().astype(str).tolist())
    text = text.lower().strip()

    if len(text) < 10:
        return None

    return text


def build_features(texts_train: list[str], texts_dev: list[str]):
    """TF-IDF vectorizer train এ fit করে, দুটো split এ transform করে।"""
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(
        max_features=VOCAB_SIZE,
        ngram_range=(1, 2),      # unigram + bigram
        min_df=2,
        sublinear_tf=True,
    )
    X_train = vectorizer.fit_transform(texts_train).toarray().astype(np.float32)
    X_dev   = vectorizer.transform(texts_dev).toarray().astype(np.float32)
    return X_train, X_dev, vectorizer


def process_split(label_df: pd.DataFrame) -> tuple[list, list]:
    texts, labels = [], []
    skipped = 0

    for _, row in label_df.iterrows():
        pid   = int(row["Participant_ID"])
        label = int(row["PHQ8_Binary"])

        text = extract_participant_text(pid)
        if text is None:
            skipped += 1
            continue

        texts.append(text)
        labels.append(label)

    print(f"  Loaded: {len(texts)}  Skipped: {skipped}")
    return texts, labels


def preprocess():
    if not DAIC_RAW.exists():
        print(f"DAIC-WOZ not found: {DAIC_RAW}")
        _synthetic_fallback()
        return

    print(f"DAIC-WOZ found: {DAIC_RAW}")

    print("\nProcessing train split...")
    train_labels = load_labels("train")
    texts_train, y_train = process_split(train_labels)

    print("\nProcessing dev split...")
    dev_labels = load_labels("dev")
    texts_dev, y_dev = process_split(dev_labels)

    if not texts_train:
        print("No text data found!")
        return

    print("\nBuilding TF-IDF features...")
    X_train, X_dev, vectorizer = build_features(texts_train, texts_dev)

    TEXT_OUT.mkdir(parents=True, exist_ok=True)
    np.save(TEXT_OUT / "X_train.npy", X_train)
    np.save(TEXT_OUT / "y_train.npy", np.array(y_train, dtype=np.int64))
    np.save(TEXT_OUT / "X_dev.npy",   X_dev)
    np.save(TEXT_OUT / "y_dev.npy",   np.array(y_dev,   dtype=np.int64))

    with open(TEXT_OUT / "vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    dep_tr = sum(y_train)
    dep_dv = sum(y_dev)
    print(f"\nTrain: {X_train.shape} | depressed: {dep_tr}  not: {len(y_train)-dep_tr}")
    print(f"Dev:   {X_dev.shape}   | depressed: {dep_dv}  not: {len(y_dev)-dep_dv}")
    print(f"Vocab size: {X_train.shape[1]}")
    print(f"\nSaved to: {TEXT_OUT}")


def _synthetic_fallback():
    rng = np.random.default_rng(SEED)
    n   = 200
    X   = rng.random((n, VOCAB_SIZE), dtype=np.float32)
    y   = np.array([0]*(n//2) + [1]*(n//2), dtype=np.int64)
    idx = rng.permutation(n)
    X, y = X[idx], y[idx]

    TEXT_OUT.mkdir(parents=True, exist_ok=True)
    np.save(TEXT_OUT / "X_train.npy", X[:160])
    np.save(TEXT_OUT / "y_train.npy", y[:160])
    np.save(TEXT_OUT / "X_dev.npy",   X[160:])
    np.save(TEXT_OUT / "y_dev.npy",   y[160:])
    print(f"Synthetic text fallback saved — shape: ({n}, {VOCAB_SIZE})")


if __name__ == "__main__":
    preprocess()
