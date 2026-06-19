"""
Phase 1 Training — Face CNN
===========================
FER2013 দিয়ে face CNN train করে।

গুরুত্বপূর্ণ: এই script এমনভাবে লেখা যে FER2013 না থাকলেও
fake (synthetic) data দিয়ে চলবে — যাতে তুমি আজই দেখতে পারো পুরো pipeline কাজ করে।
আসল data আসলে শুধু load_fer2013() function টা সত্যি data দিয়ে ভরবে।

চালানোর নিয়ম:
    python src/training/train_face.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from configs.config import (IMG_SIZE, NUM_EMOTION_CLASSES, BATCH_SIZE,
                            LEARNING_RATE, EPOCHS, SEED, DATA_RAW)
from src.models.face_cnn import FaceCNN


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_fer2013():
    """
    FER2013 load করার চেষ্টা করে। ফাইল: data/raw/fer2013.csv
    না পেলে synthetic data ফেরত দেয় (pipeline test করার জন্য)।

    আসল data: https://www.kaggle.com/datasets/msambare/fer2013
    নামিয়ে fer2013.csv কে data/raw/ এ রাখবে।
    """
    csv_path = DATA_RAW / "fer2013.csv"

    if csv_path.exists():
        import pandas as pd
        print(f"✅ আসল FER2013 পাওয়া গেছে: {csv_path}")
        df = pd.read_csv(csv_path)
        pixels = df["pixels"].apply(
            lambda s: np.array(s.split(), dtype=np.float32))
        X = np.stack(pixels.values).reshape(-1, 1, IMG_SIZE, IMG_SIZE) / 255.0
        y = df["emotion"].values.astype(np.int64)
        return X, y

    # ---- Fallback: synthetic data ----
    print("⚠️  FER2013 পাওয়া যায়নি — synthetic data দিয়ে pipeline test হচ্ছে।")
    print("   আসল data নামিয়ে data/raw/fer2013.csv এ রাখলে আসল training হবে।")
    n = 2000
    X = np.random.rand(n, 1, IMG_SIZE, IMG_SIZE).astype(np.float32)
    y = np.random.randint(0, NUM_EMOTION_CLASSES, size=n).astype(np.int64)
    return X, y


def main():
    set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ---- Data ----
    X, y = load_fer2013()
    n_train = int(len(X) * 0.8)
    Xtr, ytr = X[:n_train], y[:n_train]
    Xte, yte = X[n_train:], y[n_train:]

    train_ds = TensorDataset(torch.tensor(Xtr), torch.tensor(ytr))
    test_ds = TensorDataset(torch.tensor(Xte), torch.tensor(yte))
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE)

    # ---- Model ----
    model = FaceCNN(num_classes=NUM_EMOTION_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ---- Train (এখানে demo তে 2 epoch; আসল run এ EPOCHS) ----
    epochs = 2 if len(X) <= 2000 else EPOCHS
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} — loss: {total_loss/len(train_dl):.4f}")

    # ---- Evaluate ----
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in test_dl:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb).argmax(1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    print(f"\nTest Accuracy: {100*correct/total:.2f}%")
    print("(synthetic data হলে ~14% random হবে — এটা normal, structure কাজ করছে এটাই point)")


if __name__ == "__main__":
    main()
