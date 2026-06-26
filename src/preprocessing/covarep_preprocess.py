"""
Phase 8 — COVAREP + FORMANT Clinical Audio Preprocessing
=========================================================
DAIC-WOZ provides two clinically-validated acoustic feature sets beyond MFCC:

  COVAREP (74 features/frame):
    - F0 (fundamental frequency / pitch) — depression = reduced F0 variability
    - NAQ, QOQ (normalized amplitude/quasi-open quotient) — glottal closure quality
    - H1H2 (harmonic difference) — breathiness, a marker of flat affect
    - PSP, MDQ, peakSlope, Rd — glottal source parameters
    - MCEP 0-23 (24 mel cepstral prediction coefficients) — spectral envelope
    - HMPDM 0-24, HMPDD 0-12 — harmonic phase distortion

  FORMANT (5 features/frame):
    - F1-F5 formant frequencies — vocal tract shape
    - Reduced vowel space in depression (Scherer et al. 2015)

Combined with existing MFCC+delta+delta2 (120), total audio dim = 199.

Output:
    data/processed/daic_audio_covarep/X_train.npy  (N, T, 199)  float32
    data/processed/daic_audio_covarep/X_dev.npy
    data/processed/daic_audio_covarep/y_train.npy
    data/processed/daic_audio_covarep/y_dev.npy
    data/processed/daic_audio_covarep/phq8_train.npy    (N,)  float32
    data/processed/daic_audio_covarep/symptoms_train.npy (N,8) float32
    data/processed/daic_audio_covarep/gender_train.npy   (N,)  str
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd

from configs.config import (
    DATA_PROCESSED, N_MFCC, N_COVAREP, N_FORMANT, AUDIO_FEAT_DIM,
    DAIC_RAW, DAIC_LABELS, SAMPLE_RATE, AUDIO_TIME_STEPS, SEED,
)

OUT_DIR        = DATA_PROCESSED / "daic_audio_covarep"
MAX_T          = AUDIO_TIME_STEPS   # 300
HOP_LENGTH     = 512
COVAREP_FPS    = 100                # COVAREP extracted at 100 Hz

PHQ_SYMPTOM_COLS = [
    "PHQ8_NoInterest", "PHQ8_Depressed", "PHQ8_Sleep", "PHQ8_Tired",
    "PHQ8_Appetite", "PHQ8_Failure", "PHQ8_Concentrating", "PHQ8_Moving",
]


def load_labels(split: str) -> pd.DataFrame:
    csv_path = DAIC_LABELS / f"{split}_split_Depression_AVEC2017.csv"
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    cols = ["Participant_ID", "PHQ8_Binary", "PHQ8_Score", "Gender"] + PHQ_SYMPTOM_COLS
    return df[cols].copy()


def _pad_trim(arr: np.ndarray, target_T: int) -> np.ndarray:
    T, D = arr.shape
    if T >= target_T:
        return arr[:target_T]
    pad = np.zeros((target_T - T, D), dtype=np.float32)
    return np.concatenate([arr, pad], axis=0)


def _zscore(arr: np.ndarray) -> np.ndarray:
    mean = arr.mean(axis=0, keepdims=True)
    std  = arr.std(axis=0, keepdims=True) + 1e-8
    return (arr - mean) / std


def _resample_to_T(arr: np.ndarray, target_T: int) -> np.ndarray:
    """Linearly interpolate time axis from any length to target_T."""
    T, D = arr.shape
    if T == target_T:
        return arr
    x_old = np.linspace(0, 1, T)
    x_new = np.linspace(0, 1, target_T)
    out = np.zeros((target_T, D), dtype=np.float32)
    for d in range(D):
        out[:, d] = np.interp(x_new, x_old, arr[:, d])
    return out


def extract_mfcc_stream(audio_path: Path) -> np.ndarray:
    """Returns (MAX_T, 120) — MFCC + delta + delta2, z-scored."""
    import librosa
    y, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    mfcc   = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC,
                                   hop_length=HOP_LENGTH, n_fft=1024)
    delta  = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    feat   = np.concatenate([mfcc, delta, delta2], axis=0).T  # (T, 120)
    feat   = _zscore(feat)
    return _pad_trim(feat.astype(np.float32), MAX_T)


def extract_covarep_stream(pid: int) -> np.ndarray:
    """Returns (MAX_T, N_COVAREP) — COVAREP features, z-scored."""
    path = DAIC_RAW / f"{pid}_COVAREP.csv"
    if not path.exists():
        return np.zeros((MAX_T, N_COVAREP), dtype=np.float32)

    df = pd.read_csv(path, header=None)
    arr = df.values.astype(np.float32)  # (T_raw, 74)
    if arr.shape[1] < N_COVAREP:
        pad = np.zeros((arr.shape[0], N_COVAREP - arr.shape[1]), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=1)
    arr = arr[:, :N_COVAREP]

    # replace inf/nan
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = _zscore(arr)

    # resample from COVAREP_FPS to MAX_T steps
    arr = _resample_to_T(arr, MAX_T)
    return arr.astype(np.float32)


def extract_formant_stream(pid: int) -> np.ndarray:
    """Returns (MAX_T, N_FORMANT) — formant frequencies, z-scored."""
    path = DAIC_RAW / f"{pid}_FORMANT.csv"
    if not path.exists():
        return np.zeros((MAX_T, N_FORMANT), dtype=np.float32)

    df = pd.read_csv(path, header=None)
    arr = df.values.astype(np.float32)
    if arr.shape[1] < N_FORMANT:
        pad = np.zeros((arr.shape[0], N_FORMANT - arr.shape[1]), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=1)
    arr = arr[:, :N_FORMANT]

    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = _zscore(arr)
    arr = _resample_to_T(arr, MAX_T)
    return arr.astype(np.float32)


def extract_participant(pid: int, audio_path: Path) -> np.ndarray:
    """
    Returns (MAX_T, AUDIO_FEAT_DIM=199):
        cols 0:120    MFCC + delta + delta2
        cols 120:194  COVAREP (74)
        cols 194:199  FORMANT (5)
    """
    mfcc    = extract_mfcc_stream(audio_path)       # (300, 120)
    covarep = extract_covarep_stream(pid)            # (300, 74)
    formant = extract_formant_stream(pid)            # (300,  5)
    return np.concatenate([mfcc, covarep, formant], axis=1)  # (300, 199)


def process_split(label_df: pd.DataFrame, split_name: str):
    X_list, y_list, phq8_list, sym_list, gender_list = [], [], [], [], []

    for _, row in label_df.iterrows():
        pid    = int(row["Participant_ID"])
        label  = int(row["PHQ8_Binary"])
        phq8   = float(row["PHQ8_Score"])
        gender = str(row["Gender"])
        symptoms = [float(row[c]) for c in PHQ_SYMPTOM_COLS]

        audio_path = DAIC_RAW / f"{pid}_AUDIO.wav"
        if not audio_path.exists():
            print(f"  [SKIP] {pid} — audio not found")
            continue

        feat = extract_participant(pid, audio_path)
        X_list.append(feat)
        y_list.append(label)
        phq8_list.append(phq8)
        sym_list.append(symptoms)
        gender_list.append(gender)
        print(f"  {pid} — feat {feat.shape}  label={label}  PHQ8={phq8:.0f}  {gender}")

    if not X_list:
        print(f"  No samples for {split_name}"); return

    X        = np.stack(X_list).astype(np.float32)          # (N, 300, 199)
    y        = np.array(y_list, dtype=np.int64)
    phq8_arr = np.array(phq8_list, dtype=np.float32)
    sym_arr  = np.array(sym_list, dtype=np.float32)         # (N, 8)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / f"X_{split_name}.npy",        X)
    np.save(OUT_DIR / f"y_{split_name}.npy",        y)
    np.save(OUT_DIR / f"phq8_{split_name}.npy",     phq8_arr)
    np.save(OUT_DIR / f"symptoms_{split_name}.npy", sym_arr)

    pd.DataFrame({"participant_id": [row["Participant_ID"] for _, row in label_df.iterrows()
                                     if (DAIC_RAW / f"{int(row['Participant_ID'])}_AUDIO.wav").exists()],
                  "gender": gender_list}).to_csv(
        OUT_DIR / f"gender_{split_name}.csv", index=False)

    dep = (y == 1).sum()
    print(f"\n  [{split_name}] N={len(y)} | depressed={dep} | "
          f"not-depressed={len(y)-dep}")
    print(f"  Feature shape: {X.shape}  (T={MAX_T}, D={X.shape[2]})")


def preprocess():
    if not DAIC_RAW.exists():
        print(f"DAIC-WOZ not found at: {DAIC_RAW}")
        _synthetic_fallback()
        return

    print(f"DAIC-WOZ: {DAIC_RAW}")
    print(f"Audio feature dim: MFCC(120) + COVAREP({N_COVAREP}) + FORMANT({N_FORMANT}) = {AUDIO_FEAT_DIM}")
    for split in ["train", "dev"]:
        csv_path = DAIC_LABELS / f"{split}_split_Depression_AVEC2017.csv"
        if not csv_path.exists():
            print(f"[SKIP] {csv_path} not found"); continue
        print(f"\nProcessing {split} split...")
        process_split(load_labels(split), split)

    print(f"\nDone. Output: {OUT_DIR}")


def _synthetic_fallback():
    """Pipeline smoke-test without real DAIC-WOZ."""
    rng = np.random.default_rng(SEED)
    N   = 141

    X = rng.standard_normal((N, MAX_T, AUDIO_FEAT_DIM)).astype(np.float32)
    y = np.array([0] * (N - 40) + [1] * 40, dtype=np.int64)
    phq8 = rng.uniform(0, 24, N).astype(np.float32)
    phq8[y == 0] = np.clip(phq8[y == 0], 0, 9)
    phq8[y == 1] = np.clip(phq8[y == 1], 10, 24)
    symptoms = rng.integers(0, 4, (N, 8)).astype(np.float32)
    genders  = np.array(["M" if i % 2 == 0 else "F" for i in range(N)])

    idx = rng.permutation(N)
    X, y, phq8, symptoms = X[idx], y[idx], phq8[idx], symptoms[idx]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for split, sl in [("train", slice(0, 107)), ("dev", slice(107, 141))]:
        np.save(OUT_DIR / f"X_{split}.npy",        X[sl])
        np.save(OUT_DIR / f"y_{split}.npy",        y[sl])
        np.save(OUT_DIR / f"phq8_{split}.npy",     phq8[sl])
        np.save(OUT_DIR / f"symptoms_{split}.npy", symptoms[sl])
        pd.DataFrame({"participant_id": range(len(y[sl])),
                      "gender": genders[idx][sl]}).to_csv(
            OUT_DIR / f"gender_{split}.csv", index=False)

    print(f"Synthetic data: train={107}, dev={34}  shape=({MAX_T},{AUDIO_FEAT_DIM})")


if __name__ == "__main__":
    preprocess()
