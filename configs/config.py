"""
All settings in one place. Change any number or path only here.
"""

from pathlib import Path

# ---- Paths ----
ROOT           = Path(__file__).resolve().parent.parent
DATA_RAW       = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS        = ROOT / "results"
FIGURES        = RESULTS / "figures"
METRICS        = RESULTS / "metrics"

# ---- DAIC-WOZ paths (actual data location) ----
DAIC_RAW       = ROOT / "daicwoz" / "daicwoz"   # flat folder with all participant files
DAIC_LABELS    = ROOT                             # train/dev CSV files are in project root

# ---- Face branch — FER2013 CNN ----
IMG_SIZE            = 48
NUM_EMOTION_CLASSES = 7
DEPRESSION_CLASSES  = 2

# ---- Face branch — CLNF Action Units (DAIC-WOZ) ----
# AU columns from CLNF_AUs.txt (continuous r + binary c)
AU_COLS = [
    "AU01_r","AU02_r","AU04_r","AU05_r","AU06_r","AU09_r","AU10_r",
    "AU12_r","AU14_r","AU15_r","AU17_r","AU20_r","AU25_r","AU26_r",
    "AU04_c","AU12_c","AU15_c","AU23_c","AU28_c","AU45_c",
]
N_AU_FEATURES  = len(AU_COLS)   # 20 facial action unit features
AU_TIME_STEPS  = 200            # frames per participant (pad/trim)

# ---- Audio branch ----
SAMPLE_RATE      = 16000
N_MFCC           = 40
AUDIO_WINDOW_SEC = 3.0
AUDIO_TIME_STEPS = 300          # MFCC time steps per participant

# ---- COVAREP / FORMANT (clinical-grade audio) ----
N_COVAREP        = 74           # COVAREP features per frame (F0, NAQ, QOQ, MCEP...)
N_FORMANT        = 5            # formant frequencies F1-F5
AUDIO_FEAT_DIM   = N_MFCC * 3 + N_COVAREP + N_FORMANT  # 120 + 74 + 5 = 199

# ---- Multi-task learning ----
N_PHQ_SYMPTOMS   = 8            # PHQ-8 has 8 individual symptom items (0-3 each)
PHQ8_MAX         = 24.0         # max possible PHQ-8 score

# ---- Modality dropout (missing-modality robustness) ----
MODALITY_DROPOUT_P = 0.15       # prob of zeroing each modality per batch during training

# ---- Multi-task loss weights ----
LAMBDA_SCORE    = 0.3           # PHQ-8 score regression weight
LAMBDA_SYMPTOM  = 0.2           # 8-symptom classification weight
LAMBDA_FAIRNESS = 0.1           # equalized odds fairness weight
FAIRNESS_WARMUP = 8             # epochs before fairness term activates

# ---- Text branch ----
MAX_TEXT_LEN  = 128             # max tokens per participant transcript
VOCAB_SIZE    = 1000            # TF-IDF features (small-N: keep << #samples to avoid overfit)

# ---- Fusion model capacity ----
# With only ~107 training participants, a large model memorizes instantly.
# Keep the shared embedding dim small and regularization high.
EMBED_DIM     = 64              # shared modality embedding size
FUSION_DROPOUT = 0.6           # heavy dropout for tiny clinical dataset
WEIGHT_DECAY   = 1e-2          # strong L2 to combat overfitting

# ---- Training ----
BATCH_SIZE    = 32
LEARNING_RATE = 1e-3
EPOCHS        = 30
SEED          = 42
DEVICE        = "cuda"

# ---- Split ----
TEST_SIZE = 0.2
VAL_SIZE  = 0.1
