"""
সব setting এক জায়গায়। কোনো number বা path পরিবর্তন করতে হলে শুধু এখানে করবে।
এতে code এর ভেতরে hunt করতে হবে না।
"""

from pathlib import Path

# ---- Paths ----
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
METRICS = RESULTS / "metrics"

# ---- Face branch (CNN) ----
IMG_SIZE = 48          # FER2013 images are 48x48 grayscale
NUM_EMOTION_CLASSES = 7  # angry, disgust, fear, happy, sad, surprise, neutral

# Depression-related grouping: আমরা পরে emotion থেকে risk আনব
# (sad, fear, angry, disgust → high-risk leaning ; happy, surprise, neutral → low-risk leaning)
DEPRESSION_CLASSES = 2   # low-risk / high-risk

# ---- Audio branch ----
SAMPLE_RATE = 16000
N_MFCC = 40
AUDIO_WINDOW_SEC = 3.0

# ---- Training ----
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 30
SEED = 42
DEVICE = "cuda"  # Colab/Kaggle GPU; CPU হলে "cpu" দিবে

# ---- Train/test split ----
TEST_SIZE = 0.2
VAL_SIZE = 0.1
