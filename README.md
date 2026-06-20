# Multimodal Depression Risk Detection

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/Status-Active%20Research-green?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge"/>
</p>

<p align="center">
  <b>A multimodal deep learning system for depression risk screening from facial expressions and voice patterns.</b><br/>
  CNN (face) · LSTM (audio) · Feature Fusion · Grad-CAM Explainability
</p>

---

> ⚠️ **Disclaimer:** This is an academic research prototype for *risk screening only* — NOT a clinical diagnostic tool. Always consult a licensed mental health professional.

---

## Overview

Depression affects over **280 million people** globally (WHO, 2023), yet most cases go undetected. This project builds a **non-invasive, automated screening system** that analyzes:

- **Facial expressions** → Convolutional Neural Network (CNN) trained on FER2013
- **Speech patterns** → LSTM over MFCC features from DAIC-WOZ audio
- **Combined decision** → Feature-level fusion with a classification head
- **Why the model decided** → Grad-CAM heatmaps for explainability

The goal is an accessible, interpretable tool that could assist clinicians in early detection — not replace them.

---

## Architecture

```
                    ┌─────────────────────────────────────────────┐
                    │           MULTIMODAL PIPELINE               │
                    └─────────────────────────────────────────────┘

  Face (48×48 image)                     Voice (raw audio)
        │                                       │
        ▼                                       ▼
  ┌───────────┐                         ┌───────────────┐
  │  Conv2D   │  ×3 blocks              │  MFCC Extract │
  │  BN+ReLU  │  (32→64→128 ch)        │  (40 coeffs)  │
  │  MaxPool  │                         └───────┬───────┘
  └─────┬─────┘                                 │
        │                                       ▼
        ▼                                 ┌───────────┐
  ┌───────────┐                          │ Bi-LSTM   │  2 layers
  │  FC 256   │  ← face feature          │ hidden=128│
  │  Dropout  │                          └─────┬─────┘
  └─────┬─────┘                                │
        │                               audio feature
        └──────────────┬────────────────┘
                       ▼
              ┌─────────────────┐
              │  Feature Concat │  (256 + 256 = 512 dim)
              │  FC 256 → 64   │
              │  Classifier    │
              └────────┬────────┘
                       ▼
              Low Risk / High Risk

                       │  (Grad-CAM)
                       ▼
              🔥 Heatmap on face
              showing which region
              drove the prediction
```

---

## Phased Roadmap

| Phase | Task | Dataset | Status |
|:-----:|------|---------|:------:|
| 0 | Project scaffolding (models, configs, training scripts) | — | ✅ Done |
| 1 | Face branch — CNN emotion classifier | FER2013 | ✅ Done |
| 2 | DAIC-WOZ face frame extraction + fine-tune CNN | DAIC-WOZ | ✅ Done |
| 3 | Audio branch — MFCC + Bi-LSTM (DAIC-WOZ) | DAIC-WOZ audio | ✅ Done |
| 4 | Multimodal fusion (face + audio) | DAIC-WOZ | ⏳ Next |
| 5 | Temporal modeling (LSTM over frame sequences) | DAIC-WOZ | ⏳ |
| 6 | Grad-CAM explainability visualizations | — | ⏳ |
| 7 | Evaluation — confusion matrix, ROC, F1 | — | ⏳ |
| 8 | Paper draft + preprint submission | — | ⏳ |

Each phase is gated on the previous. No phase is started until the prior one is validated.

---

## Repository Structure

```
depression_thesis/
├── configs/
│   └── config.py            # All hyperparameters and paths in one place
├── src/
│   ├── preprocessing/
│   │   ├── face_preprocess.py   # FER2013 loading + augmentation
│   │   ├── daic_preprocess.py   # DAIC-WOZ frame extraction + labels
│   │   └── audio_preprocess.py  # MFCC + delta + delta2 extraction
│   ├── models/
│   │   ├── face_cnn.py          # CNN for facial emotion features
│   │   ├── audio_lstm.py        # Bi-LSTM for MFCC audio features
│   │   └── fusion.py            # Feature-level fusion head
│   ├── training/
│   │   ├── train_face.py        # Phase 1 — FER2013 emotion CNN
│   │   ├── train_face_daic.py   # Phase 2 — fine-tune on DAIC-WOZ
│   │   └── train_audio.py       # Phase 3 — Bi-LSTM on MFCC features
│   └── explainability/
│       └── gradcam.py           # Grad-CAM heatmap generator
├── notebooks/
│   ├── phase1_face_cnn.ipynb        # Phase 1 — FER2013 CNN training
│   ├── phase2_daic_faceframes.ipynb # Phase 2 — DAIC-WOZ fine-tuning
│   └── phase3_audio_lstm.ipynb      # Phase 3 — Bi-LSTM audio training
├── data/
│   ├── raw/                     # FER2013 CSV, DAIC-WOZ files (not committed)
│   └── processed/               # Preprocessed arrays
├── results/
│   ├── figures/                 # Confusion matrices, ROC curves
│   └── metrics/                 # Accuracy, F1, AUC logs
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Phase 1 (Face CNN)

**Without data** — runs on synthetic data to verify the pipeline works:
```bash
python src/training/train_face.py
```

**With real FER2013 data:**
1. Download `fer2013.csv` from [Kaggle FER2013](https://www.kaggle.com/datasets/msambare/fer2013)
2. Place it at `data/raw/fer2013.csv`
3. Run the same command — it auto-detects and uses real data

### 3. Verify model sanity
```bash
python src/models/face_cnn.py      # prints output shape + param count
python src/explainability/gradcam.py  # prints Grad-CAM heatmap shape
```

---

## Datasets

| Dataset | Used For | Access |
|---------|----------|--------|
| [FER2013](https://www.kaggle.com/datasets/msambare/fer2013) | Phase 1 — face emotion CNN | Public (Kaggle) |
| [DAIC-WOZ](https://dcapswoz.isi.edu/) | Phase 2-5 — depression labels + audio | Request required (USC/AVEC) |

---

## Key Design Decisions

- **Why FER2013 first?** It is fully public, well-benchmarked, and lets us build + validate the face pipeline before introducing the restricted DAIC-WOZ dataset.
- **Why feature-level fusion?** Decision-level fusion loses rich signal; pixel-level fusion is intractable. Feature concat at 256+256 dimensions is the sweet spot for this scale.
- **Why Grad-CAM?** Clinical credibility — a model that can show *which part of the face* or *which audio segment* drove a decision is far more useful to a clinician than a black box.
- **Why Bi-LSTM for audio?** Depression manifests in temporal speech patterns (long pauses, falling pitch). Bidirectional context captures these better than a feed-forward approach.

---

## Results

> Phase 1 in progress — results will be updated here after training on FER2013.

| Model | Dataset | Accuracy | F1 (macro) | AUC |
|-------|---------|----------|------------|-----|
| Face CNN (Phase 1) | FER2013 | — | — | — |
| Audio Bi-LSTM (Phase 3) | DAIC-WOZ | — | — | — |
| Fusion Model (Phase 4) | DAIC-WOZ | — | — | — |

---

## Author

**Md. Mursalin**  
Department of Computer Science & Engineering  
Netrokona University  
Supervisor: Md. Shovon

---

## Citation

If you find this work useful, please cite:

```bibtex
@misc{mursalin2025depression,
  title   = {Multimodal Depression Risk Detection from Face and Voice},
  author  = {Mursalin, Md.},
  year    = {2025},
  note    = {Undergraduate thesis, Netrokona University}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
