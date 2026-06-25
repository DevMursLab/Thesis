# Tri-Modal Depression Risk Detection

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/Dataset-DAIC--WOZ-orange?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Modalities-Face%20%7C%20Audio%20%7C%20Text-purple?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Status-Active%20Research-green?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge"/>
</p>

<p align="center">
  <b>A tri-modal deep learning system for depression risk screening<br/>
  from facial action units, speech acoustics, and linguistic patterns.</b><br/><br/>
  CLNF Action Units (face) · MFCC Bi-LSTM (audio) · TF-IDF + Bigrams (text) · Cross-Modal Attention Fusion · Grad-CAM XAI · Fairness-Aware Training
</p>

---

> ⚠️ **Disclaimer:** This is an academic research prototype for *risk screening only* — NOT a clinical diagnostic tool. Always consult a licensed mental health professional.

---

## Overview

Depression affects over **280 million people** globally (WHO, 2023), yet fewer than half receive adequate treatment. Existing automated screening systems rely on a single modality — missing the rich cross-modal signal that clinicians naturally integrate.

This project builds a **tri-modal, fairness-aware, explainable** depression risk screening system trained on the **DAIC-WOZ clinical interview corpus** (USC Institute for Creative Technologies):

| Modality | Signal | Feature Extraction |
|----------|--------|-------------------|
| **Face** | Facial muscle dynamics | 20-channel CLNF Action Units (OpenFace) |
| **Audio** | Vocal acoustics & prosody | MFCC + Δ + ΔΔ → Bi-LSTM |
| **Text** | Linguistic depression markers | Participant transcript → TF-IDF bigrams |

**Novel contributions:**
- Cross-modal attention fusion (learns *when* each modality is more informative)
- Gender fairness constraints during training (equal F1 across demographic groups)
- Grad-CAM + attention rollout for clinical explainability
- Ablation study validating each modality's independent contribution

> ⚠️ **Clinical disclaimer:** This is an academic *screening* prototype — NOT a diagnostic tool. All outputs must be reviewed by a licensed mental health professional.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    TRI-MODAL DEPRESSION SCREENING PIPELINE                │
└──────────────────────────────────────────────────────────────────────────┘

  FACE (CLNF AUs)          AUDIO (_AUDIO.wav)         TEXT (transcript)
  20 Action Units           MFCC + Δ + ΔΔ             Participant speech
  200 time steps            300 time steps             TF-IDF bigrams
        │                         │                          │
        ▼                         ▼                          ▼
  ┌───────────┐           ┌──────────────┐          ┌──────────────┐
  │  AU-LSTM  │           │   Bi-LSTM    │          │  FC 256      │
  │  Bi-dir   │           │   2 layers   │          │  ReLU        │
  │  128 hid  │           │   128 hid    │          │  Dropout     │
  └─────┬─────┘           └──────┬───────┘          └──────┬───────┘
        │                        │                          │
        │   face_feat (256)      │  audio_feat (256)        │  text_feat (256)
        └────────────────────────┼──────────────────────────┘
                                 ▼
                    ┌─────────────────────────┐
                    │  Cross-Modal Attention  │  ← learns which modality
                    │  Fusion Layer           │    matters most per sample
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │  Fairness-Aware Head    │  ← gender-equalized training
                    │  FC 512 → 128 → 2      │
                    └────────────┬────────────┘
                                 ▼
                         Not Depressed / Depressed

                                 │
                    ┌────────────▼────────────┐
                    │    Explainability       │
                    │  Grad-CAM (face AUs)   │
                    │  Attention rollout      │
                    └─────────────────────────┘
```

---

## Phased Roadmap

| Phase | Task | Dataset | Status |
|:-----:|------|---------|:------:|
| 0 | Project scaffolding — models, configs, training pipeline | — | ✅ Done |
| 1 | Face emotion pre-training (CNN on FER2013) | FER2013 | ✅ Done |
| 2 | DAIC-WOZ face branch — CLNF Action Unit preprocessing | DAIC-WOZ | ✅ Done |
| 3 | Audio branch — MFCC+Δ+ΔΔ extraction + Bi-LSTM | DAIC-WOZ | ✅ Done |
| 3+ | Text branch — transcript TF-IDF bigram features | DAIC-WOZ | ✅ Done |
| 4 | Cross-modal attention fusion (tri-modal) + Colab notebook | DAIC-WOZ | ✅ Done |
| 5 | Fairness-aware training (gender-equalized F1) | DAIC-WOZ | ⏳ Next |
| 6 | Grad-CAM + attention explainability | — | ⏳ |
| 7 | Ablation study + SOTA comparison + statistical tests | — | ⏳ |
| 8 | Paper draft → IEEE submission | — | ⏳ |

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
│   ├── preprocessing/
│   │   ├── face_preprocess.py   # FER2013 loading + augmentation
│   │   ├── daic_preprocess.py   # CLNF AU extraction (DAIC-WOZ)
│   │   ├── audio_preprocess.py  # MFCC + delta + delta2 extraction
│   │   └── text_preprocess.py   # Transcript TF-IDF bigram extraction
│   ├── models/
│   │   ├── face_cnn.py          # CNN for FER2013 emotion pre-training
│   │   ├── audio_lstm.py        # Bi-LSTM for MFCC audio features
│   │   ├── encoders.py          # FaceAUEncoder · AudioEncoder · TextEncoder (256-dim each)
│   │   └── fusion_attention.py  # CrossModalAttention + TriModalFusionModel (7.8M params)
│   ├── training/
│   │   ├── train_face.py        # Phase 1 — FER2013 emotion CNN
│   │   ├── train_face_daic.py   # Phase 2 — CLNF AU branch fine-tune
│   │   ├── train_audio.py       # Phase 3 — Bi-LSTM on MFCC features
│   │   └── train_fusion.py      # Phase 4 — tri-modal fairness-aware training
│   └── explainability/
│       └── gradcam.py           # Grad-CAM heatmap generator
├── notebooks/
│   ├── phase1_face_cnn.ipynb        # Phase 1 — FER2013 CNN training
│   ├── phase2_daic_faceframes.ipynb # Phase 2 — DAIC-WOZ fine-tuning
│   ├── phase3_audio_lstm.ipynb      # Phase 3 — Bi-LSTM audio training
│   └── phase4_fusion.ipynb          # Phase 4 — Tri-modal cross-attention fusion
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

Preliminary results on DAIC-WOZ (34-participant dev split, CPU baseline, 5 epochs):

| Model | Modalities | Dataset | F1 | AUC |
|-------|-----------|---------|-----|-----|
| Face AU Bi-LSTM (Phase 2) | Face only | DAIC-WOZ | — | — |
| Audio Bi-LSTM (Phase 3) | Audio only | DAIC-WOZ | — | — |
| **Tri-Modal Fusion (Phase 4)** | **Face + Audio + Text** | **DAIC-WOZ** | *in progress* | **0.66** |

> AUC improved from 0.49 (random) → 0.66 in 5 CPU epochs — validates the cross-modal signal.
> GPU training (30 epochs) expected on Colab: see `notebooks/phase4_fusion.ipynb`.

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
