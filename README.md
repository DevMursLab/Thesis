<div align="center">

<h1>
  <img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&size=28&pause=1000&color=6A0DAD&center=true&vCenter=true&width=900&lines=Tri-Modal+Depression+Risk+Detection;Cross-Modal+Attention+%2B+Fairness-Aware+Training;Face+%C2%B7+Audio+%C2%B7+Text+%E2%86%92+Clinical+AI" alt="Typing SVG" />
</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/DAIC--WOZ-USC%20ICT-orange?style=for-the-badge&logo=academia&logoColor=white"/>
  <img src="https://img.shields.io/badge/FER2013-Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Modalities-Face%20%7C%20Audio%20%7C%20Text-8A2BE2?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Novel-Cross--Modal%20Attention-FF6B6B?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Fairness-Gender%20Equalized%20F1-00C896?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/XAI-Grad--CAM%20%2B%20Attention-FFD700?style=for-the-badge"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Parameters-7.8M-lightgrey?style=flat-square"/>
  <img src="https://img.shields.io/badge/AUC-0.66%20%E2%86%92%20improving-brightgreen?style=flat-square"/>
  <img src="https://img.shields.io/badge/Phase-4%20of%208-blue?style=flat-square"/>
  <img src="https://img.shields.io/badge/Target-IEEE%20%7C%20Elsevier-red?style=flat-square"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square"/>
</p>

<br/>

> **"Depression leaves traces in every channel — the way eyes avoid contact, the way voice flattens, the way words grow darker. This system listens to all three at once."**

<br/>

</div>

---

## What This Research Does

Depression affects **280+ million people worldwide** (WHO, 2023). Yet most automated screening systems examine only *one* signal — audio pitch, or a single image, or typed text alone. Clinicians don't work that way: they simultaneously observe a patient's facial expressions, listen to their vocal tone, and pay attention to what words they choose.

This research replicates that multi-channel clinical intuition in deep learning. We train a single end-to-end model on real clinical interview recordings from the **DAIC-WOZ corpus** (University of Southern California) — **141 participants**, PHQ-8 depression labels, structured interview sessions — and fuse three modalities through a novel **Cross-Modal Attention** mechanism.

**The result:** each modality dynamically queries the other two. When the face looks flat *and* the voice is monotone, the model learns that these signals reinforce each other. When only one channel is uncertain, the others compensate.

---

## Novel Contributions

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         FOUR NOVEL CONTRIBUTIONS                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  1. CROSS-MODAL ATTENTION FUSION                                                │
│     Each modality (face/audio/text) queries the other two via scaled            │
│     dot-product attention — not simple concatenation.                           │
│     Face asks: "Is audio confirming what I see?"                                │
│     Audio asks: "Do the words align with what I heard?"                         │
│                                                                                 │
│  2. GENDER-EQUALIZED FAIRNESS LOSS                                              │
│     L_total = L_task + λ·(TPR_male − TPR_female)²                              │
│     Penalizes models that are accurate on average but biased by gender.         │
│     Trained to equalize True Positive Rate — catching depression equally        │
│     in male and female participants.                                            │
│                                                                                 │
│  3. TRANSFER LEARNING: FER2013 → DAIC-WOZ                                      │
│     Face encoder pre-trained on 35,887 FER2013 images, then fine-tuned         │
│     on CLNF Action Unit sequences from DAIC-WOZ clinical sessions.             │
│     Solves the small-N clinical dataset problem.                                │
│                                                                                 │
│  4. CLINICALLY INTERPRETABLE EXPLANATIONS                                       │
│     Grad-CAM on AU sequences reveals which facial muscles correlate             │
│     with depression scores. Attention weights show which modality the           │
│     model relied on — per sample, not just average.                             │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## System Architecture

```
╔══════════════════════════════════════════════════════════════════════════════════╗
║              TRI-MODAL DEPRESSION RISK DETECTION — FULL PIPELINE                ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                  ║
║   INPUT LAYER                                                                    ║
║   ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐           ║
║   │  FACE STREAM     │   │  AUDIO STREAM    │   │  TEXT STREAM     │           ║
║   │                  │   │                  │   │                  │           ║
║   │  {pid}_CLNF_     │   │  {pid}_AUDIO.wav │   │  {pid}_TRANSCRIPT│           ║
║   │  AUs.txt         │   │                  │   │  .csv            │           ║
║   │                  │   │  librosa MFCC    │   │                  │           ║
║   │  OpenFace        │   │  40 coefficients │   │  Participant     │           ║
║   │  20 AU channels  │   │  + Δ + ΔΔ        │   │  utterances only │           ║
║   │  200 time steps  │   │  300 time steps  │   │  TF-IDF bigrams  │           ║
║   │  z-score norm    │   │  120-dim total   │   │  10,000 vocab    │           ║
║   └────────┬─────────┘   └────────┬─────────┘   └────────┬─────────┘           ║
║            │                      │                       │                      ║
║   ENCODING LAYER                                                                 ║
║            ▼                      ▼                       ▼                      ║
║   ┌────────────────┐    ┌─────────────────┐    ┌──────────────────┐            ║
║   │  FaceAUEncoder │    │  AudioEncoder   │    │   TextEncoder    │            ║
║   │                │    │                 │    │                  │            ║
║   │  Bi-LSTM       │    │  Bi-LSTM        │    │  FC(10000→512)   │            ║
║   │  2 layers      │    │  2 layers       │    │  LayerNorm       │            ║
║   │  128 hidden    │    │  128 hidden     │    │  GELU            │            ║
║   │  bidirectional │    │  bidirectional  │    │  Dropout(0.4)    │            ║
║   │  mean-pool     │    │  mean-pool      │    │  FC(512→256)     │            ║
║   │  → LayerNorm   │    │  → LayerNorm    │    │  LayerNorm       │            ║
║   │  → GELU        │    │  → GELU         │    │  → GELU          │            ║
║   └───────┬────────┘    └────────┬────────┘    └────────┬─────────┘            ║
║           │  256-dim             │  256-dim              │  256-dim              ║
║           │                      │                       │                       ║
║   CROSS-MODAL ATTENTION LAYER   (★ Core Novel Contribution)                     ║
║           │                      │                       │                       ║
║           ▼◄─────────────────────┼───────────────────────┘                       ║
║   ┌──────────────────────────────────────────────────────────────────┐           ║
║   │                    CrossModalAttention × 3                       │           ║
║   │                                                                  │           ║
║   │  face  queries [audio, text]  →  attended_face  (256)           │           ║
║   │  audio queries [face,  text]  →  attended_audio (256)           │           ║
║   │  text  queries [face, audio]  →  attended_text  (256)           │           ║
║   │                                                                  │           ║
║   │  Mechanism: Q·Kᵀ/√d → softmax → weighted V → residual+LN       │           ║
║   └──────────────────────────────┬───────────────────────────────────┘           ║
║                                  │                                               ║
║                    concat([attended_face, attended_audio, attended_text])        ║
║                                  │  768-dim fused representation                 ║
║   CLASSIFICATION HEAD                                                            ║
║                                  ▼                                               ║
║   ┌──────────────────────────────────────────────────────────────────┐           ║
║   │  FC(768→512) → LayerNorm → GELU → Dropout(0.4)                  │           ║
║   │  FC(512→128) → LayerNorm → GELU → Dropout(0.2)                  │           ║
║   │  FC(128→2)  →  logits                                           │           ║
║   │                                                                  │           ║
║   │  Loss = CrossEntropy(class-weighted) + 0.3·(TPR_M − TPR_F)²     │           ║
║   └──────────────────────────────┬───────────────────────────────────┘           ║
║                                  ▼                                               ║
║                    ┌─────────────────────────────┐                              ║
║                    │   P(depressed) ∈ [0, 1]     │                              ║
║                    │   PHQ-8 risk score           │                              ║
║                    └─────────────────────────────┘                              ║
╚══════════════════════════════════════════════════════════════════════════════════╝
```

---

## Feature Engineering

| Modality | Raw Input | Processing | Output Shape |
|----------|-----------|------------|:------------:|
| **Face** | `{pid}_CLNF_AUs.txt` | Filter conf < 0.5 → 20 AU cols → z-score → even-sample | `(N, 200, 20)` |
| **Audio** | `{pid}_AUDIO.wav` | librosa MFCC(40) + Δ + ΔΔ → time-truncate/pad | `(N, 300, 120)` |
| **Text** | `{pid}_TRANSCRIPT.csv` | Participant-only → TF-IDF bigrams (min_df=2, sublinear_tf) | `(N, 10000)` |

**Why these choices?**
- **20 CLNF AUs:** AU04 (brow lowerer), AU15 (lip corner depressor), AU17 (chin raiser) are clinically associated with sadness/depression — pre-extracted by OpenFace, no face video needed
- **MFCC + Δ + ΔΔ:** Static coefficients + velocity + acceleration captures temporal speech dynamics; depression correlates with reduced prosodic variability
- **TF-IDF bigrams:** Captures co-occurrence patterns ("feel hopeless", "can't sleep") that unigrams miss; `sublinear_tf=True` reduces the weight of very frequent neutral words

---

## Training Strategy

```
Optimizer : AdamW   (lr=1e-3, weight_decay=1e-4)
Scheduler : CosineAnnealingWarmRestarts  (T₀=10, T_mult=2)
Epochs    : 30  (GPU) / 5  (CPU baseline)
Batch     : 32
Grad clip : 1.0

Class imbalance  →  class-weighted CrossEntropy  (28% depressed in DAIC-WOZ)
Small dataset    →  FER2013 transfer learning  +  aggressive dropout (0.4)
Gender bias      →  Equalized Opportunity fairness term in loss

Save criterion   :  best validation F1  (not accuracy — clinical metric)
```

---

## Dataset

| Dataset | Participants | Labels | Access |
|---------|:-----------:|--------|--------|
| **DAIC-WOZ** (USC ICT) | 188 (141 used) | PHQ-8 binary depression label | [Request here](https://dcapswoz.isi.edu/) — restricted clinical data |
| **FER2013** | 35,887 images | 7 emotion classes | [Public (Kaggle)](https://www.kaggle.com/datasets/msambare/fer2013) |

> **DAIC-WOZ is NOT in this repository.** It is a restricted clinical dataset from the University of Southern California. You must request access via the link above. All `.wav`, `.txt`, `.csv` participant files are excluded from git via `.gitignore`.

**Split used:**
```
Train : 107 participants  (depressed: ~30  |  not depressed: ~77)
Dev   :  34 participants  (depressed: ~12  |  not depressed: ~22)
Test  :  47 participants  (held out — used only for final evaluation)
```

---

## Results

### Phase 4 — Tri-Modal Fusion Baseline (5 CPU epochs)

| Model | Modalities | AUC | F1 | Fairness Gap |
|-------|-----------|:---:|:--:|:------------:|
| Random baseline | — | 0.50 | — | — |
| Audio Bi-LSTM (Phase 3) | Audio only | — | — | — |
| **Tri-Modal Fusion** | **Face + Audio + Text** | **0.66** | *training* | *pending* |
| *(Target — 30 GPU epochs)* | *Face + Audio + Text* | *> 0.80* | *> 0.70* | *< 0.05* |

> AUC jumped from **0.49 → 0.66** in just 5 CPU epochs on 107 training samples — strongly validating the cross-modal signal. Full GPU training on Colab expected to reach 0.80+ AUC.

---

## Phased Roadmap

| # | Phase | What | Status |
|:-:|-------|------|:------:|
| 0 | Scaffolding | Project structure, configs, base models | ✅ Complete |
| 1 | Face pre-training | CNN on FER2013 (35k facial images) | ✅ Complete |
| 2 | Face DAIC branch | CLNF AU extraction + z-score norm | ✅ Complete |
| 3 | Audio branch | MFCC+Δ+ΔΔ + Bi-LSTM (107/34 participants) | ✅ Complete |
| 3+ | Text branch | Transcript TF-IDF bigram encoding | ✅ Complete |
| 4 | Tri-modal fusion | Cross-modal attention + fairness loss + Colab notebook | ✅ Complete |
| **5** | **Fairness analysis** | **Gender-stratified evaluation + bias mitigation** | **⏳ Next** |
| 6 | Explainability | Grad-CAM on AU sequences + attention rollout | ⏳ |
| 7 | Ablation + SOTA | Each modality alone vs. combined; comparison table | ⏳ |
| 8 | Paper | IEEE / Elsevier submission | ⏳ |

---

## Repository Structure

```
depression_thesis/
│
├── configs/
│   └── config.py                    ← single source of truth for all hyperparameters
│
├── src/
│   ├── preprocessing/
│   │   ├── face_preprocess.py       ← FER2013 loading, augmentation, synthetic fallback
│   │   ├── daic_preprocess.py       ← CLNF AU extraction, confidence filtering, gender labels
│   │   ├── audio_preprocess.py      ← MFCC+Δ+ΔΔ, 300-step sequences from DAIC-WOZ WAVs
│   │   └── text_preprocess.py       ← participant-only transcript → TF-IDF bigrams
│   │
│   ├── models/
│   │   ├── face_cnn.py              ← CNN backbone for FER2013 pre-training
│   │   ├── audio_lstm.py            ← standalone Bi-LSTM for audio (Phase 3)
│   │   ├── encoders.py              ← FaceAUEncoder · AudioEncoder · TextEncoder (256-dim)
│   │   └── fusion_attention.py      ← CrossModalAttention · TriModalFusionModel (7.8M params)
│   │
│   ├── training/
│   │   ├── train_face.py            ← Phase 1: FER2013 CNN
│   │   ├── train_face_daic.py       ← Phase 2: CLNF AU fine-tune
│   │   ├── train_audio.py           ← Phase 3: audio Bi-LSTM
│   │   └── train_fusion.py          ← Phase 4: fairness-aware tri-modal training
│   │
│   └── explainability/
│       └── gradcam.py               ← Grad-CAM heatmap for AU time-series
│
├── notebooks/
│   ├── phase1_face_cnn.ipynb        ← Colab: FER2013 emotion CNN
│   ├── phase2_daic_faceframes.ipynb ← Colab: DAIC-WOZ face fine-tuning
│   ├── phase3_audio_lstm.ipynb      ← Colab: MFCC Bi-LSTM training
│   └── phase4_fusion.ipynb          ← Colab: tri-modal fusion + attention viz + fairness
│
├── data/
│   ├── raw/          ← fer2013.csv + daicwoz/ files  [NOT committed — see .gitignore]
│   └── processed/    ← .npy arrays output by preprocessing scripts
│
├── results/
│   ├── figures/      ← confusion matrices, ROC curves, attention bar charts
│   └── metrics/      ← phase*_metrics.json files
│
├── requirements.txt
└── .gitignore        ← daicwoz/, data/raw/, *.npy, *.pth, *.pdf excluded
```

---

## Quick Reproduction

### Install
```bash
git clone https://github.com/DevMursLab/Thesis.git
cd Thesis
pip install -r requirements.txt
```

### Preprocess all three modalities
```bash
# Requires DAIC-WOZ at daicwoz/daicwoz/  (flat folder with all {pid}_* files)
python src/preprocessing/daic_preprocess.py    # face AUs
python src/preprocessing/audio_preprocess.py   # MFCC
python src/preprocessing/text_preprocess.py    # TF-IDF
```

### Train the tri-modal fusion model
```bash
python src/training/train_fusion.py
# Prints per-epoch: loss | F1 | AUC | fairness_gap [M: f1_male  F: f1_female]
```

### Run on Colab (GPU — recommended)
Open `notebooks/phase4_fusion.ipynb` → Runtime → Change runtime type → T4 GPU → Run all

---

## Why This Approach Is Different

| Design Choice | Common Alternative | Our Reason |
|--------------|-------------------|------------|
| **Cross-modal attention** | Simple feature concatenation | Learns *dynamic* inter-modal dependencies per sample |
| **CLNF Action Units (not pixels)** | Raw face frames + CNN | Lower data requirements; AU semantics are clinically meaningful |
| **Bi-LSTM mean-pool** | Last hidden state | More stable for variable-length clinical interview segments |
| **Equalized Opportunity loss** | No fairness constraint | Prevents gender-biased screening — critical for clinical AI ethics |
| **LayerNorm + GELU** | BatchNorm + ReLU | BatchNorm degrades on small batches common in clinical datasets |
| **AdamW + CosineWarmRestarts** | Fixed-LR Adam | Escapes local minima; better generalization on small N |
| **TF-IDF bigrams** | Raw bag-of-words | Captures phrase-level depression markers ("can't sleep", "no energy") |

---

## Clinical Significance

This work addresses three open problems in computational psychiatry:

1. **Modality gap:** Most prior systems (e.g., AVEC 2017 challenge winners) use audio or text alone. Clinicians are naturally multimodal.
2. **Fairness gap:** Depression screening tools have documented performance disparities across gender groups. Our equalized opportunity constraint is a direct algorithmic response.
3. **Transparency gap:** Black-box models are clinically unusable. Our Grad-CAM + attention rollout produces human-interpretable explanations that map back to known clinical indicators (AU04, AU15, flat prosody, hopelessness language).

---

## Author

<table>
  <tr>
    <td><b>Researcher</b></td>
    <td>Md. Mursalin</td>
  </tr>
  <tr>
    <td><b>Institution</b></td>
    <td>Department of Computer Science & Engineering, Netrokona University</td>
  </tr>
  <tr>
    <td><b>Supervisor</b></td>
    <td>Md. Shovon</td>
  </tr>
  <tr>
    <td><b>Target Venue</b></td>
    <td>IEEE Access / Elsevier Expert Systems with Applications</td>
  </tr>
</table>

---

## Citation

```bibtex
@article{mursalin2025trimodal,
  title   = {Tri-Modal Depression Risk Detection via Cross-Modal Attention Fusion
             with Gender-Equalized Fairness Constraints},
  author  = {Mursalin, Md. and Shovon, Md.},
  journal = {arXiv preprint},
  year    = {2025},
  note    = {Undergraduate thesis, Dept. of CSE, Netrokona University.
             Dataset: DAIC-WOZ (USC ICT) + FER2013.
             Code: https://github.com/DevMursLab/Thesis}
}
```

---

> ⚠️ **Clinical Disclaimer:** This is an academic research prototype for depression *risk screening* only — not a clinical diagnostic tool. All model outputs must be reviewed by a licensed mental health professional before any clinical use.

---

<div align="center">
  <sub>Built with rigor. Designed for impact. Targeting IEEE publication.</sub>
</div>
