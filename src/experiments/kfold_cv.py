"""
K-Fold Cross-Validation on Full DAIC-WOZ Pool (N=188)
=======================================================
The paper's headline metrics come from a single fixed split
(train=107 / dev=34 / test=47). Reviewers rightly flag N=34 as
underpowered for the ablation claims. This experiment combines ALL
188 labeled participants (train+dev+test) into one pool and runs
stratified k-fold cross-validation, giving a much larger effective
held-out sample across folds than any single fixed split.

Test-origin participants (N=47) only have binary label + PHQ-8 total
score + gender (see full_test_split.csv) -- no per-symptom items.
We therefore mask the symptom-classification loss to zero for those
samples (valid_symptom=0); only train/dev-origin samples (N=141)
contribute to that auxiliary term. All 188 samples contribute to the
binary and PHQ-8 score losses.

Run:
    python -m src.experiments.kfold_cv
"""

import sys, json, warnings, pickle
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

from configs.config import (
    DATA_PROCESSED, DAIC_RAW,
    AU_COLS, AU_TIME_STEPS,
    N_MFCC, N_COVAREP, N_FORMANT, AUDIO_FEAT_DIM, AUDIO_TIME_STEPS,
    VOCAB_SIZE, EMBED_DIM, FUSION_DROPOUT, WEIGHT_DECAY,
    LAMBDA_SCORE, LAMBDA_SYMPTOM, MODALITY_DROPOUT_P, N_PHQ_SYMPTOMS,
)
from src.models.multitask_fusion import MultiTaskFusionModel
from src.preprocessing.covarep_preprocess import (
    extract_mfcc_stream, extract_covarep_stream, extract_formant_stream,
)

ROOT        = Path(__file__).resolve().parent.parent.parent
TEST_CSV    = ROOT / "full_test_split.csv"
FACE_DIR    = DATA_PROCESSED / "daic_faces"
AUDIO_DIR   = DATA_PROCESSED / "daic_audio_covarep"
TEXT_DIR    = DATA_PROCESSED / "daic_text"
VECTORIZER  = TEXT_DIR / "vectorizer.pkl"
CACHE_DIR   = DATA_PROCESSED / "daic_test_raw_cache"
OUT_JSON    = ROOT / "results" / "metrics" / "kfold_cv.json"

BATCH_SIZE   = 16
LR           = 5e-4
EPOCHS       = 50
PATIENCE     = 15
MAX_CW_RATIO = 1.8
N_SPLITS     = 5
CV_SEEDS     = [42, 7]     # 2 independent 5-fold partitions = 10 fold-runs total
VAL_FRAC     = 0.15        # internal early-stopping slice carved from each training fold


# ── Test-set feature extraction (cached) ────────────────────────────────────

def extract_au(pid: int):
    au_file = DAIC_RAW / f"{pid}_CLNF_AUs.txt"
    if not au_file.exists():
        return None
    df = pd.read_csv(au_file, sep=",", skipinitialspace=True)
    df.columns = df.columns.str.strip()
    df = df[df["confidence"] >= 0.5].reset_index(drop=True)
    if len(df) < 10 or any(c not in df.columns for c in AU_COLS):
        return None
    feat = df[AU_COLS].values.astype(np.float32)
    mean = feat.mean(axis=0, keepdims=True)
    std  = feat.std(axis=0, keepdims=True) + 1e-8
    feat = (feat - mean) / std
    T = feat.shape[0]
    if T >= AU_TIME_STEPS:
        idx  = np.linspace(0, T - 1, AU_TIME_STEPS, dtype=int)
        feat = feat[idx]
    else:
        pad  = np.zeros((AU_TIME_STEPS - T, feat.shape[1]), dtype=np.float32)
        feat = np.concatenate([feat, pad], axis=0)
    return feat


def extract_audio(pid: int):
    audio_path = DAIC_RAW / f"{pid}_AUDIO.wav"
    mfcc = extract_mfcc_stream(audio_path) if audio_path.exists() \
           else np.zeros((AUDIO_TIME_STEPS, N_MFCC * 3), dtype=np.float32)
    cov  = extract_covarep_stream(pid)
    fmt  = extract_formant_stream(pid)
    return np.concatenate([mfcc, cov, fmt], axis=1)


def extract_text(pid: int, vectorizer):
    transcript = DAIC_RAW / f"{pid}_TRANSCRIPT.csv"
    if not transcript.exists():
        return np.zeros(VOCAB_SIZE, dtype=np.float32)
    df = pd.read_csv(transcript, sep="\t")
    df.columns = df.columns.str.strip()
    rows = df[df["speaker"].str.strip() == "Participant"]
    if len(rows) == 0:
        return np.zeros(VOCAB_SIZE, dtype=np.float32)
    text = " ".join(rows["value"].dropna().astype(str).tolist()).lower().strip()
    if len(text) < 10:
        return np.zeros(VOCAB_SIZE, dtype=np.float32)
    return vectorizer.transform([text]).toarray().astype(np.float32)[0]


def load_or_build_test_arrays():
    """Extract (or load cached) raw feature arrays for the 47 test participants."""
    cache_ok = all((CACHE_DIR / f"{n}.npy").exists() for n in
                   ["X_face", "X_audio", "X_text", "y", "gender", "phq8"])
    if cache_ok:
        return tuple(np.load(CACHE_DIR / f"{n}.npy", allow_pickle=False)
                     for n in ["X_face", "X_audio", "X_text", "y", "gender", "phq8"])

    print("Extracting test-set raw features (first run, will cache)...")
    df_test = pd.read_csv(TEST_CSV)
    df_test.columns = df_test.columns.str.strip()
    df_test = df_test.rename(columns={
        "PHQ_Binary": "label", "PHQ8_Binary": "label",
        "PHQ_Score": "phq_score", "PHQ8_Score": "phq_score",
    })
    with open(VECTORIZER, "rb") as f:
        vectorizer = pickle.load(f)

    Xf, Xa, Xt, y, gender, phq8 = [], [], [], [], [], []
    for _, row in df_test.iterrows():
        pid = int(row["Participant_ID"])
        au = extract_au(pid)
        if au is None:
            print(f"  [SKIP] {pid}")
            continue
        Xf.append(au)
        Xa.append(extract_audio(pid))
        Xt.append(extract_text(pid, vectorizer))
        y.append(int(row["label"]))
        gender.append(int(row["Gender"]))
        phq8.append(float(row["phq_score"]))

    arrays = {
        "X_face": np.stack(Xf).astype(np.float32),
        "X_audio": np.stack(Xa).astype(np.float32),
        "X_text": np.stack(Xt).astype(np.float32),
        "y": np.array(y, dtype=np.int64),
        "gender": np.array(gender, dtype=np.int64),
        "phq8": np.array(phq8, dtype=np.float32),
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for name, arr in arrays.items():
        np.save(CACHE_DIR / f"{name}.npy", arr)
    return (arrays["X_face"], arrays["X_audio"], arrays["X_text"],
            arrays["y"], arrays["gender"], arrays["phq8"])


# ── Combine train + dev + test into one pool ────────────────────────────────

def build_full_pool():
    def load_split(split):
        X_face  = np.load(FACE_DIR  / f"X_{split}.npy").astype(np.float32)
        y       = np.load(FACE_DIR  / f"y_{split}.npy").astype(np.int64)
        gender  = np.load(FACE_DIR  / f"gender_{split}.npy").astype(np.int64)
        X_audio = np.load(AUDIO_DIR / f"X_{split}.npy").astype(np.float32)
        sym_raw = np.load(AUDIO_DIR / f"symptoms_{split}.npy").astype(np.float32)
        symptoms = np.nan_to_num(sym_raw, nan=0.0).clip(0, 3).astype(np.int64)
        phq8    = np.load(AUDIO_DIR / f"phq8_{split}.npy").astype(np.float32)
        X_text  = np.load(TEXT_DIR  / f"X_{split}.npy").astype(np.float32)
        n = min(len(X_face), len(X_audio), len(X_text))
        return X_face[:n], X_audio[:n], X_text[:n], y[:n], gender[:n], phq8[:n], symptoms[:n]

    tr = load_split("train")
    dv = load_split("dev")
    Xf_t, Xa_t, Xt_t, y_t, g_t, phq_t = load_or_build_test_arrays()
    # test set has no symptom items -> zero-fill + invalid mask
    n_test = len(y_t)
    sym_t = np.zeros((n_test, N_PHQ_SYMPTOMS), dtype=np.int64)

    X_face  = np.concatenate([tr[0], dv[0], Xf_t], axis=0)
    X_audio = np.concatenate([tr[1], dv[1], Xa_t], axis=0)
    X_text  = np.concatenate([tr[2], dv[2], Xt_t], axis=0)
    y       = np.concatenate([tr[3], dv[3], y_t], axis=0)
    gender  = np.concatenate([tr[4], dv[4], g_t], axis=0)
    phq8    = np.concatenate([tr[5], dv[5], phq_t], axis=0)
    symptoms = np.concatenate([tr[6], dv[6], sym_t], axis=0)
    valid_symptom = np.concatenate([
        np.ones(len(tr[3]), dtype=np.float32),
        np.ones(len(dv[3]), dtype=np.float32),
        np.zeros(n_test, dtype=np.float32),
    ])

    print(f"Full pool: {len(y)} participants "
          f"(train={len(tr[3])}, dev={len(dv[3])}, test={n_test})")
    print(f"Depressed: {(y==1).sum()} / {len(y)} ({100*(y==1).mean():.1f}%)")
    return X_face, X_audio, X_text, y, gender, phq8, symptoms, valid_symptom


# ── Training utilities ───────────────────────────────────────────────────────

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def compute_class_weights_capped(y, max_ratio=MAX_CW_RATIO):
    cw = compute_class_weight("balanced", classes=np.unique(y), y=y)
    if cw[1] / cw[0] > max_ratio:
        cw[1] = cw[0] * max_ratio
    return torch.tensor(cw, dtype=torch.float32)


def masked_multitask_loss(logits_binary, score_raw, symptom_logits,
                           y_binary, y_score, y_symptoms, valid_symptom,
                           class_weights=None, lambda_score=0.3, lambda_symptom=0.2):
    l_binary = F.cross_entropy(logits_binary, y_binary, weight=class_weights)

    score_target = y_score / 24.0
    l_score = F.mse_loss(score_raw, score_target)

    if valid_symptom.sum() > 0:
        B = symptom_logits.size(0)
        per_sample = F.cross_entropy(
            symptom_logits.view(B * 8, 4),
            y_symptoms.view(B * 8).long(),
            reduction="none",
        ).view(B, 8).mean(dim=1)
        l_symptom = (per_sample * valid_symptom).sum() / valid_symptom.sum()
    else:
        l_symptom = torch.tensor(0.0, device=logits_binary.device)

    return l_binary + lambda_score * l_score + lambda_symptom * l_symptom


def best_threshold(y_true, probs):
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.2, 0.8, 0.02):
        f1 = f1_score(y_true, (probs >= t).astype(int), average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t


@torch.no_grad()
def evaluate(model, loader, threshold=0.5):
    model.eval()
    probs_all, labels_all = [], []
    for Xf, Xa, Xt, yb, *_ in loader:
        lb, _, _ = model(Xf, Xa, Xt)
        probs_all.extend(torch.softmax(lb, 1)[:, 1].tolist())
        labels_all.extend(yb.tolist())
    probs, labels = np.array(probs_all), np.array(labels_all)
    preds = (probs >= threshold).astype(int)
    auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.5
    f1  = f1_score(labels, preds, average="macro", zero_division=0)
    acc = accuracy_score(labels, preds)
    return auc, f1, acc, probs, labels


def tpr(y_true, y_pred):
    dep = y_true == 1
    return float("nan") if dep.sum() == 0 else float((y_pred[dep] == 1).sum() / dep.sum())


def train_one_fold(fold_data, fold_seed):
    (Xf_tr, Xa_tr, Xt_tr, y_tr, g_tr, phq_tr, sym_tr, vs_tr,
     Xf_te, Xa_te, Xt_te, y_te, g_te, phq_te, sym_te, vs_te) = fold_data

    set_seed(fold_seed)

    # carve internal validation slice (stratified) for early stopping
    n = len(y_tr)
    rng = np.random.RandomState(fold_seed)
    idx = np.arange(n)
    pos_idx, neg_idx = idx[y_tr == 1], idx[y_tr == 0]
    rng.shuffle(pos_idx); rng.shuffle(neg_idx)
    n_val_pos = max(1, int(len(pos_idx) * VAL_FRAC))
    n_val_neg = max(1, int(len(neg_idx) * VAL_FRAC))
    val_idx = np.concatenate([pos_idx[:n_val_pos], neg_idx[:n_val_neg]])
    trn_idx = np.setdiff1d(idx, val_idx)

    def subset(idx_):
        return (Xf_tr[idx_], Xa_tr[idx_], Xt_tr[idx_], y_tr[idx_],
                g_tr[idx_], phq_tr[idx_], sym_tr[idx_], vs_tr[idx_])

    trn = subset(trn_idx)
    val = subset(val_idx)

    def make_loader(data, shuffle):
        Xf, Xa, Xt, y, g, phq, sym, vs = data
        ds = TensorDataset(
            torch.tensor(Xf), torch.tensor(Xa), torch.tensor(Xt),
            torch.tensor(y), torch.tensor(g),
            torch.tensor(phq), torch.tensor(sym), torch.tensor(vs),
        )
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

    trn_loader = make_loader(trn, True)
    val_loader = make_loader(val, False)
    test_loader = make_loader(
        (Xf_te, Xa_te, Xt_te, y_te, g_te, phq_te, sym_te, vs_te), False)

    class_weights = compute_class_weights_capped(trn[3])

    model = MultiTaskFusionModel(
        n_au=Xf_tr.shape[2], n_audio_feat=Xa_tr.shape[2],
        vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM,
        dropout=FUSION_DROPOUT, n_symptoms=N_PHQ_SYMPTOMS,
        modality_dropout_p=MODALITY_DROPOUT_P,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_f1, best_state, patience_cnt = 0.0, None, 0
    AUX_WARMUP = 10

    for ep in range(1, EPOCHS + 1):
        model.train()
        aux_scale = min(1.0, (ep - AUX_WARMUP) / 5.0) if ep > AUX_WARMUP else 0.0
        for Xf, Xa, Xt, yb, g, phq, sym, vs in trn_loader:
            optimizer.zero_grad()
            lb, sr, sl = model(Xf, Xa, Xt)
            loss = masked_multitask_loss(
                lb, sr, sl, yb, phq, sym, vs,
                class_weights=class_weights,
                lambda_score=aux_scale * LAMBDA_SCORE,
                lambda_symptom=aux_scale * LAMBDA_SYMPTOM,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        _, _, _, tr_probs, tr_labels = evaluate(model, trn_loader)
        tau = best_threshold(tr_labels, tr_probs)
        val_auc, val_f1, val_acc, _, _ = evaluate(model, val_loader, tau)

        if val_f1 > best_f1:
            best_f1, best_state, patience_cnt = val_f1, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                break

    model.load_state_dict(best_state)
    _, _, _, tr_probs_f, tr_labels_f = evaluate(model, trn_loader)
    tau = best_threshold(tr_labels_f, tr_probs_f)
    test_auc, test_f1, test_acc, test_probs, test_labels = evaluate(model, test_loader, tau)

    preds = (test_probs >= tau).astype(int)
    male_mask, female_mask = g_te == 0, g_te == 1
    tpr_m = tpr(y_te[male_mask], preds[male_mask])
    tpr_f = tpr(y_te[female_mask], preds[female_mask])
    gap = abs(tpr_m - tpr_f) if not (np.isnan(tpr_m) or np.isnan(tpr_f)) else float("nan")

    return {"auc": test_auc, "f1": test_f1, "acc": test_acc,
            "tpr_male": tpr_m, "tpr_female": tpr_f, "tpr_gap": gap,
            "n_test": len(y_te), "n_dep": int((y_te == 1).sum())}


def main():
    print("=" * 60)
    print(f"  K-FOLD CV on Full DAIC-WOZ Pool ({N_SPLITS}-fold x {len(CV_SEEDS)} partitions)")
    print("=" * 60)

    X_face, X_audio, X_text, y, gender, phq8, symptoms, valid_symptom = build_full_pool()

    all_folds = []
    for partition_seed in CV_SEEDS:
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=partition_seed)
        for fold_i, (train_idx, test_idx) in enumerate(skf.split(X_face, y)):
            print(f"\n--- Partition seed={partition_seed}  Fold {fold_i+1}/{N_SPLITS} ---")
            fold_data = (
                X_face[train_idx], X_audio[train_idx], X_text[train_idx],
                y[train_idx], gender[train_idx], phq8[train_idx],
                symptoms[train_idx], valid_symptom[train_idx],
                X_face[test_idx], X_audio[test_idx], X_text[test_idx],
                y[test_idx], gender[test_idx], phq8[test_idx],
                symptoms[test_idx], valid_symptom[test_idx],
            )
            result = train_one_fold(fold_data, fold_seed=partition_seed * 100 + fold_i)
            result["partition_seed"] = partition_seed
            result["fold"] = fold_i
            all_folds.append(result)
            print(f"  F1={result['f1']:.3f}  AUC={result['auc']:.3f}  "
                  f"Acc={result['acc']:.3f}  TPR_gap={result['tpr_gap']:.3f}  "
                  f"(n_test={result['n_test']}, n_dep={result['n_dep']})")

    f1s   = [f["f1"] for f in all_folds]
    aucs  = [f["auc"] for f in all_folds]
    accs  = [f["acc"] for f in all_folds]
    gaps  = [f["tpr_gap"] for f in all_folds if not np.isnan(f["tpr_gap"])]

    print(f"\n{'='*60}")
    print(f"  K-FOLD CV SUMMARY  ({len(all_folds)} folds, N=188 total)")
    print(f"{'='*60}")
    print(f"  F1  : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"  AUC : {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"  Acc : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"  TPR gap: {np.mean(gaps):.4f} ± {np.std(gaps):.4f}")

    summary = {
        "n_splits": N_SPLITS,
        "cv_seeds": CV_SEEDS,
        "n_folds_total": len(all_folds),
        "n_pool": int(len(y)),
        "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s)),
        "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
        "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
        "tpr_gap_mean": float(np.mean(gaps)), "tpr_gap_std": float(np.std(gaps)),
        "folds": all_folds,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {OUT_JSON}")
    return summary


if __name__ == "__main__":
    main()
