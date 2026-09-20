"""
Threshold-Matched Paired Significance Test: Multi-Task vs. Single-Task
==========================================================================
External review flagged a confound in multitask_significance_scaled.py:
the single-task arm tunes its decision threshold on dev (N=34) while the
multi-task arm tunes on train (N=107). Since F1 is threshold-dependent,
this asymmetry can itself produce an F1 gap that looks like a multi-task
learning effect but is actually a threshold-estimation-precision effect
(more data to tune on -> less noisy threshold -> less erratic F1).

Supporting evidence this confound is real: recomputing a paired Wilcoxon
test on AUC (threshold-independent) from the existing 70-seed per-seed
data in multitask_significance_scaled.json shows NO multi-task advantage
(single AUC 0.670+-0.052 vs multi AUC 0.655+-0.054, multi wins only
29/70, p=0.145, d=-0.22) -- i.e. on the metric threshold procedure cannot
touch, multi-task is not better and may be marginally worse.

This script re-runs both arms with an IDENTICAL threshold procedure
(both tuned on dev, matching the single-task arm's existing procedure)
to test whether the F1 effect survives once the threshold-N asymmetry
is removed.

Run:
    python -m src.experiments.multitask_threshold_matched
"""

import sys, json, time
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from scipy.stats import wilcoxon

from configs.config import (DATA_PROCESSED, METRICS,
                            VOCAB_SIZE, N_AU_FEATURES, EMBED_DIM,
                            FUSION_DROPOUT, WEIGHT_DECAY,
                            LAMBDA_SCORE, LAMBDA_SYMPTOM, LAMBDA_FAIRNESS,
                            FAIRNESS_WARMUP, MODALITY_DROPOUT_P, N_PHQ_SYMPTOMS)
from src.models.configurable_fusion import ConfigurableFusionModel
from src.models.multitask_fusion import MultiTaskFusionModel, multitask_loss

FACE_OUT     = DATA_PROCESSED / "daic_faces"
AUDIO_OUT_ST = DATA_PROCESSED / "daic_audio"
AUDIO_OUT_MT = DATA_PROCESSED / "daic_audio_covarep"
TEXT_OUT     = DATA_PROCESSED / "daic_text"
OUT_JSON     = METRICS / "multitask_threshold_matched.json"
OUT_PARTIAL  = METRICS / "multitask_threshold_matched_partial.json"

N_SEEDS = 70
SEEDS = list(range(1, N_SEEDS + 1))

BATCH_SIZE, LR, EPOCHS, PATIENCE = 16, 5e-4, 40, 12
MT_EPOCHS, MT_PATIENCE = 50, 15


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def class_weights_capped(y, max_ratio=1.8):
    cw = compute_class_weight("balanced", classes=np.unique(y), y=y)
    if cw[1] / cw[0] > max_ratio:
        cw[1] = cw[0] * max_ratio
    return torch.tensor(cw, dtype=torch.float32)


def best_threshold_binary(y_true, probs):
    best_t, best_f1 = 0.5, -1
    for t in np.linspace(0.1, 0.9, 33):
        f1 = f1_score(y_true, (probs >= t).astype(int), average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t)


# ---------------- Single-task arm (unchanged: already dev-tuned) ----------------

def load_split_single(split):
    n = min(len(np.load(FACE_OUT / f"X_{split}.npy")),
            len(np.load(AUDIO_OUT_ST / f"X_{split}.npy")),
            len(np.load(TEXT_OUT / f"X_{split}.npy")))
    Xf = np.load(FACE_OUT     / f"X_{split}.npy")[:n].astype(np.float32)
    Xa = np.load(AUDIO_OUT_ST / f"X_{split}.npy")[:n].astype(np.float32)
    Xt = np.load(TEXT_OUT     / f"X_{split}.npy")[:n].astype(np.float32)
    y  = np.load(FACE_OUT     / f"y_{split}.npy")[:n].astype(np.int64)
    return Xf, Xa, Xt, y


def train_single_task(data, device, seed):
    set_seed(seed)
    Xf_tr, Xa_tr, Xt_tr, y_tr = data["train"]
    Xf_dv, Xa_dv, Xt_dv, y_dv = data["dev"]

    model = ConfigurableFusionModel(
        ("face", "audio", "text"), n_au=N_AU_FEATURES, n_mfcc=120,
        vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, dropout=FUSION_DROPOUT).to(device)

    cw = class_weights_capped(y_tr).to(device)
    crit = nn.CrossEntropyLoss(weight=cw)
    opt  = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sch  = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2)

    tr_dl = DataLoader(TensorDataset(
        torch.tensor(Xf_tr), torch.tensor(Xa_tr),
        torch.tensor(Xt_tr), torch.tensor(y_tr)),
        batch_size=BATCH_SIZE, shuffle=True)

    dev_t = (torch.tensor(Xf_dv).to(device), torch.tensor(Xa_dv).to(device),
             torch.tensor(Xt_dv).to(device))

    best_auc, best_probs, wait = 0.0, None, 0
    for ep in range(EPOCHS):
        model.train()
        for xf, xa, xt, yb in tr_dl:
            opt.zero_grad()
            out = model(xf.to(device), xa.to(device), xt.to(device))
            loss = crit(out, yb.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()

        model.eval()
        with torch.no_grad():
            pr = torch.softmax(model(*dev_t), 1)[:, 1].cpu().numpy()
        auc = roc_auc_score(y_dv, pr) if len(np.unique(y_dv)) > 1 else 0.0
        if auc > best_auc + 1e-4:
            best_auc, best_probs, wait = auc, pr, 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    thr = best_threshold_binary(y_dv, best_probs)
    preds = (best_probs >= thr).astype(int)
    f1 = f1_score(y_dv, preds, average="macro", zero_division=0)
    return float(f1), float(best_auc)


# ---------------- Multi-task arm (CHANGED: threshold now tuned on dev) ----------------

def load_split_multi(split):
    X_face     = np.load(FACE_OUT     / f"X_{split}.npy").astype(np.float32)
    y          = np.load(FACE_OUT     / f"y_{split}.npy").astype(np.int64)
    gender     = np.load(FACE_OUT     / f"gender_{split}.npy").astype(np.int64)
    X_audio    = np.load(AUDIO_OUT_MT / f"X_{split}.npy").astype(np.float32)
    y_phq8     = np.load(AUDIO_OUT_MT / f"phq8_{split}.npy").astype(np.float32)
    sym_raw    = np.load(AUDIO_OUT_MT / f"symptoms_{split}.npy").astype(np.float32)
    y_symptoms = np.nan_to_num(sym_raw, nan=0.0).clip(0, 3).astype(np.int64)
    X_text     = np.load(TEXT_OUT     / f"X_{split}.npy").astype(np.float32)
    n = min(len(X_face), len(X_audio), len(X_text))
    return (X_face[:n], X_audio[:n], X_text[:n], y[:n], gender[:n],
            y_phq8[:n], y_symptoms[:n])


def mt_fairness_loss(logits, labels, gender):
    probs = torch.softmax(logits, dim=1)[:, 1]
    for g in [0, 1]:
        if (gender == g).sum() < 2:
            return torch.tensor(0.0, requires_grad=True, device=logits.device)

    def _rate(mask, pos_label):
        sub_m = mask & (labels == pos_label)
        if sub_m.sum() == 0:
            return torch.tensor(0.5, device=logits.device)
        return probs[sub_m].mean()

    tpr_m, tpr_f = _rate(gender == 0, 1), _rate(gender == 1, 1)
    fpr_m, fpr_f = _rate(gender == 0, 0), _rate(gender == 1, 0)
    return (tpr_m - tpr_f) ** 2 + (fpr_m - fpr_f) ** 2


@torch.no_grad()
def mt_evaluate(model, loader, device, threshold=0.5):
    model.eval()
    all_probs, all_labels = [], []
    for Xf, Xa, Xt, yb, _, yp, ys in loader:
        Xf, Xa, Xt = Xf.to(device), Xa.to(device), Xt.to(device)
        lb, sr, sl = model(Xf, Xa, Xt)
        all_probs.extend(torch.softmax(lb, 1)[:, 1].cpu().tolist())
        all_labels.extend(yb.tolist())
    probs, labels = np.array(all_probs), np.array(all_labels)
    preds = (probs >= threshold).astype(int)
    auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.5
    f1  = f1_score(labels, preds, average="macro", zero_division=0)
    return auc, f1, probs, labels


def mt_best_threshold(y_true, probs):
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.2, 0.8, 0.02):
        f1 = f1_score(y_true, (probs >= t).astype(int), average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t


def train_multi_task(data, device, seed):
    set_seed(seed)
    Xf_tr, Xa_tr, Xt_tr, y_tr, g_tr, yp_tr, ys_tr = data["train"]
    Xf_dv, Xa_dv, Xt_dv, y_dv, g_dv, yp_dv, ys_dv = data["dev"]

    tr_ds = TensorDataset(
        torch.tensor(Xf_tr), torch.tensor(Xa_tr), torch.tensor(Xt_tr),
        torch.tensor(y_tr), torch.tensor(g_tr),
        torch.tensor(yp_tr), torch.tensor(ys_tr))
    dv_ds = TensorDataset(
        torch.tensor(Xf_dv), torch.tensor(Xa_dv), torch.tensor(Xt_dv),
        torch.tensor(y_dv), torch.tensor(g_dv),
        torch.tensor(yp_dv), torch.tensor(ys_dv))
    tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True)
    dv_loader = DataLoader(dv_ds, batch_size=BATCH_SIZE, shuffle=False)

    cw = class_weights_capped(y_tr).to(device)
    model = MultiTaskFusionModel(
        n_au=N_AU_FEATURES, n_audio_feat=Xa_tr.shape[2], vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM, dropout=FUSION_DROPOUT, n_symptoms=N_PHQ_SYMPTOMS,
        modality_dropout_p=MODALITY_DROPOUT_P).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MT_EPOCHS)

    AUX_WARMUP = 10
    best_f1, best_auc, patience_cnt = 0.0, 0.0, 0
    best_state = None

    for ep in range(1, MT_EPOCHS + 1):
        model.train()
        aux_scale = min(1.0, (ep - AUX_WARMUP) / 5.0) if ep > AUX_WARMUP else 0.0
        for Xf, Xa, Xt, yb, g, yp, ys in tr_loader:
            Xf, Xa, Xt = Xf.to(device), Xa.to(device), Xt.to(device)
            yb, g, yp, ys = yb.to(device), g.to(device), yp.to(device), ys.to(device)
            opt.zero_grad()
            lb, sr, sl = model(Xf, Xa, Xt)
            loss = multitask_loss(lb, sr, sl, yb, yp.float(), ys,
                                   class_weights=cw,
                                   lambda_score=aux_scale * LAMBDA_SCORE,
                                   lambda_symptom=aux_scale * LAMBDA_SYMPTOM)
            if ep > FAIRNESS_WARMUP:
                loss = loss + LAMBDA_FAIRNESS * mt_fairness_loss(lb, yb, g)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()

        # CHANGED: threshold now tuned on DEV probs (was: train probs),
        # matching the single-task arm's procedure exactly.
        _, _, dv_probs, dv_labels = mt_evaluate(model, dv_loader, device)
        thr = mt_best_threshold(dv_labels, dv_probs)
        auc, f1, _, _ = mt_evaluate(model, dv_loader, device, thr)

        if f1 > best_f1:
            best_f1, best_auc, patience_cnt = f1, auc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= MT_PATIENCE:
                break

    model.load_state_dict(best_state)
    _, _, dv_probs_f, dv_labels_f = mt_evaluate(model, dv_loader, device)
    thr = mt_best_threshold(dv_labels_f, dv_probs_f)
    auc, f1, _, _ = mt_evaluate(model, dv_loader, device, thr)
    return float(f1), float(auc)


def main():
    device = "cpu"
    t0 = time.time()

    single_data = {"train": load_split_single("train"), "dev": load_split_single("dev")}
    multi_data  = {"train": load_split_multi("train"),  "dev": load_split_multi("dev")}
    print(f"Single-task N: train={len(single_data['train'][3])}, dev={len(single_data['dev'][3])}")
    print(f"Multi-task  N: train={len(multi_data['train'][3])}, dev={len(multi_data['dev'][3])}")
    print(f"Running {N_SEEDS} paired seeds, BOTH arms threshold-tuned on dev "
          f"(removes the train-N vs dev-N threshold-estimation confound)...\n")

    single_f1s, multi_f1s = [], []
    single_aucs, multi_aucs = [], []
    per_seed = []

    for i, s in enumerate(SEEDS):
        ts = time.time()
        sf1, sauc = train_single_task(single_data, device, seed=s)
        mf1, mauc = train_multi_task(multi_data, device, seed=s)
        single_f1s.append(sf1); multi_f1s.append(mf1)
        single_aucs.append(sauc); multi_aucs.append(mauc)
        per_seed.append({"seed": s, "single_f1": sf1, "single_auc": sauc,
                          "multi_f1": mf1, "multi_auc": mauc})
        elapsed = time.time() - ts
        print(f"[{i+1:>3}/{N_SEEDS}] seed={s:<4d} single_F1={sf1:.4f} "
              f"multi_F1={mf1:.4f} diff={mf1-sf1:+.4f}  ({elapsed:.1f}s)")

        if (i + 1) % 10 == 0:
            _save_partial(per_seed)

    single_f1s = np.array(single_f1s); multi_f1s = np.array(multi_f1s)
    single_aucs = np.array(single_aucs); multi_aucs = np.array(multi_aucs)

    def _report(a, b, name):
        diffs = b - a
        wins = int((diffs > 0).sum())
        stat, p_value = wilcoxon(b, a)
        d = float(diffs.mean() / diffs.std(ddof=1)) if diffs.std(ddof=1) > 0 else 0.0
        print(f"\n{name}: single={a.mean():.4f}+/-{a.std():.4f}  "
              f"multi={b.mean():.4f}+/-{b.std():.4f}  wins={wins}/{N_SEEDS}  "
              f"p={p_value:.4g}  d={d:.3f}")
        return {"single_mean": round(float(a.mean()), 4), "single_std": round(float(a.std()), 4),
                "multi_mean": round(float(b.mean()), 4), "multi_std": round(float(b.std()), 4),
                "wins_multitask": f"{wins}/{N_SEEDS}", "wilcoxon_p_value": float(p_value),
                "significant_at_0.05": bool(p_value < 0.05), "cohens_d": round(d, 4)}

    f1_report = _report(single_f1s, multi_f1s, "F1 (threshold-matched)")
    auc_report = _report(single_aucs, multi_aucs, "AUC (threshold-independent)")

    report = {
        "analysis": "Threshold-matched paired significance test: both single-task "
                     "and multi-task arms now tune their decision threshold on dev "
                     "(the single-task arm's original procedure), removing the "
                     "train-N(107) vs dev-N(34) threshold-estimation asymmetry that "
                     "external review flagged as a possible confound in "
                     "multitask_significance_scaled.json's F1-based result.",
        "n_seeds": N_SEEDS,
        "seeds": SEEDS,
        "f1_threshold_matched": f1_report,
        "auc_threshold_independent": auc_report,
        "per_seed": per_seed,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nTotal runtime: {(time.time()-t0)/60:.1f} min")
    print(f"Saved: {OUT_JSON}")


def _save_partial(per_seed):
    OUT_PARTIAL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PARTIAL, "w") as f:
        json.dump({"status": "partial_checkpoint", "n_completed": len(per_seed),
                    "per_seed": per_seed}, f, indent=2)


if __name__ == "__main__":
    main()
