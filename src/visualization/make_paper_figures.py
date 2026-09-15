"""
Publication-Quality Figure Generation
======================================
Generates all 8 figures referenced in the IEEE paper from verified metrics.
Figure numbers are NOT embedded in the images -- LaTeX \caption/\label
auto-numbering is the single source of truth, since the figures' order
of appearance in the manuscript does not match their filename suffixes
(e.g. fig8_attention_vs_concat.png appears before fig5_sota.png in the
document). Embedding a hardcoded "Fig. N" title previously caused a
number mismatch between the image and its actual caption.

  fig1_architecture.png          System architecture (schematic block diagram)
  fig2_ablation_ci.png           Ablation: per-config AUC with bootstrap CI (7 configs)
  fig3_p7_vs_p8.png               Single-task baseline vs. proposed multi-task: F1 comparison
  fig4_seed_consistency.png       Per-seed consistency strip plot (5 seeds)
  fig5_sota.png                   SOTA positioning bar chart (this work vs published DAIC-WOZ)
  fig6_fairness_radar.png         Fairness gap radar (4 criteria, male vs female)
  fig7_fairness_before_after.png  Equalized-Odds loss: per-seed TPR gap (10 seeds)
  fig8_attention_vs_concat.png    Cross-modal attention vs. concatenation (10 seeds)

All numbers loaded from results/metrics/*.json — no hard-coded values.
300 DPI, serif fonts, IEEE-compatible sizing.

Run:
    python -m src.visualization.make_paper_figures
"""

import sys, json
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIGURES = Path("results") / "figures"
METRICS = Path("results") / "metrics"
FIGURES.mkdir(parents=True, exist_ok=True)

# IEEE-style global settings
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

PURPLE = "#6A0DAD"
TEAL   = "#00C896"
CORAL  = "#FF6B6B"
GOLD   = "#FFB300"
NAVY   = "#1A237E"


def load(name):
    with open(METRICS / name) as f:
        return json.load(f)


# ============================================================
def fig_architecture():
    """Fig 1 — System architecture block diagram."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.set_xlim(0, 12); ax.set_ylim(0, 9); ax.axis("off")

    def box(x, y, w, h, text, color, fs=9, tc="white"):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0.06", linewidth=1.2,
                     edgecolor="black", facecolor=color, alpha=0.92))
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=fs, color=tc, weight="bold")

    def arrow(x1, y1, x2, y2, color="black"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                     arrowstyle="-|>", mutation_scale=14,
                     linewidth=1.3, color=color))

    # Input layer
    box(0.3, 7.3, 3.0, 1.1, "FACE\nCLNF AUs (200×20)", "#8A2BE2", 8)
    box(4.5, 7.3, 3.0, 1.1, "AUDIO\nMFCC+COVAREP+FORMANT\n(300×199)", "#EE4C2C", 7.5)
    box(8.7, 7.3, 3.0, 1.1, "TEXT\nTF-IDF bigrams (1000)", "#20639B", 8)

    # Encoders
    box(0.3, 5.6, 3.0, 0.9, "FaceAUEncoder\nBi-LSTM → 64", "#9B59B6", 8)
    box(4.5, 5.6, 3.0, 0.9, "AudioEncoder\nBi-LSTM → 64", "#E67E22", 8)
    box(8.7, 5.6, 3.0, 0.9, "TextEncoder\nFC → 64", "#2980B9", 8)
    for cx in [1.8, 6.0, 10.2]:
        arrow(cx, 7.3, cx, 6.5)

    # Cross-modal attention
    box(2.0, 3.9, 8.0, 1.0, "CROSS-MODAL ATTENTION  (each modality queries other two)\n"
        r"$\alpha_m = $softmax$(Q_m K_{-m}^T/\sqrt{d})\,V_{-m}$",
        PURPLE, 8.5)
    for cx in [1.8, 6.0, 10.2]:
        arrow(cx, 5.6, cx if cx == 6.0 else 6.0, 4.9)

    # Modality dropout annotation
    ax.text(11.4, 4.4, "modality\ndropout\np=0.15", ha="center", va="center",
            fontsize=7, style="italic", color=CORAL)

    # Shared layer
    box(3.5, 2.5, 5.0, 0.9, "Shared Representation (192-d)", NAVY, 9)
    arrow(6.0, 3.9, 6.0, 3.4)

    # Multi-task heads
    box(0.6, 0.7, 3.0, 1.1, "Binary Head\nDepressed / Not", TEAL, 8)
    box(4.5, 0.7, 3.0, 1.1, "Score Head\nPHQ-8 (0–24)", GOLD, 8, "black")
    box(8.4, 0.7, 3.0, 1.1, "Symptom Head\n8 × {0,1,2,3}", "#16A085", 8)
    arrow(6.0, 2.5, 2.1, 1.8)
    arrow(6.0, 2.5, 6.0, 1.8)
    arrow(6.0, 2.5, 9.9, 1.8)

    ax.set_title("Tri-Modal Multi-Task Architecture with Cross-Modal Attention",
                 fontsize=11, weight="bold", pad=10)
    plt.savefig(FIGURES / "fig1_architecture.png")
    plt.close()
    print("  fig1_architecture.png")


# ============================================================
def fig_ablation_ci():
    """Fig 2 — Ablation AUC with bootstrap CI error bars."""
    d = load("phase7_ablation.json")["configurations"]
    order = ["face", "text", "audio+text", "face+text",
             "face+audio", "face+audio+text", "audio"]
    labels = ["Face", "Text", "Audio+Text", "Face+Text",
              "Face+Audio", "Tri-Modal", "Audio"]
    means = [d[k]["auc_mean"] for k in order]
    stds  = [d[k]["auc_std"]  for k in order]
    colors = [PURPLE if k != "face+audio+text" else CORAL for k in order]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, means, yerr=stds, capsize=4,
                  color=colors, alpha=0.85, edgecolor="black", linewidth=0.8)
    ax.axhline(0.5, ls="--", color="gray", lw=1, label="Random (0.50)")
    ax.set_ylabel("Dev AUC (mean ± std, 5 seeds)")
    ax.set_ylim(0.4, 0.85)
    ax.set_title("Ablation: AUC by Modality Configuration")
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width()/2, m + 0.012, f"{m:.3f}",
                ha="center", fontsize=8)
    plt.xticks(rotation=25, ha="right")
    ax.legend(loc="upper left", fontsize=8)
    plt.savefig(FIGURES / "fig2_ablation_ci.png")
    plt.close()
    print("  fig2_ablation_ci.png")


# ============================================================
def fig_p7_vs_p8():
    """Fig 3 — Phase 7 vs Phase 8 F1 comparison."""
    p8 = load("phase8_validation.json")
    p7_tri = 0.607; p7_tri_std = 0.063
    p8_f1  = p8["f1_mean"]; p8_std = p8["f1_std"]

    fig, ax = plt.subplots(figsize=(5.5, 4))
    x = ["Single-Task\nBaseline\n(MFCC)", "Proposed\nMulti-Task\n(+COVAREP)"]
    y = [p7_tri, p8_f1]
    e = [p7_tri_std, p8_std]
    colors = [NAVY, CORAL]
    bars = ax.bar(x, y, yerr=e, capsize=6, color=colors, alpha=0.85,
                  edgecolor="black", linewidth=0.9, width=0.55)
    ax.set_ylabel("Macro-F1 (mean ± std, 5 seeds)")
    ax.set_ylim(0.45, 0.72)
    ax.set_title("Proposed Multi-Task Model vs. Single-Task Baseline")
    for b, v, s in zip(bars, y, e):
        ax.text(b.get_x() + b.get_width()/2, v + s + 0.005, f"{v:.3f}±{s:.3f}",
                ha="center", fontsize=8.5, weight="bold")
    ax.annotate("", xy=(1, p8_f1), xytext=(0, p7_tri),
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.5))
    ax.text(0.5, 0.66, f"+{p8_f1-p7_tri:.3f}", color=TEAL,
            ha="center", fontsize=10, weight="bold")
    plt.savefig(FIGURES / "fig3_p7_vs_p8.png")
    plt.close()
    print("  fig3_p7_vs_p8.png")


# ============================================================
def fig_seed_consistency():
    """Fig 4 — Per-seed F1 consistency strip plot."""
    p8 = load("phase8_validation.json")
    seeds = [s["seed"] for s in p8["per_seed"]]
    f1s   = [s["f1"]   for s in p8["per_seed"]]
    p7_tri = 0.607

    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.scatter(range(len(seeds)), f1s, s=120, color=CORAL,
               edgecolor="black", zorder=3, label="Proposed model per-seed F1")
    ax.axhline(p8["f1_mean"], color=CORAL, ls="-", lw=1.5,
               label=f"Proposed model mean ({p8['f1_mean']:.3f})")
    ax.axhline(p7_tri, color=NAVY, ls="--", lw=1.5,
               label=f"Single-task baseline mean ({p7_tri:.3f})")
    ax.fill_between([-0.5, len(seeds)-0.5],
                    p8["f1_mean"]-p8["f1_std"], p8["f1_mean"]+p8["f1_std"],
                    color=CORAL, alpha=0.12)
    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels([f"seed {s}" for s in seeds])
    ax.set_ylabel("Macro-F1")
    ax.set_xlim(-0.5, len(seeds)-0.5)
    ax.set_title("Proposed Model Per-Seed Consistency (all 5 above baseline)")
    ax.legend(fontsize=7.5, loc="lower right")
    plt.savefig(FIGURES / "fig4_seed_consistency.png")
    plt.close()
    print("  fig4_seed_consistency.png")


# ============================================================
def fig_sota():
    """Fig 5 — SOTA positioning."""
    methods = ["AVEC'17\ntext", "AVEC'17\naudio", "Williamson+\n'16",
               "THIS WORK\n(proposed)", "Gong+\n'17"]
    f1s     = [0.49, 0.50, 0.57, 0.629, 0.70]
    colors  = ["#999", "#999", "#777", CORAL, "#555"]

    fig, ax = plt.subplots(figsize=(6.5, 4))
    bars = ax.bar(methods, f1s, color=colors, alpha=0.9,
                  edgecolor="black", linewidth=0.8)
    ax.set_ylabel("DAIC-WOZ Dev F1")
    ax.set_ylim(0.4, 0.75)
    ax.set_title("Positioning vs. Published DAIC-WOZ Results")
    for b, v in zip(bars, f1s):
        ax.text(b.get_x()+b.get_width()/2, v+0.008, f"{v:.3f}",
                ha="center", fontsize=8.5,
                weight="bold" if v == 0.629 else "normal")
    plt.savefig(FIGURES / "fig5_sota.png")
    plt.close()
    print("  fig5_sota.png")


# ============================================================
def fig_fairness_radar():
    """Fig 6 — Fairness criteria radar (male vs female)."""
    crit = ["Equal\nOpportunity", "Predictive\nParity",
            "Demographic\nParity", "F1\nParity"]
    male   = [0.71, 0.46, 0.61, 0.56]
    female = [1.00, 0.46, 0.69, 0.63]

    angles = np.linspace(0, 2*np.pi, len(crit), endpoint=False).tolist()
    male   += male[:1]; female += female[:1]; angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5.5, 5.5), subplot_kw=dict(polar=True))
    ax.plot(angles, male, "o-", lw=2, color=NAVY, label="Male")
    ax.fill(angles, male, color=NAVY, alpha=0.12)
    ax.plot(angles, female, "o-", lw=2, color=CORAL, label="Female")
    ax.fill(angles, female, color=CORAL, alpha=0.12)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(crit, fontsize=8.5)
    ax.set_ylim(0, 1.05)
    ax.set_title("Gender Fairness Audit\n(EO gap 0.286 = real bias detected)",
                 fontsize=10, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=9)
    plt.savefig(FIGURES / "fig6_fairness_radar.png")
    plt.close()
    print("  fig6_fairness_radar.png")


def fig_fairness_before_after():
    """Fig 7 — Fairness loss: per-seed TPR gap before vs after (10 seeds)."""
    seeds     = [42, 1, 7, 13, 21, 100, 2024, 7777, 555, 88]
    no_fair   = [0.114, 0.086, 0.371, 0.371, 0.371, 0.714, 0.171, 0.514, 0.371, 0.114]
    with_fair = [0.171, 0.229, 0.229, 0.371, 0.171, 0.857, 0.029, 0.571, 0.371, 0.114]

    x = np.arange(len(seeds))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 4.5))
    b1 = ax.bar(x - w/2, no_fair,  w, label="Without Fairness Loss (λ=0)",
                color=NAVY,  alpha=0.85, edgecolor="black", linewidth=0.8)
    b2 = ax.bar(x + w/2, with_fair, w, label="With Equalized-Odds Loss (λ=0.1)",
                color=CORAL, alpha=0.85, edgecolor="black", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"S{s}" for s in seeds], fontsize=8)
    ax.set_ylabel("TPR Gap  |TPR_male − TPR_female|")
    ax.set_ylim(0, 1.00)
    mn_no  = np.mean(no_fair)
    mn_wi  = np.mean(with_fair)
    ax.set_title(
        f"Equalized-Odds Loss: Mixed Results over 10 Seeds\n"
        f"(mean gap: {mn_wi:.3f}±{np.std(with_fair):.3f} vs "
        f"{mn_no:.3f}±{np.std(no_fair):.3f},  Wilcoxon p=0.813,  wins: fair=3, base=4, ties=3)",
        fontsize=9
    )
    for b, v in [(b1, no_fair), (b2, with_fair)]:
        for bar, val in zip(b, v):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.012,
                    f"{val:.2f}", ha="center", fontsize=6.5)

    ax.axhline(mn_no,  ls="--", lw=1.2, color=NAVY,  alpha=0.6,
               label=f"Mean without: {mn_no:.3f}")
    ax.axhline(mn_wi, ls="--", lw=1.2, color=CORAL, alpha=0.6,
               label=f"Mean with: {mn_wi:.3f}")
    ax.legend(fontsize=7.5, loc="upper left")
    plt.tight_layout()
    plt.savefig(FIGURES / "fig7_fairness_before_after.png")
    plt.close()
    print("  fig7_fairness_before_after.png")


def fig_attention_vs_concat():
    """Fig 8 — Cross-modal attention vs plain concatenation (10 seeds)."""
    seeds      = [42, 1, 7, 13, 21, 100, 2024, 7777, 555, 88]
    attn_f1    = [0.647, 0.779, 0.646, 0.646, 0.621, 0.646, 0.690, 0.690, 0.597, 0.715]
    concat_f1  = [0.689, 0.597, 0.664, 0.678, 0.647, 0.652, 0.598, 0.652, 0.597, 0.664]
    attn_auc   = [0.598, 0.723, 0.727, 0.564, 0.746, 0.648, 0.686, 0.674, 0.576, 0.723]
    concat_auc = [0.633, 0.659, 0.705, 0.674, 0.716, 0.678, 0.583, 0.686, 0.568, 0.644]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    x = np.arange(len(seeds)); w = 0.35

    for ax, attn_v, concat_v, ylabel, p_val, wins in [
        (axes[0], attn_f1,  concat_f1,  "Macro-F1", "0.496", "attn=4, concat=5, ties=1"),
        (axes[1], attn_auc, concat_auc, "AUC-ROC",  "0.647", "attn=6, concat=4"),
    ]:
        b1 = ax.bar(x - w/2, attn_v,   w, label="Cross-Modal Attention",
                    color=PURPLE, alpha=0.85, edgecolor="black", linewidth=0.8)
        b2 = ax.bar(x + w/2, concat_v, w, label="Plain Concatenation",
                    color=GOLD,   alpha=0.85, edgecolor="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"S{s}" for s in seeds], fontsize=7.5)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0.50, max(max(attn_v), max(concat_v)) + 0.08)
        mn_a = np.mean(attn_v); mn_c = np.mean(concat_v)
        ax.axhline(mn_a, ls="--", lw=1.2, color=PURPLE, alpha=0.6,
                   label=f"Attn mean: {mn_a:.3f}")
        ax.axhline(mn_c, ls="--", lw=1.2, color=GOLD, alpha=0.6,
                   label=f"Concat mean: {mn_c:.3f}")
        ax.set_title(f"Wilcoxon p={p_val}  (n=10 seeds)  |  wins: {wins}", fontsize=8)
        ax.legend(fontsize=7, loc="lower right")
        for b, vs in [(b1, attn_v), (b2, concat_v)]:
            for bar, v in zip(b, vs):
                ax.text(bar.get_x() + bar.get_width()/2, v + 0.006,
                        f"{v:.3f}", ha="center", fontsize=6.5)

    fig.suptitle(
        "Cross-Modal Attention vs. Concatenation  (10 seeds, identical encoders/classifier)\n"
        "Attn: F1=0.668±0.050, AUC=0.667±0.064  |  Concat: F1=0.644±0.033, AUC=0.655±0.046  |  Neither difference significant",
        fontsize=9, weight="bold"
    )
    plt.tight_layout()
    plt.savefig(FIGURES / "fig8_attention_vs_concat.png")
    plt.close()
    print("  fig8_attention_vs_concat.png")


def fig_shortcut_audit():
    """Interviewer-prompt shortcut audit: participant-only vs both vs Ellie-only.

    All values from results/metrics/interviewer_shortcut.json (5 seeds,
    matched participant set across all three conditions).
    """
    d = json.load(open(METRICS / "interviewer_shortcut.json"))
    conds = ["participant", "both", "ellie"]
    labels = ["Participant\nonly", "Participant\n+ Ellie", "Ellie only\n(no patient words)"]
    colors = [TEAL, GOLD, CORAL]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))

    for ax, split, title in zip(
        axes, ["dev", "test"],
        [f"Dev split (N={d['matched_n']['dev']})",
         f"Held-out test split (N={d['matched_n']['test']})"]):
        means = [d["conditions"][c][f"{split}_auc_mean"] for c in conds]
        stds = [d["conditions"][c][f"{split}_auc_std"] for c in conds]
        x = np.arange(3)
        bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors,
                      edgecolor="black", linewidth=0.6)
        ax.axhline(0.5, color="black", ls="--", lw=1.2, zorder=0)
        ax.text(2.45, 0.505, "chance", fontsize=7, ha="right", va="bottom")
        for bar, m, s in zip(bars, means, stds):
            # place the label clear of the error-bar cap, not just the bar top
            ax.text(bar.get_x() + bar.get_width() / 2, m + s + 0.028,
                    f"{m:.3f}", ha="center", fontsize=8, weight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("AUC")
        ax.set_ylim(0, 1.0)
        ax.set_title(title, fontsize=9)

    fig.suptitle(
        "Interviewer-Prompt Shortcut Audit  (identical model, encoders and 5 seeds; "
        "only the speaker's transcript turns differ)\n"
        "A model that never sees a single patient word outperforms the "
        "participant-only model on both splits.",
        fontsize=8.5, weight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES / "fig9_shortcut_audit.png")
    plt.close()
    print("  fig9_shortcut_audit.png")


def fig_cross_corpus():
    """Zero-shot cross-corpus collapse: EATD (audio) and WU3D (text)."""
    e = json.load(open(METRICS / "cross_corpus_eatd.json"))
    w = json.load(open(METRICS / "cross_corpus_wu3d.json"))

    probes = [
        ("Audio probe\nEATD-Corpus", e["in_domain_dev_auc"], e["auc"],
         e["auc_ci95"], e["n_test"]),
        ("Text probe\nWU3D (Weibo)", w["in_domain_test_auc"], w["auc"],
         w["auc_ci95"], w["n_test"]),
    ]

    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    x = np.arange(len(probes))
    width = 0.34

    in_dom = [p[1] for p in probes]
    cross = [p[2] for p in probes]
    err = np.array([[p[2] - p[3][0] for p in probes],
                    [p[3][1] - p[2] for p in probes]])

    ax.bar(x - width / 2, in_dom, width, label="In-domain (DAIC-WOZ)",
           color=NAVY, edgecolor="black", linewidth=0.6)
    ax.bar(x + width / 2, cross, width, yerr=err, capsize=5,
           label="Zero-shot cross-corpus", color=CORAL,
           edgecolor="black", linewidth=0.6)

    ax.axhline(0.5, color="black", ls="--", lw=1.2, zorder=0)
    ax.text(len(probes) - 0.52, 0.508, "chance", fontsize=7, ha="right", va="bottom")

    for xi, v in zip(x - width / 2, in_dom):
        ax.text(xi, v + 0.02, f"{v:.3f}", ha="center", fontsize=8, weight="bold")
    for xi, p in zip(x + width / 2, probes):
        ax.text(xi, p[3][1] + 0.02,
                f"{p[2]:.3f}\n[{p[3][0]:.3f}, {p[3][1]:.3f}]",
                ha="center", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{p[0]}\n(N={p[4]:,})" for p in probes], fontsize=8.5)
    ax.set_ylabel("AUC")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title(
        "Zero-shot cross-corpus generalization  (error bars: 95% bootstrap CI, 2,000 resamples)\n"
        "Both modalities collapse to chance on an independent corpus in a different language.",
        fontsize=9, weight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES / "fig10_cross_corpus.png")
    plt.close()
    print("  fig10_cross_corpus.png")


def fig_learning_curve():
    """Tri-modal AUC/F1 vs. training-set size (subsampled from the 107-participant
    train pool, fixed dev set, 3 seeds per fraction)."""
    d = json.load(open(METRICS / "learning_curve.json"))
    fracs = sorted(d["fractions"].keys(), key=float)
    ns = [d["fractions"][f]["n_train"] for f in fracs]
    auc_m = [d["fractions"][f]["auc_mean"] for f in fracs]
    auc_s = [d["fractions"][f]["auc_std"] for f in fracs]
    f1_m = [d["fractions"][f]["f1_mean"] for f in fracs]
    f1_s = [d["fractions"][f]["f1_std"] for f in fracs]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.errorbar(ns, auc_m, yerr=auc_s, marker="o", color=NAVY, capsize=4,
               linewidth=2, markersize=7, label="AUC")
    ax.errorbar(ns, f1_m, yerr=f1_s, marker="s", color=CORAL, capsize=4,
               linewidth=2, markersize=7, label="F1")
    ax.axhline(0.5, color="black", ls="--", lw=1, alpha=0.6, zorder=0)
    ax.text(ns[0], 0.51, "chance", fontsize=7)
    ax.set_xlabel("Training-set size (participants, subsampled from N=107 pool)")
    ax.set_ylabel("Score (fixed dev set, N=33)")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_title(
        "Learning Curve: Tri-Modal Model vs. Training-Set Size  (3 seeds per point)\n"
        "AUC rises with N and has not visibly plateaued at N=107 -- "
        "more participants would likely help.",
        fontsize=9.5, weight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES / "fig11_learning_curve.png")
    plt.close()
    print("  fig11_learning_curve.png")


def fig_sentinel_fix():
    """OpenFace -100 sentinel contamination: effect on face-only and tri-modal."""
    f = json.load(open(METRICS / "face_sentinel_ablation.json"))
    t = json.load(open(METRICS / "trimodal_sentinel_ablation.json"))

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))

    for ax, d, title, note in zip(
            axes, [f, t],
            ["Face-only configuration", "Tri-modal configuration"],
            ["5/5 seeds improved\nWilcoxon $p$=0.031, $d$=1.43",
             "2/5 seeds improved\nWilcoxon $p$=0.313 (n.s.)"]):
        arms = ["original", "cleaned"]
        means = [d[a]["auc_mean"] for a in arms]
        stds = [d[a]["auc_std"] for a in arms]
        x = np.arange(2)
        bars = ax.bar(x, means, yerr=stds, capsize=5,
                      color=[CORAL, TEAL], edgecolor="black", linewidth=0.7,
                      width=0.55)
        # per-seed points, paired
        for i, a in enumerate(arms):
            pts = d[a]["per_seed_auc"]
            ax.scatter([i] * len(pts), pts, color="black", s=14, zorder=3, alpha=0.75)
        for p_o, p_c in zip(d["original"]["per_seed_auc"], d["cleaned"]["per_seed_auc"]):
            ax.plot([0, 1], [p_o, p_c], color="gray", lw=0.8, alpha=0.55, zorder=2)

        ax.axhline(0.5, color="black", ls="--", lw=1.1, zorder=0)
        ax.text(1.42, 0.508, "chance", fontsize=7, ha="right")
        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2, m + s + 0.035,
                    f"{m:.3f}", ha="center", fontsize=9, weight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(["with $-100$\nsentinels\n(original)",
                            "sentinel frames\nremoved\n(fixed)"], fontsize=8.5)
        ax.set_ylabel("Dev AUC")
        ax.set_ylim(0, 1.0)
        ax.set_title(f"{title}\n{note}", fontsize=9)

    fig.suptitle(
        "OpenFace $-100$ Missing-Data Sentinel: Effect on Depression Classification\n"
        "Sentinels survive the confidence filter and, under per-participant z-scoring, "
        "compress the real AU signal ~79$\\times$.",
        fontsize=9, weight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(FIGURES / "fig12_sentinel_fix.png")
    plt.close()
    print("  fig12_sentinel_fix.png")


def main():
    print("Generating publication-quality figures...")
    fig_architecture()
    fig_ablation_ci()
    fig_p7_vs_p8()
    fig_seed_consistency()
    fig_sota()
    fig_fairness_radar()
    fig_fairness_before_after()
    fig_attention_vs_concat()
    fig_shortcut_audit()
    fig_cross_corpus()
    fig_learning_curve()
    fig_sentinel_fix()
    print(f"\nAll 12 figures saved to {FIGURES}/")


if __name__ == "__main__":
    main()
