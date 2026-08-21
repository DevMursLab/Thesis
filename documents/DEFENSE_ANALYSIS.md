# Thesis Defense Analysis — Tri-Modal Depression Risk Detection
**PhD/IEEE Level | Prepared for Viva Examination & Reviewer Criticism**

> **Read this first:** this document was substantially rewritten to match the
> current `documents/paper_draft.tex`. Earlier drafts of this file described
> the **single-task tri-modal model** (AUC 0.698, F1 0.607) as the headline
> result and marked several analyses as "planned for Phase 8." That is no
> longer accurate. The paper's actual proposed/headline model is the
> **multi-task model** (dev AUC 0.658±0.029, F1 0.629±0.021), and five new
> validation experiments (symptom-level breakdown, missing-modality
> robustness, internal audio-feature ablation, calibration/error analysis,
> and a formal multi-task significance test) have since been completed and
> added to the paper. If you memorize only one number from the old version
> of this document, assume it is now wrong — use Part VII below instead.

---

## Part I — Research Novelty Analysis

### What Genuinely Distinguishes This Work

**Claim 1: Controlled-Variable Ablation Design**

The central methodological novelty is not any single technique — it is the *experimental design*. Prior multimodal depression papers (Gong & Poellabauer 2017; Williamson et al. 2016) report tri-modal results without ablating individual modalities with the *same architecture*. They change the model when they change the modality, making it impossible to attribute performance differences to the modality vs. the model.

`ConfigurableFusionModel` instantiates all 7 configurations from a single class: identical encoders (embed_dim=64), identical attention mechanism, identical classifier, identical optimizer. A performance difference between audio-only (AUC=0.721) and face-only (AUC=0.513) is attributable to the modality signal, not to architectural confounds. This is what an IEEE reviewer means by a "controlled ablation."

**Claim 2: Symptom-Level Multi-Task Learning — the paper's most promising result, honestly reported as not statistically confirmed**

Jointly predicting the binary label, PHQ-8 total score, and all eight PHQ-8 symptom items raises mean macro-F1 from 0.607±0.063 (single-task tri-modal baseline) to 0.629±0.021 (proposed multi-task model), with markedly reduced seed variance. **This is the paper's central, title-bearing contribution** ("Multi-Task Learning for Tri-Modal Depression Detection").

However: we ran a paired Wilcoxon signed-rank test matching each multi-task seed against the single-task run from the *same* seed, and it does **not** reach significance (p=0.625, n=5, multi-task wins only 3/5 pairs). Critically, at n=5 an exact two-sided Wilcoxon test cannot reach p<0.05 under *any* possible outcome (best case: p=0.0625) — so this test was structurally underpowered to confirm the effect regardless of its true size. We report this honestly rather than hiding it, and it is the single most important thing to be ready to discuss in the viva (see Q1–Q3 below).

**Claim 3: Equalized Odds, Not Equalized Opportunity**

Hardt et al. (NeurIPS 2016) show these are distinct criteria. Equal Opportunity (TPR gap only) can be satisfied by a model that predicts positive for everyone — trivially equalizing TPR while creating catastrophic FPR inequality. Equalized Odds requires both TPR and FPR to be equal across groups.

The training loss `L_fairness = (TPR_M − TPR_F)² + (FPR_M − FPR_F)²` penalizes both failures simultaneously. A 10-seed before/after comparison (Section IX-B of the paper) shows this loss does **not** significantly reduce the TPR gap at N=34 (Wilcoxon p=0.813, fairness wins only 3/10 seeds vs. the no-fairness baseline's 4/10) — another honestly-reported negative result, not hidden or reframed.

**Claim 4: Multi-Seed, Multi-Split, Multi-Method Statistical Rigor**

DAIC-WOZ official dev set is 34 participants. A single training run on N=34 has no statistical meaning. This work trains every configuration across 5 or 10 seeds (mean±std, paired bootstrap/Wilcoxon), reports bootstrap 95% CIs, adds a pooled 10-fold cross-validation over all 188 participants (Section VIII-C) to obtain a larger effective held-out sample, and adds effect-size/power analysis (Cohen's d, post-hoc power) for the two 10-seed comparisons. The honest findings — overlapping CIs, non-significant p-values across nearly every design choice tested — are *more publishable*, not less, because they are reproducible, self-aware, and consistent with what a 2025–2026 reproducibility literature review of DAIC-WOZ (see Claim 5) predicts should happen at this sample size.

**Claim 5: Positioning Against the DAIC-WOZ Reproducibility Crisis**

A 2026 systematic review (Danylenko & Unold, *Applied Sciences*) examined DAIC-WOZ papers reporting MAE and found only 5 of 66 met minimal reproducibility standards, identifying subject-level data leakage as the dominant inflating factor. A separate 2025 ICMI reproducibility study found that DAIC-WOZ classifiers may largely learn *disorder-general* distress cues rather than markers specific to Major Depressive Disorder. This work responds to both concerns directly: (a) splits are verified participant-disjoint (no leakage) at both the fixed-split and k-fold level; (b) the multi-task symptom-level breakdown (Section VIII-A) directly tests the disorder-general-cues concern by checking whether presence/absence accuracy varies across the 8 PHQ-8 symptoms — it does (macro-F1 spread 0.209–0.424), though the best-predicted symptoms are still the most generic ones, so this is reported as *partial*, not clean, evidence against the concern.

**Claim 6: Convergent XAI, With an Important Caveat**

Three mechanistically independent methods (attention rollout, gradient × input, leave-one-out occlusion) independently identify the same top modality (audio) and the same top face feature (AU04) — but only **on the dev split**. A new test-time missing-modality-masking experiment (Section VIII-B) shows this dev-split ranking does *not* fully predict test-time necessity: masking audio does not hurt test performance, while masking text collapses test AUC from 0.706 to 0.435. Convergence across independent methods is still a stronger evidential standard than any single method alone, but "most attended on dev" and "most necessary at test time" are now shown by this paper's own data to be different questions. Be ready to explain this distinction — it is a genuinely interesting finding, not a contradiction to hide.

---

### Gaps in Existing Approaches Addressed

| Gap | Prior Work | This Work |
|-----|-----------|-----------|
| No controlled ablation | Different models per modality subset | Single `ConfigurableFusionModel` class |
| Fairness not addressed | No fairness constraints in AVEC winners | Equalized Odds loss, 4-criterion audit, honestly reported as not significant at N=34 |
| Single-run results | Most papers report 1 seed | 5/10-seed, mean±std, paired bootstrap/Wilcoxon, effect size + power analysis |
| Single fixed split only | Standard practice on DAIC-WOZ | Pooled 10-fold CV over all 188 participants, showing the fixed split's fairness result is not stable |
| Black-box fusion | Concatenate → classify | Attention rollout + gradient saliency + occlusion, cross-checked against a real test-time masking experiment |
| Reproducibility unaddressed | Subject leakage common in published work (Danylenko & Unold 2026) | Participant-disjoint splits verified at fixed-split and k-fold level |
| Disorder-general-cues concern unaddressed | Aggregate binary label only (ICMI 2025 critique) | Symptom-level multi-task breakdown, partial evidence reported honestly |

---

## Part II — System Design Decisions

*(Unchanged from earlier drafts — these design-rationale answers remain accurate.)*

### Why Bi-LSTM, Not Transformer?

On 107 training samples, a transformer's self-attention (O(n²) parameters in attention layers) would overfit immediately. Bi-LSTM with mean-pooling:
- Parameters scale as O(hidden²) per layer, not O(seq_len²)
- Mean-pool is invariant to interview length (DAIC-WOZ sessions vary 7–33 min)
- Bidirectional context captures both onset and recovery of depressive episodes within a session

Transformer would be correct with 10,000+ samples (full DAIC + AVEC + CMDC combined).

### Why embed_dim=64, Not 256?

The original 7.8M-parameter model (embed_dim=256) memorized 107 training samples by epoch 5 — AUC on dev was 0.49 (random), AUC on train was 1.00. Shrinking to embed_dim=64 (~592K–638K params across model variants) forces the model to share representational capacity across samples rather than assigning separate capacity to each training point. This is the clinical-small-N equivalent of the bias-variance tradeoff.

### Why Fairness Warmup (8 epochs)?

Adding `L_fairness` at epoch 0 means the model optimizes demographic parity of a random classifier — gradient noise dominates the fairness signal. By waiting 8 epochs for the classifier to converge to a reasonable solution, the fairness gradient targets a meaningful TPR/FPR gap. This is analogous to learning rate warmup.

### Why TF-IDF Instead of BERT?

1. **N=107 training texts:** BERT fine-tuning requires thousands of samples to update the 110M-parameter backbone without catastrophic forgetting.
2. **Vocabulary restriction to 1,000:** prevents the model from memorizing rare interview-specific phrases.
3. **Speed:** TF-IDF preprocessing is deterministic; BERT adds significant overhead on CPU.

BERT is the correct choice when DAIC-WOZ is combined with larger corpora (combined N ≈ 500+).

### Why Modality Dropout, and What Did We Actually Find?

Modality dropout (p=0.15 per channel, training-time only) is included as a regularizer motivated by the clinical constraint that a modality may be missing at deployment. We do **not** just assert this works — Section VIII-B directly tests it with real test-time masking, and the result is mixed: robust to losing face, moderately costly to lose audio, and critically dependent on text (test AUC collapses to 0.435 without it). Be ready to state this as a finding, not a solved problem.

---

## Part III — Mathematical & Technical Depth

*(Unchanged — still accurate.)*

### Cross-Modal Attention: Full Derivation

Let modality encoders produce $\mathbf{h}_f, \mathbf{h}_a, \mathbf{h}_t \in \mathbb{R}^d$ where $d = 64$.

For modality $m \in \{f, a, t\}$, define:
- Query: $\mathbf{Q}_m = \mathbf{W}_Q \mathbf{h}_m$
- Keys: $\mathbf{K}_{-m} = \mathbf{W}_K [\mathbf{h}_{o_1} | \mathbf{h}_{o_2}]^T \in \mathbb{R}^{(M-1) \times d}$
- Values: $\mathbf{V}_{-m} = \mathbf{W}_V [\mathbf{h}_{o_1} | \mathbf{h}_{o_2}]^T \in \mathbb{R}^{(M-1) \times d}$

Attention weights:
$$\boldsymbol{\alpha}_m = \text{softmax}\!\left(\frac{\mathbf{Q}_m \mathbf{K}_{-m}^T}{\sqrt{d}}\right) \in \mathbb{R}^{M-1}$$

Attended context:
$$\mathbf{c}_m = \boldsymbol{\alpha}_m \mathbf{V}_{-m} \in \mathbb{R}^d$$

Residual + layer norm:
$$\tilde{\mathbf{h}}_m = \text{LayerNorm}(\mathbf{c}_m + \mathbf{h}_m)$$

Fused representation:
$$\mathbf{z} = [\tilde{\mathbf{h}}_f \| \tilde{\mathbf{h}}_a \| \tilde{\mathbf{h}}_t] \in \mathbb{R}^{3d = 192}$$

### Equalized Odds: Why Both Terms Are Necessary

**Equal Opportunity (EO):** $\text{EOpp} = \text{TPR}_M - \text{TPR}_F$

**Equalized Odds (EOdds):** EO **AND** $\text{FPR}_M = \text{FPR}_F$

A model that predicts $\hat{Y}=1$ for all samples satisfies EO trivially but catastrophically violates EOdds. The FPR term in the loss directly penalizes this collapse.

### Bootstrap CI and Paired Bootstrap Test: Exact Algorithms

```
Bootstrap CI (per-configuration):
  For b in 1..2000: resample N indices with replacement, compute AUC*[b]
  CI_95 = (percentile(AUC*, 2.5), percentile(AUC*, 97.5))

Paired bootstrap test (tri-modal vs. best unimodal, dev N=34):
  δ_obs = AUC(tri-modal) − AUC(best unimodal) = 0.698 − 0.721 = −0.023
  For b in 1..2000: resample jointly, compute δ*[b]
  p = P(δ* ≥ 0) = 0.859   ← still accurate, unchanged by this session's edits
```

p=0.859 means 85.9% of bootstrap resamples show tri-modal ≥ audio-only AUC — this does *not* say tri-modal is better, it says the two are statistically indistinguishable on N=34.

### Wilcoxon Signed-Rank Test at Small n — the n=5 Floor (NEW, important)

For the multi-task-vs-single-task paired comparison (n=5 seeds), an *exact* two-sided Wilcoxon signed-rank test has only $2^5=32$ equally likely sign patterns under the null. The most extreme possible result (all 5 differences the same sign) gives:
$$p = 2 \times \frac{1}{32} = 0.0625$$
This is *already above* α=0.05. **No possible outcome at n=5 could ever reach conventional significance**, regardless of the true effect size. The observed p=0.625 is therefore not just "not significant" — the test itself was underpowered by construction. Contrast with the n=10 attention/fairness tests, which additionally report Cohen's d and post-hoc power (16%/6%/9%) — a stronger, more honest standard that the n=5 test does not yet have. If asked why the two tests are held to different standards, say so plainly: this is a real limitation, and more seeds would be needed to make the n=5 test informative.

---

## Part IV — Defense Preparation: Viva Questions & Strong Answers

---

**Q1: "Your paper's title is about multi-task learning, but your own significance test says the multi-task improvement isn't real. Isn't that a problem for your whole thesis?"**

**Strong Answer:**

No — it is the honest, correctly-reported state of the evidence, and it is *more* defensible than claiming false certainty. The mean improvement (0.607→0.629) and variance reduction (±0.063→±0.021) are real, reproducible observations — I re-ran the single-task baseline independently this session and got 0.607±0.063, exactly matching the original. What I do *not* claim is that a formal significance test confirms this at n=5, because it doesn't (p=0.625), and I show mathematically that n=5 could never have shown significance regardless of the true effect (best-case p=0.0625). The correct scientific statement is: "the direction is consistently positive and the mechanism is theoretically motivated, but this sample size cannot statistically confirm it." I report exactly that, rather than either hiding the test or overclaiming from the mean alone.

---

**Q2: "If the multi-task result isn't confirmed, what IS actually validated in this thesis?"**

**Strong Answer:**

Three things are genuinely well-supported, not just observed: (1) the controlled-ablation *methodology* itself — using one model class across all 7 modality subsets is methodologically sound regardless of which configuration wins; (2) the case-level finding that 5/34 dev participants are correctly classified by tri-modal fusion but missed by the best single modality, while only 1/34 is wrong under every configuration — a specific, checkable observation, not an aggregate significance claim; (3) the negative/inconclusive findings themselves are validated in the sense that the statistical tests behind them (Wilcoxon, bootstrap, power analysis) are correctly computed and honestly reported — that rigor is the actual contribution of this thesis, not a specific performance number.

---

**Q3: "Why report a test (n=5 Wilcoxon) that you admit could never reach significance? Isn't that a wasted analysis?"**

**Strong Answer:**

Reporting it is more honest than omitting it. Before running this test, the paper's only evidence for the multi-task improvement was "all 5 seeds beat the baseline's *mean*," which sounds stronger than it is — it doesn't test seed-matched superiority. Running the paired test, even knowing its floor, converts a vague claim into a precise, falsifiable one, and surfaces the seed-matched result (only 3/5 wins) which is materially different from "all 5 seeds are above the mean." Reporting a test's limitation alongside its result is standard good practice, not wasted effort.

---

**Q4: "Your tri-modal AUC (single-task, 0.698) is LOWER than audio-only (0.721). Why is fusion worse?"**

**Strong Answer:**

The difference (−0.023 AUC) is not statistically significant: paired bootstrap p=0.859, confidence intervals overlap substantially. Tri-modal achieves *higher* F1 (0.607 vs. 0.592) and accuracy (0.688 vs. 0.676), and rescues 5/34 dev participants that audio-only misclassifies, while only 1/34 is wrong under every configuration. AUC measures ranking quality across all samples; F1 at a clinically-tuned threshold is closer to the deployment metric. Both are reported; neither is cherry-picked. Separately, a leave-one-modality-out occlusion experiment on the *test* split (Section VIII-B) shows audio removal does not hurt test performance either — audio's dominance is a dev-split-specific finding, and I'm explicit about that scope in the paper.

---

**Q5: "You found that removing audio doesn't hurt on test, but your explainability section says audio is the most-attended modality. Which is it?"**

**Strong Answer:**

Both are true, and they answer different questions. Attention rollout measures *how much the model looks at* a modality during a forward pass on the dev split — that's audio, on average. The test-time masking experiment measures *how much the model's test-set performance depends on* a modality being present — that's text, where removal collapses AUC to 0.435 (below chance). A modality can receive high average attention while another modality is more critical for generalization to unseen data. I treat this as a genuine, reportable finding about the difference between attribution and necessity, not an inconsistency to explain away.

---

**Q6: "N=34/47 is too small. Your results are meaningless."**

**Strong Answer:**

N=34/47 is the official DAIC-WOZ dev/test partition used by essentially every published paper on this benchmark. To address the small-N limitation rigorously, beyond what's standard in the field, this work adds: multi-seed training with mean±std and bootstrap CIs; paired significance tests with explicit acknowledgment of their power limits (including the n=5 Wilcoxon floor); and a pooled 10-fold cross-validation over all 188 available participants (Section VIII-C), which shows the fixed-split fairness result (TPR gap=0.000) is *not* stable across folds (mean gap 0.415±0.211, 7/10 folds with zero female recall). Reporting that the fixed split is unreliable, using a larger resampled estimate as corroboration, is a stronger response to "N is too small" than simply asserting the numbers are fine.

---

**Q7: "The fairness training loss doesn't actually reduce the TPR gap (p=0.813). Isn't that a failed contribution?"**

**Strong Answer:**

It's a negative result, reported as one, and it is still a real contribution in the "responsible AI development cycle" sense: audit → measure a real gap (Equal Opportunity gap 0.286 on dev) → design and integrate a training-time constraint → re-measure honestly → find it insufficient at this sample size, with a post-hoc power analysis confirming the study is underpowered (not just null) for the observed effect size. That is a complete, honestly-reported fairness investigation. If I had hidden the negative result or only reported the dev-split TPR gap of 0.000 without the 10-fold corroboration showing it isn't stable, that would be the actual failure — not the negative finding itself.

---

**Q8: "AU04 as a depression marker — isn't this circular? You trained on labels from those participants' faces."**

**Strong Answer:**

Gradient saliency is a post-hoc attribution method — the model was never told to look at AU04. It emerged from training because AU04 variation correlated with PHQ-8 labels across the training participants. The validation is that this data-driven finding matches independent clinical literature (Ekman's FACS work identifying AU04/Brow Lowerer as a sadness/distress marker, established decades before DAIC-WOZ existed). Circularity would exist only if AU04 had been hand-selected as a feature *because* it's a known marker; instead all 20 AUs are used equally and AU04 emerges from the gradient across three independent attribution methods (rollout, gradient×input, occlusion).

---

**Q9: "Why does your paper cite reproducibility-crisis papers about DAIC-WOZ? Isn't that undermining your own dataset choice?"**

**Strong Answer:**

The opposite — it's demonstrating awareness of the field's actual state and positioning this work as a response to it, not ignoring it. Danylenko & Unold (2026) found only 5/66 DAIC-WOZ papers meet minimal reproducibility standards, largely due to subject-level leakage; I explicitly verify and state that all splits here (fixed and k-fold) are participant-disjoint. The ICMI 2025 study raises a concern that DAIC-WOZ classifiers learn disorder-general rather than disorder-specific cues; the symptom-level breakdown in this thesis (Section VIII-A) directly tests that concern with real per-symptom data rather than asserting the model is fine. I report the result as partial evidence, not a clean rebuttal, because the best-predicted symptoms are still the most generic ones — that honesty is the point.

---

**Q10: "Why TF-IDF, not BERT?"**

**Strong Answer:**

BERT fine-tuning on 107 training transcripts risks catastrophic forgetting — 110M parameters vs. 107 samples is roughly six orders of magnitude past any reasonable few-shot fine-tuning regime. TF-IDF with a 1,000-word vocabulary is the regularized, overfitting-resistant choice at this N. A frozen-BERT-embedding + small-head approach becomes reasonable once combined corpora reach N≈500+ (noted as future work in the paper).

---

**Q11: "Your model uses CPU training. Doesn't that limit your results?"**

**Strong Answer:**

All experiments use the official DAIC-WOZ splits to ensure comparability with published results; CPU training affects speed, not the fundamental learning dynamics for a ~600K-parameter model on ~100 training samples. All hyperparameters and architecture are GPU-ready; CPU was used specifically to guarantee reproducibility of the ablation study (35+ runs) without depending on specific hardware. This is disclosed as a limitation in the paper, not hidden.

---

## Part V — Possible Reviewer Attacks & Responses (updated — three of four are now RESOLVED, with honest/negative findings)

**Attack 1 (RESOLVED): "The ablation study doesn't include a no-attention baseline."**

Resolved. A 10-seed cross-modal-attention-vs-plain-concatenation comparison is now in the paper (`ConcatFusionModel`, Section on Cross-Modal Attention). Result: attention shows a positive trend (F1 +0.024, AUC +0.012) but does **not** reach significance (Wilcoxon p=0.496/0.647), with an explicit Cohen's d + post-hoc power analysis showing the study is underpowered (16%/9% power) rather than showing a true null effect. If asked, be ready to state this as an honest inconclusive result, not a proof either way.

**Attack 2 (RESOLVED): "Why not use the full 188 DAIC-WOZ participants, or report the test split?"**

Resolved. Full test-split results (F1=0.585, AUC=0.706, N=47) are reported and are available to registered DAIC-WOZ users under the standard data-use agreement (no challenge-submission requirement, contrary to what earlier drafts of this document said). Additionally, a pooled 10-fold cross-validation over all 188 participants (train+dev+test combined) is now reported (Section VIII-C / k-fold section), giving F1=0.554±0.093, AUC=0.612±0.120 — lower than the fixed-split numbers, which is itself an important, honestly-reported finding about fixed-split reliability at this N.

**Attack 3: "Multi-seed mean±std is necessary but not sufficient. You need cross-validation."**

Resolved, same as Attack 2 — the pooled 10-fold CV directly addresses this. Be ready to explain *why* the k-fold numbers are lower than the fixed-split numbers: the fixed AVEC-2017 split may be a moderately favorable partition relative to the full pool, and/or the model is sensitive to which participants land in train vs. test — both are stated honestly as limitations rather than reasons to prefer the more favorable fixed-split numbers.

**Attack 4 (RESOLVED): "You haven't shown the fairness constraint actually reduces the TPR gap."**

Resolved, with a negative result. A 10-seed before/after comparison (λ_fair=0 vs. 0.1) is reported: mean TPR gap 0.320→0.311, Wilcoxon p=0.813, fairness wins only 3/10 seeds vs. the baseline's 4/10 (3 ties). This is reported plainly as a negative result, alongside a power analysis showing the study is underpowered for the observed effect size. Do not describe this as "the fairness loss works" in the viva — the honest answer is "it does not show a significant effect at this N, and here is the power analysis explaining why that is not the same as proving no effect."

**Attack 5 (NEW): "Your test-time missing-modality experiment and your dev-split explainability section seem to disagree about which modality matters most."**

See Q5 above. Prepared answer: attention weight (dev, average) and inference-time necessity (test, ablative) are different properties; the paper reports both without reconciling them into a single ranking, because they measure different things.

**Attack 6 (NEW): "Why does removing audio slightly *improve* test AUC in your robustness table?"**

Honest answer: the effect is small (AUC 0.706→0.725, ΔAUC≈0.02 at N=47) and is most plausibly sampling noise at this test size rather than a real effect — no significance test was run on this specific comparison, and it should not be oversold as "audio hurts the model." The F1 effect of removing audio is unambiguously negative on both dev and test (0.622→0.583 dev, 0.585→0.527 test), so the more defensible summary is "removing audio costs F1 on both splits; its effect on AUC specifically is small and likely noise at N=47."

---

## Part VI — Publication/Defense Readiness Checklist (updated)

| Criterion | Status | Evidence |
|-----------|:------:|---------|
| Clear novel contribution | ✅ | Controlled ablation + multi-task learning + honest statistical treatment throughout |
| Real clinical dataset | ✅ | DAIC-WOZ, 188 participants, USC ICT, verified participant-disjoint splits |
| Reproducible code | ✅ | Full pipeline, deterministic seeds, requirements.txt (`data/processed/daic_test_raw_cache/` is currently built ad hoc — see Part VIII note) |
| Baselines compared | ✅ | AVEC 2017, Williamson 2016, Gong 2017, Ray 2019, Shen 2022 (six baselines, non-comparable ones flagged as such) |
| Statistical validation | ✅ | 5/10-seed mean±std, bootstrap CI, paired Wilcoxon, Cohen's d + power analysis (10-seed tests); n=5 floor explicitly acknowledged |
| Ablation study | ✅ | 7 configs × 5 seeds = 35 runs, plus 6-config internal audio-feature ablation |
| Cross-validation | ✅ | Pooled 10-fold CV over all 188 participants |
| Fairness audit | ✅ | 4 criteria, real gap found, training constraint tested, negative result reported honestly |
| Explainability | ✅ | 3 converging dev-split methods + test-time necessity cross-check |
| Symptom-level analysis | ✅ | Presence/absence breakdown across 8 PHQ-8 items, partial evidence re: disorder-general-cues concern |
| Calibration analysis | ✅ | Brier score, ECE, reliability table, dev/test consistent FN-skewed error profile |
| Honest limitations | ✅ | N=34/47 power, overlapping CIs, n=5 Wilcoxon floor, CPU training, threshold-leakage bug found and fixed this session |
| Mathematical formulation | ✅ | Attention, fairness loss, bootstrap, Wilcoxon-floor equations |
| Positioning vs. reproducibility literature | ✅ | Explicit citation and response to 2025/2026 DAIC-WOZ reproducibility critiques |

---

## Part VII — Key Numbers to Memorize for Viva (REPLACES the old Part VII — old numbers described the wrong model)

```
DAIC-WOZ: 188 total, 107 train, 34 dev, 47 test (participant-disjoint, verified)
PHQ-8 threshold: >= 10 = depressed; prevalence ~28% in training split

PROPOSED MODEL = MULTI-TASK (not the single-task tri-modal baseline):
  Dev:  F1 = 0.629 +/- 0.021   AUC = 0.658 +/- 0.029   Acc = 68.8%
  Test: F1 = 0.585             AUC = 0.706             Acc = 70.2%
  Params: ~592K (embed_dim=64)

Single-task tri-modal baseline (for comparison only, NOT the proposed model):
  Dev:  F1 = 0.607 +/- 0.063   AUC = 0.698 +/- 0.047

Multi-task vs single-task significance: Wilcoxon p=0.625, n=5, 3/5 seeds favor
  multi-task. NOTE: n=5 exact test cannot reach p<0.05 under ANY outcome
  (floor = 0.0625) -- inconclusive by construction, not a disconfirmation.

Ablation (7 configs x 5 seeds, single-task, dev):
  Face only      : AUC 0.513+/-0.113  F1 0.525
  Audio only     : AUC 0.721+/-0.013  F1 0.592  <- best AUC
  Text only      : AUC 0.561+/-0.043  F1 0.539
  Face+Audio     : AUC 0.712+/-0.028  F1 0.563
  Face+Text      : AUC 0.667+/-0.069  F1 0.570
  Audio+Text     : AUC 0.664+/-0.057  F1 0.437
  Face+Audio+Text: AUC 0.698+/-0.047  F1 0.607  <- best single-task F1/Acc
  Paired bootstrap (tri-modal vs audio-only): p=0.859, n.s. (unchanged)

Cross-modal attention vs concat (10 seeds): F1 +0.024 (p=0.496), AUC +0.012
  (p=0.647) -- n.s., underpowered (16%/9% power)

Equalized Odds fairness loss (10 seeds, before/after): TPR gap 0.320->0.311,
  p=0.813 -- n.s., 3/10 wins vs baseline's 4/10, underpowered

Pooled 10-fold CV (N=188): F1=0.554+/-0.093  AUC=0.612+/-0.120
  TPR gap = 0.415+/-0.211 (7/10 folds: TPR_female=0)
  -> shows the fixed-split TPR gap=0.000 is NOT a stable result

Test-time missing-modality masking (dev-derived threshold, no leakage):
  Baseline:      dev F1=0.622 AUC=0.663 | test F1=0.585 AUC=0.706
  Face masked:   dev F1=0.622 AUC=0.655 | test F1=0.585 AUC=0.712  (no cost)
  Audio masked:  dev F1=0.583 AUC=0.580 | test F1=0.527 AUC=0.725  (F1 cost, AUC noise)
  Text masked:   dev F1=0.555 AUC=0.610 | test F1=0.415 AUC=0.435  (collapse)

Symptom-level breakdown (dev, presence/absence macro-F1 across 8 PHQ-8 items):
  Range: 0.209 (Psychomotor Change) -- 0.424 (Tired/Low Energy)
  Best-predicted symptoms are the most GENERIC ones -> only partial evidence
  against the "disorder-general cues" critique (ICMI 2025)

Internal audio-feature ablation (3 seeds, dev AUC):
  MFCC only 0.710 | COVAREP only 0.619 | FORMANT only (5-dim!) 0.688
  MFCC+COVAREP 0.682 | COVAREP+FORMANT 0.569 | Full (199-dim) 0.711
  -> MFCC alone ~= full combination; COVAREP/FORMANT not a clear performance add

Calibration (Brier / ECE): dev 0.233/0.102, test 0.227/0.157 -- moderate
  miscalibration. Error profile FN-skewed on BOTH splits (dev 8FN/2FP,
  test 10FN/4FP out of 14/47 misclassified) -- consistent, clinically
  relevant (less-safe failure mode)

Fairness (dev): Equal Opportunity gap = 0.286 (TPR_F=1.00, TPR_M=0.71)
XAI (dev): audio attention weight 0.39 (top); AU04 top gradient-attributed
  face feature -- but test-time masking shows audio is NOT most necessary
  at test time (see above) -- attribution != necessity

Error analysis (single-task ablation): tri-modal rescues 5/34 dev cases vs
  best unimodal; 1/34 wrong under every configuration

SOTA (DAIC-WOZ dev F1, six baselines now, non-comparable ones flagged):
  AVEC-2017 audio: 0.50 | AVEC-2017 text: 0.49 | Williamson+: 0.57
  Gong+: 0.70 | Ray+ (AUC 0.80, not F1-comparable) | Shen+ (AUC 0.83, ext. split)
  THIS WORK (multi-task, dev): F1 0.629 | (test): F1 0.585
```

---

## Part VIII — Known Loose Ends (say these proactively if asked "what would you do with more time?")

1. **`data/processed/daic_test_raw_cache/` is not regenerated by any committed script** — it exists on the development machine from an earlier interactive run of the k-fold experiment. A clean checkout cannot currently re-run the missing-modality-robustness or calibration scripts without first rebuilding this cache. Fix: extract the test-set-feature-building logic already in `src/experiments/kfold_cv.py` into a standalone, committed script.
2. **The internal audio-feature ablation uses only 3 seeds** (vs. 5/10 elsewhere), to bound compute cost across 6 additional configurations — its numbers should be read as suggestive, not as precisely estimated as the rest of the paper.
3. **The single-task-vs-multi-task significance test bundles two changes** (multi-task heads AND the extended 199-dim COVAREP+FORMANT audio vs. the single-task baseline's 120-dim MFCC-only audio) — even a significant result would not have cleanly isolated multi-task learning as the sole causal factor. An ideal follow-up would train a multi-task model on the *old* 120-dim audio to isolate the two effects.
4. **E-DAIC external validation** is named as future work in the paper but has not been attempted — requires a separate USC ICT data-use request, and any cross-dataset comparison must explicitly exclude participants who overlap between DAIC-WOZ and E-DAIC to avoid reintroducing the leakage this thesis otherwise takes care to avoid.

---

*This document is a living reference for the thesis defense and IEEE reviewer response. It was last synchronized with `documents/paper_draft.tex` in the session that added Sections VIII-A through VIII-D (symptom breakdown, missing-modality robustness, calibration, audio-feature ablation) and the multi-task significance test. If the paper changes again, re-check Parts I, IV, V, VI, and VII against it before relying on this document in a real defense.*
