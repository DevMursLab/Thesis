# Cover Letter — Draft

Dear Editor,

We submit our manuscript, "Multi-Task Learning for Tri-Modal Depression
Detection: A Controlled Ablation Study on DAIC-WOZ," for consideration for
publication.

**Summary of contribution.** This work presents a controlled study of
multimodal depression screening on the DAIC-WOZ corpus. Our primary validated
result is that symptom-level multi-task learning (jointly predicting the
binary depression label, PHQ-8 severity score, and eight individual symptom
items) improves macro-F1 from 0.607±0.063 to 0.629±0.021 across five random
seeds, with substantially reduced variance, using an identical model class
across all seven modality-subset configurations to isolate genuine modality
contribution from architectural confounds. We additionally report held-out
test-set results (N=47): F1=0.585, AUC=0.706, and a gender TPR gap of 0.000,
providing evidence that the model generalizes beyond the development split.

**On honest negative results.** In the interest of methodological integrity,
we report two design choices — cross-modal attention fusion and an Equalized
Odds fairness loss — that we evaluated rigorously (10-seed paired Wilcoxon
signed-rank tests) but which did **not** reach statistical significance
(p=0.496 and p=0.813, respectively) at this sample size. Rather than
reframing these as successes, we present them as transparent negative and
inconclusive findings that characterize the real limits of attention-based
fusion and fairness-constrained training on small clinical cohorts (N=34–47).
We believe this transparency — increasingly called for in the machine
learning community given widespread concerns about selective reporting and
irreproducibility in small-sample benchmarks — is itself a contribution to
the literature, particularly for clinical AI where overclaiming carries real
downstream risk.

**Positioning.** We do not claim to advance the state of the art in raw
performance; recent transformer-based approaches (Ray et al., 2019; Shen et
al., 2022) report higher AUC using pre-trained language models and
larger-scale training that are outside the compute budget of this study. Our
contribution is methodological: a controlled-variable ablation protocol,
symptom-level multi-task supervision as a genuine performance gain, and
convergent explainability evidence (three independent attribution methods
agreeing on audio and AU04 as primary cues) — combined with candid reporting
of where two additional techniques did not work as hypothesized.

We believe this combination of rigor, transparency, and clinically grounded
interpretability is well suited to [JOURNAL NAME]'s readership, and we hope
the manuscript will be considered for peer review.

Thank you for your time and consideration.

Sincerely,
Md. Mursalin
Md. Shovon
Department of Computer Science and Engineering, Netrokona University
