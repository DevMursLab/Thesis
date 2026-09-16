# Cover Letter — Draft

Dear Editor,

We submit our manuscript, "Why Tri-Modal Depression Detection on DAIC-WOZ
Doesn't Replicate: A Systematic Validity Audit of Sentinel Artifacts,
Interviewer Shortcuts, and Cross-Corpus Transfer," for consideration for
publication in IEEE Access.

**Summary of contribution.** Reported performance for automated depression
screening on the DAIC-WOZ corpus varies by more than 20 AUC points across the
literature under seemingly minor pipeline differences, and few studies
directly test whether their models capture depression-specific signal as
opposed to interview structure, tracking-software artifacts, or
corpus-specific acoustic idiosyncrasies. Using a controlled tri-modal
(facial action units, acoustic features, transcript text) ablation testbed
built on a single model class instantiated across seven modality subsets, we
surface and directly test three failure modes that plausibly account for a
substantial share of this variance:

1. A silent data-corruption bug in the visual stream (an OpenFace missing-data
   sentinel that compresses the genuine facial signal ~79x), found and fixed.
2. Direct confirmation, in our own pipeline, that DAIC-WOZ text models can
   learn interview structure instead of depression language: an
   interviewer-only model that never sees a patient's words outperforms our
   participant-only model.
3. Cross-corpus transfer collapsing to chance for both audio (EATD-Corpus,
   Chinese) and text (WU3D, 32,540 Weibo users) modalities, indicating the
   learned representations are corpus/language-specific rather than
   transferable biomarkers.

Within the same testbed we evaluate three common design choices. The most
promising, symptom-level multi-task learning, raises mean macro-F1 from
0.607±0.063 to 0.629±0.021 at an original 5-seed sample; a power analysis of
that comparison showed it was underpowered (n≈67 seeds needed for 80% power),
so we rescaled the paired significance test to 70 seeds. This confirms the
effect (51/70 seeds favor multi-task, Wilcoxon p=8.1×10⁻⁷, d=0.72, larger
than the d=0.34 the original power calculation assumed) — a concrete,
within-pipeline demonstration of how an underpowered n=5 significance test
can leave a real, confirmable effect looking merely suggestive, which is the
same seed-sampling risk this paper raises as a broader methodological
concern. Cross-modal attention and an
Equalized Odds fairness loss remain evaluated at 10 seeds and do not reach
significance at this sample size (p=0.496 and p=0.813); we report both as
transparent negative/inconclusive findings rather than reframing them as
successes.

**On honest, transparent reporting.** In the interest of methodological
integrity, we report every non-significant and negative result alongside the
positive ones, including a post-hoc power analysis wherever a comparison is
underpowered, and we resolve the one comparison where added seeds were
feasible (multi-task vs. single-task) rather than leaving it as an
unconfirmed trend. We believe this transparency — increasingly called for in
the machine learning community given widespread concerns about selective
reporting and irreproducibility in small-sample benchmarks — is itself a
contribution to the literature, particularly for clinical AI where
overclaiming carries real downstream risk.

**Positioning.** We do not claim to advance the state of the art in raw
performance; recent transformer-based approaches (Ray et al., 2019; Shen et
al., 2022) report higher AUC using pre-trained language models and
larger-scale training outside this study's compute budget. Our contribution
is methodological: a systematic validity audit that surfaces and directly
tests three failure modes capable of explaining a substantial share of
cross-study variance in this literature, a controlled-variable ablation
protocol that makes those audits possible, a statistically confirmed
multi-task learning result obtained by resolving rather than reporting an
underpowered test, and convergent explainability evidence (three independent
attribution methods agreeing on audio as the most informative modality) —
combined with candid reporting of every technique that did not work as
hypothesized.

We believe this combination of rigor, transparency, and reproducibility
focus is well suited to IEEE Access's readership, and we hope the manuscript
will be considered for peer review.

This work is original, has not been published previously, and is not under
consideration for publication elsewhere. Both authors have approved the
manuscript and agree with its submission to the journal.

Thank you for your time and consideration.

Sincerely,
Md. Mursalin (Corresponding Author)
Department of Computer Science and Engineering
Netrokona University, Bangladesh
Email: mursalinshuvo27@gmail.com

On behalf of: Md. Shovon (Supervisor)
