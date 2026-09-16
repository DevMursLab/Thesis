Subject: Dataset Access Request — CMDC (Chinese Multimodal Depression Corpus)

Dear Dr. Zou and co-authors,

My name is Md. Mursalin, and I am a researcher in the Department of
Computer Science and Engineering at Netrokona University, Bangladesh,
working under the supervision of Md. Shovon. We are preparing a
manuscript on validity auditing for tri-modal (facial action units,
audio, text) depression-detection models trained on DAIC-WOZ, and I am
writing to request access to CMDC, described in your paper
"Semi-structural Interview-Based Chinese Multimodal Depression Corpus
Towards Automatic Preliminary Screening of Depressive Disorders"
(IEEE Transactions on Affective Computing, vol. 14, no. 4, 2023).

**Purpose of the request.** Our manuscript audits three validity
concerns in DAIC-WOZ-based depression detection: (1) a data-corruption
sentinel in OpenFace facial Action Unit features, (2) an
interviewer-prompt shortcut in transcript-based text models, and (3)
non-transferable, corpus-specific representations in audio and text,
which we demonstrate via zero-shot cross-corpus evaluation on
EATD-Corpus (audio) and WU3D (text). We have not been able to run an
equivalent cross-corpus probe for the facial modality, because we have
not identified another public, OpenFace-compatible, depression-labeled
corpus besides CMDC. Since our own facial-AU branch is DAIC-WOZ-trained
and English/Wizard-of-Oz-specific, evaluating it zero-shot on CMDC's
independent Chinese-language interviews would let us test whether the
same corpus-specificity we found in audio and text also holds for the
visual modality — directly extending our validity audit rather than
introducing a new use case.

**Intended use.** We would use only the facial Action Unit
features (or raw video, if AUs are not directly distributed, from
which we would extract OpenFace AUs ourselves under the same
pipeline used for DAIC-WOZ) and the associated depression-severity
labels, strictly for this non-commercial academic research. We would
not redistribute the data, and would report only aggregate,
de-identified statistical results (e.g., AUC, confidence intervals) in
the manuscript, consistent with standard human-subjects data handling
and mirroring the terms we already comply with for DAIC-WOZ (USC ICT
EULA).

Could you let us know the process for requesting access (e.g., a
EULA or data-use agreement to sign) and the expected timeline? We
would be glad to provide any additional information about our
institution or research purpose that would help.

Thank you very much for your time and for making this valuable corpus
available to the research community.

Sincerely,
Md. Mursalin
Department of Computer Science and Engineering
Netrokona University, Bangladesh
Email: mursalinshuvo27@gmail.com

On behalf of: Md. Shovon (Supervisor)
