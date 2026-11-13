# Closest prior work to FCT in polyp / endoscopy segmentation (2022-2026): consistency, self-distillation, multi-view training, plug-in losses across hosts, and unseen-centre generalization

Scope note: the draft related work (paper/sections/related.tex, paper/references.bib) already cites PraNet, SANet, UACANet, Polyp-PVT, ASGNet, PraNet-V2, CTNet, Polyper, UM-Net, ASPS, VANet, PolypGen, EndoCV2021, illumination augmentation, DCLPS, ACL-Net and the general consistency / distillation literature. Everything below was found beyond that list unless marked "(already cited)". Titles, venues and years were checked against a primary source (MICCAI open-access pages, CVF open access, PMLR, PMC full text, arXiv abstract pages, or Crossref metadata). MICCAI 2023-2026 were screened by title over the full open-access indexes ([2026](https://papers.miccai.org/miccai-2026/), [2025](https://papers.miccai.org/miccai-2025/), [2024](https://papers.miccai.org/miccai-2024/), [2023](https://conferences.miccai.org/2023/papers/)), and the relevant pages and PDFs were then read.

## 1. Which fully supervised polyp/endoscopy segmentation papers use consistency regularization, flip/rotation equivariance, self-distillation, mutual learning or multi-view training? What exactly do they do, and do they target generalization to unseen centres?

### Takeaway
Under full supervision this line of work is thin but not empty. The closest paper is **Segmentation Consistency Training (SCT, IEEE ISM 2022)**: a training loss that keeps predictions consistent between augmented and unaugmented polyp images, discounts the changes the label predicts, is weighted adaptively by in-distribution IoU, is tested on four networks with 10 runs each and gains most on ETIS. Two more recent fully supervised polyp papers use self-distillation as a training-only plug-in on three networks and test on an unseen PolypGen centre: **PRISM** (Healthcare Technology Letters 2025, EMA teacher) and **DCSD** (arXiv 2025), whose distillation term is weighted by the Dice score of the earlier prediction against the label. **AAC (MICCAI 2023)** adds a fully supervised, uncertainty-weighted consistency term between adversarially and anti-adversarially perturbed polyp images. I found no fully supervised polyp paper that uses flip consistency, or the label per pixel, to choose which view teaches which.

### Cited Findings
**A. Consistency regularization under full supervision (polyp)**
- **SCT**: "Segmentation Consistency Training: Out-of-Distribution Generalization for Medical Image Segmentation", Torpmann-Hagen, Thambawita, Riegler, Halvorsen, Glette. Published at the 2022 IEEE International Symposium on Multimedia (ISM), pp. 42-49 — [Crossref record](https://api.crossref.org/works/10.1109/ISM55400.2022.00012); preprint [arXiv:2205.15428](https://arxiv.org/abs/2205.15428).
  - Method: the network sees an unaugmented image and an augmented copy. The "Segmentation Inconsistency Loss" (SIL) is a soft, Jaccard-style ratio over the four maps {label y, prediction ŷ, augmented label a, augmented prediction â}. It counts pixels whose prediction changed under the perturbation, minus the change the label predicts. It is combined as L = (1 − IoU)·L_seg + IoU·L_c, with IoU computed per batch — [arXiv PDF](https://arxiv.org/pdf/2205.15428).
  - Augmentations (albumentations): RandomRotate90, GaussNoise, ImageCompression, OpticalDistortion and ColorJitter. These are a mix of geometric and photometric changes, not flips only — [arXiv PDF](https://arxiv.org/pdf/2205.15428).
  - Networks and protocol: DeepLabV3+, FPN, UNet and Tri-Unet from segmentation-models-pytorch, ten instances each. Trained on Kvasir-SEG (80/10/10 split) and tested out-of-distribution on ETIS-LaribDB, CVC-ClinicDB and EndoCV2020. Statistics use independent t-tests per model and a Mann–Whitney U-test across models (p<0.01 on all three OOD sets vs. conventional augmentation) — [arXiv PDF](https://arxiv.org/pdf/2205.15428).
  - Results (mean IoU, vanilla augmentation → SCT): the largest gains are on ETIS (DeepLabV3+ 0.472→0.505, FPN 0.440→0.475, TriUnet 0.410→0.434, Unet 0.447→0.481). CVC-ClinicDB gains +0.007 to +0.013; EndoCV2020 gains 0 to +0.017. In-distribution Kvasir is unchanged (about 0.85) — [arXiv PDF](https://arxiv.org/pdf/2205.15428).
  - Source inconsistency: the text and Table 2 disagree on which per-model comparisons are significant. The text lists FPN/Unet on ClinicDB as non-significant, while the table stars them — [arXiv PDF](https://arxiv.org/pdf/2205.15428).
- **AAC**: "Anti-Adversarial Consistency Regularization for Data Augmentation: Applications to Robust Medical Image Segmentation", Cho, Han, Kim, MICCAI 2023, LNCS pp. 555–566 — [Crossref](https://api.crossref.org/works/10.1007/978-3-031-43901-8_53).
  - Method: the object (e.g., polyps) is perturbed with near-imperceptible adversarial noise, and the network is trained to expand discriminative regions under the guidance of anti-adversarial noise. This is implemented as a consistency regularization between the two contrasting inputs, whose strength is set adaptively by prediction uncertainty. The paper reports multiple backbones on two datasets — [MICCAI 2023 page](https://conferences.miccai.org/2023/papers/069-Paper2200.html).
  - Evidence: the datasets were Kvasir-SEG and ETIS with 80/10/10 within-dataset splits, as stated in the follow-up DASP paper ([DASP PDF](https://papers.miccai.org/miccai-2024/paper/2811_paper.pdf)). A reviewer noted the gain over a U-Net with basic augmentation was only about 0.02 Dice and asked for Wilcoxon tests ([MICCAI 2023 reviews](https://conferences.miccai.org/2023/papers/069-Paper2200.html)). There is no cross-centre test.
- **DDCNet**, "A Dual Domain Collaborative Network for Polyp Segmentation", IEEE JBHI 30 (2026) 6719–6731 ([Crossref](https://api.crossref.org/works/10.1109/jbhi.2025.3649384)). A search snippet describes a hybrid loss with boundary, cross-entropy and frequency-consistency terms; I did not read the paper. It is an architecture paper — [DOI link](https://doi.org/10.1109/jbhi.2025.3649384).

**B. Self-distillation (fully supervised polyp)**
- **PRISM**: "Past-Regularized Iterative Self-Distillation With Momentum for Polyp Segmentation", Erol, Caglikantar, Sarikaya, *Healthcare Technology Letters* 12, e70050 (2025) — [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC12706544/); [Crossref](https://api.crossref.org/works/10.1049/htl2.70050).
  - Method: the teacher is an EMA of the student, giving temperature-scaled sigmoid targets. The loss is Dice + BCE + λ·distillation, with distillation starting after epoch 6. The same input goes to student and teacher; no augmented or flipped views are described — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12706544/).
  - Protocol: plugged into PraNet, TransNetR and ShallowNet, trained on PolypGen centres C1–C5 and tested on unseen C6 (88 images) — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12706544/).
- **DCSD**: "The Power of Certainty: How Confident Models Lead to Better Segmentation", Erol, Caglikantar, Sarikaya, arXiv:2507.10490 (July 2025). No venue is listed, and the code is to be released after acceptance — [arXiv abstract](https://arxiv.org/abs/2507.10490).
  - Method: the previous mini-batch and its output are stored. The current model is re-run on that batch, and an MSE consistency term (temperature T = 4) is applied, weighted by the Dice score of the stored prediction against its ground truth. The weight is therefore label-based, but it is one scalar per batch or image, the "views" are previous iterations rather than spatial transformations, and no flips are used — [arXiv HTML](https://arxiv.org/html/2507.10490).
  - Unseen-data tests: C1–C5 → C6, and Kvasir+ClinicDB → Kvasir, ClinicDB, ColonDB, EndoScene, ETIS, BKAI-IGH — [arXiv HTML](https://arxiv.org/html/2507.10490).
- **BDKD-Net**: "Boundary-Probability Knowledge Distillation for Compact Polyp Segmentation", *Journal of Imaging* 12(7):306 (2026). The teacher is obtained by self-distillation on Kvasir+ClinicDB; a compact student learns the teacher's boundary-band probabilities, and the teacher is discarded at inference. External tests: ColonDB, CVC-300, ETIS, BKAI. Statistics: five-seed means ± SD, eight-seed paired 95% CIs, Cohen's dz and exact sign-flip p-values — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC13412844/).
- **Boundary self-knowledge distillation** for colonoscopy polyp segmentation: Meng et al., *Journal of Image and Graphics* 30(2):589–600 (2025). I could read only the metadata — [CJIG page](https://www.cjig.cn/en/article/doi/10.11834/jig.240175/).
- **CD-PolypNet** (MICCAI 2025): a SAM-based polyp network with "Semantic Supervision via Feature Distillation" (SAM features distilled into early-stage features, plus normalized self-distillation across encoder blocks) and an edge branch. It is evaluated on the five standard datasets. The distillation is part of the architecture, not a host-agnostic loss — [MICCAI 2025 page](https://papers.miccai.org/miccai-2025/0139-Paper2291.html); [PDF](https://papers.miccai.org/miccai-2025/paper/2291_paper.pdf).
- **SAM-MaGuP** (MICCAI 2025): a boundary distillation module and a 1D-2D Mamba adapter inside SAM, framed as "generalized polyp segmentation", on five datasets — [MICCAI 2025 page](https://papers.miccai.org/miccai-2025/0522-Paper3165.html).

**C. Cross-network consistency / mutual learning**
- **BoxPolyp** (MICCAI 2022, LNCS pp. 67–77) — [DOI](https://doi.org/10.1007/978-3-031-16437-8_7); [arXiv](https://arxiv.org/abs/2212.03498).
  - Method: its "Image Consistency" loss is a squared feature difference between two SANet models (Res2Net and PVT backbones) on the *same* image, restricted to uncertain regions left unlabeled by box-derived pseudo-labels. Training needs extra box-annotated data — [arXiv HTML](https://arxiv.org/html/2212.03498).
  - Evidence: it is claimed to be plug-and-play, but it is tested only inside SANet with its two backbones. The paper uses single runs and no statistics — [arXiv HTML](https://arxiv.org/html/2212.03498).
- I found no fully supervised two-network mutual-learning polyp segmentation paper in 2022–2026. Searches returned only semi- or weakly supervised work (see Q4) — [search hits included S2ME, PedSemiSeg, CVPR 2025 depth-guided mutual learning](https://openaccess.thecvf.com/content/CVPR2025/html/Li_Boost_the_Inference_with_Co-training_A_Depth-guided_Mutual_Learning_Framework_CVPR_2025_paper.html).

**D. Equivariance, orientation and viewpoint**
- **Rotation-equivariance benchmark** (arXiv:2412.09182, Dec 2024, no venue listed): on Kvasir-SEG, C8-equivariant U-Nets beat vanilla U-Nets, e.g. large-data/large-model IoU 76.7 → 79.0 and small-data/small-model 38.0 → 44.1 (5-fold CV). The authors conclude that equivariant U-Nets "are an interesting choice" for such tasks — [arXiv HTML](https://arxiv.org/html/2412.09182v1).
  - Conflict: an AI-generated search summary claimed that rotation equivariance was "detrimental" on Kvasir-SEG. The paper's own Table 4 contradicts this — [arXiv HTML](https://arxiv.org/html/2412.09182v1).
- **VANet** (MedIA 2024, already cited): splits colonoscopy viewpoints into "parallel" and "vertical" classes and builds viewpoint-aware transformers, i.e. it handles viewpoint in the architecture, not in the loss. The README reports state-of-the-art results "especially on the unseen datasets" — [GitHub README](https://github.com/1024803482/Viewpoint-Aware-Network).
- **MVOA-Net**, "Multi-view orientational attention network combining point-based affinity for polyp segmentation", *Expert Systems with Applications* 249:123663 (2024) — [Crossref](https://doi.org/10.1016/j.eswa.2024.123663). It is an architecture with orientation-aware attention modules (CAFM, PBAM), not a multi-view training loss — [AMiner abstract](https://www.aminer.cn/pub/65f9068e13fb2c6cf6785174/multi-view-orientational-attention-network-combining-point-based-affinity-for-polyp-segmentation).
- ScribblePolyp states explicitly that "in fully supervised polyp segmentation, the transformation consistency is guaranteed by the same augmentation strategy of both images and labels". This is the default view that FCT argues against — [ScribblePolyp arXiv PDF](https://arxiv.org/pdf/2311.05122).

**E. Endoscopy beyond polyp segmentation, and non-polyp MICCAI precedent**
- **BWCR**, "Boundary-weighted logit consistency improves calibration of segmentation networks" (Karani, Dey, Golland, MICCAI 2023). It shows that logit consistency across stochastic (geometric) transformations acts as a *spatially varying regularizer* that prevents overconfidence at pixels with ambiguous labels. Its boundary-weighted version gives state-of-the-art calibration on prostate and heart MRI under full supervision. Its target is calibration, not unseen-centre accuracy — [MICCAI 2023 page and reviews](https://conferences.miccai.org/2023/papers/094-Paper2691.html).
- **MAGE** (MICCAI 2026, gastric neoplasm *classification* in endoscopy): an auxiliary branch sees masked achromatic views, and its logits and attention maps are distilled into the main branch, which sees the full white-light image. Inference needs no masks. This is a fully supervised multi-view distillation in endoscopy, but for a different task — [MICCAI 2026 page](https://papers.miccai.org/miccai-2026/0605-Paper4527.html).
- **"Smarter Self-Distillation"** (MICCAI 2025): self-distillation for surgical video recognition, with teachers ensembled across seeds and epochs. Not segmentation — [MICCAI 2025 page](https://papers.miccai.org/miccai-2025/0838-Paper1323.html).

### Inferences
- SCT is the paper MICCAI polyp reviewers are most likely to raise as "this has been done". It shares FCT's setting (full supervision, consistency between transformed and untransformed views, labels inside the consistency measure, adaptive weighting, several hosts, multiple runs, largest gains on ETIS). FCT must cite it and set itself apart on mechanism: per-pixel, *directed*, label-gated pulls with stop-gradient targets over an exact flip group with three supervised views, versus SCT's symmetric soft-set inconsistency with one batch-level IoU weight and mixed augmentations.
- PRISM and DCSD show that "training-only plug-in on several polyp hosts + unseen PolypGen centre" is already a recognised framing (2025). DCSD's Dice-to-label weighting of a distillation term is the nearest prior instance of "use the label to weight self-agreement". It differs in granularity (one scalar per batch vs. per pixel), source of the second view (previous iteration vs. flipped image) and direction (fixed past→present vs. chosen per pixel by comparing errors).
- BWCR is the strongest MICCAI precedent for reading consistency as a spatially varying reweighting of the per-pixel loss. It parallels the draft's "identity" argument (consistency term = per-pixel loss reweighting), so the identity needs to be positioned against it.

### Gaps
- I could not read the AAC full text (Springer), so its exact backbones and loss form are unverified beyond the MICCAI page, the reviews and DASP's description.
- I could not read the full texts of DDCNet (JBHI 2026) and the JIG boundary self-distillation paper; their content claims are unverified.
- I found no fully supervised polyp paper whose explicit loss is flip or rotation consistency. Some may exist in lower-visibility venues not reached by searches.

## 2. Which papers propose training losses or plug-in modules for polyp segmentation and evaluate them on several host networks? What gains and evidence do they report (single run vs several seeds, statistics)?

### Takeaway
Multi-host plug-ins are common in polyp segmentation, but most are evaluated on generic or lightweight hosts (U-Net, U-Net++, LinkNet, DeepLabV3+, FPN, PraNet, TransNetR, ShallowNet), use within-dataset splits or one unseen centre, and report single runs. Gains on unseen data are typically +1–4 Dice/IoU points, larger on ETIS. Repeated-run statistics are the exception: SCT (10 runs, t-test and Mann–Whitney), BDKD-Net (5–8 seeds, paired CIs), LRBP (5 seeds on one non-polyp dataset), and PRISM (Wilcoxon over test images). None applied an unchanged loss to recent state-of-the-art polyp hosts (Polyp-PVT, PraNet-V2, SAM2-UNet-class, ASGNet) with each host's own recipe.

### Cited Findings
| Method (venue) | What is plugged in | Hosts | Train → test | Reported gain | Runs / statistics | Source |
|---|---|---|---|---|---|---|
| SCT (IEEE ISM 2022) | Consistency loss (SIL) + IoU-adaptive weighting | DeepLabV3+, FPN, UNet, Tri-Unet | Kvasir → ETIS, ClinicDB, EndoCV2020 | ETIS mIoU +2.4 to +3.5 pts vs. augmentation | 10 runs per config; t-test; Mann–Whitney p<0.01 | [arXiv PDF](https://arxiv.org/pdf/2205.15428) |
| PRISM (HTL 2025) | EMA self-distillation | PraNet, TransNetR, ShallowNet | PolypGen C1–C5 → C6 | C6 Dice: PraNet 0.7824→0.8093; TransNetR 0.7189→0.7615; ShallowNet 0.7637→0.7874 | Runs/seeds not reported; Wilcoxon: significant only for TransNetR (Dice p=0.03), PraNet/ShallowNet p≈0.2–0.6 | [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12706544/) |
| DCSD (arXiv 2025) | Dice-weighted temporal self-distillation | own PVTv2 model, TransNetR, ShallowNet | C1–C5 → C6; Kvasir+ClinicDB → 6 sets | C6 Dice 0.7864→0.8151; ETIS +2.6, BKAI +3.8, ColonDB +1.9; TransNetR C6 0.7189→0.7538 | Single numbers, no variance or tests | [arXiv HTML](https://arxiv.org/html/2507.10490) |
| AAC (MICCAI 2023) | Uncertainty-weighted anti-adversarial consistency | "multiple backbones" (U-Net family) | Kvasir-SEG, ETIS within-dataset | about +0.02 Dice over U-Net + basic augmentation (reviewer) | Reviewer asked for Wilcoxon | [MICCAI page](https://conferences.miccai.org/2023/papers/069-Paper2200.html) |
| DASP (MICCAI 2024) | Diffusion-based adversarial augmentation, uncertainty-weighted per pixel | U-Net, U-Net++, LinkNet, DeepLabv3+ | Kvasir-SEG, ETIS within-dataset (80/10/10) | up to +1.09 (Kvasir) and +1.12 (ETIS) mIoU over the second-best method | Single numbers | [PDF](https://papers.miccai.org/miccai-2024/paper/2811_paper.pdf) |
| BoxPolyp (MICCAI 2022) | Box pseudo-labels + cross-network IC loss (extra box data) | SANet-Res2Net, SANet-PVT only | Kvasir+ClinicDB → standard 5 | ETIS Dice +0.079 / +0.092; ColonDB +0.067 / +0.066 vs SANet | Single runs, no statistics | [arXiv HTML](https://arxiv.org/html/2212.03498) |
| PraNet-V2 (CVM 2026; already cited) | DSRA module | three segmentation models (hosts not confirmed from the paper) | not checked | mDice +0.50% to +1.36% | not checked | [arXiv:2504.10986](https://arxiv.org/abs/2504.10986) |
| Lite-π (MICCAI 2026) | Foundation-model prototype induction + transformer fusion (adds parameters) | U-Net, U-Net++, PraNet | Kvasir+ClinicDB (1,450) → ColonDB, ETIS, CVC-300 | e.g. PraNet 30.4M→34.9M params; U-Net++ 9.1M→10.1M | not found in text | [PDF](https://papers.miccai.org/miccai-2026/paper/5591_paper.pdf) |
| Polyp-DiFoM (WACV 2026) | Distils SAM / DINOv2 / OneFormer / Mask2Former priors | U-Net, U-Net++ | five standard datasets | "nearly 9× less computation" than a SOTA model | not checked | [CVF](https://openaccess.thecvf.com/content/WACV2026/html/Agnihotri_From_SAM_to_DINOv2_Towards_Distilling_Foundation_Models_to_Lightweight_WACV_2026_paper.html); [poster](https://wacv.thecvf.com/virtual/2026/poster/845) |
| LRBP plugin (MICCAI 2026) | Low-rank background prior + gate | ResNet-34 U-Net, MedNeXt-Small, + one more | Kvasir-SEG 8:1:1; BUSI | ASD reductions | 5 seeds on BUSI only (ASD 17.81±6.41 → 14.67±5.54) | [PDF](https://papers.miccai.org/miccai-2026/paper/0629_paper.pdf) |
| HadBalance (MICCAI 2026) | Plug-and-play shape-prior loss with gradient-conflict balancing | built on the strongest backbone (nnU-Net) | CVC-ClinicDB, CUBS, DRIONS-DB in-domain | small DSC/IoU gains | rolling mean ± SD curves | [PDF](https://papers.miccai.org/miccai-2026/paper/0859_paper.pdf) |
| WeakPolyp (MICCAI 2023) | M2B + scale-consistency loss (box supervision) | "plug-and-play... other backbones" (Res2Net, PVT) | SUN-SEG etc. | comparable to fully supervised | not checked | [MICCAI page](https://conferences.miccai.org/2023/papers/721-Paper0147.html) |
| TextPolyp (MICCAI 2024) | Point supervision + SAM mutual learning + discrepancy weights | "various backbones and segmentation methods" | polyp sets | not checked | not checked | [MICCAI page](https://papers.miccai.org/miccai-2024/768-Paper1293.html) |
| Bayesian uncertainty-weighted loss (FAIMI @ MICCAI 2023) | per-pixel CE × (1+σ)^κ | BayDeepLabV3+ only | C1–C5 → C6 single / sequence | C6-SEQ Dice 0.676→0.697; C1–5-SEQ 0.819→0.810 (drop) | ± over test samples only; no repeated runs | [arXiv HTML](https://arxiv.org/html/2309.06807v2); [DOI](https://doi.org/10.1007/978-3-031-45249-9_15) |

- Data-synthesis plug-ins evaluated on several segmenters:
  - ArSDM (MICCAI 2023) improved PraNet, SANet and Polyp-PVT — [MICCAI 2023 page](https://conferences.miccai.org/2023/papers/072-Paper1249.html); host list as described in [SIDM PDF](https://papers.miccai.org/miccai-2025/paper/3072_paper.pdf).
  - SIDM (MICCAI 2025) tested U-Net, U-Net++ and FPN on a shared split, and PraNet and FCBFormer on unseen EndoScene, ColonDB and ETIS. A reviewer criticised the lack of transformer hosts — [MICCAI 2025 page](https://papers.miccai.org/miccai-2025/0818-Paper3072.html); [PDF](https://papers.miccai.org/miccai-2025/paper/3072_paper.pdf).
  - Polyp-LDM (MICCAI 2025) — [page](https://papers.miccai.org/miccai-2025/0037-Paper2379.html); SFD-Polyp (MICCAI 2026) "consistently improves segmentation across multiple baseline models" — [page](https://papers.miccai.org/miccai-2026/0557-Paper4818.html).
- Other statistics practice in MICCAI polyp papers:
  - Endoscopic Artifact Inpainting (MICCAI 2025) reported Wilcoxon p-values, e.g. ETIS p<0.0001 — [page](https://papers.miccai.org/miccai-2025/0296-Paper1575.html).
  - KAN-AINet (MICCAI 2026) used a Brown–Forsythe variance test — [page](https://papers.miccai.org/miccai-2026/0524-Paper5427.html).
  - S2ME (MICCAI 2023) reported mean ± SD over multiple seeds — [arXiv PDF](https://arxiv.org/pdf/2306.00451).

### Inferences
- FCT's evaluation design stands out in this niche: an unchanged loss on four recent state-of-the-art polyp hosts, each with its own recipe, with six paired seeds and confidence intervals on standard sets plus PolypGen C1–C6. The typical comparison papers use one or two hosts, one recipe and single runs.
- It is **not** the first multi-seed study. SCT used 10 runs per configuration with significance tests, and BDKD-Net used paired seed CIs. The honest claim is "paired-seed CIs across four modern hosts and six unseen centres", not "first with statistics".
- The typical magnitude of unseen-set gains from training-only plug-ins (≈+2–4 points, largest on ETIS) gives reviewers a reference for judging FCT's ETIS/PolypGen gains. SCT's pattern of ETIS gains much larger than in-distribution gains matches what FCT reports.

### Gaps
- I did not verify the exact host networks of PraNet-V2's DSRA transfer: the names MERIT, MIST and EMCAD come only from a secondary summary.
- I did not verify per-host gains for WeakPolyp or TextPolyp. Both are weakly supervised, so they are less central.
- I found no seeds or CIs in the text of Lite-π (the MICCAI page keywords match only the review-form template).

## 3. Which polyp papers evaluate on PolypGen (or other multi-centre sets) as unseen-centre data, and with what protocols?

### Takeaway
Four protocols recur:
- (a) the official PolypGen/EndoCV protocol: train on C1–C5 single frames, test on C6 single frames and sequences (PRISM, DCSD, the Bayesian loss);
- (b) multi-centre training that includes PolypGen C1–C5, testing on PolypGen-C6 plus ETIS/ColonDB (ESPNet MICCAI 2025, KAN-AINet MICCAI 2026);
- (c) training on another dataset and testing on PolypGen as a whole or per centre (TransNetR: Kvasir-SEG → all six centres; S2ME and PedSemiSeg: SUN-SEG/Kvasir → PolypGen);
- (d) federated or leave-one-centre-out DG across the six centres (LFDG, FDG).

FCT's protocol (Kvasir+ClinicDB → all six PolypGen centres, per centre, with paired seeds) is closest to TransNetR's (c). Among verified papers, I found none that combines the standard PraNet split with per-centre PolypGen and repeated-seed CIs.

### Cited Findings
- **Centre sizes**: PolypGen single frames per centre are C1 256, C2 301, C3 457, C4 227, C5 208, C6 88 (1,449 in C1–C5; 1,537 in total) — [PRISM (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12706544/); [TransNetR](https://proceedings.mlr.press/v227/jha24a/jha24a.pdf); dataset paper (already cited) [Scientific Data 2023](https://www.nature.com/articles/s41597-023-01981-y).
- **Official protocol (a)**: train on 1,449 single frames from C1–C5. Test on C6-SIN (88 frames, unseen), C6-SEQ (432 sequence frames, unseen) and C1–5-SEQ (124 frames from seen centres, with negatives and corruptions) — [Bayesian uncertainty-weighted loss, arXiv HTML](https://arxiv.org/html/2309.06807v2). This follows the EndoCV2021 generalisability study (already cited) — [arXiv:2202.12031](https://arxiv.org/abs/2202.12031).
- **PRISM / DCSD (a)**: C1–C5 (90/10 train/validation) → C6 (88). DCSD additionally runs Kvasir+ClinicDB → Kvasir, ClinicDB, ColonDB, EndoScene, ETIS and BKAI-IGH — [PRISM PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12706544/); [DCSD arXiv HTML](https://arxiv.org/html/2507.10490).
- **TransNetR (MIDL 2023, PMLR 227:1372–1384), protocol (c)** — [PMLR PDF](https://proceedings.mlr.press/v227/jha24a/jha24a.pdf).
  - Protocol: train on Kvasir-SEG (880 images). Test out-of-distribution on PolypGen (all 1,537 frames and each centre C1–C6), 23 PolypGen video sequences, CVC-ClinicDB and BKAI-IGH.
  - TransNetR mDSC: All 0.6668; C1 0.7204, C2 0.7232, C3 0.7874, C4 0.5042, C5 0.4214, C6 0.6917. C4 and C5 are the hardest centres for every model.
  - Single runs, no seeds or statistics.
- **ESPNet (MICCAI 2025, Toman, Subramanian, Ali), protocol (b)**: trained on a 4,000-frame multicentre set (Kvasir-SEG, 1,182 PolypGen C1–C5 images, 2,014 other images). OOD test sets: ETIS, CVC-ColonDB, PolypGen-C6 and an in-house IBD set. Reported mIoU gains vs. the best recent method: +2.84 (ETIS), +1.26 (ColonDB), +1.90 (C6), +3.52 (IBD) — [MICCAI 2025 page](https://papers.miccai.org/miccai-2025/0307-Paper4428.html); [PDF](https://papers.miccai.org/miccai-2025/paper/4428_paper.pdf).
- **KAN-AINet (MICCAI 2026), protocol (b)**: trained on Kvasir-SEG, SUN-SEG and PolypGen C1–C5; tested on Kvasir-Sessile, ColonDB, ETIS and PolypGen-C6. mDice gains over ESPNet: +3.24% (ETIS) and +7.26% (C6). A Brown–Forsythe test supports lower prediction variance — [MICCAI 2026 page](https://papers.miccai.org/miccai-2026/0524-Paper5427.html).
- **S2ME (MICCAI 2023, scribble-supervised), protocol (c)**: trained on SUN-SEG; zero-shot OOD on Kvasir-SEG, CVC-ClinicDB and PolypGen; multiple seeds — [arXiv PDF](https://arxiv.org/pdf/2306.00451); [MICCAI page](https://conferences.miccai.org/2023/papers/559-Paper0318.html).
- **PedSemiSeg (CMIG 124:102591, 2025, semi-supervised), protocol (c)**: trained on SUN-SEG / Kvasir-SEG; external test on all six PolypGen centres; best in 5 of 6 centres; three independent training runs — [author PDF](https://discovery.ucl.ac.uk/10211481/7/Islam_1-s2.0-S0895611125001004-main.pdf); [DOI](https://doi.org/10.1016/j.compmedimag.2025.102591).
- **Federated / DG protocols (d)**:
  - LFDG (Federated self-supervised DG, accepted at the ADSMI @ MICCAI 2024 workshop per arXiv comments) was validated on six centres; the split is not described on the abstract page — [arXiv](https://arxiv.org/abs/2502.07951).
  - FDG polyp (ICASSP 2025) — [Crossref](https://api.crossref.org/works/10.1109/ICASSP49660.2025.10889662); [arXiv](https://arxiv.org/abs/2410.02044).
  - DINO "keys" (arXiv 2025, "under review at MIDL 2026") uses Domain Generalization and Extreme Single-DG protocols on a multi-centre dataset and beats nnU-Net and UM-Net — [arXiv](https://arxiv.org/abs/2512.13376v1).
- **Other cross-domain protocols without PolypGen**:
  - VPTTA (CVPR 2024): test-time adaptation from one of BKAI-IGH, CVC-ClinicDB, ETIS, Kvasir-SEG to the other three — [arXiv HTML](https://arxiv.org/html/2311.18363v4); [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Chen_Each_Test_Image_Deserves_A_Specific_Prompt_Continual_Test-Time_Adaptation_CVPR_2024_paper.html).
  - SCT: Kvasir → ETIS, ClinicDB, EndoCV2020 — [arXiv PDF](https://arxiv.org/pdf/2205.15428).

### Inferences
- Reviewers who know the EndoCV/PolypGen line, especially the Leeds/Oxford group behind PolypGen and ESPNet, may ask why FCT does not use the official C1–C5 → C6 protocol. Precedent for FCT's "train elsewhere, test every centre" choice comes from TransNetR (MIDL 2023), S2ME and PedSemiSeg.
- TransNetR's per-centre table shows C4 and C5 as the hardest centres. FCT's per-centre results can be read against that pattern: if FCT's gains concentrate on C4/C5, that is a natural story.
- Most PolypGen-based papers report single runs. FCT's paired-seed CIs per centre are a real methodological advantage, worth stating plainly.

### Gaps
- I could not verify whether VANet (MedIA 2024, already cited) reports per-centre PolypGen results. ScienceDirect returned 403; a search-engine summary claimed per-centre results, but the README does not mention PolypGen.
- I did not read the exact centre splits of LFDG and FDG (ICASSP 2025), or the DINO-keys protocol.
- A 2026 multi-centre video benchmark (Ghatwary et al., arXiv:2603.04288) exists, but its protocol was not visible on the abstract page — [arXiv](https://arxiv.org/abs/2603.04288).

## 4. Are there semi-supervised or domain-adaptation polyp papers using flip/transformation consistency that a reviewer could confuse with FCT? Note briefly how they differ.

### Takeaway
Yes, several. The most confusable is **PedSemiSeg** (CMIG 2025). It uses horizontal/vertical flips as "weak" views, explicitly motivated as mimicking endoscope viewpoint changes, and evaluates on all six PolypGen centres. But it is semi-supervised, its consistency targets unlabeled images, and its direction (weak teacher → strong students) is fixed and gated by confidence, not by the label. Weakly supervised scale or transformation consistency (WeakPolyp, ScribblePolyp), cross-branch mutual teaching (S2ME), mean-teacher consistency (Park et al. 2022, PolypMixNet, DEC-Seg, CVPR 2025 depth-guided mutual learning) and domain-adaptation methods (VPTTA, TI-ST, MFC) can all be told apart by supervision regime and by how the direction or weight of the consistency is set.

### Cited Findings
- **PedSemiSeg** (Wang, Ma, Bai, Wu, Xu, Zhang, Islam, Ren; CMIG 124:102591, 2025) — [author PDF](https://discovery.ucl.ac.uk/10211481/7/Islam_1-s2.0-S0895611125001004-main.pdf).
  - Method: weak perturbations are Resize, Crop, HorizontalFlip and VerticalFlip, described as mimicking "viewpoint changes caused by endoscope movement". Strong perturbations are ColorJitter. The weak branch is the "teacher" for two strongly augmented "students" through pseudo-labels and complementary labels (positive and negative learning), with confidence threshold τ = 0.8 and entropy-guided peer tutoring.
  - Evaluation: unseen PolypGen centres, three runs.
  - Differs from FCT: semi-supervised; consistency on unlabeled data; geometric views are only the weak/teacher side; confidence rather than the label picks the target.
- **Park et al., Scientific Reports 12:14626 (2022)**: a semi-supervised mean teacher whose consistency over random scaling, Gaussian noise and rotation is applied to unlabeled images, with continuously averaged pseudo-labels. Kvasir-SEG / CVC-612 within-dataset; no unseen centres — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9418164/).
- **PolypMixNet** (CBM 170:108006, 2024): mean teacher with polyp-aware mixup and image- and feature-level consistency (PMPC/PMAC) plus polyp-directed soft pseudo-labels — [Crossref](https://api.crossref.org/works/10.1016/j.compbiomed.2024.108006); [PubMed](https://www.pubmed.gov/38325216/).
- **DEC-Seg** (Pattern Recognition 158:110962, 2025): semi-supervised scale-consistency and cross-generative consistency, applied to polyp, skin and glioma — [arXiv](https://arxiv.org/abs/2312.16039); [Crossref](https://api.crossref.org/works/10.1016/j.patcog.2024.110962).
- **Depth-guided mutual learning** (CVPR 2025, pp. 10394–10403): a semi-supervised mean teacher with a depth-input auxiliary student, cross-modal mutual learning and depth-guided patch augmentation — [CVF](https://openaccess.thecvf.com/content/CVPR2025/html/Li_Boost_the_Inference_with_Co-training_A_Depth-guided_Mutual_Learning_Framework_CVPR_2025_paper.html).
- **FPGM** (arXiv:2508.06517, 2025, no venue): semi-supervised polyp segmentation that aligns the amplitude spectra of unlabeled images to a frequency prior learned from labeled polyp edges. It claims zero-shot generalization on six datasets — [arXiv](https://arxiv.org/abs/2508.06517).
- **Other semi-supervised work**:
  - Online pseudo-labelling with momentum networks (KSE 2022): a slow-copy teacher for pseudo-labels with 20% labels — [arXiv](https://arxiv.org/abs/2209.14599).
  - WS-DefSegNet (CVPRW 2023): weakly / semi-supervised with a weighted consistency loss between two predicted maps — [CVF PDF](https://openaccess.thecvf.com/content/CVPR2023W/VISION/papers/Ren_Towards_Automated_Polyp_Segmentation_Using_Weakly-_and_Semi-Supervised_Learning_and_CVPRW_2023_paper.pdf); [DOI](https://doi.org/10.1109/cvprw59228.2023.00458).
- **Weakly supervised**:
  - WeakPolyp (MICCAI 2023, box labels): scale-consistency loss across rescaled copies — [MICCAI page](https://conferences.miccai.org/2023/papers/721-Paper0147.html).
  - ScribblePolyp (BIBM 2023, pp. 2385–2388): transformation consistency through resizing and cropping, plus affinity propagation, with no teacher — [arXiv PDF](https://arxiv.org/pdf/2311.05122); [DOI](https://doi.org/10.1109/bibm58861.2023.10385811).
  - S2ME (MICCAI 2023, scribbles): mutual teaching between a spatial and a spectral branch — [MICCAI page](https://conferences.miccai.org/2023/papers/559-Paper0318.html).
  - TextPolyp (MICCAI 2024, points): SAM-based mutual learning — [page](https://papers.miccai.org/miccai-2024/768-Paper1293.html).
  - BoxPolyp (MICCAI 2022, extra boxes): cross-network feature consistency — [arXiv HTML](https://arxiv.org/html/2212.03498).
- **Detection, not segmentation**: Consisaug (MLMI 2023, LNCS pp. 64–73) enforces a flip-consistency constraint for polyp *detection*: the class label should match and the boxes should agree under flipping — [arXiv](https://arxiv.org/abs/2404.11355); [DOI](https://doi.org/10.1007/978-3-031-45676-3_7).
- **Domain adaptation**:
  - VPTTA (CVPR 2024): per-image visual prompts with a frozen model, across polyp datasets used as domains — [arXiv HTML](https://arxiv.org/html/2311.18363v4).
  - TI-ST (MICCAI 2023): DA self-training that filters pseudo-labels by invariance to *non-spatial* transformations. Reviewers asked why flips were not used. Not polyp — [MICCAI page](https://conferences.miccai.org/2023/papers/217-Paper1780.html).
  - Masked Frequency Consistency (MICCAI 2023): sim-to-real DA for laparoscopic segmentation (CholecSeg8K) — [MICCAI page](https://conferences.miccai.org/2023/papers/400-Paper0416.html).
  - DCLPS (TMI 2025, already cited): cross-domain polyp DA — [draft related.tex](/home/user/BACFR/paper/sections/related.tex).

### Inferences
- Shortest honest separation: these methods use consistency to *create* supervision where labels are missing (unlabeled pixels or images, or target domains), and they decide trust by confidence, entropy or a teacher. FCT works only where labels exist and uses the label to decide, per pixel, which view is trustworthy. That is something semi-supervised and DA methods cannot do on their unlabeled targets.
- PedSemiSeg's flip-as-viewpoint motivation and its PolypGen-six-centre evaluation overlap closely with FCT's story. Cite it in the semi-supervised paragraph so a reviewer does not see the motivation as unacknowledged.

### Gaps
- I did not read the full texts of PolypMixNet, FPGM or WS-DefSegNet, so their exact augmentation lists (whether flips are used as consistency views) are unverified.
- VPTTA's details beyond its domain list were not needed and were not checked.

## 5. Which 2024-2026 MICCAI / MedIA / TMI papers are closest, and what is the strongest honest distinction for FCT?

### Takeaway
Among 2024–2026 MICCAI papers, none matches FCT's mechanism. The closest are:
- **in framing**: Lite-π (MICCAI 2026, plug-in on lightweight hosts for "generalized polyp segmentation") and ESPNet / KAN-AINet (MICCAI 2025/2026, PolypGen-C6 as unseen);
- **in mechanism**: DASP (MICCAI 2024, multi-host training augmentation with per-pixel uncertainty weighting) and the MICCAI 2023 pair AAC (fully supervised consistency for polyps) and BWCR (fully supervised geometric logit consistency read as per-pixel regularization).

MedIA/TMI 2024–2026 offer VANet (viewpoint-aware architecture), UM-Net and DCLPS, all already cited and all architectural or DA work. The substantively closest papers are outside these venues: SCT (IEEE ISM 2022), PRISM (HTL 2025), DCSD (arXiv 2025) and PedSemiSeg (CMIG 2025).

The strongest honest distinction: FCT is a fully supervised, training-only loss in which the **ground-truth label, compared per pixel against the errors of two exactly related flip views, decides the direction and weight of a stop-gradient consistency pull**. It adds no parameters or test-time cost, is applied unchanged to four recent state-of-the-art polyp hosts each with its own recipe, and is evaluated with paired seeds and CIs on the standard split plus all six PolypGen centres.

### Cited Findings
**2024–2026 MICCAI papers (screened from the full open-access indexes)**
- **Lite-π** (MICCAI 2026; Agnihotri, Majhi, Nayak, Mahapatra, Jha): improves U-Net, U-Net++ and PraNet with foundation-model prototype induction and transformer fusion, adding parameters. Standard Kvasir+ClinicDB training; unseen ColonDB, ETIS, CVC-300 — [MICCAI 2026 page](https://papers.miccai.org/miccai-2026/0495-Paper5591.html); [PDF](https://papers.miccai.org/miccai-2026/paper/5591_paper.pdf). Precursor: Polyp-DiFoM (WACV 2026) — [CVF](https://openaccess.thecvf.com/content/WACV2026/html/Agnihotri_From_SAM_to_DINOv2_Towards_Distilling_Foundation_Models_to_Lightweight_WACV_2026_paper.html).
- **KAN-AINet** (MICCAI 2026): architecture with illumination modulation; multicentre training including PolypGen C1–C5; C6, ETIS and ColonDB unseen — [page](https://papers.miccai.org/miccai-2026/0524-Paper5427.html).
- **Adaptive Frequency-Guided Parallel Mamba** (MICCAI 2026): architecture claiming cross-domain generalization; compares with PraNet, Polyp-PVT, VM-UNet and PraNet-V2 — [page](https://papers.miccai.org/miccai-2026/0032-Paper4551.html); [PDF](https://papers.miccai.org/miccai-2026/paper/4551_paper.pdf).
- **LRBP plugin** and **HadBalance** (MICCAI 2026): plug-in modules or losses evaluated on Kvasir-SEG or CVC-ClinicDB within-dataset — [LRBP](https://papers.miccai.org/miccai-2026/0498-Paper0629.html); [HadBalance](https://papers.miccai.org/miccai-2026/0441-Paper0859.html).
- **SFD-Polyp** (MICCAI 2026): diffusion data improves multiple baselines — [page](https://papers.miccai.org/miccai-2026/0557-Paper4818.html).
- Other MICCAI 2026 polyp papers are SAM / one-shot or video work: SD-SAM — [page](https://papers.miccai.org/miccai-2026/0928-Paper1385.html); RPG-SAM — [page](https://papers.miccai.org/miccai-2026/0907-Paper2529.html).
- **MICCAI 2025**: ESPNet (multicentre training, PolypGen-C6 unseen) — [page](https://papers.miccai.org/miccai-2025/0307-Paper4428.html); CD-PolypNet (SAM feature self-distillation) — [page](https://papers.miccai.org/miccai-2025/0139-Paper2291.html); SAM-MaGuP (boundary distillation, "generalized") — [page](https://papers.miccai.org/miccai-2025/0522-Paper3165.html); SIDM / Polyp-LDM (data synthesis on several segmenters) — [SIDM](https://papers.miccai.org/miccai-2025/0818-Paper3072.html), [Polyp-LDM](https://papers.miccai.org/miccai-2025/0037-Paper2379.html); architectures PolyMamba, HGM and FPN-in-FPN — [PolyMamba](https://papers.miccai.org/miccai-2025/0704-Paper3872.html), [HGM](https://papers.miccai.org/miccai-2025/0423-Paper1195.html), [FPN-in-FPN](https://papers.miccai.org/miccai-2025/0351-Paper4899.html).
- **MICCAI 2024**:
  - DASP: four hosts, per-pixel uncertainty-weighted adversarial augmentation, within-dataset splits — [page](https://papers.miccai.org/miccai-2024/809-Paper2811.html).
  - TextPolyp: plug-and-play point supervision — [page](https://papers.miccai.org/miccai-2024/768-Paper1293.html).
  - ASPS: already cited.
  - Architectures Polyp-Mamba, LSSNet and QueryNet, on the standard five sets — [Polyp-Mamba](https://papers.miccai.org/miccai-2024/612-Paper0697.html), [LSSNet](https://papers.miccai.org/miccai-2024/482-Paper1926.html), [QueryNet](https://papers.miccai.org/miccai-2024/634-Paper1037.html).
- **Single-source DG for medical segmentation at MICCAI 2024–2026** mostly targets fundus, prostate or other modalities, not polyps: HSD — [page](https://papers.miccai.org/miccai-2024/369-Paper3893.html); LRFS (prostate) — [page](https://papers.miccai.org/miccai-2024/704-Paper3740.html); ConStyX (fundus) — [page](https://papers.miccai.org/miccai-2025/0172-Paper0279.html); SSUG (fundus and prostate) — [page](https://papers.miccai.org/miccai-2026/0934-Paper4696.html).

**MedIA / TMI 2024–2026 (found through searches, not full table-of-contents scans)**
- VANet (MedIA 2024; viewpoint classes parallel/vertical, architecture) — [README](https://github.com/1024803482/Viewpoint-Aware-Network).
- UM-Net (MedIA 2025) and DCLPS (TMI 2025): already cited.
- "Text-Image Co-Alignment for Weakly Supervised Polyp Segmentation", TMI 45:3394–3408 (2026). Weakly supervised; not read — [DOI](https://doi.org/10.1109/tmi.2026.3674592).

**Substantively closest papers outside those venues** (details in Q1–Q4)
- SCT (IEEE ISM 2022) — [arXiv PDF](https://arxiv.org/pdf/2205.15428)
- PRISM (HTL 2025) — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12706544/)
- DCSD (arXiv 2025) — [arXiv HTML](https://arxiv.org/html/2507.10490)
- PedSemiSeg (CMIG 2025) — [author PDF](https://discovery.ucl.ac.uk/10211481/7/Islam_1-s2.0-S0895611125001004-main.pdf)
- AAC and BWCR (MICCAI 2023) — [AAC](https://conferences.miccai.org/2023/papers/069-Paper2200.html); [BWCR](https://conferences.miccai.org/2023/papers/094-Paper2691.html)

### Inferences
**Ranking of closest prior work** (most to least likely to be raised by a MICCAI polyp reviewer):
1. **SCT (2022)**: the same paradigm (fully supervised consistency instead of plain augmentation, label-aware inconsistency, adaptive weighting, four hosts, 10 runs, ETIS gains).
   - Distinction: FCT's consistency is *directed and per pixel*. The label decides, per pixel, whether the flipped view is pulled toward the upright view or the upright toward the closest flipped view, and how strongly, through comparison of each view's error. Targets use stop-gradient. The views form an exact, lossless flip group, each supervised by the host loss at weight 1/3. SCT applies a symmetric soft-set inconsistency with one batch-level IoU weight to a single augmented view from a mixed pipeline (RandomRotate90 + photometric), on generic segmentation-models-pytorch networks trained on Kvasir only.
2. **PRISM (2025)**: training-only plug-in on three polyp hosts with an unseen PolypGen centre and Wilcoxon tests.
   - Distinction: no EMA teacher and no temporal targets; cross-view targets within one forward pass; the label gates direction. FCT reports paired-seed CIs on four state-of-the-art hosts and six centres, against one centre for PRISM.
3. **DCSD (2025, unrefereed)**: label (Dice) confidence weighting of self-distillation.
   - Distinction: a per-pixel comparison of two views' errors against the label, versus a scalar per-batch Dice weight on previous-iteration targets.
4. **AAC (MICCAI 2023)**: fully supervised polyp consistency regularization on several backbones.
   - Distinction: the weight comes from model uncertainty, not the label. The perturbation is adversarial noise, not an exact geometric transform. Evaluation is within-dataset only.
5. **BWCR (MICCAI 2023, MRI)**: geometric logit consistency read as a spatially varying regularizer under full supervision.
   - Distinction: a fixed boundary-distance weight aimed at calibration, versus an error-comparison weight with direction aimed at unseen-centre accuracy. It also anticipates FCT's "consistency = per-pixel reweighting" identity, so cite it next to that identity.
6. **PedSemiSeg (2025)**: flips as viewpoint proxies plus PolypGen six-centre testing, but semi-supervised with confidence gating.
7. **Lite-π / Polyp-DiFoM (2026)**: "plug into several hosts for generalized polyp segmentation", but they add modules or foundation-model teachers and use lightweight hosts. FCT adds nothing at inference and uses state-of-the-art hosts.
8. **TransNetR (2023)**: protocol precedent for testing every PolypGen centre as unseen.
9. **VANet (2024)**: motivation precedent (viewpoint), handled in the architecture.

**Claims FCT should avoid**, because the sources above contradict them:
- "first consistency training for fully supervised polyp segmentation" (SCT);
- "first plug-in training loss evaluated on several polyp hosts" (SCT, PRISM, DCSD, AAC, DASP);
- "first to weight consistency or self-distillation by the label" (SCT's label-aware SIL, DCSD's Dice weight; plus CA-MKD-style label-closeness weighting already cited in the draft);
- "first unseen-centre / PolypGen evaluation" (TransNetR, PRISM, the Bayesian loss, ESPNet, KAN-AINet, S2ME, PedSemiSeg);
- "first with repeated-seed statistics" (SCT, BDKD-Net, S2ME, PedSemiSeg).

**Defensible novelty statement**, supported by the absence of any counter-example in this search: in polyp segmentation, no verified prior work (a) uses the label per pixel to compare two exactly related views' errors and so set both the *direction* and the *weight* of a stop-gradient consistency pull, and (b) does so as an unchanged, parameter-free, training-only loss across four recent state-of-the-art polyp hosts with their own recipes, (c) evaluated per centre on all six PolypGen centres with paired seeds and CIs.

**Likely reviewer requests** that follow from this map (use FCT's component dissection to answer them):
- compare with or ablate against SCT-style symmetric consistency with batch-IoU weighting;
- compare with an EMA / self-distillation baseline (PRISM-style);
- show that the gains are not just from three supervised views (an effectively larger batch / repeated augmentation);

**Citations to add to the draft**, since none are in references.bib: SCT, PRISM, DCSD, AAC, BWCR, PedSemiSeg, TransNetR, DASP. Optional: BoxPolyp, WeakPolyp, S2ME, ScribblePolyp, Lite-π, ESPNet, KAN-AINet, BDKD-Net, the Bayesian uncertainty-weighted loss and Consisaug.

### Gaps
- MICCAI 2024–2026 were screened by title only (856, 1,027 and 1,165 entries). A paper with a non-obvious title (for example, an equivariance regularizer named only by its acronym) could have been missed.
- MedIA and TMI 2024–2026 were covered by keyword searches, not by scanning their tables of contents.
- I did not read the full texts of the following, so their methods were not checked for flip consistency: TMI 2026 "Text-Image Co-Alignment"; TMI 2025 "Robust Polyp Detection and Diagnosis Through Compositional Prompt-Guided Diffusion Models" ([DOI](https://doi.org/10.1109/tmi.2025.3589456)); UM-Net and VANet (both already cited).
- DCSD's peer-review status is unknown beyond arXiv v1 (July 2025), so it should be cited as a preprint — [arXiv](https://arxiv.org/abs/2507.10490).
