# Views drive FCT's gains; label arbitration makes it new

FCT's methodological novelty is moderate and narrow, and the story as written steers reviewers away from it. Supervising several views with a cross-view consistency term under full supervision already exists for polyps in Segmentation Consistency Training (SCT, IEEE ISM 2022), whose label-aware loss also gains most on ETIS. FACT (CVPR 2021) already uses two directed pulls. The lag-or-margin weight is a soft teacher-bounded weight with close precedents in distillation, and the gradient identity behind it is the binary case of a known distillation result. What no searched source contains is the label deciding, pixel by pixel, which of one network's exactly related flip views teaches which, with the test-time view as the default teacher. Because pull (b) keeps its label gate in every both-pull variant, that element already sits inside the story's "core" without being named. The evidence cannot yet isolate it. On the Polyp-PVT five-set mean all arms are statistically indistinguishable: views alone reach +0.37 of FCT's +0.52 points (+1.48 of +1.65 on ETIS), pull (a) alone matches FCT head to head, and a single-pull arm with a different weight clears zero. So the data do not support the claim that both pulls are needed. For MICCAI 2027:

- Position FCT as a parameter-free, training-only loss in which the label decides which flip view teaches which.
- Sell it on unseen-centre gains: ETIS +1.65, and PolypGen pooled +0.86 on Polyp-PVT and +2.36 on PraNet-V2, positive on every seed.
- Recast the robustness to the weight and margin as "nothing to tune".
- Credit the identity and the weight to distillation work.
- Add the one arm the ablation lacks: both pulls with no label at all.

## Every ingredient has a precedent; the single-network, per-pixel combination does not

Supervising several transformed copies of each image is standard practice. Batch augmentation replicates samples with different augmentations inside a batch and "acts as a regularizer" ([Hoffer et al.](https://arxiv.org/abs/1901.09335)). Drawing several augmentations per image raises accuracy "even if different numbers of augmentations per image perform the same number of parameter updates and gradient evaluations" ([Fort et al.](https://arxiv.org/abs/2105.13343)). Adding an agreement term between views under full supervision is equally established, including in medical segmentation. Bortsova et al.'s SupTC applies a transformation-consistency loss to labeled chest X-rays only. Both branches are supervised, an inverse-transform layer maps the predictions to a common frame, and SupTC beats the same-augmentation baseline in **24 of 25 experiments** ([Bortsova et al., MICCAI 2019](https://ar5iv.labs.arxiv.org/html/1911.01218)). Other examples:

- SHADE and Ouyang et al. supervise two views and add a pixel-wise divergence for domain-generalized segmentation ([SHADE](https://ar5iv.labs.arxiv.org/html/2204.02548); [Ouyang et al.](https://ar5iv.labs.arxiv.org/html/2111.12525)).
- A NeurIPS 2022 loss enforces flip-and-crop equivariance in fully supervised instance segmentation ([Wang et al.](https://ar5iv.labs.arxiv.org/html/2210.00911)).
- Consisaug enforces flip consistency for polyp *detection* ([Consisaug](https://ar5iv.labs.arxiv.org/html/2404.11355)).

The draft cites Bortsova et al. only for consistency on "unlabeled scans" ([related.tex](/home/user/BACFR/paper/sections/related.tex)). A reviewer who knows SupTC will notice.

The most damaging single paper is SCT. It trains on Kvasir-SEG with a label-aware Segmentation Inconsistency Loss between an unaugmented and an augmented view, and applies it to four networks with ten runs each. It gains most on unseen ETIS (DeepLabV3+ **0.472 → 0.505 mIoU**, FPN 0.440 → 0.475) while Kvasir stays flat, which is FCT's own pattern ([SCT](https://arxiv.org/pdf/2205.15428); [DOI](https://doi.org/10.1109/ism55400.2022.00012)). The differences from FCT are real but mechanical. SCT's loss:

- is symmetric, and is computed in the image frame without mapping the views back;
- uses one random view from a pipeline that mixes 90-degree rotations with photometric changes;
- balances consistency against segmentation with one batch-level IoU;
- does not control for the number of views, a limitation its authors state ([SCT](https://arxiv.org/pdf/2205.15428)).

Three more papers sit around SCT:

- **PRISM and DCSD (2025).** Training-only polyp plug-ins tested on three hosts with an unseen PolypGen centre. PRISM uses an EMA teacher ([PRISM](https://pmc.ncbi.nlm.nih.gov/articles/PMC12706544/)). DCSD, a preprint, weights its self-distillation term by one Dice score per batch against the label ([DCSD](https://arxiv.org/html/2507.10490)).
- **BWCR (MICCAI 2023).** It already reads logit consistency across geometric transforms as a spatially varying, per-pixel regularizer under full supervision ([BWCR](https://conferences.miccai.org/2023/papers/094-Paper2691.html)).
- **PedSemiSeg.** It uses horizontal and vertical flips as proxies for changes in endoscope viewpoint and tests all six PolypGen centres, though in a semi-supervised setting ([PedSemiSeg](https://discovery.ucl.ac.uk/10211481/7/Islam_1-s2.0-S0895611125001004-main.pdf)).

The direction of the pull and the role of the label also have precedents. FACT runs two directed pulls between original and amplitude-mixed views and finds both directions together better than either alone (PACS **83.44 → 84.51**; single directions 83.67 and 83.84). But both of its pulls are always on and target an EMA teacher whether or not it is correct ([FACT](https://ar5iv.labs.arxiv.org/html/2105.11120)). KDCL-Naive takes as its target the peer whose cross-entropy with the label is smallest, and each peer sees its own distortion of the image ([KDCL](https://openaccess.thecvf.com/content_CVPR_2020/papers/Guo_Online_Knowledge_Distillation_via_Collaborative_Learning_CVPR_2020_paper.pdf)).

Closest to pull (b) is the pixel-wise bidirectional selective distillation of Zhu et al. A binary map sends knowledge from a CNN student to a ViT student wherever the CNN's pixel cross-entropy with the ground truth is lower, and the other way elsewhere ([Zhu et al., ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Zhu_A_Good_Student_is_Cooperative_and_Reliable_CNN-Transformer_Collaborative_Learning_ICCV_2023_paper.html)). Binary cross-entropy is monotone in |p − y|, so this is **exactly FCT's "closer to the label" criterion**. Zhu et al. apply it between two networks on one input; FCT applies it between flip views of one network. Other label-aware relatives:

- CTRCL, evaluated on Kvasir-SEG, corrects each peer's wrong pixels with the ground truth before mutual learning ([CTRCL](https://arxiv.org/abs/2408.13698)).
- MDCS and MART gate or weight a cross-view consistency term by one view's correctness, at the sample level ([MDCS](https://arxiv.org/abs/2308.09922); [MART](https://iclr.cc/virtual_2020/poster_rklOg6EFwS.html)).

The weight, the identity and the gradient measurement are the least original parts.

- **The weight.** Giving full weight where the flipped view lags is the teacher-bounded condition of Chen et al. with zero margin ([Chen et al., NIPS 2017](https://papers.nips.cc/paper/6676-learning-efficient-object-detection-models-with-knowledge-distillation.pdf)). A ramp that falls as the target's own error grows is the attentive imitation loss ([Saputra et al., ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Saputra_Distilling_Knowledge_From_a_Deep_Pose_Regressor_Network_ICCV_2019_paper.html)). At a = 2 the ramp is zero where e_u ≥ 0.5, which is the teacher-correctness mask of adaptive cross-entropy in segmentation distillation ([Park and Heo](https://www.mdpi.com/1424-8220/20/16/4616)). The closest single relative is weighted soft labels, w = 1 − exp(−L_s/L_t) ([WSL, ICLR 2021](https://arxiv.org/abs/2102.00650)). FCT's weight differs because, once the student leads, it depends on the target's absolute error rather than on the ratio of the two errors.
- **The identity.** ∂ℓ(z_s; p_t)/∂z_s = (1 − e_t/e_s)(p_s − y) is Tang et al.'s gradient-rescaling ratio at T = 1; they call the two-class case "trivial to show" ([Tang et al.](https://arxiv.org/abs/2002.03532)). FCT's combined factor β + γω is PS-KD's Eq. 11, 1 − α·γ_{t−1}/γ_t, where γ is the probability of being incorrect ([PS-KD](https://arxiv.org/abs/2006.12000)). The sign condition appears as WSL's "regularization samples" and as DHO's gradient-conflict ball ([DHO](https://arxiv.org/abs/2505.07675)).
- **The cosine.** The cosine between an auxiliary gradient and the main gradient is a standard diagnostic ([Du et al.](https://arxiv.org/abs/1812.02224); [PCGrad](https://arxiv.org/abs/2001.06782)). Negative values have been reported between supervised and consistency gradients in semi-supervised segmentation ([POS, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Sun_Two_Losses_One_Goal_Balancing_Conflict_Gradients_for_Semi-supervised_Semantic_ICCV_2025_paper.pdf)).

| FCT element | Closest prior work | What it already does | What FCT changes | Novelty |
|---|---|---|---|---|
| Three supervised views | Batch and multi-sample augmentation ([Hoffer](https://arxiv.org/abs/1901.09335); [Fort](https://arxiv.org/abs/2105.13343)) | Several augmented copies per image, each supervised | Fixed flip orbit, weight 1/3 each | None |
| Consistency under full supervision | SCT ([ISM 2022](https://arxiv.org/pdf/2205.15428)); SupTC ([MICCAI 2019](https://ar5iv.labs.arxiv.org/html/1911.01218)) | Polyp consistency with a label-aware loss and ETIS gains; supervised-only consistency with inverse mapping | Directed, detached, per-pixel pulls after exact mapping back | Low |
| Two directed pulls | FACT ([CVPR 2021](https://ar5iv.labs.arxiv.org/html/2105.11120)) | Original ↔ augmented, both always on, EMA teacher | The label gates the direction | Low |
| The label decides who teaches (pull (b)) | Zhu et al. ([ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Zhu_A_Good_Student_is_Cooperative_and_Reliable_CNN-Transformer_Collaborative_Learning_ICCV_2023_paper.html)); KDCL ([CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Guo_Online_Knowledge_Distillation_via_Collaborative_Learning_CVPR_2020_paper.pdf)) | Same per-pixel criterion between two networks; label-chosen target per sample across distorted peers | One network, flip views, upright anchor | Moderate |
| Lag-or-margin weight (pull (a)) | Teacher-bounded loss ([NIPS 2017](https://papers.nips.cc/paper/6676-learning-efficient-object-detection-models-with-knowledge-distillation.pdf)); WSL ([ICLR 2021](https://arxiv.org/abs/2102.00650)); ACE ([Sensors 2020](https://www.mdpi.com/1424-8220/20/16/4616)) | Weight by student versus teacher error; zero where the teacher is wrong | Ramp in the target's absolute error; constant set by the 0.5 threshold | Low to moderate |
| Gradient identity | Tang et al. ([arXiv 2020](https://arxiv.org/abs/2002.03532)); PS-KD ([ICCV 2021](https://arxiv.org/abs/2006.12000)) | Same ratio and combined factor; binary case "trivial" | Exact per pixel, applied to view consistency | Low, as theory |
| Gradient cosine | Du et al. ([arXiv 2018](https://arxiv.org/abs/1812.02224)); POS ([ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Sun_Two_Losses_One_Goal_Balancing_Conflict_Gradients_for_Semi-supervised_Semantic_ICCV_2025_paper.pdf)) | Standard diagnostic; negative cosines reported | Shows the weights double the alignment | None, as a method |
| Multi-host plug-in, unseen centres | SCT; PRISM ([HTL 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12706544/)); DCSD ([arXiv 2025](https://arxiv.org/html/2507.10490)) | Training-only plug-ins on 3–4 hosts, one unseen centre | Unchanged loss on state-of-the-art hosts with their own recipes; paired-seed CIs per PolypGen centre | Low to moderate (empirical) |

The table leaves one cell empty in the literature searched. Per pixel, within one network, between exact flip views mapped back to the upright frame, the label decides which view teaches which. The test-time view acts as an anchor and gives up that role only where a flipped view is closer to the label. FCT's methodological novelty is therefore **moderate and narrow: a new combination and setting, not a new principle**.

The draft claims that FCT "is the first consistency loss that uses the label to set, pixel by pixel, the direction in which the views of an image are drawn together" ([introduction.tex](/home/user/BACFR/paper/sections/introduction.tex)). That claim survives the search, provided the paper adds "within one network" and cites Zhu et al. and KDCL as the nearest label-arbitrated schemes. The claim rests on absence of evidence, so it needs a final check before submission: the search screened MICCAI 2023–2026 by title only, did not query Google Scholar, and could not read one 2026 polyp self-distillation paper ([Neurocomputing 2026](https://doi.org/10.1016/j.neucom.2026.134234)).

## Pull (b)'s label gate already puts the novel element inside the core

The story as written places FCT's novelty where reviewers will not look for it:

- Its first clause leads with views plus consistency, the least novel ingredients.
- Its second credits the label-aware weights on pull (a), which have the closest precedents.
- Its third tells the reviewer that changing those weights does not matter.

Read in that order, the paper invites the summary "SCT or FACT applied to flips, with a WSL-style weight that the authors show is optional".

The code fact changes what the clauses mean. In the variant named "FCT without the weights" (S3FUP), only pull (a) loses its weight. Pull (b) still uses the label to pick the best flipped view, and acts only where that view beats the upright one. Pixel by pixel, this variant pulls one way (flipped toward upright) where the upright view is closer to the label, and both ways where some flipped view is closer. **The label decides where the reverse pull switches on.** So it is not symmetric consistency of the [R-Drop](https://arxiv.org/abs/2106.14448) or SupTC kind; that equivalence holds only if pull (b) loses its gate.

The lag-only variant (S3RELUP) is the purest form. It pulls each pixel only from the less accurate view toward the more accurate one, which is Zhu et al.'s selective rule moved inside one network. Every both-pull arm that "holds up" keeps this label arbitration. The third clause therefore says that the gain survives changes to *how pull (a) uses the label*. It does not say the label is dispensable.

| Arm (code) | Pull (a), flipped → upright | Pull (b), upright → best flipped | Label decides who teaches? | Five-set gain, points (p) | ETIS gain |
|---|---|---|---|---|---|
| Views only (S3) | none | none | no consistency term | +0.37 (0.052) | **+1.48** |
| + pull (b) only | none | label-gated | reverse pull only | +0.28 (0.074) | not in records read |
| + pull (a) only (S3LM) | lag-or-margin | none | no, weight only | +0.47 (0.058) | **+1.97** |
| + pull (a) only, graded (S3SIG) | confidence-graded | none | no, weight only | **+0.40 (0.013)** | **+1.56** |
| FCT, a = 2 (S3LMUP) | lag-or-margin | label-gated | yes | **+0.52 (0.016)** | **+1.65** |
| FCT, a = 0.5 / 1 / 1.5 | lag-or-margin | label-gated | yes | **+0.51 (0.002)** / +0.33 (0.059) / **+0.36 (0.042)** | **+2.15 / +1.58 / +1.83** |
| Lag only (S3RELUP) | hard gate | label-gated | yes, both ways | **+0.34 (0.015)** | not in records read |
| "Unweighted" (S3FUP) | w = 1 | label-gated | yes, through pull (b) | **+0.33 (0.034)** | not in records read |
| Label-free both pulls | w = 1 | toward flipped views everywhere | no | not run | not run |

Polyp-PVT, six paired seeds, gains over vanilla. Bold marks paired 95% CIs that exclude zero. Five-set gains come from the project brief. The S3SIG row, the a-sweep p-values and the ETIS gains come from [facts.md](/home/user/BACFR/paper/facts/facts.md) and [tab_rules.tex](/home/user/BACFR/paper/tables/tab_rules.tex). The six-seed ETIS values for S3RELUP and S3FUP were not in the records read.

Two consequences follow:

- **Row (f) must be described accurately.** The paper must say plainly that row (f) keeps pull (b)'s label gate. The draft labels it "FCT, unweighted" ([tab_rules.tex](/home/user/BACFR/paper/tables/tab_rules.tex)), and method.tex calls w = 1 "the unweighted transformation consistency of semi-supervised segmentation" ([method.tex](/home/user/BACFR/paper/sections/method.tex)). Both invite readers to assume the label has left the consistency term.
- **The decisive arm is missing.** The ablation never removes the label from the consistency term while keeping both pulls. The missing arm keeps pull (a) unweighted and pulls the upright view toward the flipped views everywhere, with no best-view selection. That is symmetric consistency in SCT's and R-Drop's sense, and it is **the only arm that tests the novelty claim directly**.

## Six paired seeds cannot attribute the gain among views, pulls and weights

The arms differ by less than their noise:

- Views alone give +0.37 [−0.01, +0.74] on the five-set mean and +1.48 [+0.32, +2.65] on ETIS. FCT gives +0.52 [+0.15, +0.90] and +1.65 [+0.88, +2.42].
- The consistency term's own effect over the views is +0.16 [−0.30, +0.62] ([tab_rules.tex](/home/user/BACFR/paper/tables/tab_rules.tex)).
- FCT minus pull (a) alone is +0.05 [−0.36, +0.47] on the five-set mean, and −0.32 on ETIS, where pull (a) alone reaches +1.97 ([facts.md](/home/user/BACFR/paper/facts/facts.md)).
- The draft itself puts the CI of a paired difference between two variants at about ±0.4 to ±0.7 points with six seeds ([experiments.tex](/home/user/BACFR/paper/sections/experiments.tex)).

Two records make the pattern "both-pull arms clear zero, reduced arms do not" weaker than the brief states:

- **A single-pull arm clears zero.** The arm with a confidence-graded weight (S3SIG, row (h)) reaches **+0.40 [+0.13, +0.68], p 0.013, positive on all six seeds**. It also reaches +0.84 [+0.59, +1.09] pooled on PolypGen, level with FCT's +0.86 ([facts.md](/home/user/BACFR/paper/facts/facts.md); [tab_polypgen.tex](/home/user/BACFR/paper/tables/tab_polypgen.tex)). The pattern therefore holds only for single-pull arms built with FCT's own weight.
- **The threshold depends on the test.** The project's permutation p-values put views alone at 0.026 and pull (a) alone at 0.032, both below 0.05. ASGNet's five-set gain passes the t-test (0.049) and fails the permutation test (0.203) ([facts.md](/home/user/BACFR/paper/facts/facts.md)).

The within-noise result that supports the robustness clause is the same result that rules out claiming both pulls are necessary.

The gradient records explain why the arms collapse together. The consistency gradient's norm is **7–12% of the supervised gradient's** on Polyp-PVT, 5% on PraNet-V2 and under 2% on ASGNet ([facts.md](/home/user/BACFR/paper/facts/facts.md)). The three-view supervision supplies nearly all of the training signal and the consistency term only steers it, so the views carry most of the measured gain.

The margin factor acts on fewer pixels still. In the PraNet-V2 and ASGNet runs that logged it, a flipped view leads the upright on about 44% of pixels at the end of training, but leads a *wrong* upright on only **0.24%** ([facts.md](/home/user/BACFR/paper/facts/facts.md)). Only on that 0.24% does a = 2 zero a pull that a smaller a keeps. Elsewhere, a scales the weight 1 − a·e_u, which stays close to 1 wherever the upright prediction is confidently correct. This is the most direct explanation of why the results barely change with a. Indeed a = 0.5, which never zeroes the pull, gives the tightest interval and smallest p-value of any arm: +0.51 [+0.30, +0.73], p 0.002. The guarantee never to pull toward a wrong upright is a sound design property, but it is not what moves Dice.

The weights do have one large, reproducible effect. With lag-based weights, the cosine between the consistency gradient on the flipped views and the gradient of the rest of the loss is **0.53–0.56**. It is 0.23 without weights and 0.26 with the graded weight, and the two groups are separated on every seed ([experiments.tex](/home/user/BACFR/paper/sections/experiments.tex)). Three qualifications belong beside this result:

- The measurement follows established practice; it does not introduce a new diagnostic ([Du et al.](https://arxiv.org/abs/1812.02224)).
- The identity makes each pixel's logit gradient parallel to its supervised gradient. The parameter-space cosine also depends on cross-pixel Jacobian terms, so the doubling is empirical support, not a corollary of the proposition.
- The cosine depends on the host. FCT's own value is 0.53 on Polyp-PVT, 0.40 on PraNet-V2 and 0.20 on ASGNet; the last equals the unweighted Polyp-PVT value ([facts.md](/home/user/BACFR/paper/facts/facts.md)). "The weights double the alignment" is therefore a within-host statement.

Distillation shows the same insensitivity to the exact form of the gate. In WSL's CIFAR-100 experiments a hard gate reached 74.59 against standard KD's 74.92 ([WSL](https://arxiv.org/pdf/2102.00650)). FCT's null differences among w = 1, the lag-only gate and a ∈ {0.5, 1, 1.5, 2} fit that pattern.

Prior work makes FCT's dissection plausible without rescuing its strong form. With the number of views held fixed, a symmetric consistency term adds to K-view loss averaging in classification: CIFAR-100 **75.0 → 75.9%**, ImageNet NF-ResNet-101 79.3 → 80.1% ([Botev et al.](https://ar5iv.labs.arxiv.org/html/2203.03304)). Consistency also adds to cross-entropy on both views in SHADE (35.68 → 39.67 mean mIoU on unseen domains) ([SHADE](https://ar5iv.labs.arxiv.org/html/2204.02548)). Against the strong form, each of FACT's single directions beats no consistency ([FACT](https://ar5iv.labs.arxiv.org/html/2105.11120)), and one-way consistency beats two-way in semi-supervised segmentation ([Grubišić et al.](https://arxiv.org/abs/2106.07075)). On six seeds, a reviewer will not accept that single pulls fail.

The signal lives on unseen centres, as it did for SCT:

- **Polyp-PVT.** ETIS alone contributes 0.33 of the five-set mean's 0.52 points. Each of the other four sets moves by less than 0.6 points, with a CI that spans zero ([facts.md](/home/user/BACFR/paper/facts/facts.md)).
- **PolypGen.** Pooled over the six centres, FCT gains **+0.86 [+0.37, +1.34]** on Polyp-PVT and **+2.36 [+1.00, +3.71]** on PraNet-V2, positive on every seed.
- **PraNet-V2, five sets.** The gain of +1.68 [+0.69, +2.67] includes +3.01 on CVC-ColonDB, positive on all six seeds.
- **ASGNet.** Its own recipe already applies random flips, and it gains least: +0.31 on the five-set mean, with only ETIS (+1.69 [+0.08, +3.30]) clearly up ([facts.md](/home/user/BACFR/paper/facts/facts.md)).

Large gains on unseen sets with a flat in-distribution score is SCT's pattern too. SCT, PRISM and DCSD report unseen-set gains of roughly 2–4 points ([SCT](https://arxiv.org/pdf/2205.15428); [PRISM](https://pmc.ncbi.nlm.nih.gov/articles/PMC12706544/); [DCSD](https://arxiv.org/html/2507.10490)). FCT's Polyp-PVT gains are smaller, but they start from a much stronger baseline: vanilla Polyp-PVT already scores 78.6% Dice on ETIS.

The flip gap does not separate the components either. In single runs, views alone reached 8.2%, and Polyp-PVT's official random flip-and-rotation augmentation reached 6.8%, below FCT's 7.5%. That augmentation run scored 85.75% five-set Dice, below every vanilla seed ([experiments.tex](/home/user/BACFR/paper/sections/experiments.tex)).

## Position FCT as label-arbitrated flip training for unseen centres

Rewrite the three clauses so that each rests on the evidence that supports it.

- **The core.** It becomes label-arbitrated flip consistency on top of multi-view supervision. One network sees three flip views. The upright view, the one used at test time, teaches each flipped view, except where that flipped view is closer to the label; there, the closest flipped view teaches the upright one. The design rationale is the per-pixel identity, credited to distillation: it shows that the label determines, at every pixel, whether a pull moves a view toward the label or away from it. Semi-supervised consistency never has that information.
- **The weights.** Claim what the weights demonstrably do. They align the consistency gradient with supervision, never pull toward a wrong upright prediction, and bound the harmful movement on a leading pixel by (1 − 2e_u)·e_u ≤ 1/8. Do not claim that they raise Dice.
- **The robustness.** It becomes "nothing to tune", explained by the 0.24% wrong-lead share.

The views' large share of the gain belongs in the main text, framed by Fort et al. and Hoffer et al. The draft already says that multi-view supervision accounts for most of the ETIS gain; it should say the same for the five-set mean and, once measured, for PolypGen. A share that readers discover in a table reads as concealment; the same share stated in the narrative reads as rigor.

Lead with the unseen centres and with the evaluation design. The search found no prior polyp plug-in applied unchanged to recent state-of-the-art hosts, each with its own recipe. SCT used generic segmentation-models-pytorch networks, and PRISM and DCSD used PraNet, TransNetR and ShallowNet ([SCT](https://arxiv.org/pdf/2205.15428); [PRISM](https://pmc.ncbi.nlm.nih.gov/articles/PMC12706544/); [DCSD](https://arxiv.org/html/2507.10490)).

The protocol has a precedent worth citing. TransNetR also trained on Kvasir-SEG and tested every PolypGen centre, and found C4 and C5 hardest for every model (mDSC 0.50 and 0.42) ([TransNetR](https://proceedings.mlr.press/v227/jha24a/jha24a.pdf)). The project's vanilla Polyp-PVT is also weakest there, at 68.3% and 65.7% ([tab_polypgen.tex](/home/user/BACFR/paper/tables/tab_polypgen.tex)). FCT's per-centre table should be read against that pattern.

MICCAI reviewers have asked polyp plug-in papers for significance tests when gains are small, and for transformer hosts ([AAC reviews](https://conferences.miccai.org/2023/papers/069-Paper2200.html); [SIDM reviews](https://papers.miccai.org/miccai-2025/0818-Paper3072.html)). FCT's paired-seed design answers both, provided the SAM2-UNet and ASGNet results appear whatever they show. A title in this spirit: "Let the Label Decide Who Teaches: Flip-Consistency Training for Polyp Segmentation on Unseen Centres".

The paper should not make any of the following claims, even in softer form, because the sources contradict them:

- the first consistency training for fully supervised polyp segmentation (SCT);
- the first plug-in loss evaluated on several polyp hosts (SCT, PRISM, DCSD);
- the first label-weighted consistency (SCT's label-aware loss, DCSD's Dice weight);
- the first evaluation on unseen centres or on PolypGen (TransNetR, PRISM);
- the first study with repeated-seed statistics: SCT used ten runs with t-tests, and [BDKD-Net](https://pmc.ncbi.nlm.nih.gov/articles/PMC13412844/) used paired seed CIs.

Three more claims go beyond the evidence: that both pulls are necessary, that the weights cause the gain, and that the identity is a new theorem. The draft carries several of these in softer forms. The table lists them with replacements.

| Draft location | Current wording | Problem | Replacement |
|---|---|---|---|
| introduction.tex, ll. 69–71 | "the first consistency loss that uses the label to set, pixel by pixel, the direction in which the views of an image are drawn together" | Holds, but Zhu et al. set the direction per pixel between two networks | Add "within one network"; cite Zhu et al. and KDCL in related work |
| introduction.tex, contribution 2 | "We show that ... a consistency term ... acts pixel by pixel as the supervised loss with an adaptive weight" | Tang et al. and PS-KD give the same identity | "We use the binary, per-pixel form of the distillation gradient-rescaling identity to show where a multi-view pull helps or hurts, and derive label-aware weights from it" |
| introduction.tex, ll. 86–88; experiments.tex, l. 203 | "FCT is the only variant that improves all five test sets on average" | Four of the five per-set CIs span zero; counting positive means is not a test | Drop it as evidence; report the paired contrasts |
| related.tex, l. 46 | Bortsova et al. cited for "unlabeled scans" | SupTC applies consistency to labeled images only | Cite SupTC and SCT as fully supervised consistency in medical segmentation |
| related.tex, ll. 62–64 | "none uses the label to compare views" | SCT, [GJS](https://ar5iv.labs.arxiv.org/html/2105.04522), KDCL, MDCS and MART use the label in view consistency | "Where the label enters, it is an extra target, the expected change, or a sample-level selector across networks; none compares one network's views pixel by pixel to decide which teaches" |
| related.tex, ll. 92–95 | "neither corrects a target nor ranks separate teachers" | Pull (b) ranks flipped views by the label, as KDCL-Naive ranks peers; Zhu et al. compare errors per pixel | "Label-arbitrated distillation compares two networks on one input; FCT compares one network's predictions on flip views of one image, pixel by pixel, with the test-time view as anchor" |
| analysis.tex, Prop. 1 and l. 89 | Proposition stated without credit; "it brings no information beyond the label" | Tang's Eq. 3, PS-KD's Eq. 11 and WSL's sign condition | Credit before the proposition, or relabel it an observation. Rewrite as "it introduces no target beyond the label and reweights the label's gradient by how two views' errors compare". State the scope: binary label, detached target |
| method.tex, end of "Relation to Other Losses" | "Of these rules, only FCT compares the two errors" | WSL, KDCL and Zhu et al. compare errors with the label | "Among consistency rules, FCT compares two views' errors with the label at each pixel; distillation does so per sample (WSL) or between networks (Zhu et al.)" |
| tab_rules.tex, row (f) | "FCT, unweighted" | Pull (b) keeps its label gate | "FCT, pull (a) unweighted (pull (b) label-gated)" |

MICCAI's short format limits the reference list, so cite these first: SCT, FACT, Zhu et al., KDCL, WSL, BWCR, PRISM, PedSemiSeg, TransNetR and Fort et al. Tang et al., PS-KD and DKD are already in the bibliography. The next tier is CTRCL, PLC-KD, MDCS, MART, GJS, Botev et al., DCSD, POS and DHO.

## One free re-evaluation and five controls decide which claim survives

**The free step needs no training.** Evaluate every existing ablation checkpoint on the six PolypGen centres, and fill in the pending flip-gap cells of the table. PolypGen is the largest unseen pool (1,411 images over six centres). The draft already pre-specified its pooled Dice as the primary endpoint for row (h) ([tab_polypgen.tex](/home/user/BACFR/paper/tables/tab_polypgen.tex)). Making it, together with a single test (t or permutation, not both), the primary analysis for every arm removes the dependence on the choice of test shown above.

1. **Compute-matched vanilla (already running).** Vanilla trained three times as long matches FCT in gradient evaluations per image; FCT's wall-clock cost is 3.3–3.7 times vanilla's ([tab_cost.tex](/home/user/BACFR/paper/tables/tab_cost.tex)). If this control lands near the views-only arm, the views' gain is mostly compute, and FCT minus this control becomes the honest headline. If it lands near vanilla, training on the three views together has an effect of its own, as Fort et al. found in classification.
2. **Six-seed random-flip baseline.** Every reviewer will expect this control, and the project has it only as one run of a flip-and-rotation recipe. Sampling one view per step from the same orbit {upright, h, v} isolates random versus simultaneous views. The current partial answer is ASGNet, which already trains with random flips and still gains +1.69 on ETIS.
3. **Label-free both-pull arm.** This is the decisive experiment for the novelty claim, and the outcome decides the framing:
   - If FCT beats it on pooled PolypGen Dice, label arbitration is demonstrated and can lead the paper.
   - If the two tie on Dice, the paper claims the mechanism (gradient alignment, no pull toward a wrong prediction) and presents label arbitration as a principled, cost-free design rather than as the source of the gain.
   - If the label-free arm wins, the novelty framing goes, and FCT is submitted as a multi-view training study.
4. **SCT's loss as a drop-in.** Replace FCT's consistency term in Polyp-PVT with SCT's inconsistency loss. This is the comparison a polyp reviewer will ask for first, because SCT is the only prior polyp method in the same paradigm.
5. **Views-only on PraNet-V2.** FCT's five-set gain is three times larger there. With this arm, the consistency contrast can be pooled over twelve seed pairs instead of six, which narrows its interval by about 40% if seed variance is similar (2.571/√6 against 2.201/√12).

Lower priority, if time allows:

- a PRISM-style EMA self-distillation baseline;
- the official PolypGen C1–C5 → C6 protocol on one host, for comparability with PRISM and DCSD;
- a final literature check of the unread Neurocomputing 2026 paper and of the full MICCAI 2026 texts.

## Conclusion

FCT's novelty and its measured gain sit in different parts of the loss. Multi-view supervision carries most of the gain and none of the novelty; label arbitration carries the novelty and an unresolved share of the gain. The paper therefore cannot rest originality and usefulness on the same evidence. It should claim originality for the design, justified by the credited identity and visible in the gradient alignment, and usefulness for the whole package, shown by the gains on unseen centres across hosts. Stated in the paper, that split is a strength; discovered by a reviewer, it reads as overclaiming.

The robustness result is the hinge of the story. Every variant that holds up keeps the label-decided direction through pull (b), so "FCT holds up when the weights change" has two readings: either the label-arbitrated direction carries the gain, or any consistency on top of three supervised views does. Only the label-free both-pull arm tells them apart. That single six-seed run decides whether FCT goes to MICCAI 2027 as a label-arbitration method or as a well-validated flip-view training recipe. Either is a defensible submission once its claims match that arm's outcome.
