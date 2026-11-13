# FCT story: flipped views, consistency the label directs, lag-or-margin as an add-on

Version of 13 November 2026. **Status: confirmed by you on 13 November**, on the condition
that is now built in: the paper focuses on the flipped views and the consistency between
them, and lag-or-margin is an add-on (tracker: Decisions D12). The method and every result are
unchanged; only what the paper leads with changes. The frozen paper in BACFR/paper is untouched.

## The story in three sentences

1. A polyp network's mask changes when the image is flipped, and its accuracy drops on unseen
   centres. Consistency between flipped views is the usual remedy, but a consistency loss pulls
   one view toward another without asking which of the two is right.
2. In supervised training the label answers that question at every pixel. FCT trains on three
   flipped views and makes them consistent in the direction the label gives: the less accurate
   view is pulled toward the more accurate one, and the upright view, the one used at test
   time, is the default target.
3. FCT is a plug-in training loss with no parameters and no test-time cost. Over six paired
   seeds it improves a transformer, a CNN and a Mamba network [HGM: runs in progress], most on
   unseen centres.

This is your own description of the method ("the two lower views are pulled toward the better
view"), made the headline.

## How the method is layered

| Layer | What it is | Role in the paper | Arm that isolates it |
|---|---|---|---|
| 1. Flipped views | Each image is trained upright and in two flips, each view supervised | Focus | S3 |
| 2. Consistency between the views | Each view is pulled toward a more accurate view; the label decides which | Focus, and the new part | S3RELUP (layers 1 and 2) |
| 3. Lag-or-margin | A flipped view that is ahead keeps a weak pull toward the upright view while the upright view is right | Add-on | S3LMUP = FCT (layers 1 to 3) |

The label's decision belongs to layer 2, the consistency itself. That is what makes it new:
consistency that only makes the views agree is already published for polyps (SCT). The paper's
name for layer 2 is **label-directed consistency**. It is the two pulls of the tracker taken
together (the pull to the upright view and the pull to the best flipped view).

Each layer against vanilla, Polyp-PVT, six paired seeds, Dice points:

| Layers | Five-set mean | ETIS | Seeds won | Gradient agreement |
|---|---|---|---|---|
| 1: flipped views | +0.37 [−0.01, +0.74] | +1.48 [+0.32, +2.65] | 5/6 | – |
| 1 and 2: + consistency the label directs | +0.34 [+0.10, +0.57] | +1.40 [+0.73, +2.06] | 6/6 | 0.56 |
| 1 to 3: + lag-or-margin (FCT) | +0.52 [+0.15, +0.90] | +1.65 [+0.88, +2.42] | 6/6 | 0.53 |

Lag-or-margin adds +0.18 [−0.38, +0.74] on the five-set mean. That is within noise, which
fits its role as an add-on: FCT works with or without it.

## What changes from the last story

| | Last story | This story |
|---|---|---|
| Leads with | Three views + consistency | Flipped views + consistency whose direction the label decides |
| Three flipped views | The core | Focus, layer 1: they also give each view's error against the label for free |
| Consistency | Makes the views agree | Focus, layer 2: pulls each view toward a more accurate one |
| Lag-or-margin | "Completes" the method | Add-on, layer 3: FCT works with or without it; its constant needs no tuning |
| Novelty claim | None that survives the check | First label-decided direction between the views of one network |
| Gain claim | FCT as a whole | Unchanged: FCT as a whole |

## Why this version is novel

- The novelty check (12 Nov) and five more searches (13 Nov) found no method in which the
  label decides, pixel by pixel, which view of **one** network is pulled toward which.
- Nearest work, to cite next to the claim:
  - Zhu et al., ICCV 2023: the same per-pixel rule, between **two separate networks** (a CNN and a ViT).
  - KDCL, CVPR 2020: the network closest to the label teaches the others, per image, across several networks.
  - SCT, IEEE ISM 2022: supervised consistency for polyps with its largest gains on ETIS. It is
    symmetric, with one weight per batch, and no direction.
- Wording for the paper: "to our knowledge, the first consistency loss in which the label
  decides, pixel by pixel, the direction in which the views of a single network are pulled
  together."
- Before submission: repeat the check. One 2026 Neurocomputing paper on polyp self-distillation
  could not be read.

## Two facts from your logs that carry the story

**Fact 1. The usual target is often the less accurate view.** Plain consistency pulls every
flipped view toward the upright view. During training, a flipped view is closer to the label
than the upright view on 44% of pixels, and the better of the two flipped views is closer on
61%. (Polyp-PVT, six seeds per arm; PraNet-V2 gives the same 44% and 61%.) On most of these
pixels both views are nearly right, so this motivates the idea; it is not a Dice claim.

**Fact 2. The label's direction makes the consistency agree with supervision.** Cosine between
the gradient of the pull to the upright view and the gradient of the rest of the loss,
averaged over training, Polyp-PVT, six seeds each:

| Consistency | Arm | Cosine | Range over seeds |
|---|---|---|---|
| Plain: every flipped view pulled to the upright view | S3FUP | 0.22 | 0.14 to 0.26 |
| Label-directed (layer 2) | S3RELUP | 0.56 | 0.55 to 0.57 |
| Label-directed + lag-or-margin, a = 2 (FCT) | S3LMUP | 0.53 | 0.47 to 0.55 |
| Label-directed + lag-or-margin, a = 1.5 | S3LMUP_A15 | 0.50 | 0.44 to 0.53 |
| Label-directed + lag-or-margin, a = 1 | S3LMUP_A10 | 0.46 | 0.43 to 0.49 |
| Label-directed + lag-or-margin, a = 0.5 | S3LMUP_A05 | 0.34 | 0.30 to 0.39 |

The label's direction raises the agreement from 0.22 to 0.56, and the seed ranges do not
overlap. This is the measured effect of the new part. The lag-or-margin add-on gives a little
of it back (0.53 at a = 2), because it keeps a weak pull on views that are ahead; the smaller
its constant a, the more it gives back. In S3FUP only the pull to the upright view is plain;
its pull to the best flipped view is still label-gated and counts here as part of the rest of
the loss. The single run of fully plain consistency (S3F) also gives 0.22.

## The introduction, paragraph by paragraph

1. **Problem.** The endoscope moves freely, so a polyp can appear in any orientation. Yet a
   trained network is not consistent across orientations: on the five test sets, the mask of an
   image and the mask of its flipped copy differ by 12.5% for Polyp-PVT and 15.6% for PraNet-V2
   (1 − Dice between the two masks). Accuracy also falls on unseen centres.
2. **Blind spot.** Flip augmentation trains on flipped copies; consistency regularization also
   makes the views agree. Consistency comes from semi-supervised learning, where no label says
   which view is right, so its target is fixed in advance: the clean view, or both views
   symmetrically. Supervised uses kept that design (SupTC, SCT, FACT). Fact 1 shows the cost:
   the fixed target is the less accurate view on 44% of pixels.
3. **Idea.** With labels, "which view is right here?" has an answer at every pixel. A known
   result from distillation (Tang et al.; PS-KD) says what the answer is worth: a pull toward a
   more accurate view acts as extra supervision, and a pull toward a less accurate one works
   against supervision. So the label should decide the direction.
4. **Method.** FCT: flipped views, consistency the label directs, and the lag-or-margin add-on.
5. **Evidence.** Gains on three networks, largest on unseen centres; Fact 2; nothing to tune.

## Title options

- **Let the Label Decide: Label-Directed Flip Consistency for Polyp Segmentation on Unseen Centres** (recommended)
- Label-Directed Flip Consistency: A Plug-In Training Loss for Generalizable Polyp Segmentation
- Which View Is Right? Label-Directed Flip-Consistency Training for Polyp Segmentation

## Draft abstract (about 200 words; HGM numbers to add)

Polyp segmentation networks lose accuracy on images from unseen centres, and their predictions
change when an image is merely flipped. Consistency regularization ties the views of an image
together, but it was designed for unlabeled data: it pulls one view toward another without
asking which is right. We find that, during training, a flipped view is closer to the label
than the upright view on 44% of pixels. We propose flip-consistency training (FCT), a plug-in
loss for supervised training in which the label directs the consistency. Each image is
supervised in three orientations. At every pixel, the label then decides the direction of the
pull: a flipped view is drawn toward the upright view, the one used at test time, where the
upright view is more accurate, and the upright view is drawn toward the best flipped view where
that view is more accurate. FCT adds no parameters and no test-time cost. Over six paired
seeds, it improves Polyp-PVT, PraNet-V2 and HGM on unseen-centre data: by 1.65 and 3.61 Dice
points on ETIS, and by 0.86 and 2.36 points on the six centres of PolypGen. Directing the
consistency with the label more than doubles the agreement between its gradient and the
supervised gradient.

## Contributions

1. **Label-directed consistency.** A consistency loss for supervised segmentation in which the
   label decides, at every pixel, which of a network's flip views is pulled toward which. To our
   knowledge it is the first within a single network.
2. **FCT, a plug-in training loss built on it.** Three supervised flip views and label-directed
   consistency, with the test-time view as the default target and a lag-or-margin add-on. No
   parameters, no test-time cost, and nothing to tune.
3. **Evidence with paired seeds.** Three networks from three architecture families, six paired
   seeds each, with 95% intervals: gains concentrated on unseen centres (ETIS and the six
   PolypGen centres), a layer-by-layer comparison (longer training, random flips, flipped
   views, plain consistency, label direction, lag-or-margin), and the gradient measurement of
   Fact 2.

## Statement of novelty for the submission form (481 of the 500 characters allowed)

Consistency losses make the views of an image agree without asking which view is right. In
supervised training the label knows. FCT lets the label decide, pixel by pixel, which flip view
of a network is pulled toward which, with the test-time view as default. It adds no parameters
or test-time cost and improves transformer, CNN and Mamba polyp networks on unseen centres
(ETIS, PolypGen) over six paired seeds. Main significance: methodological novelty, validated
across centres.

## The method in the story's words (matches the code)

1. **Flipped views.** Each image is trained upright, flipped horizontally and flipped
   vertically, each with the host network's own loss, weighted 1/3. Supervising every view
   gives each view's error against the label at every pixel.
2. **Consistency the label directs.** The flipped predictions are mapped back to the upright
   frame. Then, at each pixel:
   - where the upright view is more accurate than a flipped view, that flipped view is pulled
     toward the upright view;
   - where a flipped view is more accurate than the upright view, the upright view is pulled
     toward the most accurate flipped view.
   Targets are detached, the loss is a soft binary cross-entropy on logits, and λ = 0.5.
3. **Add-on: lag-or-margin.** A flipped view that is ahead of the upright view also keeps a
   weak pull toward it, with weight 1 − 2 × (the upright view's error). The weight is zero
   when the upright view is wrong, so a view is never pulled toward a wrong prediction that it
   has already beaten.
4. **Test time.** The upright image only.

Code and run names: layers 1 and 2 are `--fct_weit 5 --fct_up 1` (S3RELUP); with the add-on,
`--fct_weit 8 --fct_up 1` (S3LMUP, FCT).

## Claims and evidence

| Claim | Evidence | Status |
|---|---|---|
| The usual consistency target is the less accurate view on 44% of pixels | Counters in the training logs (Fact 1) | Have |
| First label-decided direction between the views of one network | Novelty check, 12 and 13 Nov | Have; repeat before submission |
| FCT improves each network, most on unseen centres | Polyp-PVT: five-set +0.52 [+0.15, +0.90], ETIS +1.65 [+0.88, +2.42], PolypGen +0.86 [+0.37, +1.34]. PraNet-V2: five-set +1.68 [+0.69, +2.67], ETIS +3.61 [+0.23, +6.99], PolypGen +2.36 [+1.00, +3.71], all six centres up | Have for two networks; HGM in progress |
| The label's direction more than doubles the gradient agreement | Fact 2 | Have |
| Lag-or-margin is an add-on: FCT works with or without it | Without it: five-set +0.34 [+0.10, +0.57], ETIS +1.40. With it: +0.52, +1.65. Its own share: +0.18 [−0.38, +0.74] | Have on the five sets; PolypGen without the add-on to score (inference only) |
| Label-directed against plain consistency, in Dice | Plain consistency (`s3f`) at six seeds | To run; the stage exists |
| More than flip augmentation | Vanilla with random flips (item 13). Also ASGNet, whose recipe already has random flips: ETIS +1.69 [+0.08, +3.30] | To run; ASGNet is your call (O1) |
| More than extra training | Vanilla for three times the epochs | Queued |
| Nothing to tune | Five-set gain +0.33 to +0.52 for a = 0.5, 1, 1.5, 2 and without the add-on. PolypGen +0.81 to +1.00 for the four values of a, every interval above 0 | Have |
| No test-time cost; training costs 3.6× (Polyp-PVT), 3.3× (PraNet-V2) | Timing from the logs | Have |

**What Dice cannot say yet.** On Polyp-PVT, six seeds do not separate the layers:

- The flipped views alone give +0.37 on the five-set mean and +0.50 on PolypGen.
- Consistency the label directs changes the five-set mean by −0.03 [−0.43, +0.38] on top of
  the views. It does tighten the gain: the interval goes from [−0.01, +0.74] to
  [+0.10, +0.57], and the seeds won from 5 of 6 to 6 of 6.
- Lag-or-margin adds +0.18 [−0.38, +0.74].
- Layers 2 and 3 together add +0.16 [−0.30, +0.62] on the five-set mean and +0.36
  [−0.40, +1.12] on PolypGen.
- The flip gap falls from 12.5% to 7.8% with the views alone, and to 7.5% with FCT.

So the story claims the Dice gains for FCT as a whole. For the label's direction it claims the
design and its effect on the gradient. The paper should say this in one sentence; a reviewer
who finds it unaided will read it as overclaiming.

## How the paper is laid out for this story

| Element | Content |
|---|---|
| Figure 1 | The three views; at one pixel, the label ranks them and the arrows run from the less accurate view to the more accurate one |
| Table 1 | Five test sets, three networks: vanilla and FCT, gain with 95% interval |
| Table 2 | PolypGen, six centres and pooled, three networks |
| Table 3 | Layer by layer on Polyp-PVT: vanilla; three times the epochs; random flips; flipped views; + plain consistency; + consistency the label directs; + lag-or-margin (FCT). Columns: five-set mean, ETIS, PolypGen pooled, gradient agreement |
| One sentence each | Other values of a, and one pull only; flip gap; results without best-epoch selection; C2; training cost |

## Credit, and what not to claim

Credit in the text:

- **Tang et al. and PS-KD**, for the identity (a consistency pull is supervision with an error-dependent weight).
- **Weighted soft labels (ICLR 2021)**, as the nearest relative of the lag-or-margin weight.
- **Zhu et al., KDCL and SCT**, as above.

Do not claim:

- the first consistency training for supervised polyp segmentation (SCT);
- that both pulls are needed, or that lag-or-margin raises Dice (the arms are within noise);
- that the identity is new.

## What the story still needs

| # | Run | Why the story needs it | Cost |
|---|---|---|---|
| 1 | HGM, vanilla and FCT (in progress on your machine) | The third network. Its runs must start from ImageNet backbone weights: see O6 | Yours |
| 2 | Vanilla with random flips (item 13) | None of the three networks trains with flips in its default recipe, so "is this flip augmentation?" is the first question a reviewer will ask | ~14 GPU-h |
| 3 | Plain consistency, `s3f`, six seeds, plus PolypGen | The one row that shows what the label's direction changes in Dice | ~25 GPU-h |
| 4 | Vanilla, three times the epochs | Rules out "just more training" | Queued |
| 5 | PolypGen for S3RELUP (layers 1 and 2) | The row "without the add-on" on unseen centres | Inference only |
| 6 | Flipped views alone and plain consistency on PraNet-V2 | FCT's gain is three times larger there, so the layers are more likely to separate | ~20 GPU-h, two new stages |

What run 3 does to the story:

- **Label-directed beats plain consistency:** the abstract gains the sentence "the label's direction improves on plain consistency by X".
- **A tie:** the story stands as written.
- **Plain consistency wins:** the label's direction cannot be the new part, and the paper becomes a study of multi-view training.
