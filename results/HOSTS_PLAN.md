# New host for the main body: plan, fixed before any run (2026-11-10)

**Update, 2026-11-13.** The SAM2-UNet runs (six seeds, both arms) were discarded, and HGM (Hybrid Graph
Mamba, MICCAI 2025) takes its place as the third host, to test FCT on a Mamba-based network. The rest of
this file is the plan as written on 2026-11-10 and is kept as the record. Its note on HGM still applies:
the released code loads no ImageNet weights into the backbone and by default starts training from the
authors' finished model (`--pre_parameter ./best_parameter_HGM.pth`). The HGM runs therefore need to
state which weights they start from; both arms should start from the ImageNet PVTv2-b2 weights. HGM's
own loader applies no flips.

ASGNet leaves the main body. In its place: a 2025/2026 polyp segmentation network, chosen
below by criteria fixed before any of its runs. This file records the choice, the recipe and
the analysis, so the result is read the same way whatever it is.

## Criteria

1. Peer-reviewed in 2025 or 2026.
2. Official public code with its model and training script (not a README alone).
3. Trained and tested on the standard PraNet split (Kvasir-SEG + CVC-ClinicDB for training;
   Kvasir, ClinicDB, ColonDB, CVC-300 and ETIS for testing), so its vanilla run can be
   checked against its own published numbers.
4. Trainable on one GPU with FCT's three supervised views, from public pretrained weights.
5. Another family of network than the hosts already in (PVTv2 Polyp-PVT, Res2Net PraNet-V2).

## Candidates screened

Every repository was checked with `git ls-remote` and a shallow clone on 2026-11-10; venues
and code links from the MICCAI open-access pages.

| Network | Venue | Code | Split | Verdict |
|---|---|---|---|---|
| **SAM2-UNet** (WZH0120/SAM2-UNet) | Visual Intelligence 4, 2026 | model, train.py, test.py; Apache-2.0 | PraNet | **chosen**: meets all five; foundation-model family (SAM2 Hiera-L) |
| HGM, Hybrid Graph Mamba (YueyueZhu/HGM) | MICCAI 2025 | model, MyTrain.py; no licence | PraNet | out: the code never loads ImageNet weights for its PVTv2-b2, and training starts by default from the authors' finished checkpoint (`--pre_parameter ./best_parameter_HGM.pth`), so its numbers cannot be reproduced from the code as released; each forward also builds a dense 7744 x 7744 adjacency in NumPy; needs mamba_ssm and torch_geometric |
| CD-PolypNet (ChangpengYue/CD-PolypNet) | MICCAI 2025 | code | PraNet | out: SAM ViT-L at 1024 x 1024, batch 4, HQ-SAM training pipeline; FCT's five passes per image are out of budget |
| ESPNet (Raneem-MT/ESPNet) | MICCAI 2025 | code | own 4,000-frame multi-centre set | out: not the PraNet split (criterion 3); 4-GPU recipe |
| FPN-in-FPN (Yejin0111/FPN-in-FPN) | MICCAI 2025 | README only | PraNet | out: no code |
| PolyMamba (Parker-rfu/PolyMamba) | MICCAI 2025 | repository not public | PraNet | out: no code |
| SAM-MaGuP (deepak1113/SAM-MaGuP) | MICCAI 2025 | README only | n/a | out: no code |
| Polyp-AFMamba (B00MCLAP/Polyp-AFMamba) | MICCAI 2026 | README only | five benchmarks | out: no code |
| Lite-π (lostinrepo/Lite-Pi) | MICCAI 2026 | two notebooks | n/a | out: a distillation framework for small networks, not a host network |
| KAN-AINet (biodatlab/kanainet) | MICCAI 2026 | code; Apache-2.0 | ESPNet's multi-centre set, 512 x 512 | reserve: fails criterion 3 only; the one to add if a MICCAI 2026 host is wanted |
| SD-SAM (xxxy0523/SD-SAM) | MICCAI 2026 | code | own CPD1K benchmark | out: not the PraNet split; two-stage SAM + DINOv3 (gated weights) |

## SAM2-UNet recipe

From its paper (Sec. 3.2) and train.py: SAM2 (not 2.1) Hiera-L encoder frozen, an adapter in
each block; AdamW, lr 1e-3, weight decay 5e-4, cosine decay to 1e-7 stepped per epoch; batch
12; 352 x 352; 20 epochs; random horizontal and vertical flips; for polyps, multi-scale
training at {1, 1.25}; no gradient clipping; full precision; loss: the structure loss on each
of its three maps; prediction: the first map.

Its published numbers (mDice, trained on all 1,450 images): Kvasir 0.928, ClinicDB 0.907,
ColonDB 0.808, CVC-300 0.894, ETIS 0.796.

Ours differs only where every arm here differs: the 1,288-image training split, the best epoch
by the five-set mean, seeds 1-6. The batch of 12 runs as 2 parts of 6 (`SAM2UNET_ACCUM`,
default 2, for a 12 GB GPU). The parts are summed into one step, so the batch norm in its
decoder sees 6 images at a time instead of 12. Full precision unless `SAM2UNET_AMP=1`. The
same settings hold for both arms.

Its recipe already flips images at random, as ASGNet's did. FCT is therefore tested against a
flip-augmented baseline. That is the comparison reviewers will ask for, and its outcome is not
predictable from the hosts run so far.

## Runs and analysis

```bash
bash setup_hosts.sh sam2unet && python test_hosts.py      # clone, hydra-core, SAM2 weights; CPU checks
CUDA_VISIBLE_DEVICES=0 ARMS="sam2unet_vanilla sam2unet_s3lmup" nohup bash run_seeds.sh 1 2 3 > seeds_sam2unet_a.out 2>&1 &
CUDA_VISIBLE_DEVICES=1 ARMS="sam2unet_vanilla sam2unet_s3lmup" nohup bash run_seeds.sh 4 5 6 > seeds_sam2unet_b.out 2>&1 &
python seed_summary.py --seeds 1 2 3 4 5 6 --arms SAM2UNET_VANILLA SAM2UNET_S3LMUP
```

FCT is S3LMUP exactly as on the other hosts (λ = 0.5, a = 2, all scales). The analysis is the
same as for the other hosts: the six paired seeds, the paired difference per test set with its
95% interval, the exact permutation test, seeds won. The five-set mean is primary. PolypGen
(pooled, then per centre) is scored with `run_polypgen.sh` if it is run for any host.

Reporting. Planned: SAM2-UNet goes in the main body whatever its result. That is the point of
choosing it before its runs.

For ASGNet, note MICCAI's rule. MICCAI 2025 and 2026 allow only multimedia (avi, mp4, wmv) as
supplementary material, with no PDFs. "Proofs, analysis, additional results" there lead to desk
rejection (Paper Submission Guidelines, section 5). A supplementary ASGNet table is therefore
not possible at MICCAI. What fits in the paper itself is one row in the hosts table, or one
sentence with its numbers.
