#!/usr/bin/env bash
# =============================================================================
#  1288-image runs: vanilla Polyp-PVT baseline and FCT v2 weight variants
# =============================================================================
#  The 162-image ValidationDataset stays held out, so every model here trains on
#  dataset/TrainDataset (1288 = 800 Kvasir + 488 ClinicDB). The best epoch
#  (PolypPVT.pth) is the checkpoint scored. No EMA.
#
#  vanilla       stock Polyp-PVT loss, no FCT, orientation aug off (the stock
#                default), multi-scale on, same AMP/drop-path as the FCT arms
#  vanilla_aug   the same with Polyp-PVT's flip/rotation augmentation on
#  vanilla_3x    vanilla trained three times as long: 297 epochs (--epoch 298) against 99, the
#                supervised passes of FCT's three views. The compute control for FCT. Constant
#                learning rate, as every 99-epoch run had (--lr_decay_epoch 0: the stock decay
#                at epoch 200 compounds every epoch and would stop training there). The five
#                sets are scored every 3rd epoch (--eval_every 3), so the best epoch is picked
#                from 99 evaluations, as for every other arm, and epoch 297 is scored.
#  s3            the three views supervised, no pull at all (--fct_weight 0): deterministic
#                flip augmentation, the first step of the component dissection
#  s3f           FCT v2 with a constant weight (w = 1), the S3F recipe
#  s3err         FCT v2 with an error-aware per-pixel weight w = 1 - |p_o - y|:
#                pixels where the upright prediction is wrong are not used as
#                consistency targets
#  s3gate        error gate: w = 1 where the upright mask is right, 0 where it is
#                wrong; all pixels stay in the divisor (no re-normalisation)
#  s3bgate       boundary weight times the error gate, divided by sum(boundary)
#  s3rel         relative gate: w = 1 where the upright is closer to the label
#                than the flipped view
#  s3sig         like s3gate, but the wrong pixels are not dropped: weight 1 + p_o
#                on missed polyp pixels (y = 1), 1 - p_o on false positives (y = 0)
#  s3sigy        s3sig with the target fixed: on the wrong pixels the flipped views
#                are pulled toward the label y, not toward the wrong upright mask
#  s3lm          lag-or-margin, per flipped view: w = 1 where the view is at least as
#                far from the label as the upright (the pull then moves it toward the
#                label), else max(0, 1 - 2|p_o - y|), so a view that beats a wrong
#                upright is never pulled toward the error. No new setting.
#  s3lmup        s3lm, and the upright view also learns from the flips: where the flip
#                closest to the label beats the upright view, the upright is pulled toward
#                that flip (--fct_up 1). No new setting; about 30% slower per step.
#  s3fup         FCT without its weight: s3lmup with the flipped views pulled to the upright view at
#                every pixel, w = 1 (--fct_weit 0)
#  s3relup       FCT without the margin: w = 1 where the flipped view lags, 0 where it leads
#                (--fct_weit 5, the a -> infinity end of the sweep)
#  s3up          FCT without the pull to the upright view: the three views supervised, and only the
#                pull of the upright view to the best flipped view (--fct_flip_pull 0)
#                The component dissection runs these in order of what they answer, test first:
#                  GPU=1 nohup bash run_dissection.sh > dissection.out 2>&1 &
#  s3lmup_a05    the sweep of lag-or-margin's factor: s3lmup with w = max(0, 1 - a|p_o - y|)
#  s3lmup_a10    on the views that beat the upright, a = 0.5, 1, 1.5 and 3 (--fct_lm_a).
#  s3lmup_a15    s3lmup itself is a = 2, the only value that gives every wrong upright
#  s3lmup_a30    weight 0 and every right one weight > 0. Compare with sweep_summary.py.
#  pranetv2_vanilla  PraNet-V2 on Res2Net-50, from PRANETV2_ROOT (default
#                    ../PraNet-V2/binary_seg), with its own loss and recipe: Adam 1e-4,
#                    batch 8, --epoch 30 (the loop runs 29 epochs, as in its MyTrain_med.py)
#  pranetv2_s3lmup   the same with FCT exactly as in s3lmup
#  asgnet_vanilla    ASGNet on PVTv2-b3 (TCSVT 2026), from ASGNET_ROOT (default
#                    ../ASGNet/ASGNet_26_TCSVT), with its own loss, data loader (edges,
#                    flips, crop, rotation, colour, label noise) and schedule: Adam 1e-4,
#                    batch 40, one scale, 80 epochs (79 run), full precision. The batch runs
#                    as 8 parts of 5 with one step (--accum 8), which fits FCT on a 12 GB
#                    GPU; the short last batch of an epoch runs as fewer parts (8 = 4 + 4).
#                    ASGNET_ACCUM=4 (parts of 10) matches its 4-GPU DataParallel, where each
#                    GPU's batch norm saw 10 images, if your GPU fits it. Use one value for
#                    both arms.
#  asgnet_s3lmup     the same with FCT exactly as in s3lmup
#  sam2unet_vanilla  SAM2-UNet on SAM2's Hiera-L (Visual Intelligence 2026), from SAM2UNET_ROOT
#                    (default ../SAM2-UNet), with its own loss, data loader (random horizontal
#                    and vertical flips) and schedule, and its paper's polyp recipe: AdamW 1e-3,
#                    weight decay 5e-4, batch 12, 20 epochs (--epoch 21: the loop runs 20),
#                    cosine to 1e-7, scales {1, 1.25}, no gradient clipping, full precision.
#                    The batch runs as SAM2UNET_ACCUM parts with one step (default 2, parts of
#                    6, for a 12 GB GPU); SAM2UNET_ACCUM=1 is its single-GPU batch, if your
#                    GPU fits it. SAM2UNET_AMP=1 trains in mixed precision, about twice as
#                    fast, where its train.py is full precision. Use one value of each for
#                    both arms. A checkpoint is ~0.9 GB (best + last: ~1.8 GB per run).
#  sam2unet_s3lmup   the same with FCT exactly as in s3lmup
#                    Run setup_hosts.sh once before the host stages.
#  gate          inference only: gate_share.py on existing checkpoints
#
#  SEED=<n> sets --seed and appends _s<n> to the run name (default: unseeded).
#  pipefail: a training run that crashes or is killed returns non-zero, so its
#  (partial or stale) checkpoint is not scored.
#
#  Usage: bash run_1288.sh vanilla | vanilla_aug | vanilla_3x | s3 | s3f | s3fup | s3relup | s3up | s3err | s3gate | s3bgate | s3rel | s3sig
#                          | s3sigy | s3lm | s3lmup | s3lmup_a05 | s3lmup_a10 | s3lmup_a15
#                          | s3lmup_a30 | pranetv2_vanilla | pranetv2_s3lmup | asgnet_vanilla
#                          | asgnet_s3lmup | sam2unet_vanilla | sam2unet_s3lmup | sam2unet_tune
#                          | hgm_vanilla | hgm_s3lmup | gate
#         SEED=1 bash run_1288.sh s3err
#         bash run_1288.sh all            # = s3err
# =============================================================================
set -u -o pipefail
STAGE="${1:-all}"
SEED="${SEED:-}"
SFX=""; SEEDFLAG=""
if [ -n "$SEED" ]; then SFX="_s$SEED"; SEEDFLAG="--seed $SEED"; fi
COMMON="--multiscale 1 --drop_path 0.1 --amp 1 --train_path ./dataset/TrainDataset/ $SEEDFLAG"
HOSTARGS=""        # --host and --host_root of the model; set by the host stages below

if [ -d ./dataset/TestDataset/test ]; then
  echo "!! ./dataset/TestDataset/test exists."
  echo "   Train_noaug.py selects checkpoints on the five-set test mean (or on --val_path"
  echo "   when it is given) and no longer reads dataset/TestDataset/test at all."
  echo "   This guard is kept to ensure no future code accidentally reads test sets during"
  echo "   training. All runs committed before this change used five-set mean selection;"
  echo "   the TestDataset/test directory did not exist then, so no test-set leakage"
  echo "   occurred in any prior committed run."
  echo "   If you intentionally created this directory, remove the guard or rename it."
  exit 1
fi
n=$(ls ./dataset/TrainDataset/images/ 2>/dev/null | wc -l)
if [ "$n" != 1288 ]; then
  echo "!! dataset/TrainDataset/images has $n files, expected 1288 (see the dataset fix: ln -s gt images)"
  exit 1
fi

train_arm() {   # name, extra flags...
  local arm="$1"; shift
  echo "=== train $arm"
  python Train_noaug.py $COMMON "$@" --train_save "./model_pth/$arm/" 2>&1 | tee "logs_$arm.txt"
}

infer_arm() {   # name, checkpoint files...
  local arm="$1"; shift
  for ck in "$@"; do
    [ -f "./model_pth/$arm/$ck.pth" ] || continue
    local tag="$arm"
    echo "=== infer $arm/$ck -> result_map/$tag"
    python Test_maps.py --pth_path "./model_pth/$arm/$ck.pth" --save_root "./result_map/$tag" $HOSTARGS
    python flip_gap.py --pth_path "./model_pth/$arm/$ck.pth" --out flip_gap.tsv $HOSTARGS
  done
}

FCT="--orientation_aug 0 --fct 1 --fct_sup 1 --fct_all_scales 1 --fct_loss bce --fct_weight 0.5 \
     --fct_sync_rng 1 --grad_diag 0"
vanilla()     { train_arm PVT_VANILLA$SFX --orientation_aug 0 && infer_arm PVT_VANILLA$SFX PolypPVT; }
vanilla_aug() { train_arm PVT_VANILLA_AUG$SFX --orientation_aug 1 && infer_arm PVT_VANILLA_AUG$SFX PolypPVT; }
vanilla_3x()  { train_arm PVT_VANILLA_3X$SFX --orientation_aug 0 --epoch 298 --eval_every 3 --lr_decay_epoch 0 \
                  && infer_arm PVT_VANILLA_3X$SFX PolypPVT; }
s3()          { train_arm S3$SFX $FCT --fct_weight 0 && infer_arm S3$SFX PolypPVT; }
s3f()         { train_arm S3F$SFX $FCT --fct_weit 0 && infer_arm S3F$SFX PolypPVT; }
s3fup()       { train_arm S3FUP$SFX $FCT --fct_weit 0 --fct_up 1 && infer_arm S3FUP$SFX PolypPVT; }
s3relup()     { train_arm S3RELUP$SFX $FCT --fct_weit 5 --fct_up 1 && infer_arm S3RELUP$SFX PolypPVT; }
s3up()        { train_arm S3UP$SFX $FCT --fct_weit 8 --fct_up 1 --fct_flip_pull 0 && infer_arm S3UP$SFX PolypPVT; }
s3err()       { train_arm S3ERR$SFX $FCT --fct_weit 2 && infer_arm S3ERR$SFX PolypPVT; }
s3gate()      { train_arm S3GATE$SFX $FCT --fct_weit 3 && infer_arm S3GATE$SFX PolypPVT; }
s3bgate()     { train_arm S3BGATE$SFX $FCT --fct_weit 4 && infer_arm S3BGATE$SFX PolypPVT; }
s3rel()       { train_arm S3REL$SFX $FCT --fct_weit 5 && infer_arm S3REL$SFX PolypPVT; }
s3sig()       { train_arm S3SIG$SFX $FCT --fct_weit 6 && infer_arm S3SIG$SFX PolypPVT; }
s3sigy()      { train_arm S3SIGY$SFX $FCT --fct_weit 7 && infer_arm S3SIGY$SFX PolypPVT; }
s3lm()        { train_arm S3LM$SFX $FCT --fct_weit 8 && infer_arm S3LM$SFX PolypPVT; }
s3lmup()      { train_arm S3LMUP$SFX $FCT --fct_weit 8 --fct_up 1 && infer_arm S3LMUP$SFX PolypPVT; }
s3lmup_a05()  { train_arm S3LMUP_A05$SFX $FCT --fct_weit 8 --fct_up 1 --fct_lm_a 0.5 && infer_arm S3LMUP_A05$SFX PolypPVT; }
s3lmup_a10()  { train_arm S3LMUP_A10$SFX $FCT --fct_weit 8 --fct_up 1 --fct_lm_a 1.0 && infer_arm S3LMUP_A10$SFX PolypPVT; }
s3lmup_a15()  { train_arm S3LMUP_A15$SFX $FCT --fct_weit 8 --fct_up 1 --fct_lm_a 1.5 && infer_arm S3LMUP_A15$SFX PolypPVT; }
s3lmup_a30()  { train_arm S3LMUP_A30$SFX $FCT --fct_weit 8 --fct_up 1 --fct_lm_a 3.0 && infer_arm S3LMUP_A30$SFX PolypPVT; }
PRANETV2="--host pranetv2 --host_root ${PRANETV2_ROOT:-../PraNet-V2/binary_seg}"
PRANETV2_RECIPE="--optimizer Adam --batchsize 8 --epoch 30"
pranetv2_vanilla() { HOSTARGS="$PRANETV2"; train_arm PRANETV2_VANILLA$SFX $HOSTARGS $PRANETV2_RECIPE --orientation_aug 0 \
                       && infer_arm PRANETV2_VANILLA$SFX PolypPVT; }
pranetv2_s3lmup()  { HOSTARGS="$PRANETV2"; train_arm PRANETV2_S3LMUP$SFX $HOSTARGS $PRANETV2_RECIPE $FCT --fct_weit 8 --fct_up 1 \
                       && infer_arm PRANETV2_S3LMUP$SFX PolypPVT; }
ASGNET="--host asgnet --host_root ${ASGNET_ROOT:-../ASGNet/ASGNet_26_TCSVT}"
ASGNET_RECIPE="--optimizer Adam --batchsize 40 --epoch 80 --multiscale 0 --amp 0 --drop_path 0 --accum ${ASGNET_ACCUM:-8}"
asgnet_vanilla()   { HOSTARGS="$ASGNET"; train_arm ASGNET_VANILLA$SFX $HOSTARGS $ASGNET_RECIPE \
                       && infer_arm ASGNET_VANILLA$SFX PolypPVT; }
asgnet_s3lmup()    { HOSTARGS="$ASGNET"; train_arm ASGNET_S3LMUP$SFX $HOSTARGS $ASGNET_RECIPE $FCT --fct_weit 8 --fct_up 1 \
                       && infer_arm ASGNET_S3LMUP$SFX PolypPVT; }
SAM2UNET="--host sam2unet --host_root ${SAM2UNET_ROOT:-../SAM2-UNet}"
SAM2UNET_RECIPE="--optimizer AdamW --lr 1e-3 --weight_decay 5e-4 --batchsize 12 --epoch 21 --size_rates 1 1.25 \
                 --clip 0 --amp ${SAM2UNET_AMP:-0} --accum ${SAM2UNET_ACCUM:-2}"
sam2unet_vanilla() { HOSTARGS="$SAM2UNET"; train_arm SAM2UNET_VANILLA$SFX $HOSTARGS $SAM2UNET_RECIPE \
                       && infer_arm SAM2UNET_VANILLA$SFX PolypPVT; }
sam2unet_s3lmup()  { HOSTARGS="$SAM2UNET"; train_arm SAM2UNET_S3LMUP$SFX $HOSTARGS $SAM2UNET_RECIPE $FCT --fct_weit 8 --fct_up 1 \
                       && infer_arm SAM2UNET_S3LMUP$SFX PolypPVT; }
# sam2unet_tune: sam2unet_s3lmup with extra FCT flags, under its own run name, for tuning FCT on
# SAM2-UNet without touching the runs above. TUNE names the variant, TUNE_ARGS holds the flags
# (later flags win in argparse, so they override $FCT), for example
#   SEED=1 SAM2UNET_AMP=1 TUNE=W10 TUNE_ARGS="--fct_weight 1.0" bash run_1288.sh sam2unet_tune
#   -> SAM2UNET_S3LMUP_W10_s1
sam2unet_tune()    { [ -n "${TUNE:-}" ] || { echo "!! sam2unet_tune needs TUNE=<name> (and TUNE_ARGS)"; exit 1; }
                     HOSTARGS="$SAM2UNET"; local n="SAM2UNET_S3LMUP_${TUNE}$SFX"
                     train_arm "$n" $HOSTARGS $SAM2UNET_RECIPE $FCT --fct_weit 8 --fct_up 1 ${TUNE_ARGS:-} \
                       && infer_arm "$n" PolypPVT; }
HGM="--host hgm --host_root ${HGM_ROOT:-../HGM}"
# HGM recipe (its MyTrain.py): Adam 3e-5, 200 epochs (--epoch 201: the loop runs 1..200),
# 352 px, multi-scale {0.75,1,1.25}, gradient clip 0.5; the x0.1 steps at epochs 50/100/150
# come from HGMHost.lr_schedule. Its batch 16 was on two GPUs: HGM_BATCH (default 16) runs as
# HGM_ACCUM parts on one 12 GB card. HGM needs mamba_ssm, which only builds in the hgm conda
# env (CUDA 11.8), so its stages put that env's python first on PATH.
HGM_PY="${HGM_ENV:-/home/yassine/anaconda3/envs/hgm}/bin"
HGM_RECIPE="--optimizer Adam --lr 3e-5 --batchsize ${HGM_BATCH:-16} --accum ${HGM_ACCUM:-2} --epoch ${HGM_EPOCH:-201} \
            --clip 0.5 --multiscale 1 --drop_path 0.1 --amp ${HGM_AMP:-0}"
hgm_vanilla() { ( PATH="$HGM_PY:$PATH"; HOSTARGS="$HGM"; train_arm HGM_VANILLA$SFX $HOSTARGS $HGM_RECIPE \
                  && infer_arm HGM_VANILLA$SFX PolypPVT ); }
hgm_s3lmup()  { ( PATH="$HGM_PY:$PATH"; HOSTARGS="$HGM"; train_arm HGM_S3LMUP$SFX $HOSTARGS $HGM_RECIPE $FCT --fct_weit 8 --fct_up 1 \
                  && infer_arm HGM_S3LMUP$SFX PolypPVT ); }
gate()        { local ck=""; for a in S3F S3F_EMA S3W05 S3ERR; do [ -d "./model_pth/$a" ] && ck="$ck ./model_pth/$a"; done
                python gate_share.py --pth_path $ck; }

case "$STAGE" in
  vanilla)     vanilla ;;
  vanilla_aug) vanilla_aug ;;
  vanilla_3x)  vanilla_3x ;;
  s3)          s3 ;;
  s3f)         s3f ;;
  s3fup)       s3fup ;;
  s3relup)     s3relup ;;
  s3up)        s3up ;;
  s3err)       s3err ;;
  s3gate)      s3gate ;;
  s3bgate)     s3bgate ;;
  s3rel)       s3rel ;;
  s3sig)       s3sig ;;
  s3sigy)      s3sigy ;;
  s3lm)        s3lm ;;
  s3lmup)      s3lmup ;;
  s3lmup_a05)  s3lmup_a05 ;;
  s3lmup_a10)  s3lmup_a10 ;;
  s3lmup_a15)  s3lmup_a15 ;;
  s3lmup_a30)  s3lmup_a30 ;;
  pranetv2_vanilla) pranetv2_vanilla ;;
  pranetv2_s3lmup)  pranetv2_s3lmup ;;
  asgnet_vanilla)   asgnet_vanilla ;;
  asgnet_s3lmup)    asgnet_s3lmup ;;
  sam2unet_vanilla) sam2unet_vanilla ;;
  sam2unet_s3lmup)  sam2unet_s3lmup ;;
  sam2unet_tune)    sam2unet_tune ;;
  hgm_vanilla)      hgm_vanilla ;;
  hgm_s3lmup)       hgm_s3lmup ;;
  gate)        gate; exit 0 ;;
  all)         s3err ;;
  *)           echo "unknown stage: $STAGE"; exit 1 ;;
esac
echo "score result_map/<arm> with your evaluator."
