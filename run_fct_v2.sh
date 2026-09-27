#!/usr/bin/env bash
# =============================================================================
#  FCT v2: supervise every flipped view, add a boundary-weighted consistency
# =============================================================================
#  Stage 1  gap     flip_gap.py on the checkpoints you already have (no training).
#                   Upright-vs-flipped disagreement on TRAIN and the 5 test sets.
#  Stage 2  train   three arms, all orientation-aug OFF, flipped views supervised,
#                   FCT at every multi-scale rate:
#                     S3      supervised flips only (lambda 0)      <- control
#                     S3W05   + soft-BCE consistency, lambda 0.5, boundary-weighted
#                     S3W2    + soft-BCE consistency, lambda 2,   boundary-weighted
#  Stage 3  infer   plain inference (--views 1) + flip_gap for each new arm.
#  Stage 4  s3f L   S3F: best lambda L from S3W05/S3W2, consistency NOT weighted,
#                   i.e. the ablation of the boundary weight. Trains + infers.
#
#  Reading it:
#    S3    vs A2     what supervised flips alone buy
#    S3W*  vs S3     what consistency adds on top of supervised flips
#    S3W   vs S3F    what the boundary weight adds
#  References: PVT_AUG 0.8559, raw Polyp-PVT (local) 0.8683.
#
#  Usage: bash run_fct_v2.sh            # gap + train + infer
#         bash run_fct_v2.sh gap        # stage 1 only
#         bash run_fct_v2.sh train      # stage 2 only
#         bash run_fct_v2.sh infer      # stage 3 only
#         bash run_fct_v2.sh s3f 2      # stage 4 with lambda = 2
#
#  Training logs print `cons_kl` (the disagreement part of the soft BCE) and,
#  every 50 batches, |g_cons|/|g_sup| and their cosine. A negative cosine means
#  the consistency term is working against the supervised loss.
# =============================================================================
set -u
STAGE="${1:-all}"
COMMON="--multiscale 1 --drop_path 0.1 --amp 1 --fct_sync_rng 1 --orientation_aug 0 \
        --fct 1 --fct_sup 1 --fct_all_scales 1 --grad_diag 50"
OLD_ARMS="PVT_A2 PVT_B2 PVT_AUG PVT_AUG_FCT"
GAP_TSV="flip_gap.tsv"

existing() {
  local out=""
  for a in "$@"; do [ -d "./model_pth/$a" ] && out="$out ./model_pth/$a"; done
  echo "$out"
}

train_arm() {   # name, extra flags...
  local arm="$1"; shift
  echo "=== train $arm"
  python Train_noaug.py $COMMON "$@" --train_save "./model_pth/$arm/" 2>&1 | tee "logs_$arm.txt"
}

infer_arm() {
  local arm="$1"
  echo "=== infer $arm"
  python Test_tta.py --pth_path "./model_pth/$arm" --views 1 --save_root "./result_map/$arm"
  python flip_gap.py --pth_path "./model_pth/$arm" --out "$GAP_TSV"
}

if [ "$STAGE" = "all" ] || [ "$STAGE" = "gap" ]; then
  CK=$(existing $OLD_ARMS)
  if [ -n "$CK" ]; then
    python flip_gap.py --pth_path $CK --out "$GAP_TSV"
  else
    echo "no old checkpoints found under ./model_pth ($OLD_ARMS)"
  fi
fi

if [ "$STAGE" = "all" ] || [ "$STAGE" = "train" ]; then
  train_arm S3    --fct_weight 0
  train_arm S3W05 --fct_loss bce --fct_weit 1 --fct_weight 0.5
  train_arm S3W2  --fct_loss bce --fct_weit 1 --fct_weight 2
fi

if [ "$STAGE" = "all" ] || [ "$STAGE" = "infer" ]; then
  for arm in S3 S3W05 S3W2; do
    [ -d "./model_pth/$arm" ] && infer_arm "$arm"
  done
fi

if [ "$STAGE" = "s3f" ]; then
  LAM="${2:?usage: bash run_fct_v2.sh s3f <lambda from the better S3W arm, 0.5 or 2>}"
  train_arm S3F --fct_loss bce --fct_weit 0 --fct_weight "$LAM"
  infer_arm S3F
fi

echo "flip gaps collected in $GAP_TSV; score ./result_map/<arm> with your evaluator as before."
