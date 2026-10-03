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
#  s3f           FCT v2 with a constant weight (w = 1), the S3F recipe
#  s3err         FCT v2 with an error-aware per-pixel weight w = 1 - |p_o - y|:
#                pixels where the upright prediction is wrong are not used as
#                consistency targets
#
#  SEED=<n> sets --seed and appends _s<n> to the run name (default: unseeded).
#
#  Usage: bash run_1288.sh vanilla | vanilla_aug | s3f | s3err
#         SEED=1 bash run_1288.sh s3err
#         bash run_1288.sh all            # = s3err
# =============================================================================
set -u
STAGE="${1:-all}"
SEED="${SEED:-}"
SFX=""; SEEDFLAG=""
if [ -n "$SEED" ]; then SFX="_s$SEED"; SEEDFLAG="--seed $SEED"; fi
COMMON="--multiscale 1 --drop_path 0.1 --amp 1 --train_path ./dataset/TrainDataset/ $SEEDFLAG"

if [ -d ./dataset/TestDataset/test ]; then
  echo "!! ./dataset/TestDataset/test exists: Train_noaug.py would pick the best epoch on it"
  echo "   instead of the 5-set mean used by every earlier run. Move it away first."
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
    python Test_tta.py --pth_path "./model_pth/$arm/$ck.pth" --views 1 --save_root "./result_map/$tag"
    python flip_gap.py --pth_path "./model_pth/$arm/$ck.pth" --out flip_gap.tsv
  done
}

FCT="--orientation_aug 0 --fct 1 --fct_sup 1 --fct_all_scales 1 --fct_loss bce --fct_weight 0.5 \
     --fct_sync_rng 1 --grad_diag 50"
vanilla()     { train_arm PVT_VANILLA$SFX --orientation_aug 0 && infer_arm PVT_VANILLA$SFX PolypPVT; }
vanilla_aug() { train_arm PVT_VANILLA_AUG$SFX --orientation_aug 1 && infer_arm PVT_VANILLA_AUG$SFX PolypPVT; }
s3f()         { train_arm S3F$SFX $FCT --fct_weit 0 && infer_arm S3F$SFX PolypPVT; }
s3err()       { train_arm S3ERR$SFX $FCT --fct_weit 2 && infer_arm S3ERR$SFX PolypPVT; }

case "$STAGE" in
  vanilla)     vanilla ;;
  vanilla_aug) vanilla_aug ;;
  s3f)         s3f ;;
  s3err)       s3err ;;
  all)         s3err ;;
  *)           echo "unknown stage: $STAGE"; exit 1 ;;
esac
echo "score result_map/<arm> with your evaluator."
