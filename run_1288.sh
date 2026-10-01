#!/usr/bin/env bash
# =============================================================================
#  1288-image runs: vanilla Polyp-PVT baseline and FCT v2 with a constant weight
# =============================================================================
#  The 162-image ValidationDataset stays held out, so every model here trains on
#  dataset/TrainDataset (1288 = 800 Kvasir + 488 ClinicDB). The best epoch
#  (PolypPVT.pth / PolypPVT_ema.pth) is the checkpoint scored.
#
#  vanilla       stock Polyp-PVT loss, no FCT, orientation aug off (the stock
#                default), multi-scale on, same AMP/drop-path as the FCT arms
#  vanilla_aug   the same with Polyp-PVT's flip/rotation augmentation on
#  s3f_ema       FCT v2 with the boundary weight replaced by a constant (w = 1),
#                i.e. the S3F recipe, plus weight EMA
#
#  Usage: bash run_1288.sh vanilla | vanilla_aug | s3f_ema
#         bash run_1288.sh all            # vanilla, then s3f_ema
# =============================================================================
set -u
STAGE="${1:-all}"
COMMON="--multiscale 1 --drop_path 0.1 --amp 1 --train_path ./dataset/TrainDataset/"

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
    local tag="$arm"; [ "$ck" = PolypPVT_ema ] && tag="${arm}_ema"
    echo "=== infer $arm/$ck -> result_map/$tag"
    python Test_tta.py --pth_path "./model_pth/$arm/$ck.pth" --views 1 --save_root "./result_map/$tag"
    python flip_gap.py --pth_path "./model_pth/$arm/$ck.pth" --out flip_gap.tsv
  done
}

vanilla()     { train_arm PVT_VANILLA --orientation_aug 0 && infer_arm PVT_VANILLA PolypPVT; }
vanilla_aug() { train_arm PVT_VANILLA_AUG --orientation_aug 1 && infer_arm PVT_VANILLA_AUG PolypPVT; }
s3f_ema()     { train_arm S3F_EMA --orientation_aug 0 --fct 1 --fct_sup 1 --fct_all_scales 1 \
                  --fct_loss bce --fct_weit 0 --fct_weight 0.5 --fct_sync_rng 1 --grad_diag 50 --ema 0.999 \
                && infer_arm S3F_EMA PolypPVT PolypPVT_ema; }

case "$STAGE" in
  vanilla)     vanilla ;;
  vanilla_aug) vanilla_aug ;;
  s3f_ema)     s3f_ema ;;
  all)         vanilla && s3f_ema ;;
  *)           echo "unknown stage: $STAGE"; exit 1 ;;
esac
echo "score result_map/<arm> with your evaluator."
