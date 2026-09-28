#!/usr/bin/env bash
# =============================================================================
#  Final run: S3W05 on the full 1450-image training set, with weight EMA
# =============================================================================
#  Goal: beat raw Polyp-PVT (local) on all 5 test sets.
#    raw   Kvasir 0.9114  ClinicDB 0.9382  ColonDB 0.8003  CVC-300 0.9040  ETIS 0.7874
#  S3W05 on 1288 images already wins Kvasir, ColonDB, ETIS; it loses ClinicDB
#  (-0.0036) and CVC-300 (-0.0105).
#
#  Two changes, neither touches the method:
#    1. 1450 training images, like the released checkpoint. The 162 missing ones
#       sit in dataset/ValidationDataset (100 Kvasir + 62 ClinicDB), disjoint
#       from train and test. They are merged back into dataset/TrainDataset1450.
#    2. --ema 0.999: an exponential moving average of the weights. The stock LR
#       never decays, so the raw weights are noisy; small sets suffer most.
#
#  REPORTED MODEL, fixed before looking at results: PolypPVT_ema.pth
#  (the EMA weights at the best epoch, same selection rule as every arm so far).
#  The other three checkpoints are scored as a robustness check only.
#
#  Usage: bash run_final_1450.sh             # prep + train + infer
#         bash run_final_1450.sh prep        # build TrainDataset1450 only
#         bash run_final_1450.sh train
#         bash run_final_1450.sh infer
#         bash run_final_1450.sh ablation    # S3 (no consistency) on 1450 + infer
# =============================================================================
set -u
STAGE="${1:-all}"
DST=./dataset/TrainDataset1450
RECIPE="--multiscale 1 --drop_path 0.1 --amp 1 --fct_sync_rng 1 --orientation_aug 0 \
        --fct 1 --fct_sup 1 --fct_all_scales 1 --grad_diag 50 --ema 0.999 \
        --train_path $DST/"

prep() {
  # The pushed layout keeps the training RGB images in TrainDataset/gt/;
  # use TrainDataset/images/ instead if it exists (e.g. the symlink).
  local src_img=./dataset/TrainDataset/images
  [ -d "$src_img" ] || src_img=./dataset/TrainDataset/gt
  mkdir -p "$DST/images" "$DST/masks"
  cp -L "$src_img"/*                         "$DST/images/"
  cp -L ./dataset/ValidationDataset/images/* "$DST/images/"
  cp -L ./dataset/TrainDataset/masks/*       "$DST/masks/"
  cp -L ./dataset/ValidationDataset/gt/*     "$DST/masks/"
  local ni nm
  ni=$(ls "$DST/images" | wc -l); nm=$(ls "$DST/masks" | wc -l)
  echo "TrainDataset1450: images $ni, masks $nm (expected 1450 / 1450)"
  if [ "$ni" != 1450 ] || [ "$nm" != 1450 ]; then echo "count mismatch, stopping"; exit 1; fi
}

train_arm() {   # name, extra flags...
  local arm="$1"; shift
  echo "=== train $arm"
  python Train_noaug.py $RECIPE "$@" --train_save "./model_pth/$arm/" 2>&1 | tee "logs_$arm.txt"
}

infer_arm() {
  local arm="$1" d="./model_pth/$1"
  for ck in PolypPVT_ema PolypPVT last_ema last; do
    [ -f "$d/$ck.pth" ] || continue
    local tag="$arm"
    case "$ck" in
      PolypPVT_ema) tag="${arm}_ema" ;;
      last_ema)     tag="${arm}_last_ema" ;;
      last)         tag="${arm}_last" ;;
    esac
    echo "=== infer $arm/$ck -> result_map/$tag"
    python Test_tta.py --pth_path "$d/$ck.pth" --views 1 --save_root "./result_map/$tag"
  done
  python flip_gap.py --pth_path "$d/PolypPVT_ema.pth" "$d/PolypPVT.pth" --out flip_gap.tsv
}

case "$STAGE" in
  prep)     prep ;;
  train)    train_arm S3W05_1450 --fct_loss bce --fct_weit 1 --fct_weight 0.5 ;;
  infer)    infer_arm S3W05_1450 ;;
  ablation) train_arm S3_1450 --fct_weight 0 && infer_arm S3_1450 ;;
  all)      prep && train_arm S3W05_1450 --fct_loss bce --fct_weit 1 --fct_weight 0.5 \
              && infer_arm S3W05_1450 ;;
  *)        echo "unknown stage: $STAGE"; exit 1 ;;
esac
echo "score result_map/S3W05_1450_ema (the reported model) with your evaluator."
