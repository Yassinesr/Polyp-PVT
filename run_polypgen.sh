#!/usr/bin/env bash
# =============================================================================
#  PolypGen: score the existing seed checkpoints on six more centres
# =============================================================================
#  No training. Each PVT_VANILLA_s<n> and S3SIG_s<n> checkpoint (its best epoch, chosen on the
#  five usual test sets) predicts PolypGen's single-frame images centre by centre, and then
#  seed_summary.py compares the two arms. The analysis is fixed here, before any result:
#
#    primary    PolypGen pooled over the six centres (every image counted once)
#    secondary  each centre, C1 to C6
#    seeds      1 to 6, all of them: PolypGen played no part in choosing S3SIG or any checkpoint
#    claim      a set counts as improved only if the 95% interval of the same-seed
#               difference lies above 0
#
#  The same analysis applies to the arms scored later through ARMS: FCT (S3LMUP) and the three
#  views alone (S3). Their checkpoints were also chosen on the five usual test sets only.
#
#  Prepare the data first:   python prep_polypgen.py --src /path/to/PolypGen
#  Usage: bash run_polypgen.sh             # seeds 1-6, vanilla and S3SIG
#         bash run_polypgen.sh 1 2 3       # some seeds
#         ARMS="PVT_VANILLA S3LMUP S3" bash run_polypgen.sh   # FCT and the three views alone
#         ARMS="PRANETV2_VANILLA PRANETV2_S3LMUP" bash run_polypgen.sh   # another host: its runs
#               are built with that host's flags and compared with its own vanilla
#  Re-running skips every checkpoint whose PolypGen maps are complete.
# =============================================================================
set -u -o pipefail
SEEDS="${*:-1 2 3 4 5 6}"
ARMS="${ARMS:-PVT_VANILLA S3SIG}"     # the runs to score; each non-vanilla arm is compared with PVT_VANILLA
ROOT=./dataset/PolypGen
SETS="PolypGen_C1 PolypGen_C2 PolypGen_C3 PolypGen_C4 PolypGen_C5 PolypGen_C6"

for s in $SETS; do
  [ -d "$ROOT/$s/masks" ] || { echo "!! $ROOT/$s is missing: run prep_polypgen.py first"; exit 1; }
done

host_args() {   # arm -> the --host flags of the model its checkpoints belong to
  case "$1" in
    PRANETV2_*) echo "--host pranetv2 --host_root ${PRANETV2_ROOT:-../PraNet-V2/binary_seg}" ;;
    ASGNET_*)   echo "--host asgnet --host_root ${ASGNET_ROOT:-../ASGNet/ASGNet_26_TCSVT}" ;;
    SAM2UNET_*) echo "--host sam2unet --host_root ${SAM2UNET_ROOT:-../SAM2-UNet}" ;;
  esac
}

vanilla_of() {  # arm -> the vanilla it is compared with
  case "$1" in
    PRANETV2_*) echo PRANETV2_VANILLA ;;
    ASGNET_*)   echo ASGNET_VANILLA ;;
    SAM2UNET_*) echo SAM2UNET_VANILLA ;;
    *)          echo PVT_VANILLA ;;
  esac
}

done_maps() {   # run name: a map exists for every PolypGen mask
  local s n m
  for s in $SETS; do
    n=$(ls "$ROOT/$s/masks" | wc -l)
    m=$(ls "./result_map/$1/$s" 2>/dev/null | wc -l)
    [ "$m" -eq "$n" ] || return 1
  done
}

for seed in $SEEDS; do
  for arm in $ARMS; do
    run="${arm}_s$seed"
    if [ ! -f "./model_pth/$run/PolypPVT.pth" ]; then
      echo "!! no checkpoint for $run, skipped"
      continue
    fi
    if done_maps "$run"; then
      echo "=== $run: PolypGen maps already there"
      continue
    fi
    echo "=== $(date '+%F %T')  $run on PolypGen"
    python Test_maps.py --pth_path "./model_pth/$run/PolypPVT.pth" --data_root "$ROOT" \
        --sets $SETS --save_root "./result_map/$run" $(host_args "$arm") || echo "!! $run failed"
  done
done

for arm in $ARMS; do
  case "$arm" in *_VANILLA) continue ;; esac
  python seed_summary.py --seeds $SEEDS --arms "$(vanilla_of "$arm")" "$arm" --gt_root "$ROOT" --sets $SETS --pooled
done
