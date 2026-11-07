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
#  Prepare the data first:   python prep_polypgen.py --src /path/to/PolypGen
#  Usage: bash run_polypgen.sh             # seeds 1-6
#         bash run_polypgen.sh 1 2 3       # some seeds
#  Re-running skips every checkpoint whose PolypGen maps are complete.
# =============================================================================
set -u -o pipefail
SEEDS="${*:-1 2 3 4 5 6}"
ROOT=./dataset/PolypGen
SETS="PolypGen_C1 PolypGen_C2 PolypGen_C3 PolypGen_C4 PolypGen_C5 PolypGen_C6"

for s in $SETS; do
  [ -d "$ROOT/$s/masks" ] || { echo "!! $ROOT/$s is missing: run prep_polypgen.py first"; exit 1; }
done

done_maps() {   # run name: a map exists for every PolypGen mask
  local s n m
  for s in $SETS; do
    n=$(ls "$ROOT/$s/masks" | wc -l)
    m=$(ls "./result_map/$1/$s" 2>/dev/null | wc -l)
    [ "$m" -eq "$n" ] || return 1
  done
}

for seed in $SEEDS; do
  for arm in PVT_VANILLA S3SIG; do
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
    python Test_tta.py --views 1 --pth_path "./model_pth/$run/PolypPVT.pth" --data_root "$ROOT" \
        --sets $SETS --save_root "./result_map/$run" || echo "!! $run failed"
  done
done

python seed_summary.py --seeds $SEEDS --gt_root "$ROOT" --sets $SETS --pooled
