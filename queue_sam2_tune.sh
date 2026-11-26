#!/usr/bin/env bash
# =============================================================================
#  Screen FCT variants on SAM2-UNet, on GPU 0 only
# =============================================================================
#  Each candidate is sam2unet_s3lmup with extra FCT flags (run_1288.sh sam2unet_tune), run
#  as SAM2UNET_S3LMUP_<name>_s<seed> and compared with the existing same-seed
#  SAM2UNET_VANILLA_s<seed> (trained with SAM2UNET_AMP=1, as these are).
#
#  Takes an exclusive lock on /tmp/gpu0.lock and waits until GPU 0 has under 1000 MiB in use
#  (the GPU 0 job before it is done), then runs seed by seed,
#  every candidate on seed 1 before any on seed 2, so a partial queue still compares like
#  with like. A run whose log finished and whose maps are complete is skipped, so
#  re-running resumes. About 3 h per run.
#
#  Usage:  nohup bash queue_sam2_tune.sh > queue_sam2_tune.out 2>&1 &
#          SEEDS="1 2 3" CANDIDATES="W10:--fct_weight 1.0;W20:--fct_weight 2.0" bash queue_sam2_tune.sh
#  Progress markers:  grep '=== STEP' queue_sam2_tune.out
# =============================================================================
set -u -o pipefail
cd "$(dirname "$0")"
SEEDS="${SEEDS:-1 2 3}"
CANDIDATES="${CANDIDATES:-W10:--fct_weight 1.0;W20:--fct_weight 2.0}"   # name:flags;name:flags
SETS="CVC-300 CVC-ClinicDB Kvasir CVC-ColonDB ETIS-LaribPolypDB"
export CUDA_VISIBLE_DEVICES=0 SAM2UNET_AMP=1

echo "=== $(date '+%F %T')  acquiring GPU 0 lock"
exec 200>/tmp/gpu0.lock
flock -x 200
until [ "$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1000 ]; do
  echo "=== $(date '+%F %T')  GPU 0 still has memory in use, waiting"; sleep 60
done
echo "=== $(date '+%F %T')  GPU 0 locked; seeds: $SEEDS"

finished() {   # run name: training reached its last epoch and every test map exists
  grep -q "^=== training finished" "logs_$1.txt" 2>/dev/null || return 1
  local s n m
  for s in $SETS; do
    n=$(ls "./dataset/TestDataset/$s/masks/" 2>/dev/null | wc -l)
    m=$(ls "./result_map/$1/$s/" 2>/dev/null | wc -l)
    [ "$n" -gt 0 ] && [ "$m" -eq "$n" ] || return 1
  done
}

IFS=';' read -ra CANDS <<< "$CANDIDATES"
for seed in $SEEDS; do
  for c in "${CANDS[@]}"; do
    name="${c%%:*}"; flags="${c#*:}"
    run="SAM2UNET_S3LMUP_${name}_s$seed"
    if finished "$run"; then echo "=== STEP $run done (already) $(grep -o 'best five-set mean [0-9.]*' "logs_$run.txt")"; continue; fi
    echo "=== $(date '+%F %T')  $run  ($flags)"
    SEED=$seed TUNE="$name" TUNE_ARGS="$flags" bash run_1288.sh sam2unet_tune \
      || echo "!! $run failed (see logs_$run.txt)"
    echo "=== STEP $run done $(date '+%F %T')  $(grep -o 'best five-set mean [0-9.]*' "logs_$run.txt" 2>/dev/null || echo 'no result')"
  done
done

echo "=== $(date '+%F %T')  screening done; each candidate vs SAM2UNET_VANILLA on seeds $SEEDS"
echo "--- reference: the current recipe (SAM2UNET_S3LMUP)"
python seed_summary.py --seeds $SEEDS --arms SAM2UNET_VANILLA SAM2UNET_S3LMUP
for c in "${CANDS[@]}"; do
  name="${c%%:*}"
  echo "--- candidate $name (${c#*:})"
  python seed_summary.py --seeds $SEEDS --arms SAM2UNET_VANILLA "SAM2UNET_S3LMUP_$name"
done
echo "=== QUEUE DONE $(date '+%F %T')"
