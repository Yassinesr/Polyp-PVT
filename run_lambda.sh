#!/usr/bin/env bash
# =============================================================================
#  Sweep of FCT's consistency weight lambda on Polyp-PVT, six seeds
# =============================================================================
#  Each run is FCT (S3LMUP) exactly as the reference S3LMUP_s1-6 runs (lambda = 0.5) were
#  trained, with only --fct_weight changed:
#    --multiscale 1 --drop_path 0.1 --amp 1 --train_path ./dataset/TrainDataset/ --seed N
#    --orientation_aug 0 --fct 1 --fct_sup 1 --fct_all_scales 1 --fct_loss bce
#    --fct_sync_rng 1 --grad_diag 50 --fct_weit 8 --fct_up 1 --fct_weight <lambda>
#  (--grad_diag 50 as in the reference runs; it also logs |g_cons|/|g_sup| for each lambda.)
#  Runs: S3LMUP_L025_s<seed> (0.25), S3LMUP_L10_s<seed> (1.0), S3LMUP_L20_s<seed> (2.0).
#  Each run: Train_noaug.py -> model_pth/<run>/ and logs_<run>.txt, then Test_maps.py on
#  PolypPVT.pth -> result_map/<run>/, then flip_gap.py -> one row appended to flip_gap.tsv.
#
#  Self-contained: it does not use run_1288.sh or run_seeds.sh.
#  Order: seed by seed (every lambda on seed 1 before seed 2), so a partial queue still
#  gives paired comparisons. Before each run it waits until the GPU has no compute process
#  (FORCE=1 skips the wait). A run whose log has '=== training finished' and whose maps are
#  complete is skipped, so re-running resumes. A run without that line counts as failed:
#  no maps, no flip-gap row, and the queue moves on.
#
#  Usage (about 4.8 h per run):
#    GPU=0 nohup bash run_lambda.sh 1 2 3 > lambda_gpu0.out 2>&1 &
#    GPU=1 nohup bash run_lambda.sh 4 5 6 > lambda_gpu1.out 2>&1 &
#    bash run_lambda.sh summary        # after all six seeds: each lambda vs PVT_VANILLA and vs S3LMUP
#    DRY_RUN=1 bash run_lambda.sh 1    # print the commands only
#    LAMBDAS="L05:0.5" GPU=0 nohup bash run_lambda.sh 1 2 3 4 5 6 > lambda_l05.out 2>&1 &
#                                      # optional control: lambda = 0.5 with the current code
#  Progress: grep '=== ' lambda_gpu0.out ; tail -1 logs_S3LMUP_L025_s1.txt
# =============================================================================
set -u -o pipefail
cd "$(dirname "$0")"
GPU="${GPU:-0}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"
LAMBDAS="${LAMBDAS:-L025:0.25 L10:1.0 L20:2.0}"      # name:value, the run is S3LMUP_<name>_s<seed>
SETS="CVC-300 CVC-ClinicDB Kvasir CVC-ColonDB ETIS-LaribPolypDB"
COMMON="--multiscale 1 --drop_path 0.1 --amp 1 --train_path ./dataset/TrainDataset/"
FCT="--orientation_aug 0 --fct 1 --fct_sup 1 --fct_all_scales 1 --fct_loss bce --fct_sync_rng 1 --grad_diag 50 --fct_weit 8 --fct_up 1"

summary() {
  local nv run
  for nv in $LAMBDAS; do
    run="S3LMUP_${nv%%:*}"
    echo "=== $run (lambda ${nv#*:}) vs PVT_VANILLA"
    python seed_summary.py --seeds 1 2 3 4 5 6 --arms PVT_VANILLA "$run"
    echo "=== $run (lambda ${nv#*:}) vs S3LMUP (lambda 0.5)"
    python seed_summary.py --seeds 1 2 3 4 5 6 --arms S3LMUP "$run"
  done
}
if [ "${1:-}" = summary ]; then summary; exit 0; fi
SEEDS="${*:-1 2 3 4 5 6}"

run() {   # print the command; execute it unless DRY_RUN=1
  echo "+ $*"
  [ "$DRY_RUN" = 1 ] || "$@"
}

if [ -d ./dataset/TestDataset/test ]; then
  echo "!! ./dataset/TestDataset/test exists: Train_noaug.py would pick the best epoch on it. Move it away first."
  exit 1
fi
n=$(ls ./dataset/TrainDataset/images/ 2>/dev/null | wc -l)
if [ "$n" != 1288 ]; then
  echo "!! dataset/TrainDataset/images has $n files, expected 1288"
  exit 1
fi
python test_fct_step.py > /dev/null || { echo "!! test_fct_step.py failed: nothing launched"; exit 1; }
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU"

finished() {   # run name: the log says training finished and every test map exists
  grep -q "^=== training finished" "logs_$1.txt" 2>/dev/null || return 1
  local s n m
  for s in $SETS; do
    n=$(ls "./dataset/TestDataset/$s/masks/" 2>/dev/null | wc -l)
    m=$(ls "./result_map/$1/$s/" 2>/dev/null | wc -l)
    [ "$n" -gt 0 ] && [ "$m" -eq "$n" ] || return 1
  done
}

wait_gpu() {   # until physical GPU $GPU runs no compute process
  [ "$FORCE" = 1 ] || [ "$DRY_RUN" = 1 ] && return 0
  while [ -n "$(nvidia-smi --id="$GPU" --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]; do
    echo "=== $(date '+%F %T')  GPU $GPU busy, waiting"; sleep 600
  done
}

echo "=== $(date '+%F %T')  lambda sweep on GPU $GPU; seeds: $SEEDS; lambdas: $LAMBDAS"
for seed in $SEEDS; do
  for nv in $LAMBDAS; do
    name="${nv%%:*}"; lam="${nv#*:}"; arm="S3LMUP_${name}_s$seed"
    if finished "$arm"; then echo "=== $arm already finished, skipped"; continue; fi
    wait_gpu
    echo "=== $(date '+%F %T')  train $arm (lambda $lam)"
    if [ "$DRY_RUN" = 1 ]; then
      echo "+ python Train_noaug.py $COMMON --seed $seed $FCT --fct_weight $lam --train_save ./model_pth/$arm/ 2>&1 | tee logs_$arm.txt"
    else
      python Train_noaug.py $COMMON --seed "$seed" $FCT --fct_weight "$lam" --train_save "./model_pth/$arm/" 2>&1 | tee "logs_$arm.txt"
      if ! grep -q "^=== training finished" "logs_$arm.txt"; then
        echo "!! $arm did not finish (see logs_$arm.txt); no maps, moving on"; continue
      fi
    fi
    run python Test_maps.py --pth_path "./model_pth/$arm/PolypPVT.pth" --save_root "./result_map/$arm" \
      || { echo "!! maps for $arm failed"; continue; }
    run python flip_gap.py --pth_path "./model_pth/$arm/PolypPVT.pth" --out flip_gap.tsv
    echo "=== $(date '+%F %T')  done $arm"
  done
done
echo "=== $(date '+%F %T')  lambda sweep finished on GPU $GPU; when both GPUs are done: bash run_lambda.sh summary"
