#!/usr/bin/env bash
# =============================================================================
#  Component dissection of FCT: the 18 runs of the core result, in the order of what they answer
# =============================================================================
#   1. s3              the three views supervised, no pull, seeds 1-6. s3 - vanilla is the gain of
#                      deterministic flip augmentation alone; nothing else can be read without it.
#   2. s3up            the three views and the pull to the best flipped view alone, seeds 1-6. With
#                      s3, s3lm and s3lmup (both already run) it decomposes FCT into the views, the
#                      pull to the best flipped view and the pull to the upright view.
#   3. s3fup s3relup   FCT with w = 1 and FCT with the lag gate only (a -> infinity): ablations of
#                      the weight function, seeds 1-3 first. Extend them to six seeds
#                      (stage "extend") only if they land near s3lmup.
#
#  Each step finishes all its seeds before the next one starts. run_seeds.sh runs its seeds in the
#  outer loop, so it is called once per step. Finished runs are skipped, so re-running the same
#  command resumes where it stopped.
#
#  Before any GPU time, test_fct_step.py must pass (CPU, a few seconds). It checks every arm's
#  gradient against an independent implementation, that each pull moves the gradient, and the
#  direction of the gate. The launcher then waits while another process holds the chosen GPU, so the
#  dissection never shares a GPU with the ASGNet seeds (FORCE=1 shares it anyway).
#
#  About 81 GPU-hours: s3 6 x 4.2 h, s3up 6 x 4.7 h, s3fup and s3relup 3 x 4.7 h each.
#
#  Usage:  GPU=1 nohup bash run_dissection.sh > dissection.out 2>&1 &            # the 18 runs
#          GPU=1 nohup bash run_dissection.sh extend > dissection_ext.out 2>&1 &   # s3fup, s3relup seeds 4-6
#          tail -f dissection.out ; nvidia-smi -l 60                              # progress, memory
# =============================================================================
set -u
GPU="${GPU:-1}"
STAGE="${1:-core}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID      # CUDA's GPU numbers then match nvidia-smi's
export CUDA_VISIBLE_DEVICES="$GPU"

python test_fct_step.py || { echo "!! test_fct_step.py failed: nothing launched"; exit 1; }

busy() { nvidia-smi --id="$GPU" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q .; }
nvidia-smi --id="$GPU" --query-gpu=index,name,memory.used,memory.total --format=csv
if [ "${FORCE:-0}" != 1 ]; then
  while busy; do
    echo "=== $(date '+%F %T')  GPU $GPU is busy; checking again in 10 min (FORCE=1 to share it):"
    nvidia-smi --id="$GPU" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
    sleep 600
  done
fi

step() {   # name, the stages to train, seeds...
  local name="$1" arms="$2"; shift 2
  echo "=== $(date '+%F %T')  $name: $arms on seeds $* (GPU $GPU)"
  ARMS="$arms" bash run_seeds.sh "$@" 2>&1 | tee -a "seeds_dissect_$name.out"
}

case "$STAGE" in
  core)
    step s3 s3 1 2 3 4 5 6
    step s3up s3up 1 2 3 4 5 6
    step weight "s3fup s3relup" 1 2 3
    ;;
  extend)
    step weight "s3fup s3relup" 4 5 6
    for a in S3FUP S3RELUP; do      # the six-seed comparison, once seeds 4-6 are in
      python seed_summary.py --seeds 1 2 3 4 5 6 --arms PVT_VANILLA "$a" | tee -a seeds_dissect_weight.out
    done
    ;;
  *) echo "usage: [GPU=n] bash run_dissection.sh [core|extend]"; exit 1 ;;
esac
echo "=== $(date '+%F %T')  done"
