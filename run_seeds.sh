#!/usr/bin/env bash
# =============================================================================
#  Multi-seed comparison: vanilla Polyp-PVT vs S3SIG (FCT v2, weight mode 6)
# =============================================================================
#  For each seed it trains vanilla, then S3SIG (run_1288.sh stages "vanilla" and
#  "s3sig", unchanged recipes), so a loop that stops early still leaves matched
#  pairs. At the end every seed run is scored with eval_dice.py and summarised
#  by seed_summary.py.
#
#  Default seeds 2-6: five seeds S3SIG was not chosen on. S3SIG became the
#  candidate because of its seed-1 result, so counting seed 1 would bias its
#  mean upward; seed_summary.py shows seed 1 separately.
#
#  Re-running the same command skips every run whose log reached the last
#  epoch and whose test maps are all there, so an interrupted loop resumes
#  where it stopped. A run that crashes is not scored (run_1288.sh uses
#  pipefail) and the loop goes on to the next one.
#
#  About 6.9 h per seed (vanilla ~2.5 h, S3SIG ~4.2 h, plus inference), so
#  ~35 h for five seeds. Start it detached so a dropped SSH session does not
#  kill it:
#      nohup bash run_seeds.sh > seeds.out 2>&1 &
#      tail -f seeds.out
#
#  Usage: bash run_seeds.sh              # seeds 2 3 4 5 6
#         bash run_seeds.sh 2 3 4        # other seeds
# =============================================================================
set -u
SEEDS="${*:-2 3 4 5 6}"
SETS="CVC-300 CVC-ClinicDB Kvasir CVC-ColonDB ETIS-LaribPolypDB"

finished() {   # run name: training reached the last epoch and every test map exists
  grep -q "Epoch \[099/100\], Step \[0081/0081\]" "logs_$1.txt" 2>/dev/null || return 1
  local s n m
  for s in $SETS; do
    n=$(ls "./dataset/TestDataset/$s/masks/" 2>/dev/null | wc -l)
    m=$(ls "./result_map/$1/$s/" 2>/dev/null | wc -l)
    [ "$n" -gt 0 ] && [ "$m" -eq "$n" ] || return 1
  done
}

for seed in $SEEDS; do
  for pair in "vanilla PVT_VANILLA" "s3sig S3SIG"; do
    set -- $pair
    stage=$1; run="$2_s$seed"
    if finished "$run"; then
      echo "=== $run already finished, skipped"
      continue
    fi
    echo "=== $(date '+%F %T')  $run"
    SEED=$seed bash run_1288.sh "$stage" || echo "!! $run failed (see logs_$run.txt); moving on"
  done
done

echo "=== $(date '+%F %T')  all seeds done; scoring"
python seed_summary.py --seeds $SEEDS
