#!/usr/bin/env bash
# =============================================================================
#  All inference passes for the whole-image no-aug / FCT ablation
# =============================================================================
#  Produces two groups of result folders.
#
#  HEADLINE (4 passes) -- the numbers you report:
#      PVT_A2        arm A, plain inference
#      PVT_A2_TTA    arm A, 4-view flip TTA
#      PVT_B2        arm B (FCT), plain inference
#      PVT_B2_TTA    arm B (FCT), 4-view flip TTA
#
#  DIAGNOSTIC (6 passes) -- why TTA behaves the way it does:
#      PVT_*_h / _v / _hv   ONE flipped view, flipped back, no averaging.
#
#  Reading the diagnostic: if the model were flip-equivariant, every single
#  view would score exactly what the plain pass scores. Whatever a single view
#  loses IS the equivariance gap that TTA averages over. Comparing that gap on
#  arm A against arm B answers the question FCT exists to answer -- did the
#  consistency term actually install equivariance, or did it just regularise?
#
#  There is an architectural reason to expect a gap. PVTv2's stem is
#  Conv2d(k=7, s=4, p=3); at a 352 input its 88 token centres run 0,4,...,348,
#  flush to one edge with 3 px left over at the other. A flip moves those 3 px
#  across, so every token samples content displaced by 3 px = 0.75 of a stride.
#
#  Baseline to beat: raw Polyp-PVT scored on this same harness = 0.8683 mean
#  Dice (0.9114 / 0.9382 / 0.8003 / 0.9040 / 0.7874).
#
#  Usage:  bash run_inference.sh                 # headline + diagnostic
#          bash run_inference.sh headline        # the 4 reported passes only
# =============================================================================
set -u

A_DIR="${A_DIR:-./model_pth/PVT_A2}"
B_DIR="${B_DIR:-./model_pth/PVT_B2}"
OUT="${OUT:-./result_map}"
MODE="${1:-all}"

# run <ckpt_dir> <views> <single_view> <out_name>
run() {
  local ck=$1 views=$2 sv=$3 out=$4
  if [ ! -d "$ck" ]; then echo "[skip] no checkpoint dir $ck  ($out)"; return 0; fi
  echo "---- $out   (ckpt $ck, views $views, single_view $sv)"
  python Test_tta.py --pth_path "$ck" --views "$views" --single_view "$sv" \
      --save_root "$OUT/$out" || echo "[FAILED] $out"
}

echo "############ HEADLINE ############"
run "$A_DIR" 1 none PVT_A2
run "$A_DIR" 4 none PVT_A2_TTA
run "$B_DIR" 1 none PVT_B2
run "$B_DIR" 4 none PVT_B2_TTA

if [ "$MODE" != "headline" ]; then
  echo
  echo "############ DIAGNOSTIC: single flipped views ############"
  for v in h v hv; do run "$A_DIR" 1 "$v" "PVT_A2_$v"; done
  for v in h v hv; do run "$B_DIR" 1 "$v" "PVT_B2_$v"; done
fi

echo
echo "=================== score these folders ==================="
ls -d "$OUT"/PVT_A2* "$OUT"/PVT_B2* 2>/dev/null
cat <<'NOTE'

Compare against raw Polyp-PVT on the same harness:
  Kvasir 0.9114 | ClinicDB 0.9382 | ColonDB 0.8003 | CVC-300 0.9040 | ETIS 0.7874 | mean 0.8683

What each comparison tells you:
  PVT_A2   vs raw          did the no-aug recipe cost anything?
  PVT_B2   vs PVT_A2       what FCT is worth on a properly trained model
  PVT_B2   vs raw          THE GOAL -- beat 0.8683 on all five
  *_TTA    vs its own arm  whether flip TTA is usable at all here
  *_h/v/hv vs its own arm  the equivariance gap TTA is averaging over
NOTE
