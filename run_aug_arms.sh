#!/usr/bin/env bash
# =============================================================================
#  Orientation-augmentation arms: what does Polyp-PVT's own augmentation do?
# =============================================================================
#  Everything so far was trained orientation-free, which is the stock DEFAULT
#  (--augmentation defaults to the bool False while the loader compares against
#  the string 'True', so the branch never fires). These two arms turn it on.
#
#    PVT_AUG       augmentation ON, no FCT
#    PVT_AUG_FCT   augmentation ON, FCT ON
#
#  Two questions get answered at once:
#
#  1. Is the ~0.018 gap to the released checkpoint the augmentation?
#     Compare PVT_AUG against A2 (0.8498) and against raw (0.8683).
#
#  2. Are augmentation and FCT redundant -- the question this whole line of
#     work opened with? FCT is worth +0.0031 with no augmentation present.
#     If PVT_AUG_FCT - PVT_AUG is ~0, they are substitutes, and the redundancy
#     thesis is confirmed on whole images even though it was refuted on patches.
#
#  CAVEAT, unresolved: the local TrainDataset holds 1288 images against the
#  standard 1450 (missing exactly the 100 Kvasir + 62 ClinicDB test counts).
#  These arms are comparable to A2/B2, which share that split. They are NOT
#  comparable to the released checkpoint until the dataset is restored.
#
#  Usage: bash run_aug_arms.sh          # train both, then score everything
#         bash run_aug_arms.sh train    # training only
#         bash run_aug_arms.sh infer    # scoring only
# =============================================================================
set -u
STAGE="${1:-all}"
COMMON="--multiscale 1 --drop_path 0.1 --amp 1 --orientation_aug 1"

if [ "$STAGE" != "infer" ]; then
  echo "######## training PVT_AUG (augmentation, no FCT) ########"
  python Train_noaug.py $COMMON --train_save ./model_pth/PVT_AUG/ \
      2>&1 | tee logs_PVT_AUG.txt
  echo "######## training PVT_AUG_FCT (augmentation + FCT) ########"
  python Train_noaug.py $COMMON --fct 1 --fct_weight 0.5 \
      --train_save ./model_pth/PVT_AUG_FCT/ 2>&1 | tee logs_PVT_AUG_FCT.txt
fi

if [ "$STAGE" != "train" ]; then
  for arm in PVT_AUG PVT_AUG_FCT; do
    d="./model_pth/$arm"
    [ -d "$d" ] || { echo "[skip] $d"; continue; }
    python Test_maps.py --pth_path "$d" --save_root "./result_map/$arm"
  done
fi

cat <<'NOTE'

Reference points when scoring:
  raw Polyp-PVT (released)   0.8683
  A2  no-aug, no FCT         0.8498
  B2  no-aug + FCT           0.8529      FCT +0.0031

Read in this order:
  PVT_AUG        vs A2      -> what orientation augmentation is worth
  PVT_AUG_FCT    vs PVT_AUG -> is FCT redundant with the augmentation?
NOTE
