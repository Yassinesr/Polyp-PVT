#!/usr/bin/env bash
# =============================================================================
#  One-time setup for FCT on other papers' models (the run_1288.sh host stages)
# =============================================================================
#  For each host named on the command line (default: both), clones its repo
#  beside this one at the commit hosts.py was tested on, installs what its model
#  code imports and puts its weights and labels in place. Then it builds each
#  named host once (python hosts.py), on the GPU if there is one.
#
#    pranetv2  PraNet-V2: Res2Net-50 weights into its binary_seg/models, where
#              its README puts them
#    asgnet    ASGNet: fvcore, einops; PVTv2-b3 weights; edge labels for the
#              training masks (python make_edges.py)
#
#  Their code is used as it is, from your clones; none of it is copied into this
#  repo. PraNet-V2 and ASGNet are both MIT.
#
#  Usage: bash setup_hosts.sh                 # both
#         bash setup_hosts.sh asgnet
#  Other folders: PRANETV2_REPO, ASGNET_REPO. Then give run_1288.sh the matching
#  PRANETV2_ROOT (.../binary_seg) and ASGNET_ROOT (.../ASGNet_26_TCSVT).
# =============================================================================
set -eu -o pipefail
HOSTS="${*:-pranetv2 asgnet}"
PRANETV2_REPO="${PRANETV2_REPO:-../PraNet-V2}"
ASGNET_REPO="${ASGNET_REPO:-../ASGNet}"

clone() {   # url, folder, commit
  [ -d "$2/.git" ] || git clone "$1" "$2"
  git -C "$2" checkout -q "$3" 2>/dev/null || { git -C "$2" fetch -q origin && git -C "$2" checkout -q "$3"; }
  echo "$2 at $(git -C "$2" rev-parse --short HEAD)"
}

fetch() {   # url, file
  [ -f "$2" ] && return 0
  mkdir -p "$(dirname "$2")"
  python -c "import sys, torch.hub; torch.hub.download_url_to_file(sys.argv[1], sys.argv[2])" "$1" "$2" || {
    echo "!! could not download $1"
    echo "   put the file at $2 by hand and run this again"
    exit 1
  }
}

setup_pranetv2() {
  clone https://github.com/ai4colonoscopy/PraNet-V2 "$PRANETV2_REPO" ca08522ae77cbda6c5056609e7951b502dbb08f2
  fetch https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_v1b_26w_4s-3cf99910.pth \
        "$PRANETV2_REPO/binary_seg/models/res2net50_v1b_26w_4s-3cf99910.pth"
}

setup_asgnet() {
  clone https://github.com/CSYSI/ASGNet "$ASGNET_REPO" 97c4e746e94bb6a184a14775b771226eed88c3e2
  pip install fvcore einops         # imported by its lib/module3_NEW_PS*.py
  fetch https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b3.pth pretrained_pth/pvt_v2_b3.pth
  python make_edges.py
}

for h in $HOSTS; do
  case "$h" in
    pranetv2|asgnet) "setup_$h" ;;
    *) echo "unknown host: $h (choose from pranetv2 asgnet)"; exit 1 ;;
  esac
done

python hosts.py --hosts $HOSTS --pranetv2_root "$PRANETV2_REPO/binary_seg" \
  --asgnet_root "$ASGNET_REPO/ASGNet_26_TCSVT"
