#!/usr/bin/env bash
# =============================================================================
#  One-time setup for FCT on the two other models (run_1288.sh cascade_* and
#  pranetv2_* stages)
# =============================================================================
#  Clones PVT-CASCADE and PraNet-V2 beside this repo at the commits hosts.py was
#  tested on, installs the packages their model code imports, links the PVTv2-b2
#  weights where PVT-CASCADE looks for them, downloads the Res2Net-50 weights to
#  PraNet-V2's binary_seg/models (where its README puts them), and builds both
#  once (python hosts.py).
#
#  Their code is used as it is, from your clones; none of it is copied into this
#  repo. PVT-CASCADE is under the UT Austin Research License (research use only,
#  no redistribution); PraNet-V2 is MIT.
#
#  Usage: bash setup_hosts.sh
#         CASCADE_ROOT=/data/CASCADE PRANETV2_REPO=/data/PraNet-V2 bash setup_hosts.sh
#  With other folders, give run_1288.sh / run_seeds.sh the same ones:
#         CASCADE_ROOT=/data/CASCADE PRANETV2_ROOT=/data/PraNet-V2/binary_seg ...
# =============================================================================
set -eu -o pipefail
CASCADE_ROOT="${CASCADE_ROOT:-../CASCADE}"
PRANETV2_REPO="${PRANETV2_REPO:-../PraNet-V2}"
RES2NET=https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_v1b_26w_4s-3cf99910.pth

clone() {   # url, folder, commit
  [ -d "$2/.git" ] || git clone "$1" "$2"
  git -C "$2" checkout -q "$3" 2>/dev/null || { git -C "$2" fetch -q origin && git -C "$2" checkout -q "$3"; }
  echo "$2 at $(git -C "$2" rev-parse --short HEAD)"
}
clone https://github.com/SLDGroup/CASCADE "$CASCADE_ROOT" d3f050ad6337ef1d8213367eefb5e141bcf2ce75
clone https://github.com/ai4colonoscopy/PraNet-V2 "$PRANETV2_REPO" ca08522ae77cbda6c5056609e7951b502dbb08f2

# imported by PVT-CASCADE's lib/networks.py and lib/decoders.py; timm and opencv
# are already needed by Polyp-PVT
pip install ml_collections seaborn scipy

if [ ! -e pretrained_pth/pvt/pvt_v2_b2.pth ]; then
  [ -f pretrained_pth/pvt_v2_b2.pth ] || { echo "!! pretrained_pth/pvt_v2_b2.pth (Polyp-PVT's PVTv2-b2 weights) is missing"; exit 1; }
  mkdir -p pretrained_pth/pvt
  ln -s ../pvt_v2_b2.pth pretrained_pth/pvt/pvt_v2_b2.pth
fi

R2N="$PRANETV2_REPO/binary_seg/models/$(basename "$RES2NET")"
if [ ! -f "$R2N" ]; then
  mkdir -p "$(dirname "$R2N")"
  python -c "import sys, torch.hub; torch.hub.download_url_to_file(sys.argv[1], sys.argv[2])" "$RES2NET" "$R2N" || {
    echo "!! could not download $RES2NET"
    echo "   put res2net50_v1b_26w_4s-3cf99910.pth in $(dirname "$R2N") by hand (PraNet-V2's README"
    echo "   links it on Google Drive) and run this again"
    exit 1
  }
fi

python hosts.py --cascade_root "$CASCADE_ROOT" --pranetv2_root "$PRANETV2_REPO/binary_seg"
