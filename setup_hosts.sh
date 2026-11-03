#!/usr/bin/env bash
# =============================================================================
#  One-time setup for FCT on other papers' models (the run_1288.sh host stages)
# =============================================================================
#  For each host named on the command line (default: all four), clones its repo
#  beside this one at the commit hosts.py was tested on, installs what its model
#  code imports and puts its weights and labels in place. Then it builds each
#  named host once (python hosts.py), on the GPU if there is one.
#
#    cascade   PVT-CASCADE: ml_collections, seaborn, scipy; links pvt_v2_b2.pth
#              where it looks for it
#    pranetv2  PraNet-V2: Res2Net-50 weights into its binary_seg/models, where
#              its README puts them
#    asgnet    ASGNet: fvcore, einops; PVTv2-b3 weights; edge labels for the
#              training masks (python make_edges.py)
#    hgm       HGM: torch_geometric, dynamic-network-architectures and mamba-ssm,
#              which must build against your torch and CUDA (this can take a while)
#
#  Their code is used as it is, from your clones; none of it is copied into this
#  repo. PVT-CASCADE is under the UT Austin Research License (research use only,
#  no redistribution); PraNet-V2 and ASGNet are MIT; HGM's repo has no licence.
#
#  Usage: bash setup_hosts.sh                 # all four
#         bash setup_hosts.sh asgnet hgm
#  Other folders: CASCADE_ROOT, PRANETV2_REPO, ASGNET_REPO, HGM_ROOT. Then give
#  run_1288.sh the matching CASCADE_ROOT, PRANETV2_ROOT (.../binary_seg),
#  ASGNET_ROOT (.../ASGNet_26_TCSVT) and HGM_ROOT.
# =============================================================================
set -eu -o pipefail
HOSTS="${*:-cascade pranetv2 asgnet hgm}"
CASCADE_ROOT="${CASCADE_ROOT:-../CASCADE}"
PRANETV2_REPO="${PRANETV2_REPO:-../PraNet-V2}"
ASGNET_REPO="${ASGNET_REPO:-../ASGNet}"
HGM_ROOT="${HGM_ROOT:-../HGM}"

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

setup_cascade() {
  clone https://github.com/SLDGroup/CASCADE "$CASCADE_ROOT" d3f050ad6337ef1d8213367eefb5e141bcf2ce75
  # imported by its lib/networks.py and lib/decoders.py; timm and opencv already come with Polyp-PVT
  pip install ml_collections seaborn scipy
  if [ ! -e pretrained_pth/pvt/pvt_v2_b2.pth ]; then
    [ -f pretrained_pth/pvt_v2_b2.pth ] || { echo "!! pretrained_pth/pvt_v2_b2.pth (Polyp-PVT's PVTv2-b2 weights) is missing"; exit 1; }
    mkdir -p pretrained_pth/pvt
    ln -s ../pvt_v2_b2.pth pretrained_pth/pvt/pvt_v2_b2.pth
  fi
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

setup_hgm() {
  clone https://github.com/YueyueZhu/HGM "$HGM_ROOT" 49ae05afc9785dbbc5a303d39e63588b068e5328
  [ -f pretrained_pth/pvt_v2_b2.pth ] || { echo "!! pretrained_pth/pvt_v2_b2.pth (Polyp-PVT's PVTv2-b2 weights) is missing"; exit 1; }
  pip install torch_geometric dynamic-network-architectures
  python -c "import mamba_ssm" 2>/dev/null || pip install mamba-ssm --no-build-isolation || {
    echo "!! mamba-ssm did not install; it needs a build that matches your torch and CUDA"
    echo "   (see https://github.com/state-spaces/mamba), then run this again"
    exit 1
  }
}

for h in $HOSTS; do
  case "$h" in
    cascade|pranetv2|asgnet|hgm) "setup_$h" ;;
    *) echo "unknown host: $h (choose from cascade pranetv2 asgnet hgm)"; exit 1 ;;
  esac
done

python hosts.py --hosts $HOSTS --cascade_root "$CASCADE_ROOT" --pranetv2_root "$PRANETV2_REPO/binary_seg" \
  --asgnet_root "$ASGNET_REPO/ASGNet_26_TCSVT" --hgm_root "$HGM_ROOT"
