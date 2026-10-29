"""The models FCT can train (--host): Polyp-PVT and two hosts from other papers.

A host says how to build its model, which of its outputs make the prediction, and the
paper's own supervised loss for one view. Train_noaug.py, Test_tta.py and flip_gap.py go
through it instead of assuming Polyp-PVT's two heads, so FCT is the same code on every
host and the vanilla arm keeps each paper's loss.

  polyppvt   Polyp-PVT (this repo). Heads P1, P2; prediction P1 + P2; loss: the
             structure loss on each head.
  cascade    PVT-CASCADE (Rahman and Marculescu, WACV 2023). Heads p1..p4; prediction
             p1 + p2 + p3 + p4, as in its test_polyp.py; loss: the same structure loss
             on each head with weight 1, as in its train_polyp.py.
  pranetv2   PraNet-V2 on Res2Net-50 (Hu et al., arXiv 2504.10986). Four foreground and
             four background maps; prediction: the sum of the foreground maps, as in its
             MyTest_med.py; loss: its foreground/background structure loss on each pair,
             as in its MyTrain_med.py.

No code from either paper is copied here: CASCADE's licence does not allow it to be
redistributed. --host_root points at your own clone, and the model class is imported
from it unchanged; setup_hosts.sh clones both at the commits this was tested on.

  python hosts.py        # build each host on the CPU and run one forward
"""
import os
import sys
import types
import argparse
import importlib
import subprocess

import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_CLASH = ('lib', 'utils')            # package names the hosts share with this repo
ROOTS = {'cascade': '../CASCADE', 'pranetv2': '../PraNet-V2/binary_seg'}
COMMITS = {'cascade': 'd3f050ad6337ef1d8213367eefb5e141bcf2ce75',
           'pranetv2': 'ca08522ae77cbda6c5056609e7951b502dbb08f2'}


def _boundary_weight(mask):
    """The stock structure_loss weight: 1 in flat regions, up to 6 near a boundary."""
    return 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)


def structure_loss(pred, mask):
    """Unchanged from the stock repo; PVT-CASCADE's train_polyp.py has the same one.

    Note the stock `reduce='none'`: `reduce` is the old boolean argument and the
    string counts as true, so the BCE term comes back as a scalar mean and
    `weit` has NO effect on it (it still weights the IoU term). Every arm, and
    the released checkpoint, share this, so it is deliberately left as is.
    """
    weit = _boundary_weight(mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def structure_loss_fgbg(pred, pred_bg, mask_fg, mask_bg):
    """PraNet-V2's loss for one pair of maps, as in its MyTrain_med.py.

    The boundary-weighted BCE and IoU of the foreground map, plus 0.8 x the
    boundary-weighted BCE of the background map against 1 - y. Unlike the stock
    loss above, its BCE really is weighted (reduction='none').
    """
    weit = _boundary_weight(mask_fg)
    wbce = F.binary_cross_entropy_with_logits(pred, mask_fg, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    wbce2 = F.binary_cross_entropy_with_logits(pred_bg, mask_bg, reduction='none')
    wbce2 = (weit * wbce2).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask_fg) * weit).sum(dim=(2, 3))
    union = ((pred + mask_fg) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou + 0.8 * wbce2).mean()


def _import_from(root, name):
    """Import module `name` (for example 'lib.networks') from the clone at `root`.

    The hosts keep their code in packages called lib and utils, as this repo does.
    While a host module is imported, this repo's packages are taken out of
    sys.modules and its folder off sys.path, so `from lib.x import y` inside the
    host finds the host's own files. Afterwards the host's modules are kept under a
    private name and this repo's are put back.
    """
    root = os.path.abspath(root)
    ours = {k: sys.modules.pop(k) for k in list(sys.modules) if k.split('.')[0] in _CLASH}
    path = sys.path[:]
    sys.path[:] = [root] + [p for p in path if os.path.realpath(p or os.curdir) != _HERE]
    importlib.invalidate_caches()
    try:
        return importlib.import_module(name)
    finally:
        tag = '_host_{}:'.format(os.path.basename(root))
        for k in [k for k in sys.modules if k.split('.')[0] in _CLASH]:
            sys.modules[tag + k] = sys.modules.pop(k)
        sys.modules.update(ours)
        sys.path[:] = path
        importlib.invalidate_caches()


def _scipy_misc_stand_in():
    """PVT-CASCADE's decoders.py imports scipy.misc, which recent SciPy releases
    dropped, and uses nothing from it; an empty module stands in when it is missing."""
    try:
        import scipy.misc  # noqa: F401
    except ImportError:
        sys.modules.setdefault('scipy.misc', types.ModuleType('scipy.misc'))


def clone_commit(root):
    """The commit checked out in the clone at `root`, or None if git cannot tell."""
    try:
        return subprocess.run(['git', '-C', root, 'rev-parse', 'HEAD'], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


class PolypPVTHost:
    """Polyp-PVT, the model the rest of this repo was written for."""
    name, paper = 'polyppvt', 'Polyp-PVT (this repo)'
    weights = './pretrained_pth/pvt_v2_b2.pth'     # hard-coded in PolypPVT.__init__
    module = None                                  # the clone file the model comes from

    def __init__(self, root=None):
        self.root = None if root is None else os.path.abspath(os.path.expanduser(root))

    def build(self):
        from lib.pvt import PolypPVT
        return PolypPVT()

    def heads(self, out):
        """The model's outputs as a list of logit maps, all at the input size."""
        return list(out)

    def logit(self, heads):
        """The prediction as a logit map: the sum of the heads, as at test time."""
        z = heads[0]
        for h in heads[1:]:
            z = z + h
        return z

    def sup_loss(self, heads, gts):
        """The paper's supervised loss for one view: the structure loss on each head."""
        loss = structure_loss(heads[0], gts)
        for h in heads[1:]:
            loss = loss + structure_loss(h, gts)
        return loss

    def set_drop_path(self, model, rate):
        """Stochastic depth of the PVTv2 backbone; returns the rate now in effect."""
        model.backbone.reset_drop_path(rate)
        return rate

    def preflight(self):
        """Fail now with a clear message rather than after the data loads."""
        if self.module and not os.path.isfile(os.path.join(self.root or '', self.module)):
            raise FileNotFoundError(
                '{} not found under {} -- point --host_root at your clone of the {} code '
                '(setup_hosts.sh makes one)'.format(self.module, self.root, self.paper))
        if self.weights and not os.path.isfile(self.weights):
            raise FileNotFoundError(
                'Missing backbone weights at {} -- the model hardcodes this path and will '
                'crash on construction{}.'.format(
                    self.weights, '' if self.module is None else ' (setup_hosts.sh puts it there)'))

    def describe(self):
        if self.module is None:
            return self.paper
        c, tested = clone_commit(self.root), COMMITS[self.name]
        return '{} from {} at {} ({})'.format(
            self.paper, self.root, (c or 'unknown')[:7],
            'the tested commit' if c == tested else '!! NOT the tested commit ' + tested[:7])


class CascadeHost(PolypPVTHost):
    """PVT-CASCADE: four heads, summed at test time, each trained with the structure loss."""
    name, paper = 'cascade', 'PVT-CASCADE (Rahman and Marculescu, WACV 2023)'
    weights = './pretrained_pth/pvt/pvt_v2_b2.pth'  # hard-coded in PVT_CASCADE.__init__
    module = 'lib/networks.py'

    def build(self):
        _scipy_misc_stand_in()
        return _import_from(self.root, 'lib.networks').PVT_CASCADE()   # n_class=1, as in train_polyp.py


class PraNetV2Host(PolypPVTHost):
    """PraNet-V2 (Res2Net-50): four foreground maps, summed at test time, and four
    background maps; each pair is trained with its foreground/background loss."""
    name, paper = 'pranetv2', 'PraNet-V2, Res2Net-50 (Hu et al., arXiv 2504.10986)'
    module = 'lib/pranet.py'
    res2net = 'res2net50_v1b_26w_4s-3cf99910.pth'   # Res2Net-50's ImageNet weights

    def __init__(self, root=None):
        super().__init__(root)
        self.weights = os.path.join(self.root or '', 'models', self.res2net)   # where its README puts them

    def build(self):
        pranet = _import_from(self.root, 'lib.pranet')
        # Its Res2Net_v1b.py loads '../models/<file>' relative to the working directory;
        # from <root>/models that is the file above, wherever training was started.
        here = os.getcwd()
        os.chdir(os.path.dirname(self.weights))
        try:
            return pranet.PraNet_V2(num_class=1)        # as in MyTrain_med.py
        finally:
            os.chdir(here)

    def logit(self, heads):
        return heads[0] + heads[1] + heads[2] + heads[3]   # MyTest_med.py: p2 + p3 + p4 + p5

    def sup_loss(self, heads, gts):
        # MyTrain_med.py: loss5 .. loss2 on the map pairs 2 .. 5, summed as loss2 + loss3 + loss4 + loss5
        fg, bg, mask_bg = heads[:4], heads[4:], 1 - gts
        loss = structure_loss_fgbg(fg[3], bg[3], gts, mask_bg)
        for k in (2, 1, 0):
            loss = loss + structure_loss_fgbg(fg[k], bg[k], gts, mask_bg)
        return loss

    def set_drop_path(self, model, rate):
        return None                                        # Res2Net has no stochastic depth


HOSTS = {h.name: h for h in (PolypPVTHost, CascadeHost, PraNetV2Host)}


def get_host(name='polyppvt', root=None):
    """The host called `name`; `root` is its clone (default ROOTS[name], beside this repo)."""
    if name not in HOSTS:
        raise ValueError('unknown host {!r}; choose from {}'.format(name, ', '.join(sorted(HOSTS))))
    return HOSTS[name](root if root is not None else ROOTS.get(name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build each host on the CPU and run one forward, '
                                                 'to check a setup before a long run.')
    parser.add_argument('--hosts', nargs='+', default=['cascade', 'pranetv2'], choices=sorted(HOSTS))
    parser.add_argument('--cascade_root', default=ROOTS['cascade'])
    parser.add_argument('--pranetv2_root', default=ROOTS['pranetv2'])
    opt = parser.parse_args()
    failed = False
    for name in opt.hosts:
        host = get_host(name, getattr(opt, name + '_root', None))
        try:
            host.preflight()
            model = host.build().eval()
            with torch.no_grad():
                heads = host.heads(model(torch.zeros(1, 3, 352, 352)))
                loss = host.sup_loss(heads, torch.zeros(1, 1, 352, 352))
                z = host.logit(heads)
            print('ok    {:<9s} {} maps of {}, prediction {}, loss {:.4f}'.format(
                name, len(heads), tuple(heads[0].shape), tuple(z.shape), float(loss)))
            print('      ' + host.describe())
        except Exception as e:                       # report every host, then fail
            failed = True
            print('FAIL  {:<9s} {}: {}'.format(name, type(e).__name__, e))
    sys.exit(1 if failed else 0)
