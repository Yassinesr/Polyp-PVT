"""The models FCT can train (--host): Polyp-PVT and hosts from other papers.

A host says how to build its model, which of its outputs make the prediction, and the
paper's own supervised loss for one view. Train_noaug.py, Test_tta.py and flip_gap.py go
through it instead of assuming Polyp-PVT's two heads, so FCT is the same code on every
host and the vanilla arm keeps each paper's loss.

  polyppvt   Polyp-PVT (this repo). Heads P1, P2; prediction P1 + P2; loss: the
             structure loss on each head.
  pranetv2   PraNet-V2 on Res2Net-50 (Hu et al., arXiv 2504.10986). Four foreground and
             four background maps; prediction: the sum of the foreground maps, as in its
             MyTest_med.py; loss: its foreground/background structure loss on each pair,
             as in its MyTrain_med.py.
  asgnet     ASGNet on PVTv2-b3 (Sun et al., IEEE TCSVT 2026). Four deep-supervision
             maps, the final map (the prediction, as in its Test.py) and four edge maps;
             loss: its structure, edge-Dice and uncertainty terms; its own data loader
             (with edges and augmentation) and learning-rate schedule, as in its
             Train_PS_R_PVTV3_edge1_new1.py.

No code from these papers is copied here: --host_root points at your own clone, and the
model (and ASGNet's data loader) is imported from it unchanged; setup_hosts.sh clones
each at the commit this was tested on.

  python hosts.py        # build each host and run one forward (on the GPU if there is one)
"""
import os
import sys
import argparse
import importlib
import subprocess

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_CLASH = ('lib', 'utils')            # package names the hosts share with this repo
ROOTS = {'pranetv2': '../PraNet-V2/binary_seg', 'asgnet': '../ASGNet/ASGNet_26_TCSVT'}
COMMITS = {'pranetv2': 'ca08522ae77cbda6c5056609e7951b502dbb08f2',
           'asgnet': '97c4e746e94bb6a184a14775b771226eed88c3e2'}


def _boundary_weight(mask):
    """The stock structure_loss weight: 1 in flat regions, up to 6 near a boundary."""
    return 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)


def structure_loss(pred, mask):
    """Unchanged from the stock repo; ASGNet's training script has the same one.

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


def asgnet_dice(predict, target):
    """ASGNet's edge loss, dice_loss in its training script (on the edge maps as they come)."""
    smooth = 1
    p = 2
    valid_mask = torch.ones_like(target)
    predict = predict.contiguous().view(predict.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    valid_mask = valid_mask.contiguous().view(valid_mask.shape[0], -1)
    num = torch.sum(torch.mul(predict, target) * valid_mask, dim=1) * 2 + smooth
    den = torch.sum((predict.pow(p) + target.pow(p)) * valid_mask, dim=1) + smooth
    loss = 1 - num / den
    return loss.mean()


def asgnet_ual(seg_logits):
    """ASGNet's uncertainty loss, cal_ual in its utils/utils.py: low where the
    prediction is confident either way."""
    sigmoid_x = seg_logits.sigmoid()
    loss_map = 1 - (2 * sigmoid_x - 1).abs().pow(2)
    return loss_map.mean()


def asgnet_ual_coef(progress):
    """ASGNet's get_coef(i / total_step, 'cos'): its weight on the uncertainty loss, rising
    from 0 to 1 over every epoch (i counts the batches of the epoch)."""
    return (1 - np.cos(progress * np.pi)) / 2


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
    n_extra = 0           # further targets sup_loss takes after the mask (ASGNet: its edges)
    loader_note = None    # what the host's own training loader does, when it has one
    lr_note = None        # how the learning rate moves, when the host has its own schedule
    progress = 0.0        # this batch's place in the epoch, set by Train_noaug.py

    def __init__(self, root=None):
        self.root = None if root is None else os.path.abspath(os.path.expanduser(root))

    def build(self):
        from lib.pvt import PolypPVT
        return PolypPVT()

    def train_loader(self, opt):
        """The paper's own training loader, or None for this repo's (Resize, ToTensor,
        Normalize; no augmentation)."""
        return None

    def lr_schedule(self, optimizer, opt):
        """A function of the epoch that sets the learning rate as the paper's script
        does, or None for the stock constant rate."""
        return None

    def heads(self, out):
        """The model's outputs as a list of logit maps, all at the input size."""
        return list(out)

    def logit(self, heads):
        """The prediction as a logit map: the sum of the heads, as at test time."""
        z = heads[0]
        for h in heads[1:]:
            z = z + h
        return z

    def sup_loss(self, heads, gts, *extra):
        """The paper's supervised loss for one view: the structure loss on each head."""
        loss = structure_loss(heads[0], gts)
        for h in heads[1:]:
            loss = loss + structure_loss(h, gts)
        return loss

    def set_drop_path(self, model, rate):
        """Stochastic depth of the PVTv2 backbone; returns the rate now in effect."""
        model.backbone.reset_drop_path(rate)
        return rate

    def preflight(self, opt=None):
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

    def sup_loss(self, heads, gts, *extra):
        # MyTrain_med.py: loss5 .. loss2 on the map pairs 2 .. 5, summed as loss2 + loss3 + loss4 + loss5
        fg, bg, mask_bg = heads[:4], heads[4:], 1 - gts
        loss = structure_loss_fgbg(fg[3], bg[3], gts, mask_bg)
        for k in (2, 1, 0):
            loss = loss + structure_loss_fgbg(fg[k], bg[k], gts, mask_bg)
        return loss

    def set_drop_path(self, model, rate):
        return None                                        # Res2Net has no stochastic depth


class ASGNetHost(PolypPVTHost):
    """ASGNet (PVTv2-b3): maps p0, f4, f3, f2 (deep supervision), f1 (the prediction) and
    edge maps e2, e3, e4, e1, trained with its own loss, data loader and schedule.

    Its training script runs one scale at 352, batch 40, Adam 1e-4, 80 epochs (the loop
    runs 79), in full precision: its FFT blocks cannot run in half precision at these
    feature sizes. Its backbone is built without stochastic depth.
    """
    name, paper = 'asgnet', 'ASGNet, PVTv2-b3 (Sun et al., IEEE TCSVT 2026)'
    weights = './pretrained_pth/pvt_v2_b3.pth'
    module = 'lib/Network_PVTV3_PS_edge_1_new.py'
    n_extra = 1
    loader_note = ("ASGNet's own loader (its utils/data_val.py): random horizontal flip, crop, "
                   "rotation, colour and label noise; edges from <train_path>/edges")
    lr_note = ('reset to --lr every epoch, then CosineAnnealingLR(T_max=20, eta_min=1e-7).step(), '
               'as in its training script')

    def build(self):
        net = _import_from(self.root, 'lib.Network_PVTV3_PS_edge_1_new')
        # Its Network.__init__ loads '/pvt_v2_b3.pth' from the filesystem root; read ours instead.
        load, weights = torch.load, os.path.abspath(self.weights)

        def redirected(f, *a, **k):
            return load(weights if f == '/pvt_v2_b3.pth' else f, *a, **k)
        torch.load = redirected
        try:
            return net.Network(channels=96)               # as in its training script
        finally:
            torch.load = load

    def logit(self, heads):
        return heads[4]                                   # Test.py: result[4]

    def sup_loss(self, heads, gts, edges):
        # Its train(): the terms in its order, the uncertainty weight from this batch's place in the epoch
        ual_loss = asgnet_ual(heads[4]) * asgnet_ual_coef(self.progress)
        loss_init = structure_loss(heads[0], gts) * 0.0625 + structure_loss(heads[1], gts) * 0.125 + \
            structure_loss(heads[2], gts) * 0.25 + structure_loss(heads[3], gts) * 0.5
        loss_final = structure_loss(heads[4], gts)
        loss_edge = asgnet_dice(heads[5], edges) * 0.0625 + asgnet_dice(heads[6], edges) * 0.125 + \
            asgnet_dice(heads[7], edges) * 0.25 + asgnet_dice(heads[8], edges) * 0.5
        return loss_init + loss_final + loss_edge + 2 * ual_loss

    def set_drop_path(self, model, rate):
        model.shared_encoder.reset_drop_path(rate)
        return rate

    def train_loader(self, opt):
        data = _import_from(self.root, 'utils.data_val')
        root = os.path.join(opt.train_path, '')
        return data.get_loader(image_root=root + 'images/', gt_root=root + 'masks/',
                               edge_root=root + 'edges/', batchsize=opt.batchsize,
                               trainsize=opt.trainsize, num_workers=8)

    def lr_schedule(self, optimizer, opt):
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=20, eta_min=1e-7)

        def at_epoch(epoch):
            for group in optimizer.param_groups:          # its adjust_lr(optimizer, lr, epoch, 0.1, 80)
                group['lr'] = 0.1 ** (epoch // 80) * opt.lr
            cosine.step()
        return at_epoch

    def preflight(self, opt=None):
        super().preflight(opt)
        if opt is not None:
            n_img = len(os.listdir(os.path.join(opt.train_path, 'images')))
            d = os.path.join(opt.train_path, 'edges')
            n_edge = len(os.listdir(d)) if os.path.isdir(d) else 0
            if n_edge != n_img:
                raise FileNotFoundError('{} has {} edge maps for {} images: run python make_edges.py '
                                        '(setup_hosts.sh does)'.format(d, n_edge, n_img))


HOSTS = {h.name: h for h in (PolypPVTHost, PraNetV2Host, ASGNetHost)}


def get_host(name='polyppvt', root=None):
    """The host called `name`; `root` is its clone (default ROOTS[name], beside this repo)."""
    if name not in HOSTS:
        raise ValueError('unknown host {!r}; choose from {}'.format(name, ', '.join(sorted(HOSTS))))
    return HOSTS[name](root if root is not None else ROOTS.get(name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build each host and run one forward (on the GPU '
                                                 'if there is one), to check a setup before a long run.')
    parser.add_argument('--hosts', nargs='+', default=['pranetv2', 'asgnet'], choices=sorted(HOSTS))
    for _name, _root in ROOTS.items():
        parser.add_argument('--{}_root'.format(_name), default=_root)
    opt = parser.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    failed = False
    for name in opt.hosts:
        host = get_host(name, getattr(opt, name + '_root', None))
        try:
            host.preflight()
            model = host.build().to(dev).eval()
            with torch.no_grad():
                heads = host.heads(model(torch.zeros(1, 3, 352, 352, device=dev)))
                gts = torch.zeros(1, 1, 352, 352, device=dev)
                loss = host.sup_loss(heads, gts, *[gts] * host.n_extra)
                z = host.logit(heads)
            print('ok    {:<9s} {} maps of {}, prediction {}, loss {:.4f}'.format(
                name, len(heads), tuple(heads[0].shape), tuple(z.shape), float(loss)))
            print('      ' + host.describe())
        except Exception as e:                       # report every host, then fail
            failed = True
            print('FAIL  {:<9s} {}: {}'.format(name, type(e).__name__, e))
    sys.exit(1 if failed else 0)
