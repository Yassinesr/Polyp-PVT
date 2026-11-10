"""Check a host's plumbing on the CPU before any GPU time.

Two parts, for SAM2-UNet:
  stand-in   hosts.py's glue against a stand-in clone with SAM2-UNet's interface (a toy
             three-map model, a FullDataset that flips): the prediction, the loss, the
             build-device redirect, the loader wrapper and its name check, the schedule
             against torch's CosineAnnealingLR, and FCT's step on a model whose prediction
             is its first map, against an independent implementation of Eqs. (7)-(11).
             Runs no third-party code.
  real       the same on the real model from your clone (random weights; Hiera-L, so
             about a minute), plus its frozen encoder, its map sizes at the paper's two
             scales, its loss against a transcription of its train.py, and its loader's
             flips (image and mask together, all four flip states).

Usage: python test_hosts.py                     # both (run after setup_hosts.sh sam2unet)
       python test_hosts.py --stand_in_only     # without the clone
       [--sam2unet_root ../SAM2-UNet]
Exits 1 on any mismatch.
"""
import argparse
import os
import sys
import tempfile
import textwrap
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import Train_noaug as T                                   # noqa: E402
from hosts import get_host, ROOTS, _import_from           # noqa: E402

OK = True


def check(name, good, detail=''):
    global OK
    OK &= bool(good)
    print('{:<4s} {}{}'.format('ok' if good else 'FAIL', name, ('   ' + detail) if detail else ''))


def rel(g1, g2):
    num = sum(((u - v) ** 2).sum() for u, v in zip(g1, g2)) ** 0.5
    den = sum((v ** 2).sum() for v in g2) ** 0.5
    return float(num / den)


def sam2unet_structure_loss(pred, mask):
    """Transcribed from SAM2-UNet's train.py (its reduce='none' included)."""
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def fct_reference(model, x, gts, lam, prediction, sup, a=2.0):
    """Eqs. (7)-(11) written from the paper (lag-or-margin flip pulls and the upright pull),
    for a host whose prediction is `prediction(heads)` and whose loss is `sup(heads, gts)`.
    Returns the gradients of the trainable parameters."""
    model.zero_grad()
    H, W = x.shape[2:]
    y = (gts > 0.5).float()
    bce = lambda z, q: -(q * F.logsigmoid(z) + (1 - q) * F.logsigmoid(-z))
    per_image_mean = lambda m: (m.sum(dim=(2, 3)) / (H * W)).mean()
    views = {'h': lambda t: torch.flip(t, [3]), 'v': lambda t: torch.flip(t, [2])}
    ho = list(model(x))
    zo = prediction(ho)
    po = torch.sigmoid(zo).detach()
    eo = (po - y).abs()
    hk = {k: [f(h) for h in model(f(x))] for k, f in views.items()}
    L = (sup(ho, gts) + sum(sup(hk[k], gts) for k in views)) / 3.0                       # Eq. (7)
    for k in views:
        zk = prediction(hk[k])
        ek = (torch.sigmoid(zk).detach() - y).abs()
        w = torch.where(ek >= eo, torch.ones_like(eo), (1 - a * eo).clamp(min=0))          # Eq. (8)
        L = L + lam / 2 * per_image_mean(w * bce(zk, po))                                 # Eq. (9)
    with torch.no_grad():
        pk = {k: torch.sigmoid(prediction([f(h) for h in model(f(x))])) for k, f in views.items()}
    ek = {k: (pk[k] - y).abs() for k in views}
    best_is_h = ek['h'] <= ek['v']
    pstar = torch.where(best_is_h, pk['h'], pk['v'])
    estar = torch.where(best_is_h, ek['h'], ek['v'])
    L = L + lam / 2 * per_image_mean((estar < eo).float() * bce(zo, pstar))               # Eq. (10)
    L.backward()
    return [p.grad.clone() for p in model.parameters() if p.requires_grad]


def fct_step(model, x, gts, lam, o):
    """Train_noaug.py's own step, as in its train loop."""
    model.zero_grad()
    flip_probs = T._flip_probs(model, x, o, None) if o.fct_up else None
    heads = T.HOST.heads(model(x))
    loss = T.HOST.sup_loss(heads, gts)

    class Rec:
        def update(self, *a):
            pass
    T._fct_v2_step(model, x, gts, heads, loss, lam, None, o, {'iter': 1}, Rec(),
                   lambda t, retain=False: t.backward(retain_graph=retain), True, flip_probs=flip_probs)
    return [p.grad.clone() for p in model.parameters() if p.requires_grad]


def write_pairs(root, names, size=(80, 60), mask_names=None):
    """Random images and masks with an off-centre box, so every flip is distinguishable."""
    rng = np.random.RandomState(0)
    os.makedirs(os.path.join(root, 'images'), exist_ok=True)
    os.makedirs(os.path.join(root, 'masks'), exist_ok=True)
    for i, n in enumerate(names):
        Image.fromarray(rng.randint(0, 256, (size[1], size[0], 3), dtype=np.uint8)).save(
            os.path.join(root, 'images', n + '.png'))
        m = np.zeros((size[1], size[0]), np.uint8)
        m[5 + i:25 + i, 8:40 + 2 * i] = 255
        Image.fromarray(m).save(os.path.join(root, 'masks', (mask_names or names)[i] + '.png'))


STAND_IN = {
    'SAM2UNet.py': '''
        import torch, torch.nn as nn
        CALLS = []                               # how build_sam2 was called

        def build_sam2(config_file, ckpt_path=None, device="cuda", mode="eval"):
            CALLS.append((config_file, ckpt_path, device))

        class SAM2UNet(nn.Module):               # SAM2-UNet's interface: three maps, a frozen encoder
            def __init__(self, checkpoint_path=None):
                super().__init__()
                build_sam2("sam2_hiera_l.yaml", checkpoint_path)
                self.encoder = nn.Conv2d(3, 8, 3, padding=1)
                for p in self.encoder.parameters():
                    p.requires_grad = False
                self.prompt_learn = nn.Conv2d(8, 8, 1)
                self.bn = nn.BatchNorm2d(8)
                self.head, self.side1, self.side2 = nn.Conv2d(8, 1, 3, padding=1), nn.Conv2d(8, 1, 5, padding=2), nn.Conv2d(8, 1, 1)

            def forward(self, x):
                h = torch.tanh(self.bn(self.prompt_learn(self.encoder(x))))
                return self.head(h), self.side1(h), self.side2(h)
        ''',
    'sam2_configs/__init__.py': '',
    'dataset.py': '''
        import os, numpy as np, torch
        from PIL import Image

        class FullDataset(torch.utils.data.Dataset):   # its interface: image_root + f, dict items
            def __init__(self, image_root, gt_root, size, mode):
                self.images = sorted(image_root + f for f in os.listdir(image_root) if f.endswith('.png'))
                self.gts = sorted(gt_root + f for f in os.listdir(gt_root) if f.endswith('.png'))
                self.size = size

            def __len__(self):
                return len(self.images)

            def __getitem__(self, i):
                im = Image.open(self.images[i]).convert('RGB').resize((self.size, self.size))
                gt = Image.open(self.gts[i]).convert('L').resize((self.size, self.size))
                return {'image': torch.from_numpy(np.asarray(im, np.float32) / 255).permute(2, 0, 1),
                        'label': torch.from_numpy(np.asarray(gt, np.float32) / 255)[None]}
        ''',
}


def make_stand_in(root):
    for name, text in STAND_IN.items():
        path = os.path.join(root, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            f.write(textwrap.dedent(text))


def test_sam2unet(root, real):
    print('== sam2unet, {} at {}'.format('the real clone' if real else 'a stand-in', root))
    host = get_host('sam2unet', root)
    T.HOST = host
    torch.manual_seed(0)
    model = host.build().cpu().train()
    mod = _import_from(host.root, 'SAM2UNet')
    check('builds from the clone', True, host.describe() if real else '')
    if not real:
        check('build_sam2 is asked for a device that exists, then restored',
              mod.CALLS[-1][2] == ('cuda' if torch.cuda.is_available() else 'cpu')
              and 'partial' not in type(mod.build_sam2).__name__, str(mod.CALLS[-1]))
        check('no SAM2 weights at {} -> encoder built without them'.format(host.weights),
              os.path.isfile(host.weights) or mod.CALLS[-1][1] is None)
    check('no stochastic depth to synchronise', host.set_drop_path(model, 0.1) is None)

    if real:
        # Only the adapters and the decoder train; the Hiera-L encoder stays frozen.
        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        n_enc = sum(p.numel() for n, p in model.named_parameters() if n.startswith('encoder.') and not p.requires_grad)
        check('encoder frozen except its adapters',
              all('prompt_learn' in n for n in trainable if n.startswith('encoder.')) and n_enc > 1e8,
              '{:.1f}M frozen, {:.2f}M trainable'.format(
                  n_enc / 1e6, sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6))
        # Map sizes at the paper's two scales: 352 and round(352 * 1.25 / 32) * 32 = 448
        with torch.no_grad():
            for s in (352, 448):
                heads = host.heads(model(torch.randn(1, 3, s, s)))
                check('three maps at the input size {}'.format(s),
                      len(heads) == 3 and all(tuple(h.shape) == (1, 1, s, s) for h in heads))

    # The loss and the prediction
    x = torch.randn(2, 3, 64, 64)
    yy, xx = torch.meshgrid(torch.arange(64), torch.arange(64), indexing='ij')
    gts = torch.stack([((yy - 20 - 8 * i) ** 2 + (xx - 38 + 6 * i) ** 2 < 150).float()
                       for i in range(2)]).unsqueeze(1)
    with torch.no_grad():
        heads = host.heads(model(x))
        ours = host.sup_loss(heads, gts)
        theirs = sum(sam2unet_structure_loss(h, gts) for h in heads)      # loss0 + loss1 + loss2
    check("loss = its train.py's loss0 + loss1 + loss2", torch.allclose(ours, theirs, rtol=1e-6),
          '{:.6f} vs {:.6f}'.format(float(ours), float(theirs)))
    check("prediction = its test.py's first output", torch.equal(host.logit(heads), heads[0]))

    # FCT's step on this host = Eqs. (7)-(11) with SAM2-UNet's prediction and loss
    o = argparse.Namespace(fct_vflip=1, fct_sub=0, fct_sup=1, fct_weit=8, fct_up=1, amp=0, fct_lm_a=2.0,
                           fct_loss='bce', grad_diag=0, fct_flip_pull=1, batchsize=2)
    lam = 0.5
    g_step = fct_step(model, x, gts, lam, o)
    g_ref = fct_reference(model, x, gts, lam, lambda hs: hs[0],
                          lambda hs, y: sum(sam2unet_structure_loss(h, y) for h in hs))
    r = rel(g_step, g_ref)
    check('FCT step = Eqs. (7)-(11), prediction = first map', r < 1e-4,
          'relative gradient difference {:.2e}'.format(r))
    r = rel(g_step, fct_step(model, x, gts, 0.0, o))
    check('the consistency terms move the gradient', r > 1e-3, 'FCT vs lambda = 0: {:.2e}'.format(r))

    # Its schedule: CosineAnnealingLR(T_max=20, eta_min=1e-7) stepped after each of 20 epochs
    p = torch.nn.Parameter(torch.zeros(1))
    o1 = torch.optim.AdamW([p], lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(o1, 20, eta_min=1.0e-7)
    ref = []
    for _ in range(20):
        ref.append(o1.param_groups[0]['lr'])
        o1.step()
        sched.step()
    o2 = torch.optim.AdamW([p], lr=1e-3)
    at_epoch = host.lr_schedule(o2, argparse.Namespace(lr=1e-3, epoch=21))   # the loop runs epochs 1..20
    got = []
    for e in range(1, 21):
        at_epoch(e)
        got.append(o2.param_groups[0]['lr'])
    d = max(abs(a - b) for a, b in zip(got, ref))
    check('cosine schedule = its CosineAnnealingLR, epochs 1..20', d < 1e-12,
          'max difference {:.1e}; {:.2e} -> {:.2e}'.format(d, got[0], got[-1]))

    # Its loader through the wrapper: (image, mask) pairs; the real one's flips; the name check
    with tempfile.TemporaryDirectory() as tmp:
        names = ['a', 'b', 'c', 'd']
        write_pairs(tmp, names)
        loader = host.train_loader(argparse.Namespace(train_path=tmp, batchsize=4, trainsize=64))
        check("its dataset.py does not take this repo's module name", 'dataset' not in sys.modules)
        data = _import_from(host.root, 'dataset')
        plain = data.FullDataset(tmp + '/images/', tmp + '/masks/', 64, mode='test')   # no flips
        ref = [(plain[i]['image'], plain[i]['label']) for i in range(len(names))]
        flips = {'none': lambda t: t, 'h': lambda t: torch.flip(t, [2]), 'v': lambda t: torch.flip(t, [1]),
                 'hv': lambda t: torch.flip(t, [1, 2])}
        seen, paired, shapes_ok = set(), True, True
        torch.manual_seed(0)
        for images, masks in (b for _ in range(10) for b in loader):     # 10 epochs of one batch
            shapes_ok &= tuple(images.shape) == (4, 3, 64, 64) and tuple(masks.shape) == (4, 1, 64, 64)
            for im, m in zip(images, masks):
                hit = [(i, k) for i, (ri, rm) in enumerate(ref) for k, f in flips.items()
                       if torch.equal(f(ri), im) and torch.equal(f(rm), m)]
                paired &= len(hit) >= 1
                if hit:
                    seen.add(hit[0][1])
        check('loader gives (image, mask) batches at --trainsize', shapes_ok)
        check('each image comes with its own mask, flipped the same way', paired)
        if real:
            check('all four flip states occur (random h and v flips)', seen == set(flips), str(sorted(seen)))
    with tempfile.TemporaryDirectory() as tmp:
        write_pairs(tmp, ['a', 'b'], mask_names=['a', 'z'])
        try:
            host.train_loader(argparse.Namespace(train_path=tmp, batchsize=2, trainsize=64))
            check('mismatched image and mask names are refused', False)
        except FileNotFoundError:
            check('mismatched image and mask names are refused', True)
    for k in ('SAM2UNet', 'sam2_configs'):          # so a second clone in this process imports afresh
        sys.modules.pop(k, None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--sam2unet_root', default=ROOTS['sam2unet'])
    parser.add_argument('--stand_in_only', action='store_true', help='skip the real clone')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as stand_in:
        make_stand_in(stand_in)
        test_sam2unet(stand_in, real=False)
    if not args.stand_in_only:
        if os.path.isfile(os.path.join(args.sam2unet_root, 'SAM2UNet.py')):
            test_sam2unet(args.sam2unet_root, real=True)
        else:
            check('SAM2-UNet clone at {}'.format(args.sam2unet_root), False,
                  'not found: bash setup_hosts.sh sam2unet, or --sam2unet_root, or --stand_in_only')
    print('ALL OK' if OK else 'MISMATCH')
    sys.exit(0 if OK else 1)
