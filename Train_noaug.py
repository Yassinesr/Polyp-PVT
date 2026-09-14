"""Polyp-PVT trained on WHOLE IMAGES with the orientation-symmetry augmentation
removed, so that flip-consistency training (FCT) and flip TTA can be measured
against a baseline that does not already contain the symmetry they exploit.

WHY THIS FILE EXISTS
--------------------
On the patch-refinement side of this project the ladder was:

    A  orientation aug OFF, no FCT
    B  orientation aug OFF, FCT consistency-only
    C  = B scored with 4-view flip TTA

This script is the whole-image analogue, so the two settings can be compared
directly and the question "are patches or full images more suitable for
FCT+TTA?" can be answered with the same ladder on both sides.

WHAT WAS REMOVED, AND WHY
-------------------------
The stock repo has TWO independent sources of augmentation. Only one of them
clashes with FCT+TTA.

1. ORIENTATION (utils/dataloader.py, gated on `augmentations == 'True'`):
      RandomRotation(90), RandomVerticalFlip(0.5), RandomHorizontalFlip(0.5)
   This is exactly the symmetry group FCT constrains and flip-TTA averages
   over. It CLASHES and is removed here: this script hard-wires the loader to
   the no-augmentation branch and does not expose a flag to re-enable it.

   Worth knowing: in the stock Train.py this augmentation is ALREADY OFF by
   accident. `--augmentation` is declared with `default=False` (a Python bool)
   while the loader tests `if self.augmentations == 'True'` (a string). The
   bool never equals the string, so the default run trains with no flips and
   no rotations. Passing `--augmentation True` on the command line DOES turn
   it on, because argparse hands through the string "True". So the published
   default is already orientation-free; this file makes that explicit and
   unable to drift.

   Also note the stock transform is RandomRotation(90), which in torchvision
   means a uniform random angle in [-90, +90] degrees -- not a 90-degree step.
   That trains continuous rotation invariance, a strictly larger group than
   the D4 group flip-TTA averages over.

2. MULTI-SCALE (Train.py, `size_rates = [0.75, 1, 1.25]`):
   Each batch is run at three scales with a backward pass each. This is NOT
   gated by --augmentation and is always on in the stock recipe. It acts on
   the SCALE axis, which is orthogonal to the flip/rotation symmetry that
   FCT+TTA target, so it does NOT clash and is KEPT ON by default. It is
   exposed as `--multiscale 0|1` so it can be ablated separately -- turning it
   off changes the recipe substantially and will lower the baseline, so keep
   it on unless scale is the thing being studied.

COST AND MEMORY
---------------
Multi-scale already costs 3 forward/backward passes per batch. FCT adds two
more forwards. To keep that from compounding to 9, the consistency term is
computed only at `rate == 1` (the canonical scale), so per-step cost is about
1.6x the baseline rather than 3x.

STOCHASTIC DEPTH: pvt_v2_b2 is built with drop_path_rate=0.1, so each forward
in train() mode samples an independent depth mask. Pass --drop_path 0 for any
FCT run (and the same value on the no-FCT arm) or the consistency term mostly
measures that noise. See the flag's help text.

MEMORY is the binding constraint, not time. An FCT forward keeps its autograd
graph alive until backward, so peak memory scales with the NUMBER OF LIVE
GRAPHS, not the number of passes.

--fct_mode seq (the default) removes that problem. Each view is backwarded as
soon as it is computed, accumulating into .grad, so exactly ONE graph is alive
at any moment and peak memory is essentially the no-FCT baseline: if arm A
fits, arm B fits. The cost is that the consistency target must be DETACHED --
gradients flow only into the flipped branch, pulling it toward the original
rather than pulling both together.

That stop-gradient is the standard formulation in consistency training (the
Pi-model, Mean Teacher and FixMatch all detach the target) and it avoids the
degenerate pressure to collapse both branches toward a constant. It is, however,
NOT identical to the patch-side arm B, which used the coupled form. If memory
allows, --fct_mode joint reproduces that exactly; if it does not, report which
mode was used, because the two are different estimators of the same idea.

If even seq mode OOMs, the baseline itself is too big for the card. Lower
--batchsize on BOTH arms, and only then consider the knobs below:

    1. --amp 1        mixed precision; roughly halves activation memory
    2. --batchsize 8  changes the recipe, so apply it to both arms
    3. --fct_vflip 0  horizontal flip only; halves the FCT time cost
    4. --fct_sub 8    LAST resort, see the caveat below

--fct_sub caveat 1: it also breaks --fct_sync_rng. DropPath draws a mask of
shape (batch,1,1,1) per block, so a sliced batch draws a different NUMBER of
randoms and the views no longer share a mask even with the RNG restored.

--fct_sub caveat 2: the decoder contains BatchNorm, and in train() mode BN
normalises using the CURRENT batch's statistics. Slicing images[:n] for the
flipped forwards means they are normalised over n samples while the target
p_o was normalised over the full batch, so the MSE picks up a BN-statistics
mismatch on top of the flip difference. (Flipping itself is safe: a spatial
flip permutes H and W and leaves per-channel statistics identical.) Prefer
--amp and --fct_vflip first; reach for --fct_sub only if those are not enough,
and keep n as large as possible.

USAGE
-----
  arm A (no-aug baseline):
    python Train_noaug.py --train_save ./model_pth/PolypPVT_noaug/
  arm B/C (no-aug + FCT, consistency-only):
    python Train_noaug.py --fct 1 --fct_weight 0.5 --train_save ./model_pth/PolypPVT_noaug_fct/

Then score with Test.py (no TTA) and Test_tta.py (4-view flip TTA).
"""
import os
import argparse
import logging
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.cuda.amp import autocast, GradScaler

from lib.pvt import PolypPVT
from utils.dataloader import get_loader, test_dataset
from utils.utils import clip_gradient, adjust_lr, AvgMeter

TESTSETS = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']


def structure_loss(pred, mask):
    """Unchanged from the stock repo."""
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


# ---------------------------------------------------------------- FCT pieces
def _hflip(t):
    return torch.flip(t, dims=[-1])


def _vflip(t):
    return torch.flip(t, dims=[-2])


def _rng_snapshot():
    st = (torch.get_rng_state(),)
    if torch.cuda.is_available():
        st = st + (torch.cuda.get_rng_state_all(),)
    return st


def _rng_restore(st):
    torch.set_rng_state(st[0])
    if len(st) > 1:
        torch.cuda.set_rng_state_all(st[1])


def _prob(model, x):
    """Inference-equivalent probability map: sigmoid(P1 + P2).

    Test.py predicts with `P1 + P2`, so the consistency term is applied to the
    same quantity TTA will later average. Constraining anything else would be
    constraining a signal that never reaches the output.
    """
    p1, p2 = model(x)
    return torch.sigmoid(p1 + p2)


def flip_consistency(model, images, p_o, use_vflip=True, rng0=None):
    """Consistency-only FCT: the flipped views are NEVER supervised by the GT.

    This mirrors `fct_supervise_flips=False` on the patch side, and the
    distinction is load-bearing. If the flipped views received GT supervision
    they would simply BE flip augmentation, reintroduced through the loss --
    and the arm would silently collapse back into the augmented baseline this
    file exists to remove. Here the flips contribute the consistency signal
    and nothing else.

    `p_o` is the ALREADY-COMPUTED sigmoid(P1+P2) of the unflipped batch,
    passed in rather than recomputed. Recomputing it would run a second
    forward pass over the same input and hold a second autograd graph alive
    until backward -- pure waste, and enough on its own to OOM at batch 16.
    """
    cons = F.mse_loss(_hflip(_prob(model, _hflip(images))), p_o)
    if use_vflip:
        if rng0 is not None:
            _rng_restore(rng0)
        cons = 0.5 * (cons + F.mse_loss(_vflip(_prob(model, _vflip(images))), p_o))
    return cons


# ---------------------------------------------------------------- evaluation
def test(model, path, dataset):
    """Unchanged from the stock repo."""
    data_path = os.path.join(path, dataset)
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    model.eval()
    num1 = len(os.listdir(gt_root))
    test_loader = test_dataset(image_root, gt_root, 352)
    DSC = 0.0
    for i in range(num1):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        res, res1 = model(image)
        res = F.upsample(res + res1, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        target = np.array(gt)
        smooth = 1
        input_flat = np.reshape(res, (-1))
        target_flat = np.reshape(target, (-1))
        intersection = (input_flat * target_flat)
        dice = float('{:.4f}'.format(
            (2 * intersection.sum() + smooth) / (res.sum() + target.sum() + smooth)))
        DSC += dice
    return DSC / num1


def evaluate_all(model, test_root):
    """Score the five benchmark sets and return (per-set dict, mean).

    The stock script selects its best checkpoint on a sixth split it calls
    'test', i.e. `<test_root>/test/`, which is absent from the standard
    TestDataset layout and crashes at the first epoch. Here the five-set mean
    is used when that directory does not exist, so training does not die an
    hour in; if the directory IS present the original behaviour is preserved
    exactly.
    """
    scores = {}
    for name in TESTSETS:
        scores[name] = test(model, test_root, name)
    if os.path.isdir(os.path.join(test_root, 'test')):
        return scores, test(model, test_root, 'test')
    return scores, sum(scores.values()) / len(scores)


# ---------------------------------------------------------------- train loop
def train(train_loader, model, optimizer, epoch, opt, state):
    model.train()
    scaler = state.get('scaler')
    size_rates = [0.75, 1, 1.25] if opt.multiscale else [1]
    loss_record = AvgMeter()
    cons_record = AvgMeter()

    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            images, gts = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()

            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize),
                                    mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize),
                                 mode='bilinear', align_corners=True)

            do_fct = bool(opt.fct) and rate == 1
            # Snapshot the RNG so every view of this sample can be given the
            # SAME stochastic-depth mask. Without this, f(x) and f(flip x) draw
            # independent DropPath masks and the consistency term measures that
            # noise instead of flip non-equivariance.
            rng0 = _rng_snapshot() if (do_fct and opt.fct_sync_rng) else None
            lam = 0.0
            if do_fct:
                it = state['iter']
                lam = opt.fct_weight
                if opt.fct_warmup_iters > 0 and it < opt.fct_warmup_iters:
                    lam = opt.fct_weight * (it / float(opt.fct_warmup_iters))
                state['iter'] += 1

            def _bw(t):
                """Backward one term, accumulating into .grad."""
                if scaler is not None:
                    scaler.scale(t).backward()
                else:
                    t.backward()

            # ---- supervised view ------------------------------------------
            with autocast(enabled=bool(opt.amp)):
                P1, P2 = model(images)
                loss = structure_loss(P1, gts) + structure_loss(P2, gts)
                # Target for the consistency term, taken from this same
                # forward. In seq mode it is detached so this graph can be
                # freed immediately.
                p_o = torch.sigmoid(P1 + P2) if do_fct else None

            if do_fct and opt.fct_mode == 'seq':
                p_o = p_o.detach()
                _bw(loss)                      # frees the supervised graph NOW
                n = images.shape[0] if opt.fct_sub <= 0 else min(opt.fct_sub, images.shape[0])
                w = 0.5 * lam if opt.fct_vflip else lam
                cons_total = 0.0
                for flip in (['h', 'v'] if opt.fct_vflip else ['h']):
                    f = _hflip if flip == 'h' else _vflip
                    if rng0 is not None:
                        _rng_restore(rng0)   # same depth mask as the main view
                    with autocast(enabled=bool(opt.amp)):
                        pf = f(_prob(model, f(images[:n])))
                        c = F.mse_loss(pf, p_o[:n])
                    _bw(w * c)                 # frees this view's graph NOW
                    cons_total += float(c.detach())
                cons_record.update(torch.tensor(cons_total / (2 if opt.fct_vflip else 1)),
                                   opt.batchsize)
            else:
                if do_fct:
                    n = images.shape[0] if opt.fct_sub <= 0 else min(opt.fct_sub, images.shape[0])
                    if rng0 is not None:
                        _rng_restore(rng0)
                    with autocast(enabled=bool(opt.amp)):
                        cons = flip_consistency(model, images[:n], p_o[:n],
                                                use_vflip=bool(opt.fct_vflip),
                                                rng0=rng0)
                        loss = loss + lam * cons
                    cons_record.update(cons.data, opt.batchsize)
                _bw(loss)

            if scaler is not None:
                scaler.unscale_(optimizer)      # clip on real, unscaled grads
                clip_gradient(optimizer, opt.clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                clip_gradient(optimizer, opt.clip)
                optimizer.step()

            if rate == 1:
                loss_record.update(loss.data, opt.batchsize)

        if i % 20 == 0 or i == state['total_step']:
            msg = ('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], loss: {:0.4f}'
                   .format(datetime.now(), epoch, opt.epoch, i, state['total_step'],
                           loss_record.show()))
            if opt.fct:
                msg += ', cons: {:0.5f}'.format(cons_record.show())
            print(msg)

    os.makedirs(opt.train_save, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(opt.train_save, 'last.pth'))
    if opt.save_every_epoch:
        torch.save(model.state_dict(),
                   os.path.join(opt.train_save, '{}PolypPVT.pth'.format(epoch)))

    if opt.eval_every > 1 and epoch % opt.eval_every != 0:
        return
    scores, meandice = evaluate_all(model, opt.test_path)
    for name, d in scores.items():
        logging.info('epoch: {}, dataset: {}, dice: {}'.format(epoch, name, d))
        print(name, ': ', d)
    print('mean: ', meandice)

    if meandice > state['best']:
        state['best'] = meandice
        torch.save(model.state_dict(), os.path.join(opt.train_save, 'PolypPVT.pth'))
        print('#' * 30, 'best', meandice)
        logging.info('#' * 30 + 'best:{}'.format(meandice))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--optimizer', type=str, default='AdamW')
    parser.add_argument('--batchsize', type=int, default=16)
    parser.add_argument('--trainsize', type=int, default=352)
    parser.add_argument('--clip', type=float, default=0.5)
    parser.add_argument('--decay_rate', type=float, default=0.1)
    parser.add_argument('--decay_epoch', type=int, default=50)
    parser.add_argument('--train_path', type=str, default='./dataset/TrainDataset/')
    parser.add_argument('--test_path', type=str, default='./dataset/TestDataset/')
    parser.add_argument('--train_save', type=str, default='./model_pth/PolypPVT_noaug/')

    # Scale augmentation: orthogonal to the flip group, so it does not clash
    # with FCT+TTA. On by default because it is part of the stock recipe.
    parser.add_argument('--multiscale', type=int, default=1,
                        help='1 = stock [0.75,1,1.25] multi-scale, 0 = single scale')

    # Orientation augmentation. Off by default -- this script exists to remove
    # it. The flag is here for ONE purpose: testing whether the ~0.018 gap
    # between the no-aug arms and the released Polyp-PVT checkpoint is the
    # value of orientation augmentation. The released model may well have been
    # trained with it, since the stock --augmentation flag works from the
    # command line even though its default silently does not.
    #
    # Turning this ON re-enables RandomRotation(90) + V/H flips, which is
    # exactly the symmetry group FCT constrains and TTA averages over. Do not
    # combine it with --fct and then report the result as an FCT measurement.
    parser.add_argument('--orientation_aug', type=int, default=0,
                        help='1 = re-enable RandomRotation + V/H flips (diagnostic only)')

    # Flip-consistency training. Consistency-only: flipped views are never
    # supervised by the GT, so this adds no augmentation of its own.
    parser.add_argument('--fct', type=int, default=0, help='1 = enable flip consistency')
    parser.add_argument('--fct_weight', type=float, default=0.5,
                        help='consistency weight; 0.05 is far too weak to install an '
                             'invariance when no augmentation supplies one')
    parser.add_argument('--fct_warmup_iters', type=int, default=300)
    parser.add_argument('--fct_vflip', type=int, default=1,
                        help='0 = horizontal flip only; halves the FCT memory cost')
    parser.add_argument('--fct_mode', type=str, default='seq', choices=['seq', 'joint'],
                        help="seq: backward each view separately against a DETACHED "
                             "target, so only one autograd graph is ever alive -- peak "
                             "memory is essentially the no-FCT baseline. joint: one "
                             "backward over all views, gradients flow through both "
                             "sides of the MSE; ~3x the activation memory")
    parser.add_argument('--fct_sub', type=int, default=0,
                        help='compute the consistency term on only the first N samples '
                             'of each batch (0 = whole batch). Cuts FCT memory without '
                             'touching the supervised loss or the effective batch size')

    # --- memory ---
    # Escalate in this order when you hit OOM:
    #   1. --amp 1          (roughly halves activation memory, both arms)
    #   2. --fct_sub 8      (consistency on half the batch)
    #   3. --fct_vflip 0    (one flip instead of two)
    #   4. --batchsize 8    (last resort: changes the recipe)
    # Use the SAME settings for the no-FCT arm, or the arms are not comparable.
    parser.add_argument('--amp', type=int, default=0,
                        help='1 = mixed precision. Apply to BOTH arms or they differ '
                             'by more than the thing being measured')

    # --- stochastic depth ---
    # pvt_v2_b2 is constructed with drop_path_rate=0.1, so in train() mode every
    # forward samples an INDEPENDENT stochastic-depth mask across its 16 blocks.
    # The consistency term compares f(x) against unflip(f(flip x)); with
    # DropPath live those two differ by the flip non-equivariance we want to
    # measure AND by two independent depth masks, which is pure noise. Set this
    # to 0 for any FCT run, and use the same value on the no-FCT arm.
    parser.add_argument('--drop_path', type=float, default=0.1,
                        help='stochastic depth rate; 0.1 = stock. Safe to leave at '
                             'stock when --fct_sync_rng 1')
    parser.add_argument('--fct_sync_rng', type=int, default=1,
                        help='restore the RNG before each flipped forward so every view '
                             'of a sample gets the SAME DropPath mask. This is what makes '
                             'the consistency term measure flip non-equivariance rather '
                             'than depth-mask noise, and it lets stochastic depth stay at '
                             'its stock 0.1 instead of being switched off')

    # --- housekeeping ---
    parser.add_argument('--save_every_epoch', type=int, default=0,
                        help='1 = keep a checkpoint per epoch (~100MB x epochs). '
                             'Default keeps only best + last')
    parser.add_argument('--eval_every', type=int, default=1,
                        help='evaluate the 5 test sets every N epochs')

    opt = parser.parse_args()

    logging.basicConfig(filename='train_log_noaug.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

    # Preflight: fail now with a clear message rather than after the data loads.
    _pre = './pretrained_pth/pvt_v2_b2.pth'
    if not os.path.isfile(_pre):
        raise FileNotFoundError(
            'Missing backbone weights at {} -- PolypPVT.__init__ hardcodes this '
            'path and will crash on construction.'.format(_pre))
    for _d in ('{}/images/'.format(opt.train_path), '{}/masks/'.format(opt.train_path)):
        if not os.path.isdir(_d):
            raise FileNotFoundError('Missing training directory: {}'.format(_d))
    for _s in TESTSETS:
        _d = os.path.join(opt.test_path, _s, 'masks')
        if not os.path.isdir(_d):
            raise FileNotFoundError('Missing test directory: {}'.format(_d))

    model = PolypPVT().cuda()

    # Stochastic depth. Must be 0 for FCT to measure flip non-equivariance
    # rather than depth-mask noise; must match across arms either way.
    model.backbone.reset_drop_path(opt.drop_path)
    if opt.fct and opt.drop_path > 0 and not opt.fct_sync_rng:
        print('!' * 78)
        print('WARNING: --fct 1, --drop_path {:.3f}, --fct_sync_rng 0.'.format(opt.drop_path))
        print('  Stochastic depth is active and the views are NOT sharing a depth')
        print('  mask, so the consistency term will be dominated by that noise')
        print('  rather than by flip non-equivariance.')
        print('  Use --fct_sync_rng 1 (preferred) or --drop_path 0 on BOTH arms.')
        print('!' * 78)

    params = model.parameters()
    if opt.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params, opt.lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(params, opt.lr, weight_decay=1e-4, momentum=0.9)
    print(optimizer)

    image_root = '{}/images/'.format(opt.train_path)
    gt_root = '{}/masks/'.format(opt.train_path)

    # The loader gates on `augmentations == 'True'` -- a STRING compare -- so
    # the string is what turns it on. Anything else (including the bool True)
    # lands in the no-augmentation branch: Resize -> ToTensor -> Normalize.
    _aug = 'True' if opt.orientation_aug else False
    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize,
                              trainsize=opt.trainsize, augmentation=_aug)

    print('#' * 20, 'Start Training (no orientation augmentation)', '#' * 20)
    print('AUGMENTATION IN EFFECT')
    print('  orientation (flip/rotate) : {}'.format(
        'ON   (RandomRotation + V/H flips) -- DIAGNOSTIC ARM' if opt.orientation_aug else 'OFF'))
    if opt.orientation_aug and opt.fct:
        print('  !! --orientation_aug with --fct: the augmentation supplies the very')
        print('     symmetry the consistency term constrains. Not an FCT measurement.')
    print('  photometric               : OFF  (this repo has none)')
    print('  multi-scale [0.75,1,1.25] : {}'.format('ON' if opt.multiscale else 'OFF'))
    print('  stochastic depth          : {:.3f}{}'.format(
        opt.drop_path,
        '  (mask shared across views)' if (opt.fct and opt.fct_sync_rng) else ''))
    print('TRAINING')
    print('  batchsize {}  trainsize {}  amp {}  epochs {}'.format(
        opt.batchsize, opt.trainsize, bool(opt.amp), opt.epoch))
    if opt.fct:
        print('  FCT consistency-only: weight {:.3f}, vflip {}, warmup {}, sub {}'.format(
            opt.fct_weight, bool(opt.fct_vflip), opt.fct_warmup_iters,
            opt.fct_sub or 'full batch'))
    else:
        print('  FCT: off')

    state = {'best': 0.0, 'iter': 0, 'total_step': len(train_loader),
             'scaler': GradScaler() if opt.amp else None}
    for epoch in range(1, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, 0.1, 200)
        train(train_loader, model, optimizer, epoch, opt, state)
