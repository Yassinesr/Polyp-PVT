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

COST NOTE
---------
Multi-scale already costs 3 forward/backward passes per batch. FCT adds two
more forwards. To keep that from compounding to 9, the consistency term is
computed only at `rate == 1` (the canonical scale). Reported per-step cost is
therefore ~1.6x the baseline, not 3x.

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


def _prob(model, x):
    """Inference-equivalent probability map: sigmoid(P1 + P2).

    Test.py predicts with `P1 + P2`, so the consistency term is applied to the
    same quantity TTA will later average. Constraining anything else would be
    constraining a signal that never reaches the output.
    """
    p1, p2 = model(x)
    return torch.sigmoid(p1 + p2)


def flip_consistency(model, images, use_vflip=True):
    """Consistency-only FCT: the flipped views are NEVER supervised by the GT.

    This mirrors `fct_supervise_flips=False` on the patch side, and the
    distinction is load-bearing. If the flipped views received GT supervision
    they would simply BE flip augmentation, reintroduced through the loss --
    and the arm would silently collapse back into the augmented baseline this
    file exists to remove. Here the flips contribute the consistency signal
    and nothing else.
    """
    p_o = _prob(model, images)
    cons = F.mse_loss(_hflip(_prob(model, _hflip(images))), p_o)
    if use_vflip:
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

            P1, P2 = model(images)
            loss = structure_loss(P1, gts) + structure_loss(P2, gts)

            # Consistency only at the canonical scale, so FCT costs 2 extra
            # forwards per step rather than 2 per scale.
            if opt.fct and rate == 1:
                it = state['iter']
                lam = opt.fct_weight
                if opt.fct_warmup_iters > 0 and it < opt.fct_warmup_iters:
                    lam = opt.fct_weight * (it / float(opt.fct_warmup_iters))
                cons = flip_consistency(model, images, use_vflip=bool(opt.fct_vflip))
                loss = loss + lam * cons
                cons_record.update(cons.data, opt.batchsize)
                state['iter'] += 1

            loss.backward()
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
    torch.save(model.state_dict(), os.path.join(opt.train_save, '{}PolypPVT.pth'.format(epoch)))

    scores, meandice = evaluate_all(model, opt.test_path)
    for name, d in scores.items():
        logging.info('epoch: {}, dataset: {}, dice: {}'.format(epoch, name, d))
        print(name, ': ', d)
    print('mean: ', meandice)

    if meandice > state['best']:
        state['best'] = meandice
        torch.save(model.state_dict(), os.path.join(opt.train_save, 'PolypPVT.pth'))
        torch.save(model.state_dict(),
                   os.path.join(opt.train_save, '{}PolypPVT-best.pth'.format(epoch)))
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

    # Flip-consistency training. Consistency-only: flipped views are never
    # supervised by the GT, so this adds no augmentation of its own.
    parser.add_argument('--fct', type=int, default=0, help='1 = enable flip consistency')
    parser.add_argument('--fct_weight', type=float, default=0.5,
                        help='consistency weight; 0.05 is far too weak to install an '
                             'invariance when no augmentation supplies one')
    parser.add_argument('--fct_warmup_iters', type=int, default=300)
    parser.add_argument('--fct_vflip', type=int, default=1)

    opt = parser.parse_args()

    logging.basicConfig(filename='train_log_noaug.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

    model = PolypPVT().cuda()
    params = model.parameters()
    if opt.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params, opt.lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(params, opt.lr, weight_decay=1e-4, momentum=0.9)
    print(optimizer)

    image_root = '{}/images/'.format(opt.train_path)
    gt_root = '{}/masks/'.format(opt.train_path)

    # augmentation=False routes the loader to its no-augmentation branch:
    # Resize -> ToTensor (-> Normalize for the image). No flips, no rotation.
    # Deliberately not configurable -- re-enabling it would defeat this script.
    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize,
                              trainsize=opt.trainsize, augmentation=False)

    print('#' * 20, 'Start Training (no orientation augmentation)', '#' * 20)
    print('multiscale:', bool(opt.multiscale),
          '| FCT:', bool(opt.fct),
          ('(weight %.3f, vflip %s, warmup %d)' % (opt.fct_weight, bool(opt.fct_vflip),
                                                   opt.fct_warmup_iters)) if opt.fct else '')

    state = {'best': 0.0, 'iter': 0, 'total_step': len(train_loader)}
    for epoch in range(1, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, 0.1, 200)
        train(train_loader, model, optimizer, epoch, opt, state)
