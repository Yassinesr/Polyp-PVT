"""Flip gap: how much does a model's prediction change when the input is flipped?

For every image, the model predicts the upright image and each flipped copy.
Each flipped prediction is flipped back, and both are thresholded at 0.5. The
score is

    gap = 1 - Dice(mask_upright, mask_flipped_back)        (both empty -> 0)

0 means the prediction is perfectly flip-equivariant. It is scored per image
and averaged over images, so the large background area of each image does not
drown the number, and it needs no ground truth: it measures agreement with
itself, not accuracy.

Run it on the TRAINING images as well as the test sets. That separates two
failures that look identical on the test set:

    FCT train gap ~ no-aug train gap   the consistency loss is not pulling
    FCT train gap << no-aug, test ~    it pulls, but does not generalise

Usage:
  python flip_gap.py --pth_path ./model_pth/PVT_A2 ./model_pth/PVT_B2
  python flip_gap.py --pth_path ./model_pth/PVT_AUG --sets train Kvasir

Images are only resized and normalised (utils/dataloader.test_dataset), so no
augmentation is applied on either the train or the test images.
"""
import os
import argparse

import torch

from lib.pvt import PolypPVT
from utils.dataloader import test_dataset
from Test_tta import resolve_checkpoint

TESTSETS = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']
FLIPS = {'h': [3], 'v': [2]}          # NCHW: dim 3 = width (horizontal), dim 2 = height


def _mask(model, x, dims=None):
    if dims is not None:
        x = torch.flip(x, dims)
    p1, p2 = model(x)
    p = torch.sigmoid(p1 + p2)
    if dims is not None:
        p = torch.flip(p, dims)
    return p > 0.5


def _one_minus_dice(a, b):
    inter = (a & b).sum().item()
    size = a.sum().item() + b.sum().item()
    return 1.0 - (2.0 * inter + 1.0) / (size + 1.0)


@torch.no_grad()
def flip_gap(model, image_root, gt_root, testsize):
    loader = test_dataset(image_root, gt_root, testsize)
    sums = {k: 0.0 for k in FLIPS}
    for _ in range(loader.size):
        image, _, _ = loader.load_data()
        x = image.cuda()
        m_o = _mask(model, x)
        for k, dims in FLIPS.items():
            sums[k] += _one_minus_dice(m_o, _mask(model, x, dims))
    n = max(loader.size, 1)
    return {k: v / n for k, v in sums.items()}, loader.size


def _roots(name, opt):
    if name == 'train':
        d = opt.train_path
    else:
        d = os.path.join(opt.test_path, name)
    return os.path.join(d, 'images/'), os.path.join(d, 'masks/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pth_path', type=str, nargs='+', required=True,
                        help='one or more checkpoint files or --train_save directories')
    parser.add_argument('--sets', type=str, nargs='+', default=['train'] + TESTSETS,
                        help="'train' and/or test-set names")
    parser.add_argument('--train_path', type=str, default='./dataset/TrainDataset/')
    parser.add_argument('--test_path', type=str, default='./dataset/TestDataset/')
    parser.add_argument('--testsize', type=int, default=352)
    parser.add_argument('--out', type=str, default='',
                        help='optional TSV file; rows are appended')
    opt = parser.parse_args()

    rows = []
    for p in opt.pth_path:
        ck = resolve_checkpoint(p)
        model = PolypPVT()
        model.load_state_dict(torch.load(ck, map_location='cpu'))
        model.cuda().eval()
        name = os.path.basename(os.path.normpath(p))
        print('== {}  ({})'.format(name, ck))
        per_set = {}
        for s in opt.sets:
            img_root, gt_root = _roots(s, opt)
            g, n = flip_gap(model, img_root, gt_root, opt.testsize)
            per_set[s] = g
            print('  {:<18s} n={:<5d} h {:.4f}  v {:.4f}  mean {:.4f}'.format(
                s, n, g['h'], g['v'], 0.5 * (g['h'] + g['v'])))
        rows.append((name, per_set))
        del model
        torch.cuda.empty_cache()

    # Summary table: mean of h and v per set; the test mean is over the 5 test sets.
    test_sets = [s for s in opt.sets if s != 'train']
    header = ['checkpoint'] + list(opt.sets) + (['test_mean'] if test_sets else [])
    lines = ['\t'.join(header)]
    for name, per_set in rows:
        vals = [0.5 * (per_set[s]['h'] + per_set[s]['v']) for s in opt.sets]
        cells = ['{:.4f}'.format(v) for v in vals]
        if test_sets:
            tm = sum(0.5 * (per_set[s]['h'] + per_set[s]['v']) for s in test_sets) / len(test_sets)
            cells.append('{:.4f}'.format(tm))
        lines.append('\t'.join([name] + cells))
    print('\nFLIP GAP (1 - Dice between upright and flipped-back masks; lower = more equivariant)')
    print('\n'.join(lines))
    if opt.out:
        new = not os.path.isfile(opt.out)
        with open(opt.out, 'a') as f:
            f.write('\n'.join(lines if new else lines[1:]) + '\n')
