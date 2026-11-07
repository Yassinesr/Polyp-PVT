"""Score result maps the way the Polyp-PVT paper did: PraNet's MATLAB evaluation, in Python.

Polyp-PVT's README: "Please note that we use the Matlab version to evaluate in our paper", i.e.
PraNet's eval/main.m with Fmeasure_calu.m. This file follows it step by step:

  ground truth  grey (rgb2gray if it has 3 channels), polyp = value > 128
  map           first channel, resized to the mask if the sizes differ (imresize, bicubic),
                min-max normalised to [0, 1] (im2double + mapminmax)
  thresholds    256, from 1 down to 0 in steps of 1/255; a pixel is polyp if map >= threshold
  Dice, IoU     per image and threshold: 2TP / (|GT| + |pred|) and TP / (TP + FP + FN),
                both 0 when TP = 0
  meanDic       mean over images at each threshold, then mean over the 256 thresholds
                (meanIoU the same way)

Pixels that lie exactly on a threshold count as polyp or not depending on the last bit of the
floating-point threshold, so the thresholds and the normalised map are computed the way MATLAB
computes them (see THRESHOLDS and image_scores), not as exact fractions k/255.

Before trusting it, score Polyp-PVT's own pre-computed maps (README section 4.8) and compare with
the paper's table; --thresholds linspace shows how much the threshold rounding alone moves it:
  python eval_dice.py --result_root ./paper_maps --runs PolypPVT
  python eval_dice.py --result_root ./paper_maps --runs PolypPVT --thresholds linspace

Usage:
  python eval_dice.py                                   # every run folder under ./result_map
  python eval_dice.py --runs PVT_VANILLA_s1 S3SIGY_s1   # some runs
  python eval_dice.py --ref PVT_VANILLA_s1              # also: per-set difference to this run
"""
import os
import argparse

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None
try:
    from PIL import Image
except ImportError:
    Image = None

TESTSETS = ['Kvasir', 'CVC-ClinicDB', 'CVC-ColonDB', 'CVC-300', 'ETIS-LaribPolypDB']
SHORT = {'Kvasir': 'Kvasir', 'CVC-ClinicDB': 'ClinicDB', 'CVC-ColonDB': 'ColonDB',
         'CVC-300': 'CVC-300', 'ETIS-LaribPolypDB': 'ETIS'}
EXTS = ('.png', '.tif', '.tiff', '.jpg', '.jpeg', '.bmp')


def short(s):
    """Column label for a test set: the usual short name, or PG-C1 for PolypGen_C1."""
    return SHORT.get(s, s.replace('PolypGen_', 'PG-'))

# main.m's Thresholds = 1:-1/255:0, built the way MATLAB's colon operator builds it: the first half
# as a + k*d and the second half as b - k*d, so a value can differ from k/255 in the last bit.
_D = -(1.0 / 255.0)
_K = np.arange(128)
THRESHOLDS = np.empty(256)
THRESHOLDS[_K] = 1.0 + _K * _D
THRESHOLDS[255 - _K] = 0.0 - _K * _D
_MATLAB = THRESHOLDS.copy()


def set_thresholds(kind):
    """'matlab' (default) or 'linspace' = np.linspace(1, 0, 256), as Python ports usually do.
    The two differ only in the last bit, which tips pixels lying exactly on a threshold; on real
    maps that moves a set's meanDic by a few 1e-4."""
    THRESHOLDS[:] = _MATLAB if kind == 'matlab' else np.linspace(1.0, 0.0, 256)


def _read(path):
    """Image as stored (uint8 or uint16), colour as RGB."""
    if Image is not None:
        im = Image.open(path)
        if im.mode == 'P':
            im = im.convert('RGB')
        a = np.asarray(im)
        return a[..., :3] if a.ndim == 3 else a
    a = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if a is None:
        raise IOError('cannot read ' + path)
    return a[..., 2::-1] if a.ndim == 3 else a          # BGR(A) -> RGB


def read_gt(path):
    """Polyp mask: rgb2gray if colour, then > 128 (main.m)."""
    g = _read(path)
    if g.dtype == bool:                                  # 1-bit mask: imread gives logical
        return g
    if g.ndim == 3:
        g = np.round(0.2989 * g[..., 0] + 0.5870 * g[..., 1] + 0.1140 * g[..., 2])
    return g > 128


def read_map(path, shape):
    """Result map: first channel, resized to the mask if needed. Returns (map, resized)."""
    m = _read(path)
    if m.ndim == 3:
        m = m[..., 0]
    if m.shape == shape:
        return m, False
    if Image is not None and m.dtype == np.uint8:     # bicubic with antialiasing, like imresize
        m = np.asarray(Image.fromarray(m).resize((shape[1], shape[0]), Image.BICUBIC))
    else:
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)
    return m, True


def image_scores(m, gt):
    """Dice and IoU of one map at the 256 thresholds, plus Dice and error areas at 0.5.

    Every pixel of grey level v gets the same normalised value, computed as MATLAB does it:
    x = v / 255 (im2double), y = (x - xmin) * gain with gain = 1 / (xmax - xmin) (mapminmax).
    A constant map normalises to 0. Histograms over the grey levels then give TP and |pred| at
    every threshold at once.
    """
    scale = 65535.0 if m.dtype == np.uint16 else 255.0
    m = m.astype(np.int64)
    vmin, vmax = int(m.min()), int(m.max())
    x = np.arange(vmax + 1) / scale
    if vmax > vmin:
        xmin = vmin / scale
        y = (x - xmin) * (1.0 / (vmax / scale - xmin))
    else:
        y = np.zeros(vmax + 1)
    pos = np.bincount(m[gt], minlength=vmax + 1).astype(np.float64)
    allp = np.bincount(m.ravel(), minlength=vmax + 1).astype(np.float64)
    passed = y[:, None] >= THRESHOLDS[None, :]           # grey level x threshold
    tp, pred = pos @ passed, allp @ passed
    p = float(gt.sum())
    ok = tp > 0
    dice = np.where(ok, 2.0 * tp / np.maximum(p + pred, 1.0), 0.0)
    iou = np.where(ok, tp / np.maximum(p + pred - tp, 1.0), 0.0)
    at5 = y >= 0.5                                       # the map thresholded at 0.5
    tp5, pred5 = float(pos @ at5), float(allp @ at5)
    return {'dice': dice, 'iou': iou, 'meandic': float(dice.mean()), 'meaniou': float(iou.mean()),
            'dice05': 2.0 * tp5 / (p + pred5) if tp5 > 0 else 0.0,
            'fn': (p - tp5) / max(p, 1.0), 'fp': (pred5 - tp5) / max(p, 1.0)}


def gt_dir(gt_root, s):
    d = os.path.join(gt_root, s, 'masks')
    return d if os.path.isdir(d) else os.path.join(gt_root, s, 'gt')


def list_images(gt_root, result_root, run, s):
    """(stem, gt file, map file) for the maps of one run on one set, and the stems without a map.

    main.m loops over the maps, so a missing map silently drops the image; here it is reported.
    """
    gd, md = gt_dir(gt_root, s), os.path.join(result_root, run, s)
    gts = {os.path.splitext(f)[0]: os.path.join(gd, f) for f in os.listdir(gd)
           if f.lower().endswith(EXTS)}
    maps = {os.path.splitext(f)[0]: os.path.join(md, f) for f in os.listdir(md)
            if f.lower().endswith(EXTS)} if os.path.isdir(md) else {}
    both = [(n, gts[n], maps[n]) for n in sorted(maps) if n in gts]
    return both, sorted(set(gts) - set(maps)), sorted(set(maps) - set(gts))


def score_run(gt_root, result_root, run, sets):
    """{set: (meanDic, meanIoU, n images)} for one run, printing any coverage problems."""
    out = {}
    for s in sets:
        imgs, no_map, no_gt = list_images(gt_root, result_root, run, s)
        if not imgs:
            print('  !! {} / {}: no maps'.format(run, s))
            continue
        dice, iou, resized = np.zeros(256), np.zeros(256), 0
        for _, g, mp in imgs:
            gt = read_gt(g)
            m, r = read_map(mp, gt.shape)
            resized += r
            sc = image_scores(m, gt)
            dice += sc['dice']
            iou += sc['iou']
        n = len(imgs)
        out[s] = (float((dice / n).mean()), float((iou / n).mean()), n)
        for msg, cnt in (('masks without a map', len(no_map)), ('maps without a mask', len(no_gt)),
                         ('maps resized to the mask', resized)):
            if cnt:
                print('  !! {} / {}: {} {}'.format(run, s, cnt, msg))
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_root', type=str, default='./result_map')
    parser.add_argument('--gt_root', type=str, default='./dataset/TestDataset')
    parser.add_argument('--runs', type=str, nargs='+', default=None,
                        help='run folders under --result_root (default: all of them)')
    parser.add_argument('--sets', type=str, nargs='+', default=TESTSETS)
    parser.add_argument('--ref', type=str, default='PVT_VANILLA_s1',
                        help='run to compare every other run against (skipped if not scored)')
    parser.add_argument('--thresholds', type=str, default='matlab', choices=['matlab', 'linspace'],
                        help="'matlab' reproduces main.m; 'linspace' is what Python ports usually use")
    opt = parser.parse_args()
    set_thresholds(opt.thresholds)

    runs = opt.runs or sorted(d for d in os.listdir(opt.result_root)
                              if os.path.isdir(os.path.join(opt.result_root, d)))
    scores = {}
    for r in runs:
        res = score_run(opt.gt_root, opt.result_root, r, opt.sets)
        if not res:
            continue
        scores[r] = res
        print('\nExpr - {}\n{:<18s} {:>8s} {:>8s} {:>6s}'.format(r, 'dataset', 'meanDic', 'meanIoU', 'n'))
        for s, (d, i, n) in res.items():
            print('{:<18s} {:>8.4f} {:>8.4f} {:>6d}'.format(s, d, i, n))

    sets = [s for s in opt.sets if all(s in v for v in scores.values())]
    if not scores or not sets:
        raise SystemExit('nothing scored')
    head = ''.join('{:>9s}'.format(short(s)) for s in sets)
    print('\nmeanDic' + ' ' * 18 + head + '     mean')
    for r, res in scores.items():
        vals = [res[s][0] for s in sets]
        print('{:<25s}'.format(r) + ''.join('{:>9.4f}'.format(v) for v in vals) +
              '{:>9.4f}'.format(np.mean(vals)))
    if opt.ref in scores:
        ref = [scores[opt.ref][s][0] for s in sets]
        print('\ndifference to {} (sets won: higher by >= 0.0001 after rounding; goal: all {} won, '
              'one by >= 0.001)'.format(opt.ref, len(sets)))
        for r, res in scores.items():
            if r == opt.ref:
                continue
            dif = [round(res[s][0], 4) - round(v, 4) for s, v in zip(sets, ref)]
            won = sum(d > 0 for d in dif)
            goal = won == len(sets) and max(dif) >= 0.001 - 1e-9
            print('{:<25s}'.format(r) + ''.join('{:>+9.4f}'.format(d) for d in dif) +
                  '{:>+9.4f}   won {}/{}{}'.format(np.mean(dif), won, len(sets), '   GOAL MET' if goal else ''))
