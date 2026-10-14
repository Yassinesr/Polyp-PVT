"""Per-image Dice across runs: is FCT's ColonDB / CVC-300 shortfall systematic, or run noise?

Reads the maps Test_tta.py saved (result_map/<run>/<set>/<image>.png) and the ground truth, so it
needs no GPU and no checkpoints. For every image and run it computes

    meandic   mean Dice over the 256 thresholds t = 0..255 (pred >= t), PraNet-style meanDic
    dice05    Dice at 0.5 (pred >= 128)

and reports:

  1. SANITY  each run's set means next to the numbers your evaluator gave. Shows which Dice
             definition matches your evaluator, and whether the maps on disk are the ones that
             were scored (the PVT_VANILLA_s1 maps may have been replaced by the stopped re-run).
  2. GAP     per set, the FCT runs' mean minus the vanilla runs' mean, overall and by polyp size,
             with an exact permutation p-value: the real labelling against every way of calling
             len(--vanilla) of the runs 'vanilla'. A large p means the gap is what run-to-run
             noise alone produces.
  3. IMAGES  how many images every FCT run scores below every vanilla run (and the reverse),
             against the same permutation null; where the gap comes from (a few images or all of
             them); error type at 0.5 (missed polyp vs false positive, complete misses); and, for
             the --focus sets, the 10 images FCT loses most on.
  4. BAR     one FCT run against one vanilla run (default S3SIGY_s1 vs PVT_VANILLA_s1): which
             images make up the difference.

Every per-image number goes to a CSV (--out).

Usage:
  python per_image_compare.py
  python per_image_compare.py --result_root ./result_map --gt_root ./dataset/TestDataset
  python per_image_compare.py --vanilla PVT_VANILLA PVT_VANILLA_s1 --fct S3F S3F_EMA S3SIGY_s1
"""
import os
import csv
import argparse
import itertools

import numpy as np

try:
    import cv2
except ImportError:                      # plain PIL works too, e.g. on the evaluation machine
    cv2 = None
    from PIL import Image

TESTSETS = ['Kvasir', 'CVC-ClinicDB', 'CVC-ColonDB', 'CVC-300', 'ETIS-LaribPolypDB']
SHORT = {'Kvasir': 'Kvasir', 'CVC-ClinicDB': 'ClinicDB', 'CVC-ColonDB': 'ColonDB',
         'CVC-300': 'CVC-300', 'ETIS-LaribPolypDB': 'ETIS'}
KEYS = ['meandic', 'dice05', 'fn', 'fp']

# What your evaluator reported for each run, in TESTSETS order.
EVALUATOR = {
    'PVT_VANILLA':    (0.9153, 0.9293, 0.7957, 0.9041, 0.8072),
    'PVT_VANILLA_s1': (0.9100, 0.9155, 0.8062, 0.8931, 0.7823),
    'S3':             (0.9210, 0.9237, 0.8002, 0.8867, 0.8136),
    'S3W2':           (0.9020, 0.9265, 0.7940, 0.8885, 0.8152),
    'S3W05':          (0.9177, 0.9346, 0.8048, 0.8935, 0.8182),
    'S3F':            (0.9212, 0.9391, 0.8026, 0.8956, 0.8215),
    'S3F_EMA':        (0.9144, 0.9424, 0.7862, 0.9032, 0.7984),
    'S3ERR':          (0.9170, 0.9380, 0.7857, 0.9078, 0.8059),
    'S3GATE_s1':      (0.9248, 0.9346, 0.8003, 0.9018, 0.7943),
    'S3BGATE_s1':     (0.9130, 0.9324, 0.7926, 0.8927, 0.8257),
    'S3SIG_s1':       (0.9189, 0.9390, 0.7986, 0.8956, 0.8065),
    'S3SIGY':         (0.9217, 0.9096, 0.7944, 0.8901, 0.8173),
    'S3SIGY_s1':      (0.9201, 0.9294, 0.8035, 0.8875, 0.8209),
}
VANILLA = ['PVT_VANILLA', 'PVT_VANILLA_s1']
FCT = ['S3', 'S3W2', 'S3W05', 'S3F', 'S3F_EMA', 'S3ERR', 'S3GATE_s1', 'S3BGATE_s1',
       'S3SIG_s1', 'S3SIGY', 'S3SIGY_s1']
MISS = 0.1                               # Dice at 0.5 below this = the polyp was missed


def _read_gray(path):
    if cv2 is not None:
        a = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if a is None:
            raise IOError('cannot read ' + path)
        return a
    return np.asarray(Image.open(path).convert('L'))


def _resize(a, shape):
    if cv2 is not None:
        return cv2.resize(a, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    return np.asarray(Image.fromarray(a).resize((shape[1], shape[0]), Image.BILINEAR))


def image_scores(pred, gt):
    """Dice of one saved map against its mask.

    The maps are min-max normalised probabilities x 255, so the threshold t on the 0..255 grid
    is the evaluator's threshold t/255. Two 256-bin histograms give the Dice at all 256
    thresholds at once: tp[t] / fp[t] = polyp / background pixels with pred >= t.
    fn and fp are the missed and the false-positive area at 0.5, as fractions of the polyp area.
    """
    pos = np.bincount(pred[gt], minlength=256)
    neg = np.bincount(pred[~gt], minlength=256)
    tp = np.cumsum(pos[::-1])[::-1].astype(np.float64)
    fp = np.cumsum(neg[::-1])[::-1].astype(np.float64)
    p = float(gt.sum())
    dice = 2.0 * tp / (tp + fp + p + 1e-8)          # 2TP / (2TP + FP + FN)
    t = 128                                          # pred >= 128  <=>  pred / 255 > 0.5
    return {'meandic': float(dice.mean()), 'dice05': float(dice[t]),
            'fn': (p - tp[t]) / max(p, 1.0), 'fp': fp[t] / max(p, 1.0)}


def load_set(opt, s, runs):
    """Per-image scores of every run on one set, on the images all runs have."""
    gt_dir = os.path.join(opt.gt_root, s, 'masks')
    if not os.path.isdir(gt_dir):
        gt_dir = os.path.join(opt.gt_root, s, 'gt')
    gts = {os.path.splitext(f)[0]: f for f in os.listdir(gt_dir)
           if f.lower().endswith(('.png', '.tif', '.jpg', '.bmp'))}
    have = {}
    for r in runs:
        d = os.path.join(opt.result_root, r, s)
        have[r] = {os.path.splitext(f)[0] for f in os.listdir(d)} if os.path.isdir(d) else set()
    names = [n for n in sorted(gts) if all(n in have[r] for r in runs)]
    area = np.zeros(len(names))
    sc = {k: np.zeros((len(runs), len(names))) for k in KEYS}
    resized = 0
    for j, n in enumerate(names):
        gt = _read_gray(os.path.join(gt_dir, gts[n])) > 127
        area[j] = gt.mean()
        for i, r in enumerate(runs):
            pred = _read_gray(os.path.join(opt.result_root, r, s, n + '.png'))
            if pred.shape != gt.shape:
                pred, resized = _resize(pred, gt.shape), resized + 1
            for k, v in image_scores(pred, gt).items():
                sc[k][i, j] = v
    return names, area, sc, len(gts) - len(names), resized


def labelings(n_runs, n_v, cap):
    """Every way of choosing n_v runs as 'vanilla' (a fixed random subset if there are more
    than cap). The first entry is the real labelling: runs 0..n_v-1."""
    combs = list(itertools.combinations(range(n_runs), n_v))
    if len(combs) > cap:
        keep = np.random.default_rng(0).choice(len(combs) - 1, cap - 1, replace=False) + 1
        combs = [combs[0]] + [combs[i] for i in sorted(keep)]
    return combs


def split(m, v_idx):
    vm = np.zeros(m.shape[0], bool)
    vm[list(v_idx)] = True
    return m[vm], m[~vm]


def perm_p(stat, m, combs, two_sided):
    """Exact permutation p-value of stat(V, F) over the labellings in combs."""
    vals = np.array([stat(*split(m, c)) for c in combs])
    if two_sided:
        return float(np.mean(np.abs(vals) >= abs(vals[0]) - 1e-12))
    return float(np.mean(vals >= vals[0] - 1e-12))


def gap(v, f):
    return f.mean() - v.mean()


def n_lose(v, f):                        # images where every FCT run is below every vanilla run
    return int((f.max(0) < v.min(0)).sum())


def n_win(v, f):
    return int((f.min(0) > v.max(0)).sum())


def sanity(results, runs, sets):
    """Set means per run against the evaluator; returns the metric that matches it best."""
    print('\n1. SANITY  set means per run: meandic | dice05 | evaluator (max |diff| per metric)')
    dev = {'meandic': [], 'dice05': []}
    for i, r in enumerate(runs):
        row, ev = [], EVALUATOR.get(r)
        dmax = {'meandic': 0.0, 'dice05': 0.0}
        for s in sets:
            sc = results[s][2]
            a, b = sc['meandic'][i].mean(), sc['dice05'][i].mean()
            if ev is not None:
                e = ev[TESTSETS.index(s)]
                for k, x in (('meandic', a), ('dice05', b)):
                    dev[k].append(abs(x - e))
                    dmax[k] = max(dmax[k], abs(x - e))
                row.append('{} {:.4f}|{:.4f}|{:.4f}'.format(SHORT[s], a, b, e))
            else:
                row.append('{} {:.4f}|{:.4f}|  -   '.format(SHORT[s], a, b))
        tail = '' if ev is None else '  max|d| {:.4f} / {:.4f}'.format(dmax['meandic'], dmax['dice05'])
        print('  {:<15s} {}{}'.format(r, '  '.join(row), tail))
    if not dev['meandic']:
        return 'meandic'
    best = min(dev, key=lambda k: np.median(dev[k]))
    print('  -> analysis uses {} (median |diff| to your evaluator: meandic {:.4f}, dice05 {:.4f})'.format(
        best, np.median(dev['meandic']), np.median(dev['dice05'])))
    off = []
    for i, r in enumerate(runs):
        ev = EVALUATOR.get(r)
        if ev is None:
            continue
        d = max(abs(results[s][2][best][i].mean() - ev[TESTSETS.index(s)]) for s in sets)
        if d > 0.002:
            off.append('{} ({:.4f})'.format(r, d))
    if off:
        print('  !! maps differ from your evaluator by more than 0.002 for: {}. They may come from '
              'another checkpoint; point --result_root at the maps you scored.'.format(', '.join(off)))
    return best


def report_set(s, names, area, sc, combs, metric, show):
    m = sc[metric]
    v, f = split(m, combs[0])
    d = f.mean(0) - v.mean(0)                        # per image, FCT mean - vanilla mean
    g = d.mean()
    print('\n== {}  ({} images; vanilla {:.4f}, FCT {:.4f}, gap {:+.4f}, permutation p {:.3f})'.format(
        SHORT[s], len(names), v.mean(), f.mean(), g, perm_p(gap, m, combs, True)))

    # where the gap comes from
    o = np.argsort(d)
    k = min(10, len(d))
    worst, best = d[o[:k]].sum() / len(d), d[o[-k:]].sum() / len(d)
    print('  from: {} worst images {:+.4f}, {} best images {:+.4f}, the other {} {:+.4f}'.format(
        k, worst, k, best, len(d) - 2 * k, g - worst - best))

    # by polyp size
    q = np.quantile(area, [0.25, 0.5, 0.75])
    b = np.digitize(area, q)
    parts = []
    for i, lab in enumerate(['smallest', 'small', 'large', 'largest']):
        idx = np.where(b == i)[0]
        if len(idx) == 0:
            continue
        ms = m[:, idx]
        vv, ff = split(ms, combs[0])
        parts.append('{} {:+.4f} (p {:.3f})'.format(lab, gap(vv, ff), perm_p(gap, ms, combs, True)))
    print('  by polyp size (quarters of the set):', ', '.join(parts))

    # consistency across runs
    lo, wi = n_lose(v, f), n_win(v, f)
    print('  images where every FCT run < every vanilla run: {} (p {:.3f});  every FCT run > every '
          'vanilla run: {} (p {:.3f})'.format(lo, perm_p(n_lose, m, combs, False),
                                             wi, perm_p(n_win, m, combs, False)))

    # error type at 0.5
    fn_v, fn_f = split(sc['fn'], combs[0])
    fp_v, fp_f = split(sc['fp'], combs[0])
    miss_v, miss_f = split((sc['dice05'] < MISS).astype(float), combs[0])
    print('  at 0.5: missed polyp area {:.3f} vs {:.3f}, false-positive area {:.3f} vs {:.3f} '
          '(vanilla vs FCT, fraction of polyp area); complete misses per run {:.1f} vs {:.1f}'.format(
              fn_v.mean(), fn_f.mean(), fp_v.mean(), fp_f.mean(),
              miss_v.sum(1).mean(), miss_f.sum(1).mean()))

    # the images FCT loses most on
    if not show:
        return g
    below = (f < v.min(0)).sum(0)                    # FCT runs below both vanilla runs, per image
    print('  10 images where FCT loses most (FCT mean - vanilla mean):')
    for j in o[:10]:
        err = 'missed' if fn_f[:, j].mean() >= fp_f[:, j].mean() else 'false pos'
        print('    {:<24s} polyp {:5.1f}%  vanilla {}  FCT {:.3f} ({:.3f}-{:.3f})  below every vanilla '
              'run in {}/{} FCT runs  FCT misses {}  FCT error: {}'.format(
                  names[j][:24], 100 * area[j], ' '.join('{:.3f}'.format(x) for x in v[:, j]),
                  f[:, j].mean(), f[:, j].min(), f[:, j].max(), int(below[j]), f.shape[0],
                  int(miss_f[:, j].sum()), err))
    return g


def report_bar(s, names, area, sc, runs, metric, fr, vr, show):
    i, j = runs.index(fr), runs.index(vr)
    m = sc[metric]
    e = m[i] - m[j]
    o = np.argsort(e)
    k = min(10, len(e))
    worst, best = e[o[:k]].sum() / len(e), e[o[-k:]].sum() / len(e)
    miss = (sc['dice05'] < MISS).sum(1)
    print('  {}: {:+.4f}  = {} worst images {:+.4f}, {} best {:+.4f}, other {} {:+.4f};  complete '
          'misses {} vs {}'.format(SHORT[s], e.mean(), k, worst, k, best, len(e) - 2 * k,
                                   e.mean() - worst - best, miss[i], miss[j]))
    for t in (o[:5] if show else []):
        print('      {:<24s} polyp {:5.1f}%  {} {:.3f}  {} {:.3f}  ({} at 0.5: missed {:.2f}, '
              'false pos {:.2f} of polyp area)'.format(names[t][:24], 100 * area[t], vr, m[j, t],
                                                       fr, m[i, t], fr, sc['fn'][i, t],
                                                       sc['fp'][i, t]))


def write_csv(path, results, runs, n_v, sets):
    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['set', 'image', 'polyp_area', 'run', 'group'] + KEYS)
        for s in sets:
            names, area, sc, _, _ = results[s]
            for j, n in enumerate(names):
                for i, r in enumerate(runs):
                    w.writerow([SHORT[s], n, '{:.5f}'.format(area[j]), r,
                                'vanilla' if i < n_v else 'fct'] +
                               ['{:.5f}'.format(sc[k][i, j]) for k in KEYS])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_root', type=str, default='./result_map')
    parser.add_argument('--gt_root', type=str, default='./dataset/TestDataset')
    parser.add_argument('--vanilla', type=str, nargs='+', default=VANILLA)
    parser.add_argument('--fct', type=str, nargs='+', default=FCT)
    parser.add_argument('--sets', type=str, nargs='+', default=TESTSETS)
    parser.add_argument('--focus', type=str, nargs='+', default=['CVC-ColonDB', 'CVC-300'],
                        help='sets whose individual images are listed')
    parser.add_argument('--metric', type=str, default='auto', choices=['auto', 'meandic', 'dice05'],
                        help='auto = whichever matches your evaluator best (see SANITY)')
    parser.add_argument('--bar', type=str, nargs=2, default=['S3SIGY_s1', 'PVT_VANILLA_s1'],
                        metavar=('FCT_RUN', 'VANILLA_RUN'))
    parser.add_argument('--max_labelings', type=int, default=20000)
    parser.add_argument('--out', type=str, default='per_image_dice.csv')
    opt = parser.parse_args()

    found = lambda r: os.path.isdir(os.path.join(opt.result_root, r))
    for r in opt.vanilla + opt.fct:
        if not found(r):
            print('!! no maps for {} under {} (skipped)'.format(r, opt.result_root))
    van = [r for r in opt.vanilla if found(r)]
    fct = [r for r in opt.fct if found(r)]
    if not van or not fct:
        raise SystemExit('need at least one vanilla and one FCT run with maps')
    runs = van + fct

    results = {}
    for s in opt.sets:
        results[s] = load_set(opt, s, runs)
        names, _, _, missing, resized = results[s]
        print('{}: {} images{}{}'.format(SHORT[s], len(names),
                                          ', {} skipped (not in every run)'.format(missing) if missing else '',
                                          ', {} maps resized to the mask size'.format(resized) if resized else ''))

    best = sanity(results, runs, opt.sets)
    metric = best if opt.metric == 'auto' else opt.metric
    combs = labelings(len(runs), len(van), opt.max_labelings)
    print('\n2-3. FCT ({} runs) vs vanilla ({} runs) on {}; p-values are exact over {} labellings '
          '(smallest possible p {:.3f})'.format(len(fct), len(van), metric, len(combs), 1.0 / len(combs)))
    for s in opt.sets:
        names, area, sc, _, _ = results[s]
        report_set(s, names, area, sc, combs, metric, s in opt.focus)

    fr, vr = opt.bar
    if fr in runs and vr in runs:
        print('\n4. BAR  {} - {} per set, and the 5 images it loses most on'.format(fr, vr))
        for s in opt.sets:
            names, area, sc, _, _ = results[s]
            report_bar(s, names, area, sc, runs, metric, fr, vr, s in opt.focus)

    write_csv(opt.out, results, runs, len(van), opt.sets)
    print('\nper-image numbers for every run: {}'.format(opt.out))
