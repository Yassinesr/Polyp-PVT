"""Per-image Dice across runs: is FCT's ColonDB / CVC-300 shortfall systematic, or run noise?

Reads the maps Test_tta.py saved (result_map/<run>/<set>/<image>.png) and the ground truth, so it
needs no GPU and no checkpoints. Every image is scored with eval_dice.py, the Python port of the
MATLAB evaluation the Polyp-PVT paper used:

    meandic   mean Dice over the 256 thresholds (what the paper reports as mDic)
    dice05    Dice with the map thresholded at 0.5

and it reports:

  1. SCORES  each run's set means (identical to eval_dice.py).
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

import eval_dice
from eval_dice import TESTSETS, SHORT, EXTS, gt_dir, read_gt, read_map, image_scores

KEYS = ['meandic', 'meaniou', 'dice05', 'fn', 'fp']

VANILLA = ['PVT_VANILLA', 'PVT_VANILLA_s1']
FCT = ['S3', 'S3W2', 'S3W05', 'S3F', 'S3F_EMA', 'S3ERR', 'S3GATE_s1', 'S3BGATE_s1',
       'S3SIG_s1', 'S3SIGY', 'S3SIGY_s1']
MISS = 0.1                               # Dice at 0.5 below this = the polyp was missed


def load_set(opt, s, runs):
    """Per-image scores of every run on one set, on the images all runs have."""
    gd = gt_dir(opt.gt_root, s)
    gts = {os.path.splitext(f)[0]: f for f in os.listdir(gd) if f.lower().endswith(EXTS)}
    have = {}
    for r in runs:
        d = os.path.join(opt.result_root, r, s)
        have[r] = {os.path.splitext(f)[0]: os.path.join(d, f) for f in os.listdir(d)
                   if f.lower().endswith(EXTS)} if os.path.isdir(d) else {}
    names = [n for n in sorted(gts) if all(n in have[r] for r in runs)]
    area = np.zeros(len(names))
    sc = {k: np.zeros((len(runs), len(names))) for k in KEYS}
    resized = 0
    for j, n in enumerate(names):
        gt = read_gt(os.path.join(gd, gts[n]))
        area[j] = gt.mean()
        for i, r in enumerate(runs):
            m, rs = read_map(have[r][n], gt.shape)
            resized += rs
            res = image_scores(m, gt)
            for k in KEYS:
                sc[k][i, j] = res[k]
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


def scores_table(results, runs, sets):
    """Set means per run, the same numbers eval_dice.py prints."""
    print('\n1. SCORES  meanDic (mean Dice over 256 thresholds, as in the Polyp-PVT paper)')
    print('  {:<15s}'.format('') + ''.join('{:>9s}'.format(SHORT[s]) for s in sets) + '     mean')
    for i, r in enumerate(runs):
        vals = [results[s][2]['meandic'][i].mean() for s in sets]
        print('  {:<15s}'.format(r) + ''.join('{:>9.4f}'.format(v) for v in vals) +
              '{:>9.4f}'.format(np.mean(vals)))


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
    parser.add_argument('--metric', type=str, default='meandic', choices=['meandic', 'dice05'])
    parser.add_argument('--thresholds', type=str, default='matlab', choices=['matlab', 'linspace'],
                        help='see eval_dice.py')
    parser.add_argument('--bar', type=str, nargs=2, default=['S3SIGY_s1', 'PVT_VANILLA_s1'],
                        metavar=('FCT_RUN', 'VANILLA_RUN'))
    parser.add_argument('--max_labelings', type=int, default=20000)
    parser.add_argument('--out', type=str, default='per_image_dice.csv')
    opt = parser.parse_args()
    eval_dice.set_thresholds(opt.thresholds)

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

    scores_table(results, runs, opt.sets)
    metric = opt.metric
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
