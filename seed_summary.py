"""Multi-seed comparison of vanilla Polyp-PVT and an FCT v2 arm, scored the published way.

The analysis below was fixed before the seed runs were trained (see this file's commit date):

  runs      <vanilla>_s<seed> and <fct>_s<seed> for the primary seeds (default 2-6), each at its
            best epoch (PolypPVT.pth, maps in result_map/<run>)
  metric    meanDic per test set and its mean over the five sets, from eval_dice.py (Python port
            of the MATLAB evaluation used in the Polyp-PVT paper)
  per set   mean +- SD over seeds for each arm; the per-seed difference fct - vanilla (the same
            seed gives both arms the same initial weights and data order) with its 95% t interval
            and paired t-test; and an exact permutation test over every split of the 2n runs into
            two arms of n, which assumes nothing about pairing or normality
  claim     a set counts as improved only if the 95% interval of the difference lies above 0
  goal      your criterion (all five sets won, at least one by >= 0.001), per seed and on the
            seed means
  seed 1    shown separately and kept out of the statistics, because S3SIG was chosen on its
            seed-1 result

Usage:
  python seed_summary.py                            # seeds 2-6, PVT_VANILLA vs S3SIG
  python seed_summary.py --seeds 2 3 4 --arms PVT_VANILLA S3SIG
"""
import os
import math
import argparse
import itertools

import numpy as np

from eval_dice import TESTSETS, SHORT, score_run, set_thresholds


def _betacf(a, b, x):
    """Continued fraction for the incomplete beta function (modified Lentz)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        for aa in (m * (b - m) * x / ((qam + m2) * (a + m2)),
                   -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))):
            d = 1.0 + aa * d
            d = 1.0 / (d if abs(d) > tiny else tiny)
            c = 1.0 + aa / c
            c = c if abs(c) > tiny else tiny
            h *= d * c
        if abs(d * c - 1.0) < 1e-15:
            break
    return h


def betainc(a, b, x):
    """Regularised incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lbt) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbt) * _betacf(b, a, 1.0 - x) / b


def t_pvalue(t, df):
    """Two-sided p-value of Student's t."""
    return betainc(df / 2.0, 0.5, df / (df + t * t))


def t_crit(df, alpha=0.05):
    """t with two-sided p = alpha, by bisection."""
    lo, hi = 0.0, 1000.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if t_pvalue(mid, df) > alpha else (lo, mid)
    return 0.5 * (lo + hi)


def perm_pvalue(f, v):
    """Exact two-sided permutation p of mean(f) - mean(v) over every split of the runs."""
    allv = np.concatenate([f, v])
    n = len(f)
    obs = f.mean() - v.mean()
    stats = []
    for idx in itertools.combinations(range(len(allv)), n):
        mask = np.zeros(len(allv), bool)
        mask[list(idx)] = True
        stats.append(allv[mask].mean() - allv[~mask].mean())
    return float(np.mean(np.abs(stats) >= abs(obs) - 1e-12))


def won(diff):
    """Your rule: a set is won if the 4-decimal scores differ by > 0; goal = all won, one >= 0.001."""
    d = np.round(diff, 4)
    return int((d > 0).sum()), bool((d > 0).all() and d.max() >= 0.001 - 1e-9)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, nargs='+', default=[2, 3, 4, 5, 6])
    parser.add_argument('--arms', type=str, nargs=2, default=['PVT_VANILLA', 'S3SIG'],
                        metavar=('VANILLA', 'FCT'))
    parser.add_argument('--result_root', type=str, default='./result_map')
    parser.add_argument('--gt_root', type=str, default='./dataset/TestDataset')
    parser.add_argument('--thresholds', type=str, default='matlab', choices=['matlab', 'linspace'])
    opt = parser.parse_args()
    set_thresholds(opt.thresholds)
    va, fa = opt.arms
    cols = [SHORT[s] for s in TESTSETS] + ['mean']

    def score(run):
        if not os.path.isdir(os.path.join(opt.result_root, run)):
            return None
        res = score_run(opt.gt_root, opt.result_root, run, TESTSETS)
        if len(res) < len(TESTSETS):
            return None
        vals = [res[s][0] for s in TESTSETS]
        return np.array(vals + [float(np.mean(vals))])

    seeds = sorted(set(opt.seeds) | {1})
    sc = {(a, s): score('{}_s{}'.format(a, s)) for a in (va, fa) for s in seeds}

    print('\nmeanDic per run' + ' ' * 10 + ''.join('{:>9s}'.format(c) for c in cols))
    for a in (va, fa):
        for s in seeds:
            x = sc[(a, s)]
            note = '   seed 1: not in the statistics' if s == 1 and 1 not in opt.seeds else ''
            print('{:<25s}'.format('{}_s{}'.format(a, s)) +
                  (''.join('{:>9.4f}'.format(v) for v in x) if x is not None else '   missing or incomplete') + note)

    primary = [s for s in opt.seeds if sc[(va, s)] is not None and sc[(fa, s)] is not None]
    missing = [s for s in opt.seeds if s not in primary]
    if missing:
        print('\n!! seeds without both runs (left out): {}'.format(missing))
    n = len(primary)
    if n < 2:
        raise SystemExit('need at least two complete seeds for the statistics')
    V = np.stack([sc[(va, s)] for s in primary])
    F = np.stack([sc[(fa, s)] for s in primary])
    D = F - V
    tc = t_crit(n - 1)

    print('\nseeds {} (n = {}); difference = {} - {} with the same seed'.format(primary, n, fa, va))
    print('{:<9s} {:>17s} {:>17s} {:>8s} {:>19s} {:>8s} {:>7s} {:>11s}  {}'.format(
        'set', va + ' mean+-SD', fa + ' mean+-SD', 'diff', '95% CI', 'p paired', 'p perm',
        'seeds won', 'improved?'))
    for j, c in enumerate(cols):
        d = D[:, j]
        se = d.std(ddof=1) / math.sqrt(n)
        t = d.mean() / se if se > 0 else float('inf')
        lo, hi = d.mean() - tc * se, d.mean() + tc * se
        print('{:<9s} {:>9.4f}+-{:.4f} {:>9.4f}+-{:.4f} {:>+8.4f} [{:>+8.4f},{:>+8.4f}] {:>8.3f} {:>7.3f} {:>8d}/{}  {}'.format(
            c, V[:, j].mean(), V[:, j].std(ddof=1), F[:, j].mean(), F[:, j].std(ddof=1), d.mean(),
            lo, hi, t_pvalue(t, n - 1) if se > 0 else 0.0, perm_pvalue(F[:, j], V[:, j]),
            int((np.round(d, 4) > 0).sum()), n, 'yes' if lo > 0 else 'no'))

    print('\nyour goal (all 5 sets won, one by >= 0.001):')
    for k, s in enumerate(primary):
        w, ok = won(D[k, :5])
        print('  seed {}: won {}/5{}'.format(s, w, '   GOAL MET' if ok else ''))
    w, ok = won(F[:, :5].mean(0) - V[:, :5].mean(0))
    print('  seed means: won {}/5{}'.format(w, '   GOAL MET' if ok else ''))

    if 1 not in opt.seeds and sc[(va, 1)] is not None and sc[(fa, 1)] is not None:
        d1 = sc[(fa, 1)] - sc[(va, 1)]
        w, ok = won(d1[:5])
        print('\nseed 1 (not in the statistics): ' + ' '.join('{} {:+.4f}'.format(c, x) for c, x in zip(cols, d1)) +
              '   won {}/5{}'.format(w, '   GOAL MET' if ok else ''))
