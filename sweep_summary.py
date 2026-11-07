"""The sweep of lag-or-margin's factor a: does the 2 in 1 - 2|p_o - y| hold up against 0.5, 1, 1.5?

Where a flipped view beats the upright, S3LMUP weights its pull by w = max(0, 1 - a e_o),
e_o = |p_o - y|, with a = 2. The upright is right only while e_o < 0.5, so a < 2 keeps some pull
toward a wrong upright, a > 2 drops the pull while the upright is still right, and only a = 2
does neither. This script checks that argument on the test sets.

The analysis below was fixed before the sweep runs were trained (see this file's commit date):

  runs      S3LMUP_A05, S3LMUP_A10, S3LMUP_A15 (a = 0.5, 1, 1.5) and S3LMUP (a = 2), each
            _s<seed> for seeds 1-6 at its best epoch (PolypPVT.pth, maps in result_map/<run>);
            S3LMUP_A30 (a = 3) is reported if present, PVT_VANILLA for reference
  metric    meanDic per test set and its mean over the five sets, from eval_dice.py
  trend     primary: how Dice changes from a = 0.5 to a = 2. Per seed, the least-squares line
            of Dice on a over its four runs; the fitted change over that range is 1.5 x the
            slope. Then the mean over seeds with its 95% t interval: each seed counts once, so
            the four runs of a seed need not be independent
  pairwise  each a against a = 2 on the same seed: mean difference, 95% t interval, and the
            number of seeds on which a = 2 scores higher
  verdict   on the five-set mean and ETIS:
              against 2     some a < 2 beats a = 2, its 95% interval above 0
              supports 2    not against, and the trend's interval on the five-set mean lies
                            above 0 (Dice rises as a goes to 2)
              consistent    neither: no a < 2 does better, but the rise is not resolved
            a = 3 tests the other side of the argument and is reported against a = 2 only
  pixels    from the training logs (logs_<run>.txt), the shares of pixels where a flipped view
            beats the upright (lead: the pixels a acts on) and where it beats a wrong upright
            (wl: where a < 2 pulls toward the error), averaged over the last epoch

Usage:
  python sweep_summary.py
  python sweep_summary.py --seeds 1 2 3
"""
import os
import re
import math
import argparse

import numpy as np

from eval_dice import TESTSETS, short, score_run, set_thresholds
from seed_summary import t_crit, t_pvalue

ARMS = [(0.5, 'S3LMUP_A05'), (1.0, 'S3LMUP_A10'), (1.5, 'S3LMUP_A15'), (2.0, 'S3LMUP'),
        (3.0, 'S3LMUP_A30')]
TREND = [0.5, 1.0, 1.5, 2.0]
VANILLA = 'PVT_VANILLA'
LOG_RE = re.compile(r'Epoch \[(\d+)/\d+\].*lead: ([0-9.]+), wl: ([0-9.]+)')


def interval(d):
    """Per column of d (rows = seeds): mean, 95% t interval and two-sided paired p."""
    n = d.shape[0]
    m = d.mean(0)
    se = d.std(0, ddof=1) / math.sqrt(n)
    tc = t_crit(n - 1)
    p = np.array([t_pvalue(mi / si, n - 1) if si > 0 else 0.0 for mi, si in zip(m, se)])
    return m, m - tc * se, m + tc * se, p


def pixel_shares(log_root, run):
    """lead and wl averaged over the last epoch logged in logs_<run>.txt, or None."""
    path = os.path.join(log_root, 'logs_{}.txt'.format(run))
    if not os.path.isfile(path):
        return None
    rows = []
    with open(path, errors='replace') as f:
        for line in f:
            m = LOG_RE.search(line)
            if m:
                rows.append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
    if not rows:
        return None
    last = max(r[0] for r in rows)
    return np.array([r[1:] for r in rows if r[0] == last]).mean(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6])
    parser.add_argument('--result_root', type=str, default='./result_map')
    parser.add_argument('--gt_root', type=str, default='./dataset/TestDataset')
    parser.add_argument('--log_root', type=str, default='.',
                        help='folder with the training logs logs_<run>.txt')
    parser.add_argument('--thresholds', type=str, default='matlab', choices=['matlab', 'linspace'])
    parser.add_argument('--sets', type=str, nargs='+', default=TESTSETS,
                        help='test-set folders under --gt_root (default: the five benchmark sets)')
    opt = parser.parse_args()
    set_thresholds(opt.thresholds)
    sets, k = opt.sets, len(opt.sets)
    cols = [short(s) for s in sets] + ['mean']
    etis = sets.index('ETIS-LaribPolypDB') if 'ETIS-LaribPolypDB' in sets else None
    key = [(k, 'the five-set mean')] + ([(etis, 'ETIS')] if etis is not None else [])
    name_of = dict(ARMS)

    def score(run):
        if not os.path.isdir(os.path.join(opt.result_root, run)):
            return None
        res = score_run(opt.gt_root, opt.result_root, run, sets)
        if len(res) < k:
            return None
        vals = [res[s][0] for s in sets]
        return np.array(vals + [float(np.mean(vals))])

    arms = ARMS + [(None, VANILLA)]
    label = {name: ('vanilla' if a is None else 'a = {:g}'.format(a)) for a, name in arms}
    sc = {(name, s): score('{}_s{}'.format(name, s)) for _, name in arms for s in opt.seeds}

    print('\nmeanDic, mean +- SD over seeds' + ' ' * 9 + ''.join('{:>17s}'.format(c) for c in cols) + '  seeds')
    for a, name in arms:
        got = [s for s in opt.seeds if sc[(name, s)] is not None]
        if not got:
            continue
        X = np.stack([sc[(name, s)] for s in got])
        sd = X.std(0, ddof=1) if len(got) > 1 else np.zeros(X.shape[1])
        print('{:<10s}{:<29s}'.format(label[name], name) +
              ''.join('{:>9.4f}+-{:.4f}'.format(m, d) for m, d in zip(X.mean(0), sd)) + '  {:>5d}'.format(len(got)))

    # primary: the trend over a = 0.5 ... 2, one fitted change per seed
    full = [s for s in opt.seeds if all(sc[(name_of[a], s)] is not None for a in TREND)]
    trend = None
    if len(full) >= 2:
        ac = np.array(TREND) - np.mean(TREND)
        Y = np.stack([np.stack([sc[(name_of[a], s)] for a in TREND]) for s in full])   # seed, a, column
        change = 1.5 * np.einsum('i,sic->sc', ac, Y) / (ac ** 2).sum()
        trend = interval(change)
        m, lo, hi, p = trend
        print('\ntrend from a = 0.5 to a = 2 (primary): fitted change in Dice, seeds {} (n = {})'.format(full, len(full)))
        print('{:<9s} {:>8s} {:>19s} {:>8s} {:>13s}'.format('set', 'change', '95% CI', 'p', 'seeds rising'))
        for j, c in enumerate(cols):
            print('{:<9s} {:>+8.4f} [{:>+8.4f},{:>+8.4f}] {:>8.3f} {:>10d}/{}'.format(
                c, m[j], lo[j], hi[j], p[j], int((np.round(change[:, j], 4) > 0).sum()), len(full)))
    else:
        print('\ntrend: needs the four runs a = 0.5, 1, 1.5, 2 on at least two seeds; '
              'complete on {}'.format(full or 'none'))

    # each a against a = 2, same seed
    base = name_of[2.0]
    beats = {}
    print('\neach a against a = 2 on the same seed: difference = a - 2 (above 0: a scores higher)')
    print('{:<8s} {:<9s} {:>8s} {:>19s} {:>8s} {:>15s}'.format('a', 'set', 'diff', '95% CI', 'p', '2 higher on'))
    for a, name in ARMS:
        if a == 2.0:
            continue
        both = [s for s in opt.seeds if sc[(name, s)] is not None and sc[(base, s)] is not None]
        if len(both) < 2:
            if any(sc[(name, s)] is not None for s in opt.seeds):
                print('{:<8s} needs at least two seeds with both runs; has {}'.format(label[name], both))
            continue
        D = np.stack([sc[(name, s)] - sc[(base, s)] for s in both])
        m, lo, hi, p = interval(D)
        for j, c in enumerate(cols):
            print('{:<8s} {:<9s} {:>+8.4f} [{:>+8.4f},{:>+8.4f}] {:>8.3f} {:>12d}/{}'.format(
                label[name] if j == 0 else '', c, m[j], lo[j], hi[j], p[j],
                int((np.round(D[:, j], 4) < 0).sum()), len(both)))
        beats[a] = [what for j, what in key if lo[j] > 0]

    print('\nverdict (fixed in advance; see the top of this file):')
    below = [a for a in TREND if a < 2.0]
    if trend is None or any(a not in beats for a in below):
        print('  incomplete: needs every a < 2 against a = 2 and the trend, on at least two seeds')
    else:
        against = ['a = {:g} on {}'.format(a, ' and '.join(beats[a])) for a in below if beats[a]]
        if against:
            print('  AGAINST 2: ' + '; '.join(against) + ' beats a = 2 with its 95% interval above 0')
        elif trend[1][k] > 0:
            print('  SUPPORTS 2: no a < 2 does better than a = 2, and Dice rises as a goes to 2 '
                  '(five-set mean {:+.4f} [{:+.4f}, {:+.4f}])'.format(trend[0][k], trend[1][k], trend[2][k]))
        else:
            print('  CONSISTENT WITH 2: no a < 2 does better than a = 2, but the rise toward a = 2 is not '
                  'resolved (five-set mean {:+.4f} [{:+.4f}, {:+.4f}])'.format(trend[0][k], trend[1][k], trend[2][k]))
    if 3.0 in beats:
        print('  a = 3 (the other side): {}'.format(
            'beats a = 2 on ' + ' and '.join(beats[3.0]) if beats[3.0] else 'does not beat a = 2'))

    print('\npixel shares in the last training epoch: lead = a flipped view beats the upright, '
          'wl = it beats a wrong upright')
    for a, name in ARMS:
        got = [v for v in (pixel_shares(opt.log_root, '{}_s{}'.format(name, s)) for s in opt.seeds)
               if v is not None]
        if got:
            print('  {:<8s} lead {:.4f}  wl {:.4f}  ({} runs)'.format(label[name], *np.mean(got, 0), len(got)))
        elif any(sc[(name, s)] is not None for s in opt.seeds):
            print('  {:<8s} not in its logs (trained before the diagnostic was added)'.format(label[name]))
