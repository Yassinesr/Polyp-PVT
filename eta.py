"""When will the multi-seed loop be done? An estimate from the logs run_seeds.sh leaves.

For each run of the plan (--arms x --seeds) it reads logs_<ARM>_s<seed>.txt. A run is
done when its log has Train_noaug.py's end marker, running when its last epoch line is
under an hour old, and to do otherwise (not started, failed or stopped: the loop trains
it again).
The minutes per epoch of each arm (evaluation included) are measured from the
timestamps of that arm's epoch lines. An arm with no measured run yet is estimated from
the other one with --ratio, FCT's time per epoch over vanilla's: 2.0 on Polyp-PVT and
2.3 on PraNet-V2 in the earlier runs. Each run adds --extra minutes for start-up and
inference. Runs go one after another, on one GPU.

Usage: python eta.py                     # ASGNet vanilla and S3LMUP, seeds 1-6
       python eta.py --arms PRANETV2_VANILLA PRANETV2_S3LMUP --seeds 1 2 3
"""
import argparse
import datetime as dt
import os
import re

LINE = re.compile(r'^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d(?:\.\d+)?) Epoch \[(\d+)/(\d+)\], Step \[(\d+)/(\d+)\]')


def read(path):
    """A log as (epoch lines as (time, epochs done), epochs the run trains, text)."""
    if not os.path.exists(path):
        return [], None, None
    pts, total = [], None
    with open(path, errors='replace') as f:
        text = f.read()
    for line in text.splitlines():
        m = LINE.match(line)
        if m:
            e, n, s, steps = map(int, m.groups()[1:])
            pts.append((dt.datetime.fromisoformat(m.group(1)), e - 1 + s / steps))
            total = n - 1                                  # the loop runs epochs 1..n-1
    return pts, total, text


def hm(minutes):
    minutes = max(0, int(round(minutes)))
    d, h, m = minutes // 1440, minutes // 60 % 24, minutes % 60
    return ('%d d %d h' % (d, h)) if d else ('%d h %02d min' % (h, m)) if h else '%d min' % m


def main():
    ap = argparse.ArgumentParser(description='Estimate when a run_seeds.sh plan is done, from its logs.')
    ap.add_argument('--arms', nargs='+', default=['ASGNET_VANILLA', 'ASGNET_S3LMUP'],
                    help='run names without _s<seed>; the first is the vanilla arm')
    ap.add_argument('--seeds', nargs='+', default=['1', '2', '3', '4', '5', '6'])
    ap.add_argument('--ratio', type=float, default=2.2,
                    help="an FCT arm's minutes per epoch over vanilla's, until one is measured")
    ap.add_argument('--extra', type=float, default=10.0,
                    help='minutes per run for start-up and inference (about 6 on PraNet-V2)')
    ap.add_argument('--now', type=str, default=None, help='count from this time (YYYY-MM-DD HH:MM) instead of now')
    opt = ap.parse_args()
    now = dt.datetime.fromisoformat(opt.now) if opt.now else dt.datetime.now()

    runs = []
    for s in opt.seeds:
        for arm in opt.arms:
            path = 'logs_%s_s%s.txt' % (arm, s)
            pts, total, text = read(path)
            if text is None:
                state = 'to do'
            elif '=== training finished' in text:
                state = 'done'
            elif 'Traceback' in text:
                state = 'to do (last try failed)'
            else:   # running if its last epoch line (or, before the first, the log) is under an hour old
                last = pts[-1][0] if pts else dt.datetime.fromtimestamp(os.path.getmtime(path))
                state = ('running' if pts else 'starting') if last > now - dt.timedelta(hours=1) else 'to do (stopped)'
            runs.append(dict(name='%s_s%s' % (arm, s), arm=arm, pts=pts, total=total, state=state))

    # minutes per epoch, measured on every run of the arm with two or more epoch lines
    per, how = {}, {}
    for arm in opt.arms:
        span = prog = 0.0
        for r in runs:
            if r['arm'] == arm and len(r['pts']) >= 2 and r['pts'][-1][1] > r['pts'][0][1]:
                span += (r['pts'][-1][0] - r['pts'][0][0]).total_seconds() / 60
                prog += r['pts'][-1][1] - r['pts'][0][1]
        if prog > 0:
            per[arm], how[arm] = span / prog, 'measured'
    van, fct = opt.arms[0], opt.arms[1:]
    for arm in fct:
        if arm not in per and van in how:
            per[arm], how[arm] = per[van] * opt.ratio, 'estimated: %.1f x vanilla' % opt.ratio
    fct_measured = [per[a] for a in fct if how.get(a) == 'measured']
    if van not in per and fct_measured:
        per[van], how[van] = min(fct_measured) / opt.ratio, 'estimated: FCT / %.1f' % opt.ratio
    epochs = next((r['total'] for r in runs if r['total']), None)

    if not per or epochs is None:
        print('No run of %s has two epoch lines yet: try again once the first run has logged a few steps.'
              % ', '.join(opt.arms))
        return
    left, unknown = 0.0, []
    print('%d runs (%s x seeds %s), one after another on one GPU\n'
          % (len(runs), ', '.join(opt.arms), ' '.join(opt.seeds)))
    for r in runs:
        m, n = per.get(r['arm']), r['total'] or epochs
        if r['state'] == 'done':
            print('  %-22s done' % r['name'])
            continue
        if m is None:
            unknown.append(r['name'])
            print('  %-22s %s, no time per epoch yet' % (r['name'], r['state']))
            continue
        if r['state'] == 'running':
            t, p = r['pts'][-1]
            mins = max(0.0, (n - p) * m - (now - t).total_seconds() / 60) + opt.extra
            print('  %-22s running, epoch %.0f of %d, ends ~%s (in %s)'
                  % (r['name'], p, n, (now + dt.timedelta(minutes=mins)).strftime('%a %d %b %H:%M'), hm(mins)))
        else:
            mins = n * m + opt.extra
            print('  %-22s %s, ~%s' % (r['name'], r['state'], hm(mins)))
        left += mins
    print('\nminutes per epoch, evaluation included: ' + ', '.join(
        '%s %.2f (%s)' % (a, per[a], how[a]) for a in opt.arms if a in per))
    if unknown:
        print('no estimate yet for: ' + ', '.join(unknown))
    elif left == 0:
        print('all runs done')
    else:
        print('all done ~%s (in %s), with %.0f min per run for start-up and inference'
              % ((now + dt.timedelta(minutes=left)).strftime('%a %d %b %H:%M'), hm(left), opt.extra))


if __name__ == '__main__':
    main()
