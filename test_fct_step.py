"""Check _fct_v2_step against an independent implementation of the paper's Eqs. (7)-(11), on CPU.
Usage: python test_fct_step.py [REPO]   (REPO defaults to this file's folder; CPU only, a few seconds)"""
import argparse, sys, warnings
import torch, torch.nn as nn, torch.nn.functional as F
warnings.filterwarnings('ignore')
import os
sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__)))
import Train_noaug as T

torch.manual_seed(0)
class Toy(nn.Module):                 # two heads, like Polyp-PVT's P1 and P2
    def __init__(self):
        super().__init__()
        self.a = nn.Conv2d(3, 8, 3, padding=1); self.b = nn.Conv2d(8, 1, 3, padding=1); self.c = nn.Conv2d(8, 1, 5, padding=2)
    def forward(self, x):
        h = torch.tanh(self.a(x)); return self.b(h), self.c(h)

B, H, W, lam = 3, 40, 40, 0.5
x = torch.randn(B, 3, H, W)
yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
gts = torch.stack([((yy - 8 - 6 * i) ** 2 + (xx - 25 + 3 * i) ** 2 < 60).float() for i in range(B)]).unsqueeze(1)
hf, vf = (lambda t: torch.flip(t, [3])), (lambda t: torch.flip(t, [2]))

def opt(**kw):
    d = dict(fct_vflip=1, fct_sub=0, fct_sup=1, fct_weit=8, fct_up=1, amp=0, fct_lm_a=2.0, fct_loss='bce',
             grad_diag=0, fct_flip_pull=1, batchsize=B)
    d.update(kw); return argparse.Namespace(**d)

def grads_from_step(model, o):
    model.zero_grad()
    flip_probs = T._flip_probs(model, x, o, None) if o.fct_up else None
    heads = T.HOST.heads(model(x))
    loss = T.HOST.sup_loss(heads, gts)
    class Rec:
        def update(self, *a): pass
    T._fct_v2_step(model, x, gts, heads, loss, lam, None, o, {'iter': 1}, Rec(),
                   lambda t, retain=False: t.backward(retain_graph=retain), True, flip_probs=flip_probs)
    return [p.grad.clone() for p in model.parameters()]

def grads_reference(model, a=2.0, flip_pull=True, up=True, weight='lm'):
    """Eqs. (7)-(11) written from the paper, without the module's helpers."""
    model.zero_grad()
    y = (gts > 0.5).float()
    def logit(hs): return hs[0] + hs[1]
    def bce(z, q): return -(q * F.logsigmoid(z) + (1 - q) * F.logsigmoid(-z))
    def per_image_mean(m): return (m.sum(dim=(2, 3)) / (H * W)).mean()
    ho = list(model(x)); zo = logit(ho); po = torch.sigmoid(zo).detach(); eo = (po - y).abs()
    views = {'h': hf, 'v': vf}
    zk, hk = {}, {}
    for k, f in views.items():
        hs = [f(h) for h in model(f(x))]; hk[k] = hs; zk[k] = logit(hs)
    L = (T.HOST.sup_loss(ho, gts) + sum(T.HOST.sup_loss(hk[k], gts) for k in views)) / 3.0      # Eq. (7)
    if flip_pull:
        for k in views:
            ek = (torch.sigmoid(zk[k]).detach() - y).abs()
            if weight == 'lm':
                w = torch.where(ek >= eo, torch.ones_like(eo), (1 - a * eo).clamp(min=0))     # Eq. (8)
            elif weight == 'lag':                       # lag only: 1 where the upright is strictly closer
                w = (eo < ek).float()
            elif weight == 'graded':                    # confidence-graded: 1 / 1 + p_o / 1 - p_o
                right = ((po > 0.5).float() == y).float()
                w = right + (1 - right) * torch.where(y > 0.5, 1 + po, 1 - po)
            else:
                w = torch.ones_like(eo)
            L = L + lam / 2 * per_image_mean(w * bce(zk[k], po))                              # Eq. (9)
    if up:
        with torch.no_grad():
            pk = {k: torch.sigmoid(logit([f(h) for h in model(f(x))])) for k, f in views.items()}
        ek = {k: (pk[k] - y).abs() for k in views}
        best_is_h = ek['h'] <= ek['v']
        pstar = torch.where(best_is_h, pk['h'], pk['v']); estar = torch.where(best_is_h, ek['h'], ek['v'])
        g = (estar < eo).float()
        L = L + lam / 2 * per_image_mean(g * bce(zo, pstar))                                  # Eq. (10)
    L.backward()
    return [p.grad.clone() for p in model.parameters()]

def rel(g1, g2):
    num = sum(((u - v) ** 2).sum() for u, v in zip(g1, g2)) ** 0.5
    den = sum((v ** 2).sum() for v in g2) ** 0.5
    return float(num / den)

model = Toy()
cases = [('FCT (both pulls, lag-or-margin)', opt(), dict()),
         ('pull to the best flip only (--fct_flip_pull 0)', opt(fct_flip_pull=0), dict(flip_pull=False)),
         ('pull to the upright only, lag-or-margin', opt(fct_up=0), dict(up=False)),
         ('both pulls, unweighted', opt(fct_weit=0), dict(weight='none')),
         ('a = 0.5', opt(fct_lm_a=0.5), dict(a=0.5)),
         ('both pulls, lag-only gate (--fct_weit 5)', opt(fct_weit=5), dict(weight='lag')),
         ('pull to the upright only, graded (6)', opt(fct_weit=6, fct_up=0), dict(weight='graded', up=False))]
ok = True
for name, o, kw in cases:
    r = rel(grads_from_step(model, o), grads_reference(model, **kw))
    ok &= r < 1e-4
    print('%-42s relative gradient difference %.2e' % (name, r))
# and the three views alone (lambda = 0) must equal Eq. (7)
model.zero_grad(); heads = T.HOST.heads(model(x)); loss = T.HOST.sup_loss(heads, gts)
class Rec:
    def update(self, *a): pass
T._fct_v2_step(model, x, gts, heads, loss, 0.0, None, opt(fct_up=0, fct_weit=0), {'iter': 1}, Rec(),
               lambda t, retain=False: t.backward(retain_graph=retain), True)
g_step = [p.grad.clone() for p in model.parameters()]
r = rel(g_step, grads_reference(model, flip_pull=False, up=False))
ok &= r < 1e-4
print('%-42s relative gradient difference %.2e' % ('three supervised views (lambda = 0)', r))
# Each pull must actually move the step's own gradient: a gate that silently zeroes a pull would
# still match a reference with the same bug in neither, so compare the step with itself.
g_none = g_step                                                             # three views, lambda = 0
g_fct = grads_from_step(model, opt())                                       # FCT
g_up = grads_from_step(model, opt(fct_flip_pull=0))                         # pull to the best flip only
g_flip = grads_from_step(model, opt(fct_up=0))                              # pull to the upright only
moves = [('the pull to the best flip (alone vs no pull)', rel(g_up, g_none)),
         ('the pull to the upright (alone vs no pull)', rel(g_flip, g_none)),
         ('the pull to the upright, on top of the pull to the best flip', rel(g_fct, g_up)),
         ('the pull to the best flip, on top of the pull to the upright', rel(g_fct, g_flip))]
for name, r_ in moves:
    print('moves the gradient: %-62s %.2e' % (name, r_))
    ok &= r_ > 1e-3
# and the gate's direction, pixel by pixel: on a pixel where the flipped view lags (e_f >= e_o) the
# lag-or-margin weight is 1; where it leads a wrong upright view it is 0; where it leads a right one
# it is 1 - 2 e_o. Checked on the weights the step itself computes, through a hook on _wsum_reduce.
seen = []
orig = T._wsum_reduce
def spy(per, w, den):
    seen.append(w.detach().clone()); return orig(per, w, den)
T._wsum_reduce = spy
model.zero_grad(); flip_probs = T._flip_probs(model, x, opt(), None)
heads = T.HOST.heads(model(x)); loss = T.HOST.sup_loss(heads, gts)
T._fct_v2_step(model, x, gts, heads, loss, lam, None, opt(), {'iter': 1}, Rec(),
               lambda t, retain=False: t.backward(retain_graph=retain), True, flip_probs=flip_probs)
T._wsum_reduce = orig
with torch.no_grad():
    y = (gts > 0.5).float(); po = torch.sigmoid(sum(model(x))); eo = (po - y).abs()
    w_h = seen[1]                       # seen[0] is the upright gate, seen[1] the horizontal flip's weight
    ph = torch.flip(torch.sigmoid(sum(model(torch.flip(x, [3])))), [3]); eh = (ph - y).abs()
    lag, lead_wrong, lead_right = eh >= eo, (eh < eo) & (eo > 0.5), (eh < eo) & (eo < 0.5)
    checks = [('lagging pixels get w = 1', bool(torch.all(w_h[lag] == 1)), int(lag.sum())),
              ('pixels leading a wrong upright view get w = 0', bool(torch.all(w_h[lead_wrong] == 0)), int(lead_wrong.sum())),
              ('pixels leading a right upright view get w = 1 - 2 e_o',
               bool(torch.allclose(w_h[lead_right], 1 - 2 * eo[lead_right], atol=1e-6)), int(lead_right.sum()))]
for name, good, n in checks:
    print('gate direction: %-56s %s (%d pixels)' % (name, 'ok' if good and n > 0 else 'FAIL', n))
    ok &= good and n > 0
print('ALL MATCH' if ok else 'MISMATCH')
sys.exit(0 if ok else 1)
