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
cases = [('FCT (lag-or-margin + upright pull)', opt(), dict()),
         ('upright pull only (--fct_flip_pull 0)', opt(fct_flip_pull=0), dict(flip_pull=False)),
         ('lag-or-margin, no upright pull', opt(fct_up=0), dict(up=False)),
         ('unweighted flip pulls + upright pull', opt(fct_weit=0), dict(weight='none')),
         ('a = 0.5', opt(fct_lm_a=0.5), dict(a=0.5)),
         ('lag-only gate + upright pull (--fct_weit 5)', opt(fct_weit=5), dict(weight='lag')),
         ('confidence-graded, no upright pull (6)', opt(fct_weit=6, fct_up=0), dict(weight='graded', up=False))]
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
# the flip pulls must matter when they are on
r_diff = rel(grads_reference(model), grads_reference(model, flip_pull=False))
print('sanity: FCT vs upright-pull-only reference differ by %.2e (must be > 0)' % r_diff)
print('ALL MATCH' if ok and r_diff > 1e-3 else 'MISMATCH')
