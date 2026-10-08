"""How much of the flip disagreement sits where the upright mask is wrong?

Inference only. For each training image (resize + normalise, no augmentation):

    p_o            upright probability
    p_f            flipped view's probability, flipped back (h and v)
    err            pixels where the upright mask (p_o > 0.5) disagrees with the label
    disagreement   |p_f - p_o|

and reports, summed over all images and both flips:

    err_pixels     share of pixels the upright mask gets wrong
    gate3_share    share of the disagreement on err pixels
                   = what --fct_weit 3 (error gate) removes from the consistency term
    gate5_share    share of the disagreement where the flipped view is at least as
                   close to the label as the upright
                   = what --fct_weit 5 (relative gate) removes

If gate3_share is only a few percent, the error gate behaves almost like the
uniform weight (S3F) at the end of training. It acts mostly early, when the
upright is wrong more often, which a final checkpoint cannot show.

Usage:
  python gate_share.py --pth_path ./model_pth/S3F ./model_pth/S3W05 ./model_pth/S3ERR
"""
import os
import argparse

import numpy as np
import torch
import torch.nn.functional as F

from lib.pvt import PolypPVT
from utils.dataloader import test_dataset
from Test_tta import resolve_checkpoint


@torch.no_grad()
def prob(model, x, dims=None):
    if dims is not None:
        x = torch.flip(x, dims)
    p1, p2 = model(x)
    z = p1 + p2
    if dims is not None:
        z = torch.flip(z, dims)
    return z


@torch.no_grad()
def shares(model, image_root, gt_root, testsize):
    loader = test_dataset(image_root, gt_root, testsize)
    tot = {'pix': 0.0, 'err': 0.0, 'dis': 0.0, 'dis_err': 0.0, 'dis_rel': 0.0}
    for _ in range(loader.size):
        image, gt, _ = loader.load_data()
        y = torch.from_numpy((np.asarray(gt, np.float32) / 255.0 > 0.5).astype(np.float32)).cuda()
        x = image.cuda()
        size = y.shape
        up = lambda z: torch.sigmoid(F.interpolate(z, size=size, mode='bilinear', align_corners=False))[0, 0]
        p_o = up(prob(model, x))
        err = ((p_o > 0.5).float() != y).float()
        tot['pix'] += y.numel()
        tot['err'] += float(err.sum())
        for dims in ([3], [2]):
            p_f = up(prob(model, x, dims))
            d = (p_f - p_o).abs()
            tot['dis'] += float(d.sum())
            tot['dis_err'] += float((d * err).sum())
            not_closer = ((p_o - y).abs() >= (p_f - y).abs()).float()
            tot['dis_rel'] += float((d * not_closer).sum())
    return {
        'err_pixels': tot['err'] / tot['pix'],
        'gate3_share': tot['dis_err'] / max(tot['dis'], 1e-12),
        'gate5_share': tot['dis_rel'] / max(tot['dis'], 1e-12),
        'n': loader.size,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pth_path', type=str, nargs='+', required=True)
    parser.add_argument('--train_path', type=str, default='./dataset/TrainDataset/')
    parser.add_argument('--testsize', type=int, default=352)
    opt = parser.parse_args()

    print('{:<28s} {:>6s} {:>11s} {:>12s} {:>12s}'.format('checkpoint', 'n', 'err_pixels', 'gate3_share', 'gate5_share'))
    for p in opt.pth_path:
        ck = resolve_checkpoint(p)
        model = PolypPVT()
        model.load_state_dict(torch.load(ck, map_location='cpu'))
        model.cuda().eval()
        r = shares(model, os.path.join(opt.train_path, 'images/'), os.path.join(opt.train_path, 'masks/'), opt.testsize)
        name = os.path.basename(os.path.normpath(p))
        print('{:<28s} {:>6d} {:>11.4f} {:>12.4f} {:>12.4f}'.format(name, r['n'], r['err_pixels'], r['gate3_share'], r['gate5_share']))
        del model
        torch.cuda.empty_cache()
