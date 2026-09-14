"""Polyp-PVT inference with 4-view flip test-time augmentation.

Identical to Test.py except that each image is predicted under the four
label-preserving flips -- identity, horizontal, vertical, horizontal+vertical
-- and the four probability maps are averaged after being flipped back.

This is the whole-image counterpart of the patch-side TTA, so the two can be
compared on equal terms. Two details are matched deliberately:

  * The averaged quantity is sigmoid(P1 + P2), which is what Test.py predicts
    with and what Train_noaug.py's consistency term constrains. Averaging any
    other signal would measure something the output never sees.

  * The per-image min-max normalisation that Test.py applies is done AFTER the
    four views are averaged, not per view. Normalising each view first would
    rescale them against four different maxima and the average would no longer
    correspond to any single prediction.

Usage:
  python Test_tta.py --views 1 --pth_path ./model_pth/PVT_B --save_root ./result_map/PVT_B
  python Test_tta.py --views 4 --pth_path ./model_pth/PVT_B --save_root ./result_map/PVT_C

--pth_path takes either a .pth file or the --train_save DIRECTORY you trained
into. Given a directory it prefers PolypPVT.pth (best), then last.pth, then the
highest-numbered per-epoch file. --views 1 is plain inference and --views 4 is
flip TTA, so both passes of the ablation come from one code path.
"""
import os
import re
import glob
import argparse

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from lib.pvt import PolypPVT
from utils.dataloader import test_dataset

TESTSETS = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']


def resolve_checkpoint(path):
    """Accept a .pth file OR a training directory, and pick sensibly.

    Train_noaug.py writes three kinds of checkpoint:
        PolypPVT.pth      best epoch so far -- ONLY written once an epoch has
                          improved on the running best, so it is absent early
                          in a run, or if training died before its first eval
        last.pth          overwritten every epoch; present after epoch 1
        <N>PolypPVT.pth   per-epoch, only with --save_every_epoch 1

    Preference: best -> last -> highest-numbered epoch. When nothing matches,
    the error lists what the directory actually holds, which is quicker to act
    on than a bare FileNotFoundError on a path that was never going to exist.
    """
    if os.path.isfile(path):
        return path

    d = path if os.path.isdir(path) else os.path.dirname(path)
    if not os.path.isdir(d):
        raise FileNotFoundError(
            'No such checkpoint or directory: {}\n'
            'Point --pth_path at the --train_save directory you trained into.'.format(path))

    for cand in ('PolypPVT.pth', 'last.pth'):
        f = os.path.join(d, cand)
        if os.path.isfile(f):
            return f

    epochs = glob.glob(os.path.join(d, '*PolypPVT.pth'))
    if epochs:
        def _n(f):
            m = re.match(r'(\d+)PolypPVT\.pth$', os.path.basename(f))
            return int(m.group(1)) if m else -1
        return max(epochs, key=_n)

    have = sorted(os.listdir(d)) or ['(empty)']
    raise FileNotFoundError(
        'No checkpoint in {}\nFound: {}\n\n'
        'PolypPVT.pth appears only after an epoch improves on the running best, '
        'and last.pth after the first epoch completes. If neither is there, '
        'training has not finished an epoch yet.'.format(d, ', '.join(have)))


@torch.no_grad()
def tta_probability(model, image, out_shape):
    """Mean of sigmoid(P1+P2) over the four flips, each flipped back first."""
    acc = None
    # (dims to flip the input, dims to flip the output back) -- same dims, since
    # every one of these transforms is its own inverse.
    for dims in ([], [3], [2], [2, 3]):
        x = torch.flip(image, dims=dims) if dims else image
        p1, p2 = model(x)
        res = F.upsample(p1 + p2, size=out_shape, mode='bilinear', align_corners=False)
        res = res.sigmoid()
        if dims:
            res = torch.flip(res, dims=dims)
        acc = res if acc is None else acc + res
    return (acc / 4.0).data.cpu().numpy().squeeze()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int, default=352)
    parser.add_argument('--pth_path', type=str, default='./model_pth/PolypPVT.pth')
    parser.add_argument('--data_root', type=str, default='./dataset/TestDataset')
    parser.add_argument('--save_root', type=str, default='./result_map/PolypPVT_TTA')
    parser.add_argument('--views', type=int, default=4, choices=[1, 4],
                        help='4 = flip TTA, 1 = plain inference (for an A/B on one script)')
    opt = parser.parse_args()

    pth = resolve_checkpoint(opt.pth_path)
    model = PolypPVT()
    model.load_state_dict(torch.load(pth))
    model.cuda()
    model.eval()
    print('checkpoint:', pth)
    print('views:', opt.views, '| out:', opt.save_root)

    for name in TESTSETS:
        data_path = os.path.join(opt.data_root, name)
        save_path = os.path.join(opt.save_root, name)
        os.makedirs(save_path, exist_ok=True)

        image_root = '{}/images/'.format(data_path)
        gt_root = '{}/masks/'.format(data_path)
        num1 = len(os.listdir(gt_root))
        test_loader = test_dataset(image_root, gt_root, opt.testsize)

        for _ in range(num1):
            image, gt, fname = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()

            if opt.views == 4:
                res = tta_probability(model, image, gt.shape)
            else:
                with torch.no_grad():
                    p1, p2 = model(image)
                    res = F.upsample(p1 + p2, size=gt.shape, mode='bilinear',
                                     align_corners=False).sigmoid()
                res = res.data.cpu().numpy().squeeze()

            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            cv2.imwrite(os.path.join(save_path, fname), res * 255)
        print(name, 'Finish!')
