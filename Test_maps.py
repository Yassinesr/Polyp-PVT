"""Save the prediction maps of a trained model on the test sets, for eval_dice.py.

The prediction is sigmoid of the host's logit map (P1 + P2 for Polyp-PVT), resized to the
image and min-max normalised per image, as Test.py does.

Usage:
  python Test_maps.py --pth_path ./model_pth/PVT_B --save_root ./result_map/PVT_B

--pth_path takes either a .pth file or the --train_save DIRECTORY you trained
into. Given a directory it prefers PolypPVT.pth (best), then last.pth, then the
highest-numbered per-epoch file.

For a checkpoint of another model trained with Train_noaug.py --host, pass the
same --host and --host_root; the prediction is then that paper's (see hosts.py).
"""
import os
import re
import glob
import argparse

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from hosts import HOSTS, get_host
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int, default=352)
    parser.add_argument('--pth_path', type=str, default='./model_pth/PolypPVT.pth')
    parser.add_argument('--data_root', type=str, default='./dataset/TestDataset')
    parser.add_argument('--save_root', type=str, default='./result_map/PolypPVT')
    parser.add_argument('--sets', type=str, nargs='+', default=TESTSETS,
                        help='test-set folders under --data_root, each with images/ and masks/')
    parser.add_argument('--host', type=str, default='polyppvt', choices=sorted(HOSTS),
                        help='the model the checkpoint belongs to (see hosts.py)')
    parser.add_argument('--host_root', type=str, default=None,
                        help='your clone of the host\'s code, as for Train_noaug.py')
    opt = parser.parse_args()

    pth = resolve_checkpoint(opt.pth_path)
    host = get_host(opt.host, opt.host_root)
    model = host.build()
    model.load_state_dict(torch.load(pth))
    model.cuda()
    model.eval()
    print('checkpoint:', pth, '| out:', opt.save_root)

    for name in opt.sets:
        data_path = os.path.join(opt.data_root, name)
        save_path = os.path.join(opt.save_root, name)
        os.makedirs(save_path, exist_ok=True)

        image_root = '{}/images/'.format(data_path)
        gt_root = '{}/masks/'.format(data_path)
        if not os.path.isdir(gt_root):           # ValidationDataset keeps its masks in gt/
            gt_root = '{}/gt/'.format(data_path)
        num1 = len(os.listdir(gt_root))
        test_loader = test_dataset(image_root, gt_root, opt.testsize)

        for _ in range(num1):
            image, gt, fname = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()

            with torch.no_grad():
                z = host.logit(host.heads(model(image)))
                r = F.upsample(z, size=gt.shape, mode='bilinear', align_corners=False).sigmoid()
            res = r.data.cpu().numpy().squeeze()

            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            cv2.imwrite(os.path.join(save_path, fname), res * 255)
        print(name, 'Finish!')
