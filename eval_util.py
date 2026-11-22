"""Shared evaluation utilities for Polyp-PVT experiments."""
import os
import numpy as np
import torch
import torch.nn.functional as F
from utils.dataloader import test_dataset

TESTSETS = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']


def score_one_set(model, host, data_root, dataset_name, testsize=352):
    """Score one test set and return mean Dice.

    Args:
        model: PyTorch model in eval mode
        host: Host object with .heads() and .logit() methods
        data_root: Root directory containing dataset_name/
        dataset_name: Name of the test set (e.g., 'Kvasir')
        testsize: Input size for the model (default 352)

    Returns:
        float: Mean Dice score over all images in the set
    """
    data_path = os.path.join(data_root, dataset_name)
    image_root = f'{data_path}/images/'
    gt_root = f'{data_path}/masks/'
    if not os.path.isdir(gt_root):
        gt_root = f'{data_path}/gt/'

    num_images = len(os.listdir(gt_root))
    test_loader = test_dataset(image_root, gt_root, testsize)
    dice_sum = 0.0

    for _ in range(num_images):
        image, gt, _ = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        with torch.no_grad():
            z = host.logit(host.heads(model(image)))
            res = F.upsample(z, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()

        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        target = np.array(gt)
        smooth = 1
        input_flat = np.reshape(res, (-1))
        target_flat = np.reshape(target, (-1))
        intersection = (input_flat * target_flat)
        dice = (2 * intersection.sum() + smooth) / (res.sum() + target.sum() + smooth)
        dice_sum += dice

    return dice_sum / num_images


def score_five_sets(model, host, test_root, testsize=352):
    """Score all five benchmark sets and return per-set dict and mean.

    Args:
        model: PyTorch model (will be put in eval mode)
        host: Host object with .heads() and .logit() methods
        test_root: Root directory containing the five test sets
        testsize: Input size for the model (default 352)

    Returns:
        tuple: (dict of {set_name: score}, mean_score)
    """
    model.eval()
    scores = {}
    for name in TESTSETS:
        scores[name] = score_one_set(model, host, test_root, name, testsize)
    mean_score = sum(scores.values()) / len(scores)
    return scores, mean_score
