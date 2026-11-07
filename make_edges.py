"""Edge labels for ASGNet (--host asgnet), made from the training masks.

ASGNet's loader reads one edge image per training image from <train_path>/edges/. They
are made here with the recipe of its utils/edge_util.py: the square root of the mask's
distance transform, set to 255 beyond 2, is subtracted from the mask, which leaves a band
about four pixels wide inside the boundary, and the result is blurred 3x3. Its loader
then dilates them and thresholds them at 0.

  python make_edges.py                  # dataset/TrainDataset/masks -> dataset/TrainDataset/edges
"""
import os
import argparse

import cv2
import numpy as np


def edge_of(mask):
    """The edge image of one 8-bit mask, as ASGNet's generate_edge() writes it."""
    body = cv2.distanceTransform(mask, distanceType=cv2.DIST_L2, maskSize=5) ** 0.5
    if (body > 0).any():
        body[body > 2] = 255
    res = cv2.blur(mask - body, (3, 3))
    # generate_edge() hands this float image to cv2.imwrite, which rounds it to 8 bits
    return np.clip(np.rint(res), 0, 255).astype(np.uint8)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--masks', type=str, default='./dataset/TrainDataset/masks/')
    parser.add_argument('--out', type=str, default='./dataset/TrainDataset/edges/')
    opt = parser.parse_args()
    os.makedirs(opt.out, exist_ok=True)
    names = sorted(f for f in os.listdir(opt.masks) if f.lower().endswith(('.png', '.jpg')))
    for name in names:
        mask = cv2.imread(os.path.join(opt.masks, name), cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(os.path.join(opt.out, name), edge_of(mask))
    print('{} edge maps written to {}'.format(len(names), opt.out))
