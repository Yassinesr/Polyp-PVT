"""Turn PolypGen's single-frame data into the test-set layout this repo uses.

PolypGen (Ali et al., Scientific Data 2023) holds polyp frames from six centres. Its single-frame
data is split by centre into data_C1 ... data_C6 (written data-C1 in the paper), each with a folder
of images, a folder of masks (both .jpg), bounding boxes and box overlays. This script finds the
image and mask folders of each centre by name, pairs every image with its mask, and writes

    <dst>/PolypGen_C1/images/<name>.jpg     the image (copied; .png kept as .png)
    <dst>/PolypGen_C1/masks/<name>.png      the mask, binarised at > 128 and saved losslessly
    ...                                     one folder per centre, C1 to C6

which is what Test_tta.py, eval_dice.py and seed_summary.py read. Sequence frames, negative
(polyp-free) frames, boxes and overlays are left out. A frame whose mask is empty is skipped too:
Dice is undefined without a polyp, and the MATLAB scorer would count it as 0.

The pairing is by file name: a mask named <image name>_mask is matched to <image name>. Every
count is printed (paired, skipped, unmatched), so check them against the dataset's own numbers.

Usage:
  python prep_polypgen.py --src /path/to/PolypGen --dst ./dataset/PolypGen
"""
import os
import re
import shutil
import argparse

import numpy as np
from PIL import Image

EXTS = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')
CENTRE = re.compile(r'(?:^|[_\-\s])c([1-6])(?:$|[_\-\s])', re.I)
SKIP = re.compile(r'seq|negative|bbox|box|overlay', re.I)


def classify(rel_dir):
    """(centre, 'image' | 'mask') for a folder path relative to --src, or None."""
    parts = rel_dir.split(os.sep)
    if any(SKIP.search(p) for p in parts):
        return None
    leaf = parts[-1].lower()
    kind = 'image' if leaf.startswith('image') else 'mask' if leaf.startswith('mask') else None
    if kind is None:
        return None
    for p in reversed(parts):
        m = CENTRE.search(p)
        if m:
            return 'C' + m.group(1), kind
    return None


def key(stem, kind):
    return re.sub(r'[_\-]?mask$', '', stem, flags=re.I) if kind == 'mask' else stem


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, required=True, help='the extracted PolypGen folder')
    parser.add_argument('--dst', type=str, default='./dataset/PolypGen')
    opt = parser.parse_args()

    found = {}                                      # (centre, kind) -> {key: path}
    dupes = 0
    for root, _, files in os.walk(opt.src):
        c = classify(os.path.relpath(root, opt.src))
        if c is None:
            continue
        for f in files:
            stem, ext = os.path.splitext(f)
            if ext.lower() not in EXTS:
                continue
            d = found.setdefault(c, {})
            k = key(stem, c[1])
            dupes += k in d
            d[k] = os.path.join(root, f)
    if not found:
        raise SystemExit('no image or mask folders found under {}'.format(opt.src))
    if dupes:
        print('!! {} file names appear in more than one folder of the same centre; the last one '
              'found was kept'.format(dupes))

    total = 0
    for centre in sorted({c for c, _ in found}):
        imgs, masks = found.get((centre, 'image'), {}), found.get((centre, 'mask'), {})
        out = os.path.join(opt.dst, 'PolypGen_' + centre)
        os.makedirs(os.path.join(out, 'images'), exist_ok=True)
        os.makedirs(os.path.join(out, 'masks'), exist_ok=True)
        kept = empty = resized = 0
        for k in sorted(set(imgs) & set(masks)):
            m = np.asarray(Image.open(masks[k]).convert('L')) > 128
            if not m.any():
                empty += 1
                continue
            with Image.open(imgs[k]) as im:
                if im.size != (m.shape[1], m.shape[0]):
                    resized += 1
            ext = os.path.splitext(imgs[k])[1].lower()
            if ext in ('.jpg', '.jpeg'):
                shutil.copyfile(imgs[k], os.path.join(out, 'images', k + '.jpg'))
            elif ext == '.png':
                shutil.copyfile(imgs[k], os.path.join(out, 'images', k + '.png'))
            else:
                Image.open(imgs[k]).convert('RGB').save(os.path.join(out, 'images', k + '.png'))
            Image.fromarray((m * 255).astype(np.uint8)).save(os.path.join(out, 'masks', k + '.png'))
            kept += 1
        total += kept
        print('{}: {} images, {} masks -> {} pairs kept, {} empty masks skipped, {} images without a '
              'mask, {} masks without an image{}'.format(
                  centre, len(imgs), len(masks), kept, empty, len(set(imgs) - set(masks)),
                  len(set(masks) - set(imgs)),
                  ', {} image/mask size mismatches'.format(resized) if resized else ''))
    print('total pairs: {}  ->  {}'.format(total, opt.dst))
