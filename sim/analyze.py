"""Precise pixel-level blob report for one observation index.

Red block centroid/bbox in top and side views; dark gripper blob centroid in
the top view (arm region). Deterministic thresholding, outputs cell coords
(448px/8) with one decimal. Usage: analyze.py <run_dir> <index>
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image

run, idx = Path(sys.argv[1]), int(sys.argv[2])


def red_mask(a):
    return (a[:, :, 0] > 140) & (a[:, :, 1] < 95) & (a[:, :, 2] < 95)


def report(cam, dark_box=None):
    a = np.asarray(Image.open(run / f'observation_{idx:03d}_{cam}.png').convert('RGB'), dtype=np.float32)
    m = red_mask(a)
    print(f'[{cam}] red: ', end='')
    if m.sum() < 20:
        print('none')
    else:
        ys, xs = np.nonzero(m)
        print(f'n={m.sum()} centroid=({xs.mean()/8:.1f},{ys.mean()/8:.1f}) '
              f'bbox=cols {xs.min()/8:.1f}-{xs.max()/8:.1f} rows {ys.min()/8:.1f}-{ys.max()/8:.1f}')
    if dark_box:
        x0, y0, x1, y1 = [int(v * 8) for v in dark_box]
        sub = a[y0:y1, x0:x1]
        v = sub.max(axis=2)
        dark = v < 90
        if dark.sum() > 20:
            ys, xs = np.nonzero(dark)
            print(f'[{cam}] dark-in-box: n={dark.sum()} '
                  f'centroid=({(xs.mean()+x0)/8:.1f},{(ys.mean()+y0)/8:.1f}) '
                  f'bbox=cols {(xs.min()+x0)/8:.1f}-{(xs.max()+x0)/8:.1f} rows {(ys.min()+y0)/8:.1f}-{(ys.max()+y0)/8:.1f}')


report('top', dark_box=(24, 0, 44, 28))
report('side')
