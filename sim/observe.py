"""Render observation PNGs as classified character grids for the policy to read.

Downsamples each camera view to a coarse grid, classifies each cell by dominant
color, and prints rulers + a color legend. Deterministic; no state access.
Usage: python observe.py <run_dir> <index>   e.g. observe.py logs_glm_session/20260914-204317 3
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image

GRID = 56  # 448/8


def classify(rgb):
    r, g, b = rgb
    v = max(r, g, b)
    sat = v - min(r, g, b)
    if sat < 30:
        if v > 205:
            return 'W'  # bright white table
        if v > 110:
            return 'm'  # mid gray
        if v > 60:
            return 'd'  # dim gray
        return 'K'  # dark: gripper / bowl / shadows
    if r > 140 and g < 95 and b < 95:
        return 'R'  # saturated red: the block
    if r > 150:
        return 'T'  # warm tan background
    return '?'  # other saturated


def render(path):
    img = np.asarray(Image.open(path).convert('RGB'), dtype=np.float32)
    h, w = img.shape[:2]
    k = h // GRID
    small = img[:GRID * k, :GRID * k].reshape(GRID, k, GRID, k, 3).mean(axis=(1, 3))
    chars = [[classify(small[i, j]) for j in range(GRID)] for i in range(GRID)]
    counts = {}
    for row in chars:
        for c in row:
            counts[c] = counts.get(c, 0) + 1
    print(f'== {Path(path).name}  ({w}x{h}, cell={k}px) ==')
    print('    ' + ''.join(str(c // 10 % 10) if c % 10 == 0 else ' ' for c in range(GRID)))
    print('    ' + ''.join(str(c % 10) if c % 10 == 0 else ' ' for c in range(GRID)))
    for i, row in enumerate(chars):
        mark = '->' if i % 10 == 0 else ' |'
        print(f'{i:3d}{mark}' + ''.join(row))
    red = np.argwhere(chars == 'R')
    if len(red):
        ys, xs = red[:, 0], red[:, 1]
        print(f'red cells: rows {ys.min()}-{ys.max()} cols {xs.min()}-{xs.max()} '
              f'centroid=({xs.mean():.1f},{ys.mean():.1f}) n={len(red)}')
    print('legend(counts): ' + ' '.join(f'{c}:{n}' for c, n in sorted(counts.items(), key=lambda kv: -kv[1])))
    print()


if __name__ == '__main__':
    run, idx = Path(sys.argv[1]), int(sys.argv[2])
    for cam in ['top', 'side']:
        p = run / f'observation_{idx:03d}_{cam}.png'
        if p.exists():
            render(p)
