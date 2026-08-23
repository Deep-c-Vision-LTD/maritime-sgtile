"""
Slice a YOLO-format dataset (full frames + .txt labels) into training tiles.

Standard SAHI training recipe: detectors learn small objects far better when
trained on crops at native resolution than on whole 4K frames downscaled to 640.
This produces an Ultralytics-ready tiled dataset:

    <out>/images/<split>/<stem>_<tx>_<ty>.jpg
    <out>/labels/<split>/<stem>_<tx>_<ty>.txt
    <out>/data.yaml

A tile keeps a box if the box centre falls inside it; the box is clipped to the
tile and re-normalised. Empty tiles are kept with probability `neg_ratio` so the
detector also learns what empty sea looks like (precision).

Needs cv2. Run offline (or as a SageMaker processing step).
"""

import argparse
import glob
import os
import random

import cv2
import numpy as np

from sys import path as _p
_p.insert(0, os.path.join(os.path.dirname(__file__), "..", "tiling"))
from geometry import generate_tiles  # noqa


def _read_yolo_label(path):
    if not os.path.exists(path):
        return np.zeros((0, 5), np.float32)
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 5:
                rows.append([float(x) for x in parts])
    return np.array(rows, np.float32).reshape(-1, 5)


def _slice_one(img, labels, tile, overlap, w, h):
    """Yield (crop, tile_labels_yolo, tx, ty) for each tile."""
    tiles = generate_tiles(w, h, tile, tile, overlap)
    for (x0, y0, x1, y1) in tiles:
        tw, th = x1 - x0, y1 - y0
        crop = img[y0:y1, x0:x1]
        keep = []
        for cls, cx, cy, bw, bh in labels:
            # to absolute
            acx, acy = cx * w, cy * h
            abw, abh = bw * w, bh * h
            if not (x0 <= acx < x1 and y0 <= acy < y1):
                continue  # centre not in tile
            # clip box to tile, re-normalise to tile size
            bx0 = max(acx - abw / 2, x0); by0 = max(acy - abh / 2, y0)
            bx1 = min(acx + abw / 2, x1); by1 = min(acy + abh / 2, y1)
            ncx = ((bx0 + bx1) / 2 - x0) / tw
            ncy = ((by0 + by1) / 2 - y0) / th
            nbw = (bx1 - bx0) / tw
            nbh = (by1 - by0) / th
            if nbw > 0 and nbh > 0:
                keep.append([int(cls), ncx, ncy, nbw, nbh])
        yield crop, keep, x0, y0


def slice_split(images_dir, labels_dir, out_dir, split, tile=640, overlap=0.2,
                neg_ratio=0.1, seed=0):
    rng = random.Random(seed)
    out_img = os.path.join(out_dir, "images", split)
    out_lbl = os.path.join(out_dir, "labels", split)
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_lbl, exist_ok=True)

    img_paths = [p for p in sorted(glob.glob(os.path.join(images_dir, "*")))
                 if p.lower().endswith((".jpg", ".jpeg", ".png"))]
    n_tiles, n_pos = 0, 0
    for ip in img_paths:
        img = cv2.imread(ip)
        if img is None:
            continue
        h, w = img.shape[:2]
        stem = os.path.splitext(os.path.basename(ip))[0]
        labels = _read_yolo_label(os.path.join(labels_dir, stem + ".txt"))
        for crop, keep, tx, ty in _slice_one(img, labels, tile, overlap, w, h):
            if not keep and rng.random() > neg_ratio:
                continue  # drop most empty tiles
            name = f"{stem}_{tx}_{ty}"
            cv2.imwrite(os.path.join(out_img, name + ".jpg"), crop)
            with open(os.path.join(out_lbl, name + ".txt"), "w") as f:
                f.write("\n".join(f"{c} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}"
                                  for c, x, y, bw, bh in keep))
            n_tiles += 1
            n_pos += 1 if keep else 0
    print(f"[slice] {split}: {len(img_paths)} frames -> {n_tiles} tiles "
          f"({n_pos} with objects)")
    return n_tiles


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--images-dir", required=True)
    p.add_argument("--labels-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--tile", type=int, default=640)
    p.add_argument("--overlap", type=float, default=0.2)
    p.add_argument("--neg-ratio", type=float, default=0.1)
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    slice_split(a.images_dir, a.labels_dir, a.out_dir, a.split,
                a.tile, a.overlap, a.neg_ratio)
