"""
Build maritime *target-aware* saliency ground truth from SeaDronesSee boxes.

Generic fixation saliency (SALICON-trained MSI-Net) under-fires on tiny swimmers,
which is why saliency gating (EXPERIMENTS.md row D) suppressed real detections.
This script renders a "target-here" density map per training image — a Gaussian
blob at every person/boat box — to use as the GT for fine-tuning the saliency
model into a SAR-specific prior (Section 5).

For each COCO image we render an H_out x W_out single-channel map (downscaled
from full frame for storage / training speed) where each annotated object
contributes a normalised 2-D Gaussian centred on the box, with sigma tied to the
box size (floored so tiny objects still produce a visible, learnable blob).

Output:
  --out-dir/<stem>.png   uint8 [0..255], one per train image with >=1 object

Usage:
  python saliency/make_target_maps.py \
      --coco-json data/seadronessee/annotations/instances_train.json \
      --out-dir   data/seadronessee/target_maps/train \
      --out-h 288 --out-w 512 --scheme binary
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
from convert_seadronessee import build_category_map  # noqa: E402


def _gaussian_blob(canvas, cx, cy, sigma):
    """Add a unit-peak 2-D Gaussian centred at (cx,cy) into canvas in-place."""
    H, W = canvas.shape
    # 3-sigma support window, clipped to the canvas
    r = int(max(1, round(3 * sigma)))
    x0, x1 = max(0, int(cx) - r), min(W, int(cx) + r + 1)
    y0, y1 = max(0, int(cy) - r), min(H, int(cy) + r + 1)
    if x0 >= x1 or y0 >= y1:
        return
    ys = np.arange(y0, y1)[:, None].astype(np.float32)
    xs = np.arange(x0, x1)[None, :].astype(np.float32)
    g = np.exp(-(((xs - cx) ** 2) + ((ys - cy) ** 2)) / (2.0 * sigma * sigma))
    # accumulate with max so overlapping objects don't blow past 1.0
    canvas[y0:y1, x0:x1] = np.maximum(canvas[y0:y1, x0:x1], g)


def build_maps(coco_json, out_dir, out_h, out_w, scheme,
               sigma_scale=0.5, min_sigma=2.0):
    with open(coco_json) as f:
        coco = json.load(f)
    _, names = build_category_map(coco["categories"], scheme)
    images = {im["id"]: im for im in coco["images"]}

    anns_by_img = {}
    for a in coco["annotations"]:
        if a.get("iscrowd", 0):
            continue
        anns_by_img.setdefault(a["image_id"], []).append(a)

    os.makedirs(out_dir, exist_ok=True)
    import cv2
    n_written, n_objs, n_empty = 0, 0, 0
    ids = list(images)
    for k, img_id in enumerate(ids):
        im = images[img_id]
        W_img, H_img = im["width"], im["height"]
        sx, sy = out_w / float(W_img), out_h / float(H_img)
        canvas = np.zeros((out_h, out_w), np.float32)
        boxes = anns_by_img.get(img_id, [])
        if not boxes:
            n_empty += 1
            continue
        for a in boxes:
            x, y, w, h = a["bbox"]
            cx = (x + w / 2.0) * sx
            cy = (y + h / 2.0) * sy
            # sigma from the (downscaled) box size, floored so tiny objects
            # still leave a learnable footprint
            bw, bh = w * sx, h * sy
            sigma = max(min_sigma, sigma_scale * 0.5 * (bw + bh) / 2.0)
            _gaussian_blob(canvas, cx, cy, sigma)
            n_objs += 1
        # normalise to 0..255 for PNG storage (KL loss renormalises anyway)
        if canvas.max() > 0:
            canvas = canvas / canvas.max()
        png = (canvas * 255.0).astype(np.uint8)
        stem = os.path.splitext(os.path.basename(im["file_name"]))[0]
        cv2.imwrite(os.path.join(out_dir, stem + ".png"), png)
        n_written += 1
        if (k + 1) % 1000 == 0:
            print(f"[target-maps] {k + 1}/{len(ids)}  written={n_written}")

    print(f"[target-maps] done: {n_written} maps, {n_objs} objects, "
          f"{n_empty} images skipped (no objects) → {out_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--coco-json", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--out-h", type=int, default=288)
    p.add_argument("--out-w", type=int, default=512)
    p.add_argument("--scheme", default="binary", choices=["binary", "native"])
    p.add_argument("--sigma-scale", type=float, default=0.5,
                   help="Gaussian sigma as a fraction of box half-size")
    p.add_argument("--min-sigma", type=float, default=2.0,
                   help="floor on sigma (px, in output scale) for tiny objects")
    a = p.parse_args()
    build_maps(a.coco_json, a.out_dir, a.out_h, a.out_w, a.scheme,
               sigma_scale=a.sigma_scale, min_sigma=a.min_sigma)


if __name__ == "__main__":
    main()
