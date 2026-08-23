"""
Download SeaDronesSee OD v2 images from Ultralytics HUB NDJSON export and
produce YOLO labels + a COCO-style val JSON for run_eval.py.

NDJSON format (one JSON object per line):
  line 0  : dataset header (type=dataset, class_names, …)
  lines 1+: image entries (type=image, file, url, width, height, split,
             annotations={boxes:[[cls,cx,cy,w,h],…]})

Outputs written to --out-dir (default: data/seadronessee):
  images/train/*.jpg
  images/val/*.jpg
  images/test/*.jpg          (no labels — held-out)
  labels/train/*.txt         (binary YOLO: 0=person, 1=boat)
  labels/val/*.txt
  annotations/instances_val.json    (COCO-style, for run_eval.py)
  annotations/instances_train.json  (COCO-style, for convert_seadronessee.py)
  data.yaml                  (Ultralytics training config)

Binary class mapping (NDJSON id → YOLO):
  0 ignored              → SKIP
  1 swimmer              → 0 (person)
  2 boat                 → 1 (boat)
  3 jetski               → 1 (boat)
  4 life_saving_appliances → 0 (person)
  5 buoy                 → 0 (person)

Usage:
  python data/download_from_ndjson.py \
      --ndjson seadronessee-object-detection-v2-dataset.ndjson \
      --out-dir data/seadronessee \
      --workers 16
"""

import argparse
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

# NDJSON class id → binary YOLO class (-1 = skip)
_NDJSON_TO_BINARY = {0: -1, 1: 0, 2: 1, 3: 1, 4: 0, 5: 0}
_BINARY_NAMES = ["person", "boat"]

# COCO category list (binary)
_COCO_CATS = [{"id": 1, "name": "person", "supercategory": "person"},
              {"id": 2, "name": "boat",   "supercategory": "vehicle"}]
# YOLO class → COCO category id
_YOLO_TO_COCO_CAT = {0: 1, 1: 2}


def parse_ndjson(path):
    """Return (header_dict, [image_dicts])."""
    header = None
    images = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d["type"] == "dataset":
                header = d
            elif d["type"] == "image":
                images.append(d)
    return header, images


def _download_one(args):
    url, dst = args
    if os.path.exists(dst):
        return dst, True, "cached"
    try:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        tmp = dst + ".tmp"
        urllib.request.urlretrieve(url, tmp)
        os.rename(tmp, dst)
        return dst, True, "ok"
    except Exception as e:
        return dst, False, str(e)


def download_images(images, out_dir, workers=16, splits=None):
    """Download images for the given splits. splits=None → all."""
    tasks = []
    for im in images:
        if splits and im["split"] not in splits:
            continue
        dst = os.path.join(out_dir, "images", im["split"], im["file"])
        tasks.append((im["url"], dst))

    total = len(tasks)
    print(f"[download] {total} images to download (workers={workers})")
    done, failed, cached = 0, 0, 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_download_one, t): t for t in tasks}
        for fut in as_completed(futs):
            dst, ok, msg = fut.result()
            done += 1
            if msg == "cached":
                cached += 1
            elif ok:
                pass
            else:
                failed += 1
                print(f"  FAIL {dst}: {msg}", file=sys.stderr)
            if done % 500 == 0 or done == total:
                elapsed = time.time() - t0
                rate = done / max(elapsed, 1)
                eta = (total - done) / max(rate, 0.01)
                print(f"[download] {done}/{total}  {rate:.1f} img/s  "
                      f"ETA {eta/60:.0f}m  (cached={cached} fail={failed})")
    print(f"[download] done in {(time.time()-t0)/60:.1f} min  "
          f"fail={failed}")
    return failed


def write_labels(images, out_dir, scheme="binary"):
    """Write YOLO .txt label files for train and val."""
    mapping = _NDJSON_TO_BINARY  # only binary supported here
    n_files, n_boxes = 0, 0
    for im in images:
        if im["split"] not in ("train", "val"):
            continue
        stem = os.path.splitext(im["file"])[0]
        ldir = os.path.join(out_dir, "labels", im["split"])
        os.makedirs(ldir, exist_ok=True)
        boxes = (im.get("annotations") or {}).get("boxes") or []
        lines = []
        for box in boxes:
            cls_orig, cx, cy, w, h = box
            cls_bin = mapping.get(int(cls_orig), -1)
            if cls_bin < 0:
                continue
            lines.append(f"{cls_bin} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            n_boxes += 1
        with open(os.path.join(ldir, stem + ".txt"), "w") as f:
            f.write("\n".join(lines))
        n_files += 1
    print(f"[labels] wrote {n_files} label files, {n_boxes} boxes")


def build_coco_json(images, split):
    """Build a minimal COCO-style dict from NDJSON image list for one split.

    Image IDs are taken from the filename (e.g. '3464.jpg' -> id 3464) so
    they stay aligned with the official SeaDronesSee COCO JSON, making
    altitude merging and leaderboard submission trivial.
    """
    mapping = _NDJSON_TO_BINARY
    coco_images = []
    coco_anns = []
    ann_id = 1
    for im in (img for img in images if img["split"] == split):
        img_id = int(os.path.splitext(im["file"])[0])
        coco_images.append({
            "id": img_id,
            "file_name": im["file"],
            "width": im["width"],
            "height": im["height"],
        })
        boxes = (im.get("annotations") or {}).get("boxes") or []
        for box in boxes:
            cls_orig, cx, cy, w_n, h_n = box
            cls_bin = mapping.get(int(cls_orig), -1)
            if cls_bin < 0:
                continue
            # YOLO normalised → COCO absolute xywh
            abs_w = w_n * im["width"]
            abs_h = h_n * im["height"]
            abs_x = cx * im["width"] - abs_w / 2
            abs_y = cy * im["height"] - abs_h / 2
            coco_anns.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": _YOLO_TO_COCO_CAT[cls_bin],
                "bbox": [round(abs_x, 2), round(abs_y, 2),
                         round(abs_w, 2), round(abs_h, 2)],
                "area": round(abs_w * abs_h, 2),
                "iscrowd": 0,
            })
            ann_id += 1
    return {"images": coco_images, "annotations": coco_anns,
            "categories": _COCO_CATS}


def write_data_yaml(out_dir):
    sds = os.path.abspath(out_dir)
    yaml = (f"path: {sds}\n"
            f"train: images/train\n"
            f"val: images/val\n"
            f"nc: 2\n"
            f"names: {{0: person, 1: boat}}\n")
    p = os.path.join(out_dir, "data.yaml")
    with open(p, "w") as f:
        f.write(yaml)
    print(f"[yaml] wrote {p}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ndjson", required=True,
                   help="path to the Ultralytics HUB .ndjson export file")
    p.add_argument("--out-dir", default="data/seadronessee",
                   help="root output dir (default: data/seadronessee)")
    p.add_argument("--workers", type=int, default=16,
                   help="parallel download workers (default: 16)")
    p.add_argument("--splits", default="train,val,test",
                   help="comma-separated splits to download (default: train,val,test)")
    p.add_argument("--labels-only", action="store_true",
                   help="skip image download, only write labels + COCO JSON")
    p.add_argument("--no-coco-json", action="store_true",
                   help="skip COCO JSON generation")
    a = p.parse_args()

    splits = set(a.splits.split(","))

    print(f"[ndjson] parsing {a.ndjson} …")
    header, images = parse_ndjson(a.ndjson)
    by_split = {}
    for im in images:
        by_split.setdefault(im["split"], []).append(im)
    print(f"[ndjson] {len(images)} images: "
          + ", ".join(f"{s}={len(v)}" for s, v in sorted(by_split.items())))

    os.makedirs(a.out_dir, exist_ok=True)

    if not a.labels_only:
        download_images(images, a.out_dir, workers=a.workers, splits=splits)

    write_labels(images, a.out_dir)

    if not a.no_coco_json:
        ann_dir = os.path.join(a.out_dir, "annotations")
        os.makedirs(ann_dir, exist_ok=True)
        for split in ("train", "val"):
            coco = build_coco_json(images, split)
            out = os.path.join(ann_dir, f"instances_{split}.json")
            with open(out, "w") as f:
                json.dump(coco, f)
            n_im = len(coco["images"])
            n_ann = len(coco["annotations"])
            print(f"[coco] {out}  ({n_im} images, {n_ann} annotations)")

    write_data_yaml(a.out_dir)
    print("[done] all outputs written to", a.out_dir)


if __name__ == "__main__":
    main()
