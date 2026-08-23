"""
Headline experiment: STATIC baseline vs SALIENCY+ALTITUDE ADAPTIVE tiling, on the
same trained detector and the same validation frames.

For every val image it:
  * loads the frame (cv2), its altitude (from the COCO meta), and its saliency PNG,
  * runs the static TiledDetector and the AdaptiveTiledDetector,
  * accumulates predictions + per-frame detector stats,
then evaluates both with eval/evaluate.py (mAP, AP_small, FP/frame, precision@recall,
compute) and prints a side-by-side report + writes results.json.

Run on the GPU box / SageMaker after training. Example:
  python run_eval.py \
      --weights detector_best.pt \
      --val-coco-json data/seadronessee/annotations/instances_val.json \
      --images-dir    data/seadronessee/images/val \
      --saliency-dir  data/seadronessee/saliency/val \
      --scheme binary --gate-alpha 0.5 --out results.json
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))

from tiling.sahi_baseline import TiledDetector
from tiling.adaptive_tiling import AdaptiveTiledDetector
from eval.evaluate import evaluate, print_report
from convert_seadronessee import build_category_map


def _altitude_of(im):
    """Best-effort altitude (m) from a SeaDronesSee COCO image entry."""
    for k in ("altitude", "meta_altitude"):
        if k in im and im[k]:
            return float(im[k])
    meta = im.get("meta") or im.get("metadata") or {}
    for k in ("altitude", "alt", "height"):
        if k in meta and meta[k]:
            return float(meta[k])
    return None


def load_gt(coco_json, scheme):
    with open(coco_json) as f:
        coco = json.load(f)
    catid_to_idx, names = build_category_map(coco["categories"], scheme)
    images = {im["id"]: im for im in coco["images"]}
    by_img = {im_id: {"boxes": [], "labels": []} for im_id in images}
    for a in coco["annotations"]:
        if a.get("iscrowd", 0):
            continue
        cls = catid_to_idx.get(a["category_id"])
        if cls is None:
            continue
        x, y, w, h = a["bbox"]
        by_img[a["image_id"]]["boxes"].append([x, y, x + w, y + h])
        by_img[a["image_id"]]["labels"].append(cls)
    return coco, images, by_img, names


def run(args):
    import cv2
    coco, images, gt_by_img, names = load_gt(args.val_coco_json, args.scheme)
    classes = list(range(len(names)))

    base = TiledDetector(args.weights, tile=args.tile, overlap=args.overlap,
                         conf=args.conf, imgsz=args.imgsz, device=args.device, backend=args.backend,
                         full_frame_pass=True)
    adpt = AdaptiveTiledDetector(args.weights, base_tile=args.tile, overlap=args.overlap,
                                 conf=args.conf, imgsz=args.imgsz, device=args.device, backend=args.backend,
                                 use_altitude=args.use_altitude,
                                 use_saliency_select=args.use_saliency_select,
                                 keep_thr=args.keep_thr, gate_alpha=args.gate_alpha,
                                 gate_mode=args.gate_mode, gate_floor=args.gate_floor)
    treat_label = (f"adaptive[alt={args.use_altitude},sel={args.use_saliency_select},"
                   f"gate={args.gate_alpha}]")

    preds_b, preds_a, gts, stats_b, stats_a = [], [], [], [], []
    n_missing_sal = 0
    items = list(images.values())
    if args.limit:
        items = items[:args.limit]

    for i, im in enumerate(items):
        ip = os.path.join(args.images_dir, os.path.basename(im["file_name"]))
        img = cv2.imread(ip)
        if img is None:
            continue
        gt = gt_by_img[im["id"]]
        gts.append({"boxes": np.array(gt["boxes"], np.float32).reshape(-1, 4),
                    "labels": np.array(gt["labels"], np.int64)})

        # baseline
        rb = base.detect(img)
        preds_b.append({"boxes": rb["boxes"], "scores": rb["scores"], "labels": rb["labels"]})
        stats_b.append(rb["stats"])

        # adaptive (needs altitude + saliency)
        stem = os.path.splitext(os.path.basename(im["file_name"]))[0]
        sal_path = os.path.join(args.saliency_dir, stem + ".png")
        if os.path.exists(sal_path):
            sal = cv2.imread(sal_path, cv2.IMREAD_GRAYSCALE)
        else:
            sal = None
            n_missing_sal += 1
        meta = {"altitude": _altitude_of(im), "saliency": sal}
        ra = adpt.detect(img, meta=meta)
        preds_a.append({"boxes": ra["boxes"], "scores": ra["scores"], "labels": ra["labels"]})
        stats_a.append(ra["stats"])

        if (i + 1) % 100 == 0:
            print(f"[eval] {i + 1}/{len(items)}")

    if n_missing_sal:
        print(f"[eval] WARNING: {n_missing_sal}/{len(items)} frames had NO saliency "
              f"map — adaptive tiling silently fell back to full tiling for those. "
              f"Run data/gen_saliency.py for --saliency-dir first.")

    rep_b = evaluate(preds_b, gts, classes, op_conf=args.conf,
                     target_recall=args.target_recall, det_stats=stats_b)
    rep_a = evaluate(preds_a, gts, classes, op_conf=args.conf,
                     target_recall=args.target_recall, det_stats=stats_a)

    print_report(rep_b, "STATIC SAHI baseline")
    print_report(rep_a, treat_label)
    d = rep_a["mAP_50_95"] - rep_b["mAP_50_95"]
    fp_b = rep_b["clutter@op_conf"]["fp_per_frame"]
    fp_a = rep_a["clutter@op_conf"]["fp_per_frame"]
    print(f"\n>>> ΔmAP(50:95) = {d:+.4f} | FP/frame {fp_b:.3f} -> {fp_a:.3f} "
          f"({(fp_b - fp_a) / (fp_b + 1e-9) * 100:+.1f}%)")

    out = {"classes": names, "baseline": rep_b, "adaptive": rep_a,
           "config": vars(args)}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"[eval] wrote {args.out}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--val-coco-json", required=True)
    p.add_argument("--images-dir", required=True)
    p.add_argument("--saliency-dir", required=True)
    p.add_argument("--scheme", default="binary", choices=["binary", "native"])
    p.add_argument("--tile", type=int, default=640)
    p.add_argument("--overlap", type=float, default=0.2)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--keep-thr", type=float, default=0.15)
    p.add_argument("--gate-alpha", type=float, default=0.5,
                   help="saliency FP-gating strength; 0 disables gating")
    p.add_argument("--gate-mode", default="mult", choices=["mult", "asym"],
                   help="'mult' = symmetric (original); 'asym' = clutter-only "
                        "(never penalises detections above --gate-floor)")
    p.add_argument("--gate-floor", type=float, default=0.1,
                   help="asym mode: detections with saliency below this are "
                        "treated as clutter and suppressed")
    # Ablation toggles for the adaptive ('treatment') detector. Turn all three off
    # (--no-altitude --no-saliency-select --gate-alpha 0) to reproduce the baseline.
    # Both spellings work: --use-altitude / --no-altitude (matches EXPERIMENTS.md).
    alt = p.add_mutually_exclusive_group()
    alt.add_argument("--use-altitude", dest="use_altitude", action="store_true",
                     help="altitude-scaled tile sizing (default on)")
    alt.add_argument("--no-altitude", dest="use_altitude", action="store_false",
                     help="disable altitude-scaled tile sizing")
    sel = p.add_mutually_exclusive_group()
    sel.add_argument("--use-saliency-select", dest="use_saliency_select",
                     action="store_true", help="run only salient tiles (default on)")
    sel.add_argument("--no-saliency-select", dest="use_saliency_select",
                     action="store_false", help="process all tiles")
    p.set_defaults(use_altitude=True, use_saliency_select=True)
    p.add_argument("--target-recall", type=float, default=0.8)
    p.add_argument("--device", default=None)
    p.add_argument("--backend", default="yolo", choices=["yolo","rtdetr","frcnn"])
    p.add_argument("--limit", type=int, default=0, help="0 = all val images")
    p.add_argument("--out", default="results.json")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
