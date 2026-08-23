"""
Altitude-stratified robustness slice: does the adaptive tiler's benefit
concentrate at higher altitude (smaller apparent objects, more clutter)?

Runs the static SAHI baseline (A) and the full xSoyel-Net system (S1) on every
val frame that has logged altitude, buckets frames into dataset-defined
altitude tertiles, and reports per-bucket FP/frame and small-object AP for both,
plus the deltas (S1 - A).

Usage:
  python eval/altitude_slice.py --weights runs/det/detector_best.pt \
      --val-coco-json data/seadronessee/annotations/instances_val.json \
      --images-dir data/seadronessee/images/val \
      --saliency-dir data/seadronessee/saliency_sar/val \
      --edges 31,48 --out results/altitude_slice.json
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
from tiling.sahi_baseline import TiledDetector
from tiling.adaptive_tiling import AdaptiveTiledDetector
from eval.evaluate import evaluate
from convert_seadronessee import build_category_map


def _alt(im):
    for k in ("altitude", "meta_altitude"):
        if im.get(k):
            return float(im[k])
    meta = im.get("meta") or im.get("metadata") or {}
    for k in ("altitude", "alt", "height"):
        if meta.get(k):
            return float(meta[k])
    return None


def main():
    import cv2
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--val-coco-json", required=True)
    p.add_argument("--images-dir", required=True)
    p.add_argument("--saliency-dir", required=True)
    p.add_argument("--edges", default="31,48", help="two altitude tertile edges (m)")
    p.add_argument("--scheme", default="binary")
    p.add_argument("--tile", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--keep-thr", type=float, default=0.05)
    p.add_argument("--device", default="mps")
    p.add_argument("--out", default="results/altitude_slice.json")
    a = p.parse_args()

    e1, e2 = [float(x) for x in a.edges.split(",")]
    bands = [("low (<%.0f m)" % e1, 0, e1),
             ("mid (%.0f-%.0f m)" % (e1, e2), e1, e2),
             ("high (>=%.0f m)" % e2, e2, 1e9)]

    coco = json.load(open(a.val_coco_json))
    catid_to_idx, names = build_category_map(coco["categories"], a.scheme)
    classes = list(range(len(names)))
    images = {im["id"]: im for im in coco["images"]}
    gt = {i: {"boxes": [], "labels": []} for i in images}
    for an in coco["annotations"]:
        if an.get("iscrowd", 0):
            continue
        c = catid_to_idx.get(an["category_id"])
        if c is None:
            continue
        x, y, w, h = an["bbox"]
        gt[an["image_id"]]["boxes"].append([x, y, x + w, y + h])
        gt[an["image_id"]]["labels"].append(c)

    base = TiledDetector(a.weights, tile=a.tile, overlap=0.2, conf=a.conf,
                         imgsz=a.tile, device=a.device, full_frame_pass=True)
    s1 = AdaptiveTiledDetector(a.weights, base_tile=a.tile, overlap=0.2,
                               conf=a.conf, imgsz=a.tile, device=a.device,
                               use_altitude=True, use_saliency_select=True,
                               keep_thr=a.keep_thr, gate_alpha=0)

    # accumulators per band: predictions/gts/stats for A and S1
    acc = {b[0]: dict(pa=[], ps=[], g=[], sa=[], ss=[]) for b in bands}
    n_done = 0
    for im in images.values():
        alt = _alt(im)
        if alt is None:
            continue
        band = next((b[0] for b in bands if b[1] <= alt < b[2]), None)
        if band is None:
            continue
        ip = os.path.join(a.images_dir, os.path.basename(im["file_name"]))
        img = cv2.imread(ip)
        if img is None:
            continue
        stem = os.path.splitext(os.path.basename(im["file_name"]))[0]
        sp = os.path.join(a.saliency_dir, stem + ".png")
        sal = cv2.imread(sp, cv2.IMREAD_GRAYSCALE) if os.path.exists(sp) else None

        g = gt[im["id"]]
        gd = {"boxes": np.array(g["boxes"], np.float32).reshape(-1, 4),
              "labels": np.array(g["labels"], np.int64)}
        rb = base.detect(img)
        ra = s1.detect(img, meta={"altitude": alt, "saliency": sal})
        d = acc[band]
        d["g"].append(gd)
        d["pa"].append({"boxes": rb["boxes"], "scores": rb["scores"], "labels": rb["labels"]})
        d["ps"].append({"boxes": ra["boxes"], "scores": ra["scores"], "labels": ra["labels"]})
        d["sa"].append(rb["stats"]); d["ss"].append(ra["stats"])
        n_done += 1
        if n_done % 150 == 0:
            print(f"[alt-slice] {n_done} frames")

    out = {"edges": [e1, e2], "names": names, "bands": {}}
    print(f"\n{'band':18} {'n':>4} | {'FP/fr A':>8} {'FP/fr S1':>9} {'dFP':>7} "
          f"| {'AP_S A':>7} {'AP_S S1':>8} {'dAP_S':>7}")
    for name, lo, hi in bands:
        d = acc[name]
        n = len(d["g"])
        if n == 0:
            continue
        ra = evaluate(d["pa"], d["g"], classes, op_conf=a.conf, det_stats=d["sa"])
        rs = evaluate(d["ps"], d["g"], classes, op_conf=a.conf, det_stats=d["ss"])
        fpa = ra["clutter@op_conf"]["fp_per_frame"]
        fps = rs["clutter@op_conf"]["fp_per_frame"]
        apa, aps = ra["AP_small_50_95"], rs["AP_small_50_95"]
        out["bands"][name] = dict(n=n, fp_A=fpa, fp_S1=fps, dFP=fps - fpa,
                                  APs_A=apa, APs_S1=aps, dAPs=aps - apa,
                                  tiles_A=ra.get("compute", {}).get("mean_tiles_processed", 0),
                                  tiles_S1=rs.get("compute", {}).get("mean_tiles_processed", 0))
        print(f"{name:18} {n:>4} | {fpa:8.2f} {fps:9.2f} {fps-fpa:+7.2f} "
              f"| {apa:7.3f} {aps:8.3f} {aps-apa:+7.3f}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2, default=float)
    print(f"[alt-slice] wrote {a.out}")


if __name__ == "__main__":
    main()
