"""
Wall-clock latency / throughput benchmark for the tiling configurations, so the
compute claim is reported in ms/frame and FPS (comparable to prior-art FPS), not
just mean tiles processed.

Times detect() end-to-end (tiling + all forward passes + merge) per frame on the
same device, after a warm-up, over a fixed subset of val frames:
  A  : static SAHI (all tiles)              -- TiledDetector
  B  : altitude tiling, all tiles           -- AdaptiveTiledDetector(alt, no select)
  S1 : altitude + xSoyel-Net selection      -- AdaptiveTiledDetector(alt, select, keep .05)

Usage:
  python eval/latency_bench.py --weights runs/det/detector_best.pt \
      --val-coco-json data/seadronessee/annotations/instances_val.json \
      --images-dir data/seadronessee/images/val \
      --saliency-dir data/seadronessee/saliency_sar/val \
      --n 100 --device mps --out results/latency.json
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
from tiling.sahi_baseline import TiledDetector
from tiling.adaptive_tiling import AdaptiveTiledDetector


def _alt(im):
    for k in ("altitude", "meta_altitude"):
        if im.get(k):
            return float(im[k])
    meta = im.get("meta") or im.get("metadata") or {}
    for k in ("altitude", "alt", "height"):
        if meta.get(k):
            return float(meta[k])
    return None


def time_config(name, detector, frames, metas, warmup=5):
    import cv2
    # warm-up (compile/caches/MPS graph)
    for i in range(min(warmup, len(frames))):
        detector.detect(frames[i], meta=metas[i]) if metas[i] is not None \
            else detector.detect(frames[i])
    ts = []
    for img, meta in zip(frames, metas):
        t0 = time.perf_counter()
        detector.detect(img, meta=meta) if meta is not None else detector.detect(img)
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts = np.array(ts)
    r = dict(config=name, n=len(ts), ms_mean=float(ts.mean()),
             ms_median=float(np.median(ts)), ms_std=float(ts.std()),
             fps=float(1000.0 / ts.mean()))
    print(f"[lat] {name:4s}  {r['ms_mean']:7.1f} ms/frame  "
          f"({r['fps']:5.2f} FPS)  median {r['ms_median']:.1f}")
    return r


def main():
    import cv2
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--val-coco-json", required=True)
    p.add_argument("--images-dir", required=True)
    p.add_argument("--saliency-dir", required=True)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--tile", type=int, default=640)
    p.add_argument("--overlap", type=float, default=0.2)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--device", default="mps")
    p.add_argument("--out", default="results/latency.json")
    a = p.parse_args()

    coco = json.load(open(a.val_coco_json))
    images = coco["images"][:a.n]

    frames, metas_alt, metas_sel = [], [], []
    for im in images:
        ip = os.path.join(a.images_dir, os.path.basename(im["file_name"]))
        img = cv2.imread(ip)
        if img is None:
            continue
        stem = os.path.splitext(os.path.basename(im["file_name"]))[0]
        sp = os.path.join(a.saliency_dir, stem + ".png")
        sal = cv2.imread(sp, cv2.IMREAD_GRAYSCALE) if os.path.exists(sp) else None
        frames.append(img)
        metas_alt.append({"altitude": _alt(im), "saliency": sal})  # B & S1 share meta
    print(f"[lat] timing {len(frames)} frames on {a.device}")

    base = TiledDetector(a.weights, tile=a.tile, overlap=a.overlap, conf=a.conf,
                         imgsz=a.tile, device=a.device, full_frame_pass=True)
    bdet = AdaptiveTiledDetector(a.weights, base_tile=a.tile, overlap=a.overlap,
                                 conf=a.conf, imgsz=a.tile, device=a.device,
                                 use_altitude=True, use_saliency_select=False,
                                 gate_alpha=0)
    s1 = AdaptiveTiledDetector(a.weights, base_tile=a.tile, overlap=a.overlap,
                               conf=a.conf, imgsz=a.tile, device=a.device,
                               use_altitude=True, use_saliency_select=True,
                               keep_thr=0.05, gate_alpha=0)

    res = []
    res.append(time_config("A",  base, frames, [None] * len(frames)))
    res.append(time_config("B",  bdet, frames, metas_alt))
    res.append(time_config("S1", s1,   frames, metas_alt))

    # speedups vs A
    base_ms = res[0]["ms_mean"]
    for r in res:
        r["speedup_vs_A"] = base_ms / r["ms_mean"]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)
    print(f"[lat] speedups vs A: " +
          ", ".join(f"{r['config']}={r['speedup_vs_A']:.2f}x" for r in res))
    print(f"[lat] wrote {a.out}")


if __name__ == "__main__":
    main()
