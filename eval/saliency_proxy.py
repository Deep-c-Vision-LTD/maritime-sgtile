"""
Saliency as a tile-level target proxy (manuscript Sec. III-D).

For every validation frame, generate the candidate tiles exactly as the detector
pipeline does (tiling/geometry.py), score each tile by its mean saliency, and
treat that score as a predictor of whether the tile contains a target. Reports,
for the generic MSI-Net prior and the fine-tuned xSoyelNet prior:

  * ROC-AUC of the tile score against tile target presence (pooled over frames),
  * at a selection threshold tau (Eq. 3: score / frame max >= tau, at least one
    tile kept): the fraction of annotated targets retained and of tiles discarded.

A tile "contains a target" when a ground-truth box centre lies inside it; a
target is "retained" when its centre lies inside at least one kept tile. Both
candidate sets are reported: fixed 640 px tiles (the static-SAHI candidates) and
altitude-sized tiles (the S1 candidates, Eq. 2).

Usage:
  python eval/saliency_proxy.py \
      --val-coco-json data/seadronessee/annotations/instances_val.json \
      --generic-dir data/seadronessee/saliency/val \
      --xsoyel-dir  data/seadronessee/saliency_sar/val \
      --taus 0.05,0.15 --out results/saliency_target_auc.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tiling.geometry import (generate_tiles, altitude_tile_size,  # noqa: E402
                             tile_saliency_scores, select_salient_tiles)


def _altitude_of(im):
    """Same lookup as eval/run_eval.py."""
    for k in ("altitude", "meta_altitude"):
        if k in im and im[k]:
            return float(im[k])
    meta = im.get("meta") or im.get("metadata") or {}
    for k in ("altitude", "alt", "height"):
        if k in meta and meta[k]:
            return float(meta[k])
    return None


def _auc(scores, labels):
    from sklearn.metrics import roc_auc_score
    labels = np.asarray(labels)
    if labels.min() == labels.max():
        return float("nan")
    return float(roc_auc_score(labels, np.asarray(scores)))


def main():
    import cv2
    p = argparse.ArgumentParser()
    p.add_argument("--val-coco-json", required=True)
    p.add_argument("--generic-dir", required=True)
    p.add_argument("--xsoyel-dir", required=True)
    p.add_argument("--taus", default="0.05,0.15")
    p.add_argument("--tile", type=int, default=640)
    p.add_argument("--overlap", type=float, default=0.2)
    p.add_argument("--out", default="results/saliency_target_auc.json")
    a = p.parse_args()
    taus = [float(t) for t in a.taus.split(",")]

    coco = json.load(open(a.val_coco_json))
    centres = {im["id"]: [] for im in coco["images"]}
    for an in coco["annotations"]:
        if an.get("iscrowd", 0):
            continue
        x, y, w, h = an["bbox"]
        centres[an["image_id"]].append((x + w / 2.0, y + h / 2.0))

    priors = {"generic": a.generic_dir, "xsoyelnet": a.xsoyel_dir}
    tilings = ("fixed640", "altitude")
    # pooled tile scores / labels, and selection bookkeeping
    acc = {(pr, tl): dict(s=[], y=[], kept={t: 0 for t in taus},
                          cand=0, tgt=0, ret={t: 0 for t in taus},
                          frac_kept_frames={t: [] for t in taus})
           for pr in priors for tl in tilings}
    n_frames = 0
    for im in coco["images"]:
        stem = os.path.splitext(os.path.basename(im["file_name"]))[0]
        maps = {}
        for pr, d in priors.items():
            m = cv2.imread(os.path.join(d, stem + ".png"), cv2.IMREAD_GRAYSCALE)
            if m is None:
                break
            if m.shape != (im["height"], im["width"]):
                m = cv2.resize(m, (im["width"], im["height"]))
            maps[pr] = m.astype(np.float32) / 255.0
        if len(maps) != len(priors):
            continue
        n_frames += 1
        W, H = im["width"], im["height"]
        alt = _altitude_of(im)
        t_alt = altitude_tile_size(alt, base_tile=a.tile) if alt else a.tile
        cands = {"fixed640": generate_tiles(W, H, a.tile, a.tile, a.overlap),
                 "altitude": generate_tiles(W, H, t_alt, t_alt, a.overlap)}
        cs = centres[im["id"]]
        for tl, tiles in cands.items():
            T = np.asarray(tiles, np.float32)
            if cs:
                C = np.asarray(cs, np.float32)
                # inside[i, j]: centre j lies in tile i
                inside = ((C[None, :, 0] >= T[:, None, 0]) & (C[None, :, 0] < T[:, None, 2]) &
                          (C[None, :, 1] >= T[:, None, 1]) & (C[None, :, 1] < T[:, None, 3]))
            else:
                inside = np.zeros((len(tiles), 0), bool)
            y = inside.any(axis=1).astype(int)
            for pr in priors:
                sc = tile_saliency_scores(maps[pr], tiles)
                d = acc[(pr, tl)]
                d["s"].extend(sc.tolist()); d["y"].extend(y.tolist())
                d["cand"] += len(tiles); d["tgt"] += inside.shape[1]
                for t in taus:
                    k = select_salient_tiles(tiles, sc, keep_thr=t, min_keep=1)
                    d["kept"][t] += len(k)
                    d["frac_kept_frames"][t].append(len(k) / len(tiles))
                    if inside.shape[1]:
                        d["ret"][t] += int(inside[k].any(axis=0).sum())

    out = {"n_frames": n_frames,
           "definition": {
               "tile_score": "mean saliency inside the tile (tiling/geometry.tile_saliency_scores)",
               "tile_label": "1 if a ground-truth box centre lies inside the tile",
               "selection": "keep tiles with score/frame-max >= tau, at least one kept (Eq. 3)",
               "retention": "fraction of annotated targets whose centre lies in a kept tile",
               "discarded": "1 - kept tiles / candidate tiles, pooled over frames"},
           "results": {}}
    for (pr, tl), d in acc.items():
        r = {"n_tiles": d["cand"], "n_targets": d["tgt"],
             "pos_rate": float(np.mean(d["y"])),
             "roc_auc": _auc(d["s"], d["y"])}
        for t in taus:
            r[f"tau{t:g}"] = {
                "target_retention": d["ret"][t] / max(1, d["tgt"]),
                "tiles_discarded_pooled": 1 - d["kept"][t] / max(1, d["cand"]),
                "tiles_discarded_mean_frame": float(1 - np.mean(d["frac_kept_frames"][t])),
                "mean_tiles_kept": d["kept"][t] / max(1, n_frames),
                "mean_tiles_candidate": d["cand"] / max(1, n_frames)}
        out["results"][f"{pr}/{tl}"] = r
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"[proxy] {n_frames} frames -> {a.out}")
    for key, r in out["results"].items():
        line = f"  {key:22s} AUC {r['roc_auc']:.3f}  pos {r['pos_rate']:.3f}"
        for t in taus:
            q = r[f"tau{t:g}"]
            line += (f" | tau={t:g}: retain {q['target_retention']*100:.1f}% "
                     f"discard {q['tiles_discarded_pooled']*100:.1f}% "
                     f"(kept {q['mean_tiles_kept']:.1f}/{q['mean_tiles_candidate']:.1f})")
        print(line)


if __name__ == "__main__":
    main()
