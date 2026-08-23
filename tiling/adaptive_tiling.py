"""
Saliency- and altitude-guided adaptive tiled detector — the NOVEL contribution.

Three additions on top of the static SAHI baseline (sahi_baseline.TiledDetector):

  1. ALTITUDE-AWARE tile sizing — tile side is chosen from the frame's UAV altitude
     (geometry.altitude_tile_size). Higher altitude -> smaller tiles -> tiny objects
     magnified after the detector's input resize.

  2. SALIENCY-GUIDED tile selection — the MSI-Net saliency map ranks tiles by how
     much structure (vs empty water) they contain; only salient tiles are run through
     the detector. This cuts compute AND removes the empty-sea tiles that are the main
     source of clutter false positives.

  3. SALIENCY-GATED confidence — surviving detections are re-weighted by the local
     saliency under each box (geometry.saliency_gate_scores). Detections on flat,
     low-saliency water (glint/whitecaps/wakes) are pushed below threshold; detections
     on salient structure are preserved. This is the precision / FP-rate lever that the
     existing tiling literature does not address.

Each component can be toggled independently, so the ablation is clean:
    use_altitude, use_saliency_select, gate_alpha (0 disables gating).

`meta` per frame:  {"altitude": <float m or None>, "saliency": <HxW uint8/float map>}
The saliency map is produced offline by data/gen_saliency.py using the MSI-Net SavedModel.
"""

import numpy as np

from .sahi_baseline import TiledDetector
from .geometry import (generate_tiles, altitude_tile_size, tile_saliency_scores,
                       select_salient_tiles, saliency_at_boxes, saliency_gate_scores)


class AdaptiveTiledDetector(TiledDetector):
    def __init__(self, weights, base_tile=640, overlap=0.2, conf=0.25,
                 iou_merge=0.5, imgsz=640, device=None, full_frame_pass=True,
                 use_altitude=True, use_saliency_select=True,
                 keep_thr=0.15, gate_alpha=0.5, gate_mode="mult", gate_floor=0.1,
                 ref_alt=100.0, min_tile=320, max_tile=1280, final_conf=None,
                 backend="yolo"):
        super().__init__(weights, tile=base_tile, overlap=overlap, conf=conf,
                         iou_merge=iou_merge, imgsz=imgsz, device=device,
                         full_frame_pass=full_frame_pass, backend=backend)
        self.base_tile = base_tile
        self.use_altitude = use_altitude
        self.use_saliency_select = use_saliency_select
        self.keep_thr = keep_thr          # saliency tile-selection threshold
        self.gate_alpha = gate_alpha      # 0 = no gating
        self.gate_mode = gate_mode        # "mult" (symmetric) or "asym" (clutter-only)
        self.gate_floor = gate_floor      # asym mode: saliency below this = clutter
        self.ref_alt = ref_alt
        self.min_tile = min_tile
        self.max_tile = max_tile
        # confidence applied AFTER saliency gating (so gating can rescue precision
        # without throwing away recall at the per-tile stage). Defaults to `conf`.
        self.final_conf = conf if final_conf is None else final_conf

    def _takes_meta(self):
        return True

    def _tiles_for(self, w, h, meta=None):
        meta = meta or {}
        # 1. altitude -> tile size
        if self.use_altitude and meta.get("altitude"):
            t = altitude_tile_size(meta["altitude"], base_tile=self.base_tile,
                                   ref_alt=self.ref_alt, min_tile=self.min_tile,
                                   max_tile=self.max_tile)
        else:
            t = self.base_tile

        cand = generate_tiles(w, h, t, t, self.overlap)
        stats = {"tile_size": t, "n_candidate": len(cand)}

        # 2. saliency -> keep only salient tiles
        sal = meta.get("saliency")
        if self.use_saliency_select and sal is not None and len(cand) > 1:
            scores = tile_saliency_scores(sal, cand)
            keep = select_salient_tiles(cand, scores, keep_thr=self.keep_thr,
                                        min_keep=1)
            cand = [cand[i] for i in keep]
        stats["n_processed"] = len(cand)
        # compute saving vs processing every candidate tile
        stats["tile_reduction"] = (1.0 - stats["n_processed"] /
                                   max(1, stats["n_candidate"]))
        return cand, stats

    def _postprocess(self, B, S, C, image, meta, stats):
        # 3. saliency-gated confidence -> suppress clutter FPs
        meta = meta or {}
        sal = meta.get("saliency")
        if self.gate_alpha > 0 and sal is not None and len(B):
            sal_box = saliency_at_boxes(sal, B)
            S = saliency_gate_scores(S, sal_box, alpha=self.gate_alpha,
                                     mode=self.gate_mode, floor=self.gate_floor)
        # final confidence threshold after gating
        if len(B):
            keep = S >= self.final_conf
            B, S, C = B[keep], S[keep], C[keep]
        return B, S, C, stats
