"""
Static (SAHI-style) tiled detector — the BASELINE.

Slices each frame into a uniform overlapping grid, runs an Ultralytics YOLO on
every tile, maps detections back to frame coordinates, and merges with class-aware
NMS. This reproduces the standard "Slicing Aided Hyper Inference" recipe that leads
the SeaDronesSee small-object literature, and is the control the adaptive tiler
(adaptive_tiling.py) is compared against.

Ultralytics/torch are imported lazily so this module can be inspected on a host
without them (the geometry it relies on is pure numpy in geometry.py).
"""

import numpy as np

from .geometry import (generate_tiles, offset_boxes, merge_tile_detections)


class TiledDetector:
    def __init__(self, weights, tile=640, overlap=0.2, conf=0.25,
                 iou_merge=0.5, imgsz=640, device=None,
                 full_frame_pass=False, backend="yolo"):
        """
        weights        : path to a trained YOLO .pt
        tile           : tile side in px (static baseline uses one fixed size)
        overlap        : fractional tile overlap (0..1)
        conf           : per-tile confidence threshold
        iou_merge      : IoU threshold for cross-tile NMS merge
        imgsz          : detector input size (tiles are resized to this)
        full_frame_pass: also run one downscaled pass on the whole frame and merge
                         (helps recover large objects spanning tile borders)
        """
        from tiling.detector_backends import load_detector
        self.model = load_detector(weights, backend=backend, device=device)
        self.backend = backend
        self.tile = tile
        self.overlap = overlap
        self.conf = conf
        self.iou_merge = iou_merge
        self.imgsz = imgsz
        self.device = device
        self.full_frame_pass = full_frame_pass

    # -- per-tile inference -------------------------------------------------
    def _infer_crop(self, crop):
        """Run the detector on a single BGR crop -> (boxes_xyxy, scores, labels)."""
        r = self.model.predict(crop, imgsz=self.imgsz, conf=self.conf,
                               device=self.device, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return (np.zeros((0, 4), np.float32),
                    np.zeros((0,), np.float32),
                    np.zeros((0,), np.int64))
        b = r.boxes.xyxy.cpu().numpy().astype(np.float32)
        s = r.boxes.conf.cpu().numpy().astype(np.float32)
        c = r.boxes.cls.cpu().numpy().astype(np.int64)
        return b, s, c

    def _tiles_for(self, w, h):
        """Tile grid for a frame. Overridden by the adaptive subclass."""
        tiles = generate_tiles(w, h, self.tile, self.tile, self.overlap)
        return tiles, {"n_candidate": len(tiles), "n_processed": len(tiles)}

    # -- full-frame detection ----------------------------------------------
    def detect(self, image, meta=None):
        """image: BGR HxWx3 numpy. Returns dict with boxes/scores/labels + stats.

        meta: optional dict (e.g. {"altitude": 80, "saliency": <HxW array>}),
        consumed by the adaptive subclass; ignored here.
        """
        h, w = image.shape[:2]
        tiles, stats = self._tiles_for(w, h, meta) if self._takes_meta() \
            else self._tiles_for(w, h)

        all_b, all_s, all_c = [], [], []
        for (x0, y0, x1, y1) in tiles:
            crop = image[y0:y1, x0:x1]
            b, s, c = self._infer_crop(crop)
            if len(b):
                all_b.append(offset_boxes(b, x0, y0)); all_s.append(s); all_c.append(c)

        if self.full_frame_pass:
            b, s, c = self._infer_crop(image)
            if len(b):
                all_b.append(b); all_s.append(s); all_c.append(c)

        if all_b:
            B = np.concatenate(all_b); S = np.concatenate(all_s); C = np.concatenate(all_c)
            B, S, C = merge_tile_detections(B, S, C, self.iou_merge)
        else:
            B = np.zeros((0, 4), np.float32); S = np.zeros((0,)); C = np.zeros((0,), np.int64)

        B, S, C, stats = self._postprocess(B, S, C, image, meta, stats)
        return {"boxes": B, "scores": S, "labels": C, "stats": stats}

    def _postprocess(self, B, S, C, image, meta, stats):
        """Hook for subclasses (e.g. saliency gating). Baseline: no-op."""
        return B, S, C, stats

    def _takes_meta(self):
        return False
