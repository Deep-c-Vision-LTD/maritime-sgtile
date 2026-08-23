"""
Pure-numpy tiling geometry: tile generation, coordinate mapping, NMS, and the
altitude -> tile-size schedule. No torch / cv2 / ultralytics here, so this is the
unit-testable core that the baseline and the adaptive tiler both build on.

Box convention everywhere: xyxy in absolute pixels, numpy float arrays.
"""

import numpy as np


# ─────────────────────────────────────────────────────────────
#  Tile grid
# ─────────────────────────────────────────────────────────────

def generate_tiles(img_w, img_h, tile_w, tile_h, overlap=0.2):
    """Overlapping tile grid covering the whole image (SAHI-style).

    Returns list of (x0, y0, x1, y1) int tiles. The last row/column is shifted
    inward so every tile is exactly tile_w x tile_h and the image edge is covered.
    """
    tile_w = min(tile_w, img_w)
    tile_h = min(tile_h, img_h)
    step_x = max(1, int(tile_w * (1 - overlap)))
    step_y = max(1, int(tile_h * (1 - overlap)))

    xs = list(range(0, max(1, img_w - tile_w + 1), step_x))
    ys = list(range(0, max(1, img_h - tile_h + 1), step_y))
    if not xs or xs[-1] != img_w - tile_w:
        xs.append(max(0, img_w - tile_w))
    if not ys or ys[-1] != img_h - tile_h:
        ys.append(max(0, img_h - tile_h))

    tiles = []
    for y0 in ys:
        for x0 in xs:
            tiles.append((int(x0), int(y0), int(x0 + tile_w), int(y0 + tile_h)))
    # de-duplicate (small images can repeat the inward-shifted edge tile)
    return sorted(set(tiles))


def offset_boxes(boxes, x0, y0):
    """Map tile-local xyxy boxes to full-frame coords by adding the tile origin."""
    if len(boxes) == 0:
        return boxes
    out = np.asarray(boxes, dtype=np.float32).copy()
    out[:, [0, 2]] += x0
    out[:, [1, 3]] += y0
    return out


# ─────────────────────────────────────────────────────────────
#  NMS (for merging detections across overlapping tiles)
# ─────────────────────────────────────────────────────────────

def iou_matrix(boxes):
    """Pairwise IoU of an (N,4) xyxy array. Returns (N,N)."""
    if len(boxes) == 0:
        return np.zeros((0, 0), np.float32)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    area = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ix1 = np.maximum(x1[:, None], x1[None, :])
    iy1 = np.maximum(y1[:, None], y1[None, :])
    ix2 = np.minimum(x2[:, None], x2[None, :])
    iy2 = np.minimum(y2[:, None], y2[None, :])
    iw = np.clip(ix2 - ix1, 0, None)
    ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    union = area[:, None] + area[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def nms(boxes, scores, iou_thr=0.5):
    """Greedy NMS. Returns kept indices (highest score first)."""
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    if len(boxes) == 0:
        return np.array([], dtype=int)
    order = scores.argsort()[::-1]
    keep = []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    area = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        iw = np.clip(xx2 - xx1, 0, None)
        ih = np.clip(yy2 - yy1, 0, None)
        inter = iw * ih
        union = area[i] + area[rest] - inter
        iou = np.where(union > 0, inter / union, 0.0)
        order = rest[iou <= iou_thr]
    return np.array(keep, dtype=int)


def merge_tile_detections(boxes, scores, labels, iou_thr=0.5):
    """Class-aware NMS over pooled cross-tile detections."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    keep_all = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        k = nms(boxes[idx], scores[idx], iou_thr)
        keep_all.extend(idx[k].tolist())
    keep_all = np.array(sorted(keep_all), dtype=int)
    return boxes[keep_all], scores[keep_all], labels[keep_all]


# ─────────────────────────────────────────────────────────────
#  Altitude -> tile size schedule  (altitude-aware component)
# ─────────────────────────────────────────────────────────────

def altitude_tile_size(altitude_m, base_tile=640, ref_alt=100.0,
                       min_tile=320, max_tile=1280):
    """Pick a tile size from UAV altitude.

    Intuition: higher altitude -> objects occupy fewer pixels -> use a SMALLER
    tile so that, after the detector's fixed input resize, objects are magnified.
    Lower altitude -> objects are already large -> a LARGER tile (fewer tiles,
    less compute) suffices.

    tile = base_tile * (ref_alt / altitude)  clamped to [min_tile, max_tile],
    rounded to a multiple of 32 (detector stride friendliness).
    """
    if not altitude_m or altitude_m <= 0:
        t = base_tile
    else:
        t = base_tile * (ref_alt / float(altitude_m))
    t = int(round(max(min_tile, min(max_tile, t)) / 32) * 32)
    return max(32, t)


# ─────────────────────────────────────────────────────────────
#  Saliency helpers  (saliency-guided component)
# ─────────────────────────────────────────────────────────────

def tile_saliency_scores(saliency, tiles):
    """Mean saliency (0..1) inside each tile. `saliency` is a HxW float/uint8 map."""
    s = np.asarray(saliency, dtype=np.float32)
    if s.max() > 1.0:
        s = s / 255.0
    out = np.empty(len(tiles), np.float32)
    for i, (x0, y0, x1, y1) in enumerate(tiles):
        crop = s[y0:y1, x0:x1]
        out[i] = float(crop.mean()) if crop.size else 0.0
    return out


def select_salient_tiles(tiles, sal_scores, keep_thr=0.15, min_keep=1):
    """Keep tiles whose mean saliency exceeds keep_thr (relative to the max).

    Always keeps at least `min_keep` tiles (the most salient) so a frame is never
    left completely unprocessed. Returns the list of kept tile indices.
    """
    sal = np.asarray(sal_scores, dtype=np.float32)
    if len(sal) == 0:
        return []
    norm = sal / (sal.max() + 1e-7)
    keep = np.where(norm >= keep_thr)[0]
    if len(keep) < min_keep:
        keep = norm.argsort()[::-1][:min_keep]
    return sorted(keep.tolist())


def saliency_at_boxes(saliency, boxes):
    """Mean saliency (0..1) under each detection box — used for FP gating."""
    s = np.asarray(saliency, dtype=np.float32)
    if s.max() > 1.0:
        s = s / 255.0
    H, W = s.shape[:2]
    out = np.empty(len(boxes), np.float32)
    for i, (x0, y0, x1, y1) in enumerate(np.asarray(boxes, dtype=int)):
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(W, x1), min(H, y1)
        crop = s[y0:y1, x0:x1]
        out[i] = float(crop.mean()) if crop.size else 0.0
    return out


def saliency_gate_scores(scores, sal_under_box, alpha=0.5, mode="mult", floor=0.1):
    """Down-weight detection confidence by local saliency to suppress clutter FPs.

    Two modes:

    "mult" (original, symmetric):
        score' = score * ((1 - alpha) + alpha * saliency_under_box)
      Every detection below max saliency is penalised. With a *generic* fixation
      prior this dims real (under-fired) targets too — ablation row D showed this
      hurts mAP and P@R. Kept for the ablation / backwards-compatibility.

    "asym" (asymmetric, clutter-only):
        score' = score                          if saliency >= floor
        score' = score * ramp(saliency/floor)   if saliency <  floor
      Detections at or above the saliency `floor` are left UNTOUCHED; only
      detections sitting on clearly-empty water (below the floor) are suppressed.
      This removes sea-clutter FPs without taxing true positives — the right
      design once the prior is target-aware (Section 5).

    alpha=0 -> no gating (baseline). alpha in (0,1] sets suppression strength.
    """
    scores = np.asarray(scores, dtype=np.float32)
    sal = np.clip(np.asarray(sal_under_box, dtype=np.float32), 0.0, 1.0)
    if mode == "mult":
        return scores * ((1.0 - alpha) + alpha * sal)
    if mode == "asym":
        ratio = np.clip(sal / (floor + 1e-7), 0.0, 1.0)   # 0..1 across [0, floor]
        factor = (1.0 - alpha) + alpha * ratio            # in [1-alpha, 1]
        factor = np.where(sal >= floor, 1.0, factor)      # above floor: no penalty
        return scores * factor
    raise ValueError(f"unknown gate mode {mode!r} (expected 'mult' or 'asym')")
