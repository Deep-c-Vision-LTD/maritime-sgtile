"""
Clutter-aware detection evaluation.

Beyond the usual mAP, this reports the metrics the maritime tiling literature omits
and that matter for SAR:

  * mAP@0.50 and AP@[.50:.05:.95]          (standard)
  * AP_small                                (gt boxes with area < small_thr^2)
  * FP per frame                            (clutter false-alarm rate)
  * Precision @ a fixed Recall (e.g. 0.80)  (precision at an operating point that
                                             keeps most true victims)
  * mean tiles processed / tile reduction   (compute, from the detector stats)

Self-contained (numpy only) so it runs without pycocotools. For an official
leaderboard number, also run the SeaDronesSee COCO evaluator; the AP here uses
all-point interpolation and matches COCO closely but is not byte-identical.

Data format (per image, index-aligned lists):
  preds: list of dict { "boxes":(N,4) xyxy, "scores":(N,), "labels":(N,) }
  gts  : list of dict { "boxes":(M,4) xyxy, "labels":(M,) }
"""

import numpy as np

from tiling.geometry import iou_matrix  # reuse the pure-numpy IoU


def _match_image(pred_boxes, pred_scores, gt_boxes, iou_thr):
    """Greedy score-ordered matching for ONE class in ONE image.
    Returns (tp_flags, fp_flags) aligned to score-sorted preds, plus n_gt."""
    n_p, n_g = len(pred_boxes), len(gt_boxes)
    if n_p == 0:
        return np.zeros(0), np.zeros(0), n_g
    order = np.argsort(-pred_scores)
    pb = pred_boxes[order]
    tp = np.zeros(n_p); fp = np.zeros(n_p)
    if n_g == 0:
        fp[:] = 1.0
        return tp, fp, 0
    ious = iou_matrix(np.vstack([pb, gt_boxes]))[:n_p, n_p:]  # (n_p, n_g)
    matched = np.zeros(n_g, dtype=bool)
    for i in range(n_p):
        # COCO-style greedy: take the highest-IoU UNMATCHED gt above threshold.
        # (Picking only the single argmax gt mislabels a pred as FP when its best
        # gt is already taken but another unmatched gt still overlaps it >= thr —
        # which undercounts recall for clustered objects, e.g. groups of swimmers.)
        assigned = False
        for j in np.argsort(-ious[i]):
            if ious[i, j] < iou_thr:
                break  # all remaining gts overlap even less
            if not matched[j]:
                tp[i] = 1.0; matched[j] = True; assigned = True
                break
        if not assigned:
            fp[i] = 1.0
    return tp, fp, n_g


def _ap_all_point(rec, prec):
    """All-point (COCO-style) AP from a recall/precision curve."""
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[0.0], prec, [0.0]])
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def _ap_for_class(preds, gts, cls, iou_thr):
    """AP for one class across all images at one IoU threshold."""
    scores_all, tp_all, fp_all, n_gt = [], [], [], 0
    for p, g in zip(preds, gts):
        pm = p["labels"] == cls
        gm = g["labels"] == cls
        pb, ps = p["boxes"][pm], p["scores"][pm]
        gb = g["boxes"][gm]
        tp, fp, ng = _match_image(pb, ps, gb, iou_thr)
        n_gt += ng
        scores_all.append(ps[np.argsort(-ps)] if len(ps) else ps)
        tp_all.append(tp); fp_all.append(fp)
    if n_gt == 0:
        return None
    scores = np.concatenate(scores_all) if scores_all else np.zeros(0)
    tp = np.concatenate(tp_all) if tp_all else np.zeros(0)
    fp = np.concatenate(fp_all) if fp_all else np.zeros(0)
    if len(scores) == 0:
        return 0.0
    order = np.argsort(-scores)
    tp, fp = tp[order], fp[order]
    tp_c, fp_c = np.cumsum(tp), np.cumsum(fp)
    rec = tp_c / (n_gt + 1e-9)
    prec = tp_c / (tp_c + fp_c + 1e-9)
    return _ap_all_point(rec, prec)


def mean_ap(preds, gts, classes, iou_thrs):
    """mAP averaged over classes and IoU thresholds. Returns (mAP, per_iou_dict)."""
    per_iou = {}
    for t in iou_thrs:
        aps = [_ap_for_class(preds, gts, c, t) for c in classes]
        aps = [a for a in aps if a is not None]
        per_iou[round(t, 2)] = float(np.mean(aps)) if aps else 0.0
    return float(np.mean(list(per_iou.values()))) if per_iou else 0.0, per_iou


def _filter_small(preds, gts, small_thr):
    """Keep only small GT boxes (area < small_thr^2); preds untouched."""
    g2 = []
    for g in gts:
        if len(g["boxes"]) == 0:
            g2.append(g); continue
        wh = g["boxes"][:, 2:] - g["boxes"][:, :2]
        area = wh[:, 0] * wh[:, 1]
        m = area < (small_thr ** 2)
        g2.append({"boxes": g["boxes"][m], "labels": g["labels"][m]})
    return preds, g2


def clutter_metrics(preds, gts, classes, conf, iou_thr=0.5):
    """TP/FP/FN, precision, recall, FP-per-frame at a single operating conf."""
    TP = FP = FN = 0
    for p, g in zip(preds, gts):
        m = p["scores"] >= conf
        pb, ps, pl = p["boxes"][m], p["scores"][m], p["labels"][m]
        for c in classes:
            tp, fp, ng = _match_image(pb[pl == c], ps[pl == c],
                                      g["boxes"][g["labels"] == c], iou_thr)
            TP += int(tp.sum()); FP += int(fp.sum()); FN += ng - int(tp.sum())
    n_frames = max(1, len(gts))
    prec = TP / (TP + FP + 1e-9)
    rec = TP / (TP + FN + 1e-9)
    return {"TP": TP, "FP": FP, "FN": FN, "precision": prec, "recall": rec,
            "fp_per_frame": FP / n_frames}


def precision_at_recall(preds, gts, classes, target_recall=0.8, iou_thr=0.5,
                        sweep=200):
    """Highest precision achievable while keeping recall >= target_recall.

    Sweeps the confidence threshold; reports the operating point. This is the
    headline clutter metric: 'at the conf that still finds 80% of victims, how
    clean are the alarms?'"""
    all_scores = np.concatenate([p["scores"] for p in preds if len(p["scores"])]) \
        if any(len(p["scores"]) for p in preds) else np.array([0.0])
    thrs = np.unique(np.quantile(all_scores, np.linspace(0, 1, sweep)))
    best = {"precision": 0.0, "recall": 0.0, "conf": 1.0, "fp_per_frame": None}
    for t in thrs:
        m = clutter_metrics(preds, gts, classes, conf=float(t), iou_thr=iou_thr)
        if m["recall"] >= target_recall and m["precision"] > best["precision"]:
            best = {"precision": m["precision"], "recall": m["recall"],
                    "conf": float(t), "fp_per_frame": m["fp_per_frame"]}
    return best


def evaluate(preds, gts, classes, small_thr=32, op_conf=0.25,
             target_recall=0.8, iou_thrs=None, det_stats=None):
    """Full report dict. det_stats: optional list of per-frame detector stats
    (with 'n_processed'/'n_candidate'/'tile_reduction') to report compute."""
    iou_thrs = iou_thrs or list(np.arange(0.5, 1.0, 0.05))
    classes = list(classes)

    map_all, per_iou = mean_ap(preds, gts, classes, iou_thrs)
    map50, _ = mean_ap(preds, gts, classes, [0.5])
    p_small, g_small = _filter_small(preds, gts, small_thr)
    ap_small, _ = mean_ap(p_small, g_small, classes, iou_thrs)

    rep = {
        "mAP_50_95": map_all,
        "mAP_50": map50,
        "AP_small_50_95": ap_small,
        "clutter@op_conf": clutter_metrics(preds, gts, classes, op_conf),
        f"precision@recall_{target_recall}": precision_at_recall(
            preds, gts, classes, target_recall),
    }
    if det_stats:
        red = [s.get("tile_reduction", 0.0) for s in det_stats if s]
        proc = [s.get("n_processed", 0) for s in det_stats if s]
        rep["compute"] = {
            "mean_tiles_processed": float(np.mean(proc)) if proc else 0.0,
            "mean_tile_reduction": float(np.mean(red)) if red else 0.0,
        }
    return rep


def print_report(rep, title=""):
    print(f"\n===== {title} =====")
    print(f"  mAP@50:95      {rep['mAP_50_95']:.4f}")
    print(f"  mAP@50         {rep['mAP_50']:.4f}")
    print(f"  AP_small@50:95 {rep['AP_small_50_95']:.4f}")
    c = rep["clutter@op_conf"]
    print(f"  [op conf] precision {c['precision']:.3f}  recall {c['recall']:.3f}  "
          f"FP/frame {c['fp_per_frame']:.3f}")
    pr = next(v for k, v in rep.items() if k.startswith("precision@recall"))
    print(f"  precision@recall: P={pr['precision']:.3f} at recall {pr['recall']:.3f} "
          f"(conf {pr['conf']:.3f}, FP/frame {pr['fp_per_frame']})")
    if "compute" in rep:
        print(f"  compute: mean tiles {rep['compute']['mean_tiles_processed']:.1f}  "
              f"reduction {rep['compute']['mean_tile_reduction']*100:.1f}%")
