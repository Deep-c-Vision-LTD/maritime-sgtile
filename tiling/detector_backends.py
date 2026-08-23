"""
Detector backends for the cross-architecture experiment (GRSL revision).

All three backends expose the SAME minimal interface the tiling pipeline uses:

    result = model.predict(crop_bgr, imgsz=?, conf=?, device=?, verbose=False)[0]
    result.boxes.xyxy   -> (N,4) tensor, absolute px in the crop
    result.boxes.conf   -> (N,)  tensor
    result.boxes.cls    -> (N,)  tensor, class ids in YOLO convention (0=person,1=boat)
    len(result.boxes)

  yolo   : Ultralytics YOLO           (unchanged default)
  rtdetr : Ultralytics RT-DETR        (transformer, single-model)
  frcnn  : torchvision Faster R-CNN   (two-stage; wrapped to match the interface)
"""
import numpy as np


def load_detector(weights, backend="yolo", device=None):
    backend = (backend or "yolo").lower()
    if backend == "yolo":
        from ultralytics import YOLO
        return YOLO(weights)
    if backend == "rtdetr":
        from ultralytics import RTDETR
        return RTDETR(weights)
    if backend == "frcnn":
        return _FRCNN(weights, device=device)
    raise ValueError(f"unknown backend {backend!r}")


# ---- torchvision Faster R-CNN wrapped to the ultralytics-style interface ----
class _Boxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy, self.conf, self.cls = xyxy, conf, cls
    def __len__(self):
        return int(self.xyxy.shape[0])


class _Result:
    def __init__(self, boxes):
        self.boxes = boxes


class _FRCNN:
    def __init__(self, weights, device=None):
        import torch, torchvision
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        self.torch = torch
        self.dev = device or ("cuda" if torch.cuda.is_available()
                              else ("mps" if torch.backends.mps.is_available() else "cpu"))
        m = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
        in_feat = m.roi_heads.box_predictor.cls_score.in_features
        m.roi_heads.box_predictor = FastRCNNPredictor(in_feat, 3)   # bg+person+boat
        sd = torch.load(weights, map_location="cpu")
        m.load_state_dict(sd)
        m.eval().to(self.dev)
        self.model = m

    @property
    def torch_no_grad(self):
        return self.torch.no_grad

    def predict(self, crop_bgr, imgsz=640, conf=0.25, device=None, verbose=False):
        torch = self.torch
        rgb = crop_bgr[:, :, ::-1].copy()                       # BGR -> RGB
        t = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0).to(self.dev)
        with torch.no_grad():
            out = self.model([t])[0]
        boxes = out["boxes"].cpu(); scores = out["scores"].cpu(); labels = out["labels"].cpu()
        keep = scores >= conf
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
        labels = (labels - 1).clamp(min=0)                      # torchvision 1/2 -> YOLO 0/1
        return [_Result(_Boxes(boxes, scores, labels))]
