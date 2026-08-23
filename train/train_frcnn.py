"""
Two-stage detector (torchvision Faster R-CNN ResNet50-FPN) trained on the tiled
SeaDronesSee dataset, for the GRSL revision's cross-architecture experiment.

SageMaker script-mode entry point. Reads the same YOLO-format tiled dataset the
YOLO/RT-DETR jobs use (images/{train,val}, labels/{train,val}, class 0=person,
1=boat), converts YOLO normalised cx,cy,w,h -> absolute xyxy, and fine-tunes a
COCO-pretrained Faster R-CNN (num_classes=3: background, person, boat).

  channel 'training' -> SM_CHANNEL_TRAINING (tiles root)
  output             -> SM_MODEL_DIR/detector_best.pt  (state_dict)
"""
import argparse, os, glob, math, random
import torch, torch.utils.data as tud
from PIL import Image
import torchvision
from torchvision.transforms.functional import to_tensor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class YoloTiles(tud.Dataset):
    def __init__(self, root, split):
        self.imgs = sorted(glob.glob(os.path.join(root, "images", split, "*.jpg")))
        self.lbl = os.path.join(root, "labels", split)

    def __len__(self): return len(self.imgs)

    def __getitem__(self, i):
        p = self.imgs[i]
        im = Image.open(p).convert("RGB"); W, H = im.size
        lp = os.path.join(self.lbl, os.path.splitext(os.path.basename(p))[0] + ".txt")
        boxes, labels = [], []
        if os.path.exists(lp):
            for ln in open(lp):
                t = ln.split()
                if len(t) != 5: continue
                c, cx, cy, w, h = int(t[0]), *(float(v) for v in t[1:])
                x0, y0 = (cx - w/2)*W, (cy - h/2)*H
                x1, y1 = (cx + w/2)*W, (cy + h/2)*H
                if x1 - x0 < 1 or y1 - y0 < 1: continue
                boxes.append([x0, y0, x1, y1]); labels.append(c + 1)   # 0->1,1->2
        if boxes:
            target = {"boxes": torch.tensor(boxes, dtype=torch.float32),
                      "labels": torch.tensor(labels, dtype=torch.int64)}
        else:
            target = {"boxes": torch.zeros((0, 4), dtype=torch.float32),
                      "labels": torch.zeros((0,), dtype=torch.int64)}
        target["image_id"] = torch.tensor([i])
        return to_tensor(im), target


def collate(b): return tuple(zip(*b))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", default=os.environ.get("SM_CHANNEL_TRAINING", "data/work/tiles"))
    p.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", "model"))
    p.add_argument("--model_dir", dest="ignore", default=None)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max-steps-per-epoch", type=int, default=0)   # 0 = full pass
    a, _ = p.parse_known_args()
    a.model_dir = os.path.abspath(a.model_dir); os.makedirs(a.model_dir, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[frcnn] device={dev} train_root={a.train} epochs={a.epochs} bs={a.batch_size}", flush=True)

    ds = YoloTiles(a.train, "train")
    print(f"[frcnn] train tiles: {len(ds)}", flush=True)
    dl = tud.DataLoader(ds, batch_size=a.batch_size, shuffle=True, num_workers=a.workers,
                        collate_fn=collate, pin_memory=True)

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, 3)   # bg+person+boat
    model.to(dev)
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                          lr=a.lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)

    for ep in range(a.epochs):
        model.train(); running = 0.0; n = 0
        for step, (imgs, tgts) in enumerate(dl):
            if a.max_steps_per_epoch and step >= a.max_steps_per_epoch: break
            imgs = [im.to(dev) for im in imgs]
            tgts = [{k: v.to(dev) for k, v in t.items()} for t in tgts]
            loss_dict = model(imgs, tgts)
            loss = sum(loss_dict.values())
            opt.zero_grad(); loss.backward(); opt.step()
            running += float(loss); n += 1
            if step % 200 == 0:
                print(f"[frcnn] ep{ep} step{step} loss={float(loss):.3f}", flush=True)
        sched.step()
        print(f"[frcnn] epoch {ep} mean loss {running/max(1,n):.3f}", flush=True)
        torch.save(model.state_dict(), os.path.join(a.model_dir, "detector_best.pt"))
    print("[frcnn] done -> detector_best.pt", flush=True)


if __name__ == "__main__":
    main()
