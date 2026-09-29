"""
Qualitative figures for the GRSL revision (reviewers 2 and 3).

Fig A  generic MSI-Net vs fine-tuned xSoyelNet saliency on the same frames.
Fig B  which tiles the xSoyelNet prior keeps vs drops (empty-water rejection).

Both are built entirely from cached saliency maps + the project's own tiling
functions, so they reproduce the exact selection the pipeline uses.
"""
import os, json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import sys; sys.path.insert(0, ".")
from tiling.geometry import (generate_tiles, altitude_tile_size,
                             tile_saliency_scores, select_salient_tiles)

ROOT = "data/seadronessee"
IMG = f"{ROOT}/images/val"
SAL_GEN = f"{ROOT}/saliency/val"       # generic MSI-Net
SAL_FT  = f"{ROOT}/saliency_sar/val"   # fine-tuned xSoyelNet
OUT = "paper/figs"; os.makedirs(OUT, exist_ok=True)
EX = ["276.jpg", "300.jpg", "189.jpg"]   # high-altitude, small targets on water

ann = json.load(open(f"{ROOT}/annotations/instances_val.json"))
meta = {i["file_name"]: i for i in ann["images"]}
gt = {}
name_by_id = {i["id"]: i["file_name"] for i in ann["images"]}
for a in ann["annotations"]:
    gt.setdefault(name_by_id[a["image_id"]], []).append(a["bbox"])

def load_sal(d, stem, shape):
    p = f"{d}/{stem}.png"
    s = np.asarray(Image.open(p).convert("L").resize((shape[1], shape[0])), np.float32)
    return s / 255.0

# ---------- Figure A: generic vs fine-tuned ----------
fig, ax = plt.subplots(len(EX), 3, figsize=(9.5, 3.0*len(EX)))
for r, fn in enumerate(EX):
    stem = fn[:-4]
    im = np.asarray(Image.open(f"{IMG}/{fn}").convert("RGB"))
    H, W = im.shape[:2]
    sg = load_sal(SAL_GEN, stem, (H, W)); sf = load_sal(SAL_FT, stem, (H, W))
    ax[r,0].imshow(im)
    for x,y,w,h in gt.get(fn, []):
        ax[r,0].add_patch(Rectangle((x,y), w, h, fill=False, edgecolor="#00e5ff", lw=1.5))
    ax[r,1].imshow(im); ax[r,1].imshow(sg, cmap="jet", alpha=0.5)
    ax[r,2].imshow(im); ax[r,2].imshow(sf, cmap="jet", alpha=0.5)
    alt = (meta[fn].get("meta") or {}).get("altitude", 0)
    ax[r,0].set_ylabel(f"{stem}  ({alt:.0f} m)", fontsize=9)
    for c in range(3): ax[r,c].set_xticks([]); ax[r,c].set_yticks([])
for c,t in enumerate(["frame + targets (cyan)", "generic MSI-Net", "xSoyelNet (fine-tuned)"]):
    ax[0,c].set_title(t, fontsize=10)
fig.tight_layout(); fig.savefig(f"{OUT}/qual_saliency.png", dpi=200, bbox_inches="tight")
print("saved qual_saliency.png")

# ---------- Figure B: kept vs dropped tiles ----------
fig, ax = plt.subplots(1, len(EX), figsize=(4.6*len(EX), 3.4))
if len(EX)==1: ax=[ax]
for c, fn in enumerate(EX):
    stem = fn[:-4]
    im = np.asarray(Image.open(f"{IMG}/{fn}").convert("RGB"))
    H, W = im.shape[:2]
    alt = (meta[fn].get("meta") or {}).get("altitude", 100)
    t = altitude_tile_size(alt)                      # same altitude sizing as S1
    tiles = generate_tiles(W, H, t, t, 0.2)
    sf = (load_sal(SAL_FT, stem, (H, W))*255).astype(np.uint8)
    scores = tile_saliency_scores(sf, tiles)
    keep = set(select_salient_tiles(tiles, scores, keep_thr=0.05))  # S1 threshold
    ax[c].imshow(im)
    for i,(x0,y0,x1,y1) in enumerate(tiles):
        kept = i in keep
        ax[c].add_patch(Rectangle((x0,y0), x1-x0, y1-y0, fill=not kept,
                        facecolor=("none" if kept else "black"), alpha=(1 if kept else 0.45),
                        edgecolor=("#2ecc40" if kept else "#ff4136"), lw=2))
    for x,y,w,h in gt.get(fn, []):
        ax[c].add_patch(Rectangle((x,y), w, h, fill=False, edgecolor="#00e5ff", lw=1.5))
    ax[c].set_title(f"{stem}: keep {len(keep)}/{len(tiles)} tiles", fontsize=10)
    ax[c].set_xticks([]); ax[c].set_yticks([])
fig.tight_layout(); fig.savefig(f"{OUT}/qual_tiles.png", dpi=200, bbox_inches="tight")
print("saved qual_tiles.png")
