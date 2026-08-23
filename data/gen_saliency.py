"""
Generate saliency maps for a folder of images using the MSI-Net saliency model
(alexanderkroner/MSI-Net, SALICON-trained) — the visual-attention prior for the
detector.

For every image  <images_dir>/foo.jpg  it writes a single-channel saliency PNG
<out_dir>/foo.png  at the SAME pixel resolution as the source image, so it lines
up channel-for-channel when the detector stacks RGB + saliency.

Loads weights in this priority order:
  1. --savedmodel path to MSI-Net SavedModel dir  (primary; data/_models/msinet)
  2. --keras  path to a re-trainable .keras checkpoint (fine-tune experiments)
Falls back to building the architecture with ImageNet VGG16 init if neither is
given (useful for a smoke test only).

Pre/post-processing matches the original MSI-Net app_base.py:
  * input : RGB->BGR, subtract ImageNet BGR mean, size divisible by 8.
  * output: min-max normalised to [0,255] uint8.

Usage:
  python gen_saliency.py \
      --images-dir data/seadronessee/images/train \
      --out-dir    data/seadronessee/saliency/train \
      --savedmodel data/_models/msinet \
      --infer-h 384 --infer-w 640
"""

import argparse
import glob
import os
import sys

import numpy as np

_BGR_MEAN = np.array([103.939, 116.779, 123.68], dtype=np.float32)
_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def _round8(v):
    return max(8, int(round(v / 8) * 8))


def load_model(keras_path, savedmodel_path):
    import tensorflow as tf  # noqa
    if savedmodel_path and os.path.exists(savedmodel_path):
        print(f"[saliency] loading SavedModel: {savedmodel_path}")
        layer = tf.saved_model.load(savedmodel_path)
        return layer, "savedmodel"
    if keras_path and os.path.exists(keras_path):
        import keras
        print(f"[saliency] loading Keras model: {keras_path}")
        # Custom layers registered via @register_keras_serializable
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "saliency"))
        import model as _m  # noqa: registers ResizeToMatch / Upsample2x
        return keras.models.load_model(keras_path, compile=False), "keras"
    # Fallback: build architecture fresh (ImageNet init) — smoke test only.
    print("[saliency] WARNING: no checkpoint given; pass --savedmodel data/_models/msinet "
          "for real runs (MSI-Net SavedModel from alexanderkroner/MSI-Net)")
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "saliency"))
    from model import build_xsoyelnet
    return build_xsoyelnet(load_vgg=True), "keras"


def _infer(model, kind, batch):
    """batch: float32 NHWC, BGR mean-subtracted. Returns NHW1 saliency."""
    if kind == "savedmodel":
        sig = model.signatures["serving_default"]
        # auto-discover input kwarg name: MSI-Net SavedModel uses 'input_1';
        # other SavedModels may differ — auto-discovery makes any SavedModel work.
        kw = list(sig.structured_input_signature[1].keys())[0]
        out = sig(**{kw: batch})
        # single output tensor regardless of key name
        return list(out.values())[0].numpy()
    return model(batch, training=False).numpy()


def generate(images_dir, out_dir, model, kind, infer_h, infer_w):
    import tensorflow as tf
    os.makedirs(out_dir, exist_ok=True)
    paths = [p for p in sorted(glob.glob(os.path.join(images_dir, "*")))
             if p.lower().endswith(_EXTS)]
    print(f"[saliency] {len(paths)} images -> {out_dir}")

    for i, p in enumerate(paths):
        raw = tf.io.read_file(p)
        img = tf.io.decode_image(raw, channels=3, expand_animations=False)
        orig_h, orig_w = int(tf.shape(img)[0]), int(tf.shape(img)[1])

        h = _round8(infer_h) if infer_h else _round8(orig_h)
        w = _round8(infer_w) if infer_w else _round8(orig_w)

        x = tf.cast(tf.image.resize(img, (h, w)), tf.float32)
        x = x[..., ::-1] - _BGR_MEAN            # RGB->BGR, mean subtract
        x = tf.expand_dims(x, 0)

        sal = _infer(model, kind, x)[0, ..., 0]  # HxW
        # min-max normalise to 0..255
        sal = np.maximum(sal, 0)
        rng = sal.max() - sal.min()
        sal = (sal - sal.min()) / (rng + 1e-7)
        sal = (sal * 255).astype(np.uint8)

        # resize saliency back to the ORIGINAL image size so it aligns with RGB
        sal_full = tf.image.resize(sal[..., None], (orig_h, orig_w),
                                   method="bilinear")
        sal_full = tf.cast(tf.round(sal_full), tf.uint8)
        png = tf.io.encode_png(sal_full)
        stem = os.path.splitext(os.path.basename(p))[0]
        tf.io.write_file(os.path.join(out_dir, stem + ".png"), png)

        if (i + 1) % 200 == 0:
            print(f"[saliency]   {i + 1}/{len(paths)}")
    print("[saliency] done.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--images-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--keras", default="")
    p.add_argument("--savedmodel", default="")
    p.add_argument("--infer-h", type=int, default=384,
                   help="inference height (div by 8); 0 = use original")
    p.add_argument("--infer-w", type=int, default=640,
                   help="inference width (div by 8); 0 = use original")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    model, kind = load_model(a.keras, a.savedmodel)
    generate(a.images_dir, a.out_dir, model, kind, a.infer_h, a.infer_w)
