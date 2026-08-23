"""
Fine-tune MSI-Net into a maritime *target-aware* saliency prior — SageMaker
script-mode entry point (also runs locally for smoke tests).

Motivation (EXPERIMENTS.md §5): generic SALICON fixation saliency under-fires on
tiny swimmers, so saliency gating (ablation row D) suppressed real detections.
We retrain the saliency net to predict "target-here" maps (Gaussian blobs on
person/boat boxes, built by make_target_maps.py) so that:
  * tile SELECTION keeps the tiles that actually contain small objects, and
  * confidence GATING boosts real targets instead of dimming them.

Data channel layout (SM_CHANNEL_TRAINING or --data):
  images/train/<stem>.jpg
  target_maps/train/<stem>.png      (GT, built by make_target_maps.py)

Loss = KL(gt||pred) - lambda_cc * CC(gt,pred)   (losses.py).

Outputs (SM_MODEL_DIR or --model-dir):
  saliency_best.weights.h5    best val-loss weights
  msinet_sar/                 exported SavedModel (drop-in for gen_saliency.py)

Local smoke:
  python saliency/finetune.py --data data/seadronessee \
      --epochs 1 --limit 64 --model-dir /tmp/sal_ft
"""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=os.environ.get("SM_CHANNEL_TRAINING",
                                                     "data/seadronessee"))
    p.add_argument("--model-dir", dest="model_dir",
                   default=os.environ.get("SM_MODEL_DIR", "saliency/_ft"))
    p.add_argument("--model_dir", dest="model_dir_s3", default=None)  # SM injects; ignore
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--infer-h", type=int, default=384)
    p.add_argument("--infer-w", type=int, default=640)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lambda-cc", type=float, default=0.5)
    # Accept both a bare flag (local: --freeze-encoder) and a value
    # (SageMaker passes hyperparameters as --freeze-encoder True/False).
    def _str2bool(v):
        if isinstance(v, bool):
            return v
        return str(v).strip().lower() in ("1", "true", "yes", "y")
    p.add_argument("--freeze-encoder", nargs="?", const=True, default=False,
                   type=_str2bool,
                   help="train only ASPP+decoder (faster, less overfit)")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--limit", type=int, default=0, help="0=all (smoke: small N)")
    return p.parse_args()


def make_dataset(data_dir, H, W, batch, val_frac, limit):
    import numpy as np
    import tensorflow as tf
    import cv2

    img_dir = os.path.join(data_dir, "images", "train")
    map_dir = os.path.join(data_dir, "target_maps", "train")
    maps = sorted(glob.glob(os.path.join(map_dir, "*.png")))
    pairs = []
    for m in maps:
        stem = os.path.splitext(os.path.basename(m))[0]
        ip = os.path.join(img_dir, stem + ".jpg")
        if os.path.exists(ip):
            pairs.append((ip, m))
    if limit:
        pairs = pairs[:limit]
    if not pairs:
        raise SystemExit(f"[finetune] no (image,target_map) pairs under {data_dir}")
    n_val = max(1, int(len(pairs) * val_frac))
    train_pairs, val_pairs = pairs[n_val:], pairs[:n_val]
    print(f"[finetune] {len(pairs)} pairs  (train={len(train_pairs)} val={len(val_pairs)})")

    # MUST match data/gen_saliency.py preprocessing exactly: the model is fed
    # BGR-order, mean-subtracted input (gen_saliency does RGB->BGR then minus
    # _BGR_MEAN). cv2.imread already returns BGR, so we keep BGR and subtract the
    # same BGR mean. Training/inference channel order must agree or the prior breaks.
    BGR_MEAN = np.array([103.939, 116.779, 123.68], "float32")

    def _load(ip, mp):
        ip, mp = ip.decode(), mp.decode()
        img = cv2.imread(ip)                       # BGR (matches gen_saliency input)
        img = cv2.resize(img, (W, H)).astype("float32")
        img -= BGR_MEAN
        tgt = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        tgt = cv2.resize(tgt, (W, H)).astype("float32")[..., None]
        return img, tgt

    def _gen(pairs):
        def g():
            for ip, mp in pairs:
                yield ip, mp
        return g

    def _build(pairs, training):
        ds = tf.data.Dataset.from_generator(
            _gen(pairs), output_signature=(
                tf.TensorSpec((), tf.string), tf.TensorSpec((), tf.string)))
        if training:
            ds = ds.shuffle(min(len(pairs), 1000))
        ds = ds.map(lambda a, b: tf.numpy_function(_load, [a, b],
                                                    [tf.float32, tf.float32]),
                    num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.map(lambda im, tg: (tf.ensure_shape(im, (H, W, 3)),
                                    tf.ensure_shape(tg, (H, W, 1))))
        return ds.batch(batch).prefetch(tf.data.AUTOTUNE)

    return _build(train_pairs, True), _build(val_pairs, False)


def main():
    a = parse_args()
    a.model_dir = os.path.abspath(a.model_dir)
    os.makedirs(a.model_dir, exist_ok=True)

    import keras
    try:  # repo package layout
        from saliency.model import build_xsoyelnet, VGG16_CONV_NAMES
        from saliency.losses import kl_divergence, correlation_coefficient
    except ImportError:  # flat SageMaker source_dir
        from model import build_xsoyelnet, VGG16_CONV_NAMES
        from losses import kl_divergence, correlation_coefficient

    H, W = a.infer_h, a.infer_w
    model = build_xsoyelnet(load_vgg=True)
    if a.freeze_encoder:
        for name in VGG16_CONV_NAMES:
            model.get_layer(name).trainable = False
        print(f"[finetune] froze {len(VGG16_CONV_NAMES)} encoder layers")

    lam = a.lambda_cc
    def loss_fn(y_true, y_pred):
        return kl_divergence(y_true, y_pred) - lam * correlation_coefficient(y_true, y_pred)

    model.compile(optimizer=keras.optimizers.Adam(a.lr), loss=loss_fn,
                  metrics=[correlation_coefficient])

    train_ds, val_ds = make_dataset(a.data, H, W, a.batch_size, a.val_frac, a.limit)

    best_path = os.path.join(a.model_dir, "saliency_best.weights.h5")
    cbs = [
        keras.callbacks.ModelCheckpoint(best_path, monitor="val_loss",
                                        save_best_only=True, save_weights_only=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=3, min_lr=1e-6),
    ]
    print(f"[finetune] training {a.epochs} epochs @ {H}x{W} lr={a.lr} lambda_cc={lam}")
    model.fit(train_ds, validation_data=val_ds, epochs=a.epochs, callbacks=cbs)

    if os.path.exists(best_path):
        model.load_weights(best_path)
        print(f"[finetune] restored best weights from {best_path}")
    export_dir = os.path.join(a.model_dir, "msinet_sar")
    model.export(export_dir)
    print(f"[finetune] exported SavedModel → {export_dir}")


if __name__ == "__main__":
    main()
