"""
MSI-Net architecture re-implemented in Keras 3 (TF2 backend).

Port of alexanderkroner/saliency (originally TensorFlow 1.13) to the modern
stack so it can train in an off-the-shelf SageMaker TF2 container and stays
re-trainable.

**For inference, use the pre-trained MSI-Net SavedModel at data/_models/msinet
via `data/gen_saliency.py --savedmodel data/_models/msinet`.** This file is
only needed for fine-tuning experiments (Section 5 of EXPERIMENTS.md).

The model is fully convolutional and accepts **arbitrary input sizes** (H, W
divisible by 8), exactly like the original HF MSI-Net — so the exported
SavedModel is a drop-in replacement in app_base.py.

Architecture (output stride 8):
  Encoder : modified VGG16 -- blocks 1-3 with max-pool (/8), blocks 4-5 keep
            resolution; block 5 uses dilation rate 2 (atrous).
  ASPP    : parallel branches at dilations 1/4/8/12 + a global-context branch.
  Decoder : 3x bilinear upsample (x8) back to input resolution, -> 1 channel.

NOTE on weights: the original encoder is a *hybrid* VGG16 (ImageNet+Places365).
Here we initialise from stock ImageNet VGG16 via keras.applications, the
pragmatic choice for this port. Results drift slightly from the paper until/
unless the hybrid weights are ported. See README.md.

━━━ TO IMPROVE CC AND NSS — encoder weight upgrade ━━━
The single biggest gain available in this file: replace the ImageNet VGG16
init with the hybrid VGG16 (ImageNet + Places365) used in the paper. The
Places365 pretraining gives the encoder a much richer understanding of scene
context, which directly improves the spatial precision that drives CC and NSS.

How to do it when ready:
  1. Download VGG16 Places365 weights:
       https://github.com/GKalliatakis/Keras-VGG16-places365
       or the alexanderkroner/saliency download.py gdown ID: 1ff0va472Xs1bvidCwRlW3Ctf7Hbyyn7p
  2. Convert the TF1 checkpoint to Keras 3 format (load_weights with name mapping)
  3. Replace transfer_vgg16_weights() below to load from the hybrid checkpoint
     instead of keras.applications.VGG16(weights="imagenet")
  4. Retrain — expect noticeable CC/NSS improvement over the ImageNet-only init

License: Places365 repo (github.com/csailvision/places365) is MIT licensed
— commercial use permitted. ✅ No blocker here.
"""

import keras
from keras import layers

MODEL_NAME = "xsoyelNet"

# Conv layer names that match keras.applications.VGG16 so ImageNet weights can
# be transferred by name. Blocks 4/5 keep the names but drop the pooling.
VGG16_CONV_NAMES = [
    "block1_conv1", "block1_conv2",
    "block2_conv1", "block2_conv2",
    "block3_conv1", "block3_conv2", "block3_conv3",
    "block4_conv1", "block4_conv2", "block4_conv3",
    "block5_conv1", "block5_conv2", "block5_conv3",
]


@keras.saving.register_keras_serializable(package="xsoyelnet")
class ResizeToMatch(layers.Layer):
    """Bilinearly resize `source` to the (dynamic) spatial size of `reference`.

    Lets the ASPP global-context branch work at arbitrary input resolutions.
    """

    def call(self, inputs):
        import tensorflow as tf
        source, reference = inputs
        size = tf.shape(reference)[1:3]
        return tf.image.resize(source, size, method="bilinear")

    def compute_output_shape(self, input_shape):
        src, ref = input_shape
        return (src[0], ref[1], ref[2], src[-1])


@keras.saving.register_keras_serializable(package="xsoyelnet")
class Upsample2x(layers.Layer):
    """Bilinear 2x upsample that traces under unknown H/W (export-safe).

    Matches the original MSI-Net decoder, which used tf.image.resize_bilinear.
    """

    def call(self, x):
        import tensorflow as tf
        h, w = tf.shape(x)[1] * 2, tf.shape(x)[2] * 2
        return tf.image.resize(x, (h, w), method="bilinear")

    def compute_output_shape(self, s):
        d = lambda v: None if v is None else v * 2
        return (s[0], d(s[1]), d(s[2]), s[3])


def _conv(x, filters, name, dilation=1, kernel=3):
    return layers.Conv2D(
        filters, kernel, padding="same", dilation_rate=dilation,
        activation="relu", name=name,
    )(x)


def build_encoder(inp):
    """Modified VGG16 encoder, output stride 8."""
    x = _conv(inp, 64, "block1_conv1")
    x = _conv(x, 64, "block1_conv2")
    x = layers.MaxPooling2D(2, name="block1_pool")(x)            # /2
    x = _conv(x, 128, "block2_conv1")
    x = _conv(x, 128, "block2_conv2")
    x = layers.MaxPooling2D(2, name="block2_pool")(x)            # /4
    x = _conv(x, 256, "block3_conv1")
    x = _conv(x, 256, "block3_conv2")
    x = _conv(x, 256, "block3_conv3")
    x = layers.MaxPooling2D(2, name="block3_pool")(x)            # /8
    x = _conv(x, 512, "block4_conv1")
    x = _conv(x, 512, "block4_conv2")
    x = _conv(x, 512, "block4_conv3")                            # /8, no pool
    x = _conv(x, 512, "block5_conv1", dilation=2)
    x = _conv(x, 512, "block5_conv2", dilation=2)
    x = _conv(x, 512, "block5_conv3", dilation=2)                # /8, dilated
    return x


def build_aspp(x, filters=256):
    """Atrous Spatial Pyramid Pooling with a global-context branch."""
    b1 = layers.Conv2D(filters, 1, padding="same", activation="relu")(x)
    b2 = layers.Conv2D(filters, 3, dilation_rate=4, padding="same", activation="relu")(x)
    b3 = layers.Conv2D(filters, 3, dilation_rate=8, padding="same", activation="relu")(x)
    b4 = layers.Conv2D(filters, 3, dilation_rate=12, padding="same", activation="relu")(x)
    g = layers.GlobalAveragePooling2D(keepdims=True)(x)
    g = layers.Conv2D(filters, 1, padding="same", activation="relu")(g)
    g = ResizeToMatch()([g, x])                                  # tile back to feature size
    x = layers.Concatenate()([b1, b2, b3, b4, g])
    x = layers.Conv2D(filters, 1, padding="same", activation="relu")(x)
    return x


def build_decoder(x):
    """3x bilinear upsample (x8) -> single-channel saliency logits."""
    x = Upsample2x()(x)      # /4
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = Upsample2x()(x)      # /2
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = Upsample2x()(x)      # /1
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    # Linear output; the KL loss normalises it to a distribution. For display,
    # min-max normalise (matches app_base.py postprocessing).
    x = layers.Conv2D(1, 1, padding="same", activation=None, name="saliency")(x)
    return x


# TF1 checkpoint variable name -> Keras layer name mapping.
# conv{block}_{layer}/weights -> block{b}_conv{l}  (Keras VGG16 naming).
_HYBRID_CKPT_MAP = {
    "conv1_1": "block1_conv1", "conv1_2": "block1_conv2",
    "conv2_1": "block2_conv1", "conv2_2": "block2_conv2",
    "conv3_1": "block3_conv1", "conv3_2": "block3_conv2", "conv3_3": "block3_conv3",
    "conv4_1": "block4_conv1", "conv4_2": "block4_conv2", "conv4_3": "block4_conv3",
    "conv5_1": "block5_conv1", "conv5_2": "block5_conv2", "conv5_3": "block5_conv3",
}
# Default local cache path for the hybrid checkpoint.
HYBRID_CKPT = "/tmp/vgg16_hybrid/vgg16_hybrid.ckpt"


def transfer_hybrid_vgg16_weights(model, ckpt_path=HYBRID_CKPT):
    """Load Places365-hybrid VGG16 weights from TF1 checkpoint into encoder.

    The checkpoint uses TF1 variable names (conv1_1/weights, conv1_1/biases…).
    We read each tensor directly and set it on the matching Keras Conv2D layer.
    Dilation (block5) does not change the kernel shape so weights transfer as-is.
    """
    import tensorflow as tf
    reader = tf.train.load_checkpoint(ckpt_path)
    n = 0
    for tf1_name, keras_name in _HYBRID_CKPT_MAP.items():
        kernel = reader.get_tensor(f"{tf1_name}/weights")   # shape [kH,kW,in,out]
        bias   = reader.get_tensor(f"{tf1_name}/biases")    # shape [out]
        model.get_layer(keras_name).set_weights([kernel, bias])
        n += 1
    print(f"[model] transferred hybrid VGG16 (ImageNet+Places365) weights into {n} layers")
    return model


def transfer_vgg16_weights(model, hybrid_ckpt=None):
    """Transfer encoder weights. Uses hybrid Places365 checkpoint when available,
    falls back to ImageNet-only keras.applications.VGG16.

    Args:
        model:       the xsoyelNet Keras model
        hybrid_ckpt: path to vgg16_hybrid.ckpt  (default: HYBRID_CKPT constant)
    """
    import os
    path = hybrid_ckpt or HYBRID_CKPT
    if os.path.exists(path + ".index"):
        return transfer_hybrid_vgg16_weights(model, path)
    # Fallback: ImageNet only (no Places365 context)
    print("[model] hybrid checkpoint not found — falling back to ImageNet VGG16")
    vgg = keras.applications.VGG16(include_top=False, weights="imagenet")
    for name in VGG16_CONV_NAMES:
        model.get_layer(name).set_weights(vgg.get_layer(name).get_weights())
    print(f"[model] transferred ImageNet VGG16 weights into {len(VGG16_CONV_NAMES)} layers")
    return model


def build_xsoyelnet(load_vgg=True, hybrid_ckpt=None):
    """Build the full xsoyelNet model. Accepts any input size divisible by 8.

    Args:
        load_vgg:    transfer VGG16 encoder weights (True by default)
        hybrid_ckpt: path to vgg16_hybrid.ckpt (without .index/.data suffix).
                     If None, uses HYBRID_CKPT constant; falls back to ImageNet.
    """
    inp = keras.Input(shape=(None, None, 3), name="image")
    x = build_encoder(inp)
    x = build_aspp(x)
    out = build_decoder(x)
    model = keras.Model(inp, out, name=MODEL_NAME)
    if load_vgg:
        transfer_vgg16_weights(model, hybrid_ckpt=hybrid_ckpt)
    return model


if __name__ == "__main__":
    build_xsoyelnet(load_vgg=False).summary()
