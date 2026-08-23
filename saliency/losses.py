"""
Saliency losses / metrics (Keras 3, backend-agnostic via keras.ops).

The original MSI-Net is trained with Kullback-Leibler divergence between the
predicted and ground-truth saliency distributions. We add Correlation
Coefficient (CC) as a monitoring metric.

━━━ TO IMPROVE CC AND NSS — next steps ━━━

Current state (v2, CAT2000 base):
  AUC ~0.903  (great — hard to push higher)
  CC  ~0.596  (room to improve)
  NSS ~1.527  (room to improve)

Why AUC is already high but CC/NSS lag:
  AUC measures rank ordering of saliency values — the model already predicts
  the RIGHT REGIONS well. CC and NSS measure the *shape* and *magnitude* of
  the distribution at fixation points, which requires finer spatial precision
  and better calibration.

Levers in this file (loss function):
─────────────────────────────────────
1. ADD CC TO THE LOSS (most impactful for CC metric):
   Currently we only minimise KL. Adding -CC as a loss term directly
   optimises for what we want to improve. Combine them:

     loss = kl_divergence(y_true, y_pred) - lambda_cc * correlation_coefficient(y_true, y_pred)
     # lambda_cc = 0.5 is a reasonable starting point; tune via val_loss

2. ADD NSS LOSS:
   NSS requires binary fixation maps (FIXATIONLOCS), not density maps.
   Formula: mean of normalised saliency at fixation locations.
   Once you have fixation point data (the .mat files in FIXATIONLOCS/),
   implement nss_loss(y_true_binary, y_pred) and add it to the combined loss.

3. COMBINED LOSS (literature best practice):
   combined = kl_divergence + alpha * (-cc) + beta * (-nss)
   Start with alpha=0.5, beta=0.5 and tune. MIT/Tübingen benchmark
   top models (DeepGaze, UNISAL) all use multi-term losses.
"""

import keras

K = keras.ops
EPS = 1e-7
_AXIS = (1, 2, 3)  # reduce over H, W, C per sample


def kl_divergence(y_true, y_pred):
    """KL(P_gt || P_pred). Both maps are normalised to sum-1 distributions.

    A relu/clamp guarantees the prediction is non-negative so the
    normalisation yields a valid distribution.
    """
    y_pred = K.maximum(y_pred, 0.0)
    y_true = y_true / (K.sum(y_true, axis=_AXIS, keepdims=True) + EPS)
    y_pred = y_pred / (K.sum(y_pred, axis=_AXIS, keepdims=True) + EPS)
    return K.sum(y_true * K.log(EPS + y_true / (EPS + y_pred)), axis=_AXIS)


def correlation_coefficient(y_true, y_pred):
    """Pearson CC between predicted and ground-truth maps (higher is better)."""
    yt = y_true - K.mean(y_true, axis=_AXIS, keepdims=True)
    yp = y_pred - K.mean(y_pred, axis=_AXIS, keepdims=True)
    num = K.sum(yt * yp, axis=_AXIS)
    den = K.sqrt(K.sum(yt * yt, axis=_AXIS) * K.sum(yp * yp, axis=_AXIS)) + EPS
    return num / den
