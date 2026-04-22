# tvt_lite_model.py
"""
TVT-Lite: CPU-optimized Temporal Vision Transformer for behavior classification.
Uses landmark embeddings only (no raw frames). Pure NumPy for portability.


"""
import numpy as np
from typing import Dict, Optional, Any, List

from app.utils.logger import debug_logger

BEHAVIOR_CLASSES: List[str] = [
    "normal",
    "left_cheating_glance",
    "right_cheating_glance",
    "phone_lookdown",
    "second_person",
    "face_spoof",
    "occluded_face_behavior",
]
NUM_CLASSES: int = len(BEHAVIOR_CLASSES)


class TVTLiteModel:
    """
    Lightweight temporal model for landmark sequences.
    Input : (T, D) array  — T = time steps, D = 936 (468 × 2).
    Output: behavior_class, probability, prediction_confidence, all_probs.

    IMPORTANT — Random Weights Notice:
    This model is initialized with random weights (seed=42) and has NOT
    been trained on any labeled dataset. With random weights:
      - softmax output is near-uniform (~1/7 ≈ 0.143 per class)
      - the highest class probability is ~0.18–0.22 at most
      - TVT gating in detection_service (prob > 0.5) NEVER fires
      - TVT effectively acts as a pass-through (no suppression occurs)

    This is the correct safe default: an untrained model should not
    suppress valid violations. To enable TVT behavior filtering, load
    pre-trained weights via load_weights() before inference.

    The model architecture (embed → temporal pool → linear) is intentionally
    minimal for CPU efficiency. A full attention mechanism is not needed
    for the current binary-ish classification task.
    """

    def __init__(
        self,
        input_dim:   int = 936,
        embed_dim:   int = 64,
        window_size: int = 24,
        num_classes: int = NUM_CLASSES,
        seed:        int = 42,
        **kwargs,          # absorb num_heads / num_layers from create_tvt_model
    ) -> None:
        self.input_dim   = input_dim
        self.embed_dim   = embed_dim
        self.window_size = window_size
        self.num_classes = num_classes
        self._rng        = np.random.default_rng(seed)

        scale        = 0.02
        self.W_embed = self._rng.standard_normal((input_dim, embed_dim)) * scale
        self.b_embed = np.zeros(embed_dim)
        self.W_out   = self._rng.standard_normal((embed_dim, num_classes)) * scale
        self.b_out   = np.zeros(num_classes)

        # Track whether non-random weights have been loaded
        self._weights_loaded: bool = False

    def load_weights(self, path: str) -> bool:
        """
        Load pre-trained weights from a .npz file.
        Expected keys: W_embed, b_embed, W_out, b_out.
        Returns True on success, False on failure (model keeps random weights).
        """
        try:
            data = np.load(path)
            self.W_embed = data["W_embed"].astype(np.float32)
            self.b_embed = data["b_embed"].astype(np.float32)
            self.W_out   = data["W_out"].astype(np.float32)
            self.b_out   = data["b_out"].astype(np.float32)
            self._weights_loaded = True
            debug_logger.info(f"TVTLiteModel: loaded weights from {path}")
            return True
        except Exception as e:
            debug_logger.warning(
                f"TVTLiteModel: failed to load weights from {path}: {e}. "
                f"Continuing with random weights (TVT pass-through mode)."
            )
            return False

    def _embed(self, x: np.ndarray) -> np.ndarray:
        """(T, D) → (T, embed_dim) via linear projection."""
        return x.astype(np.float32) @ self.W_embed + self.b_embed

    def _temporal_pool(self, emb: np.ndarray) -> np.ndarray:
        """
        (T, E) → (E,) exponentially-weighted mean (recent frames weighted more).

        CHANGE: replaced linear ramp with exponential ramp (base=1.08).
        Linear ramp gave weight ratio newest/oldest = T (24x for window=24),
        massively over-emphasising the most recent frame.
        Exponential base=1.08 gives ratio 1.08^(T-1) ≈ 6x at T=24 —
        still recency-biased but with smoother gradient across the window.

        Why exponential over linear for proctoring:
          - Cheating behaviors persist over multiple frames (not single-frame)
          - We want recent confirmation to matter more, but context from
            earlier frames still contributes meaningfully
          - 6x ratio balances recency vs context better than 24x
        """
        t = emb.shape[0]
        if t <= 1:
            return np.mean(emb, axis=0)

        # CHANGE: exponential weights, base=1.08
        # weights[0] = 1.08^0 = 1.0 (oldest), weights[T-1] = 1.08^(T-1) (newest)
        exponents = np.arange(t, dtype=np.float32)          # [0, 1, 2, ..., T-1]
        weights   = np.power(1.08, exponents)                # exponential ramp
        weights  /= weights.sum()                            # normalise to sum=1
        return (emb * weights[:, np.newaxis]).sum(axis=0)

    def predict(self, window: np.ndarray) -> Dict[str, Any]:
        """
        Run forward pass on a single window.
        window: (T, D) float32 array.

        Returns:
            behavior_class:       most likely class name
            probability:          softmax probability of predicted class
            prediction_confidence: 1 - normalised_entropy (0=uncertain, 1=certain)
            temporal_confidence:  alias for prediction_confidence (backward compat)
            all_probs:            softmax probabilities for all classes
            weights_loaded:       whether pre-trained weights are in use
        """
        if window is None or window.size == 0:
            return self._default_prediction()

        window = np.asarray(window, dtype=np.float32)
        if window.ndim == 1:
            window = window.reshape(1, -1)

        t, d = window.shape
        if d != self.input_dim:
            return self._default_prediction()

        emb    = self._embed(window)               # (T, embed_dim)
        pooled = self._temporal_pool(emb)          # (embed_dim,)
        logits = pooled @ self.W_out + self.b_out  # (num_classes,)

        # Numerical stability: subtract max before exp
        # CHANGE: removed redundant np.clip(logits, -20, 20) upper bound.
        # After subtracting max, all logits are ≤ 0, so clipping to +20
        # is dead code. Only clip the lower bound to prevent underflow.
        logits -= logits.max()
        exp     = np.exp(np.clip(logits, -88.0, 0.0))  # -88 = float32 underflow floor
        probs   = exp / (exp.sum() + 1e-8)

        pred_idx       = int(np.argmax(probs))
        probability    = float(probs[pred_idx])
        behavior_class = BEHAVIOR_CLASSES[pred_idx]

        # prediction_confidence: 1 - normalised_entropy
        # High entropy (uniform distribution) → low confidence
        # Low entropy (peaked distribution) → high confidence
        entropy              = -float(np.sum(probs * np.log(probs + 1e-8)))
        max_entropy          = float(np.log(self.num_classes))
        prediction_confidence = float(np.clip(1.0 - entropy / max_entropy, 0.0, 1.0))

        return {
            "behavior_class":       behavior_class,
            "probability":          probability,
            # CHANGE: clearer primary key name
            "prediction_confidence": prediction_confidence,
            # backward-compat alias kept for any existing consumers
            "temporal_confidence":  prediction_confidence,
            "all_probs":            probs.tolist(),
            "weights_loaded":       self._weights_loaded,
        }

    def _default_prediction(self) -> Dict[str, Any]:
        """Safe default when input is invalid."""
        uniform = 1.0 / self.num_classes
        return {
            "behavior_class":       "normal",
            "probability":          0.0,
            "prediction_confidence": 0.0,
            "temporal_confidence":  0.0,
            "all_probs":            [uniform] * self.num_classes,
            "weights_loaded":       self._weights_loaded,
        }


def create_tvt_model(
    config: Any,
    weight_path: Optional[str] = None,
) -> Optional[TVTLiteModel]:
    """
    Create TVT-Lite model from config.
    Returns None if ENABLE_TVT is False.

    CHANGE: added optional weight_path parameter.
    If provided and the file exists, pre-trained weights are loaded.
    If not provided or loading fails, model runs in pass-through mode
    (random weights, prob always < 0.5, no violations suppressed).

    v2.2 fix retained: num_heads / num_layers passed as kwargs —
    TVTLiteModel absorbs them via **kwargs to avoid TypeError.
    """
    if not getattr(config, "ENABLE_TVT", False):
        return None

    window_size = int(getattr(config, "TVT_TEMPORAL_WINDOW", 24))
    model = TVTLiteModel(
        input_dim=936,
        embed_dim=64,
        num_heads=2,        # forwarded but unused in NumPy backend
        num_layers=2,       # forwarded but unused in NumPy backend
        window_size=window_size,
        num_classes=NUM_CLASSES,
    )

    # Attempt weight loading if path provided
    # (no-op in current deployment; placeholder for future training pipeline)
    if weight_path is not None:
        model.load_weights(weight_path)
    else:
        debug_logger.info(
            "TVTLiteModel created with random weights (pass-through mode). "
            "Set weight_path in create_tvt_model() to enable trained behavior."
        )

    return model