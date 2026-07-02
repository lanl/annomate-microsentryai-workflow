"""
SAMStrategy — Qt-free wrapper around SAM 2 (ONNX encoder/decoder pair).

Follows the same interface pattern as onnx_anomaly_strategy.py: zero Qt
imports, instantiate → load() → predict_bbox(). On first load() the ONNX
encoder and decoder are downloaded from Hugging Face into
<project_root>/sam_weights/; all subsequent runs load from disk with no
network access.

Prompt encoding and pre/post-processing follow the reference implementation
in vietanhdev/samexporter (also used by AnyLabeling): a bounding box is
encoded as two decoder points labelled 2.0 (top-left) and 3.0 (bottom-right).
"""

import logging
import pathlib
import sys
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ai_strategies.ort_backend import create_session
from core.utils.geometry import simplify_polygon

logger = logging.getLogger(__name__)

# When frozen by PyInstaller (--onefile), __file__ resolves into the ephemeral
# _MEIPASS temp dir that is deleted on exit. Use the exe's own directory instead
# so weights persist between runs.
if getattr(sys, "frozen", False):
    _SAM_WEIGHTS_DIR = pathlib.Path(sys.executable).parent / "sam_weights"
else:
    _SAM_WEIGHTS_DIR = (
        pathlib.Path(__file__).resolve().parent.parent.parent / "sam_weights"
    )

_HF_REPO = "vietanhdev/segment-anything-2-onnx-models"

# variant → (encoder filename, decoder filename) in the HF repo.
_VARIANT_FILES = {
    "sam2_t": ("sam2_hiera_tiny.encoder.onnx", "sam2_hiera_tiny.decoder.onnx"),
    "sam2_s": ("sam2_hiera_small.encoder.onnx", "sam2_hiera_small.decoder.onnx"),
    "sam2_b": (
        "sam2_hiera_base_plus.encoder.onnx",
        "sam2_hiera_base_plus.decoder.onnx",
    ),
    "sam2_l": ("sam2_hiera_large.encoder.onnx", "sam2_hiera_large.decoder.onnx"),
}

VARIANTS: List[str] = list(_VARIANT_FILES)

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def weights_cached(variant: str) -> bool:
    """Return True if both ONNX files for *variant* are already on disk."""
    try:
        encoder_file, decoder_file = _VARIANT_FILES.get(
            variant, _VARIANT_FILES["sam2_t"]
        )
        return (_SAM_WEIGHTS_DIR / encoder_file).exists() and (
            _SAM_WEIGHTS_DIR / decoder_file
        ).exists()
    except Exception:
        return False


class SAMStrategy:
    """Wraps SAM 2 (ONNX) for bounding-box-prompted segmentation.

    Usage::

        strategy = SAMStrategy("sam2_t")
        strategy.load()                          # download / init once
        pts, conf = strategy.predict_bbox(bgr, (x1, y1, x2, y2))
    """

    def __init__(self, variant: str = "sam2_t") -> None:
        self._variant = variant
        self._encoder = None
        self._decoder = None
        self._encoder_hw: Tuple[int, int] = (1024, 1024)
        self._cached_image: Optional[np.ndarray] = None
        self._cached_embedding: Optional[dict] = None
        self.is_loaded: bool = False

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def load(self) -> None:
        """Load the SAM 2 ONNX pair, downloading to sam_weights/ if absent."""
        if self.is_loaded:
            return
        try:
            encoder_file, decoder_file = _VARIANT_FILES.get(
                self._variant, _VARIANT_FILES["sam2_t"]
            )

            _SAM_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

            for filename in (encoder_file, decoder_file):
                if not (_SAM_WEIGHTS_DIR / filename).exists():
                    from huggingface_hub import hf_hub_download

                    logger.info("Downloading %s → %s", filename, _SAM_WEIGHTS_DIR)
                    hf_hub_download(
                        repo_id=_HF_REPO,
                        filename=filename,
                        local_dir=str(_SAM_WEIGHTS_DIR),
                    )

            logger.info("Loading SAM 2 variant: %s", self._variant)
            self._encoder, provider = create_session(
                str(_SAM_WEIGHTS_DIR / encoder_file), "auto"
            )
            self._decoder, _ = create_session(
                str(_SAM_WEIGHTS_DIR / decoder_file), "auto"
            )

            shape = self._encoder.get_inputs()[0].shape  # [1, 3, H, W]
            self._encoder_hw = (int(shape[2]), int(shape[3]))

            self.is_loaded = True
            logger.info("SAM 2 ready on %s.", provider)
        except Exception as exc:
            raise RuntimeError(f"SAM load failed ({self._variant}): {exc}") from exc

    def set_variant(self, variant: str) -> None:
        """Switch to a different model variant; resets loaded state."""
        if variant != self._variant:
            self._variant = variant
            self._encoder = None
            self._decoder = None
            self._cached_image = None
            self._cached_embedding = None
            self.is_loaded = False

    def predict_bbox(
        self,
        image_bgr: np.ndarray,
        bbox: Tuple[float, float, float, float],
        epsilon: float = 2.0,
    ) -> Tuple[List[Tuple[float, float]], float]:
        """Run SAM inference for a single bounding-box prompt.

        The image embedding is cached by array identity, so repeated prompts
        on the same image skip the expensive encoder pass.

        Args:
            image_bgr: Full-resolution BGR image array (original pixels).
            bbox: (x1, y1, x2, y2) in original image coordinates.
            epsilon: Douglas-Peucker simplification tolerance in pixels.

        Returns:
            (polygon_pts, confidence) where polygon_pts is a list of
            (x, y) tuples in original image coordinates, and confidence
            is the SAM IoU quality score in [0, 1]. Returns ([], 0.0) when
            no mask is found.

        Raises:
            RuntimeError: If load() has not been called successfully.
        """
        if not self.is_loaded:
            raise RuntimeError("SAM model not loaded — call load() first.")

        orig_h, orig_w = image_bgr.shape[:2]
        embedding = self._encode(image_bgr)

        x1, y1, x2, y2 = bbox
        enc_h, enc_w = self._encoder_hw
        # Box prompt = two points labelled 2.0 (top-left) / 3.0 (bottom-right),
        # scaled from original image coordinates into encoder-input space.
        point_coords = np.array(
            [[[x1 * enc_w / orig_w, y1 * enc_h / orig_h],
              [x2 * enc_w / orig_w, y2 * enc_h / orig_h]]],
            dtype=np.float32,
        )
        point_labels = np.array([[2.0, 3.0]], dtype=np.float32)
        mask_input = np.zeros((1, 1, enc_h // 4, enc_w // 4), dtype=np.float32)
        has_mask_input = np.array([0.0], dtype=np.float32)

        masks, iou_predictions = self._decoder.run(
            ["masks", "iou_predictions"],
            {
                "image_embed": embedding["image_embed"],
                "high_res_feats_0": embedding["high_res_feats_0"],
                "high_res_feats_1": embedding["high_res_feats_1"],
                "point_coords": point_coords,
                "point_labels": point_labels,
                "mask_input": mask_input,
                "has_mask_input": has_mask_input,
            },
        )

        if masks is None or masks.shape[1] == 0:
            return [], 0.0

        scores = np.asarray(iou_predictions).squeeze()
        best = int(np.argmax(scores))
        mask = cv2.resize(
            masks[0, best], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR
        )
        pts, _ = self._mask_to_polygon(mask, epsilon)
        return pts, float(scores.ravel()[best])

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _encode(self, image_bgr: np.ndarray) -> dict:
        """Run the encoder for *image_bgr*, reusing the cached embedding.

        Identity comparison is safe here: the caller (SAMController) passes
        the same ndarray object for every prompt on the current image, and
        the cache holds a reference so the id cannot be reused.
        """
        if self._cached_embedding is not None and image_bgr is self._cached_image:
            return self._cached_embedding

        enc_h, enc_w = self._encoder_hw
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (enc_w, enc_h))
        tensor = (rgb.astype(np.float32) / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
        tensor = tensor.transpose(2, 0, 1)[np.newaxis]

        high_res_feats_0, high_res_feats_1, image_embed = self._encoder.run(
            ["high_res_feats_0", "high_res_feats_1", "image_embed"],
            {"image": tensor},
        )

        self._cached_image = image_bgr
        self._cached_embedding = {
            "high_res_feats_0": high_res_feats_0,
            "high_res_feats_1": high_res_feats_1,
            "image_embed": image_embed,
        }
        return self._cached_embedding

    def _mask_to_polygon(
        self,
        mask: np.ndarray,
        epsilon: float,
    ) -> Tuple[List[Tuple[float, float]], float]:
        """Convert a binary mask to the largest external polygon."""
        binary = (mask > 0).astype(np.uint8)
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return [], 0.0

        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        if area < 10:
            return [], 0.0

        pts: List[Tuple[float, float]] = [(float(p[0][0]), float(p[0][1])) for p in cnt]
        pts = simplify_polygon(pts, epsilon)

        total_pixels = float(mask.shape[0] * mask.shape[1])
        confidence = area / total_pixels if total_pixels > 0 else 0.0

        return pts, float(confidence)
