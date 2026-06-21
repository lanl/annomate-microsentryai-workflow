"""
SAMOnnxStrategy — ONNX Runtime wrapper for SAM 2 segmentation.

Uses pre-exported encoder + decoder ONNX models from
vietanhdev/segment-anything-2-onnx-models on Hugging Face. No PyTorch
required. Same public API as SAMStrategy so SAMController works unchanged.
"""

import logging
import pathlib
import sys
from typing import List, Tuple

import cv2
import numpy as np
import onnxruntime as ort

from core.utils.geometry import simplify_polygon

logger = logging.getLogger(__name__)

# Weights directory: beside the exe when frozen, else <repo-root>/sam_weights/
if getattr(sys, "frozen", False):
    _SAM_WEIGHTS_DIR = pathlib.Path(sys.executable).parent / "sam_weights"
else:
    _SAM_WEIGHTS_DIR = (
        pathlib.Path(__file__).resolve().parent.parent.parent / "sam_weights"
    )

_HF_REPO = "vietanhdev/segment-anything-2-onnx-models"

# Maps the same variant keys used by SAMStrategy → (encoder_file, decoder_file)
# Filenames match vietanhdev/segment-anything-2-onnx-models exactly (dot separators).
_ONNX_FILES = {
    "sam2_t.pt": ("sam2_hiera_tiny.encoder.onnx",      "sam2_hiera_tiny.decoder.onnx"),
    "sam2_s.pt": ("sam2_hiera_small.encoder.onnx",     "sam2_hiera_small.decoder.onnx"),
    "sam2_b.pt": ("sam2_hiera_base_plus.encoder.onnx", "sam2_hiera_base_plus.decoder.onnx"),
    "sam2_l.pt": ("sam2_hiera_large.encoder.onnx",     "sam2_hiera_large.decoder.onnx"),
}

# ImageNet normalisation constants used by the SAM2 ONNX encoder
_IMG_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
_IMG_STD  = np.array([ 58.395,  57.12,  57.375], dtype=np.float32)

_ENCODER_SIZE = 1024  # SAM2 encoder always expects 1024×1024


def weights_cached(variant: str) -> bool:
    """Return True if both ONNX files for *variant* are already on disk."""
    files = _ONNX_FILES.get(variant)
    if not files:
        return False
    enc_file, dec_file = files
    return (_SAM_WEIGHTS_DIR / enc_file).exists() and (_SAM_WEIGHTS_DIR / dec_file).exists()


class SAMOnnxStrategy:
    """Wraps vietanhdev SAM 2 ONNX models for bounding-box-prompted segmentation.

    Usage::

        strategy = SAMOnnxStrategy("sam2_t.pt")
        strategy.load()                          # download / init once
        pts, conf = strategy.predict_bbox(bgr, (x1, y1, x2, y2))
    """

    def __init__(self, variant: str = "sam2_t.pt") -> None:
        self._variant = variant
        self._enc_session: ort.InferenceSession | None = None
        self._dec_session: ort.InferenceSession | None = None
        self.is_loaded: bool = False

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def load(self) -> None:
        """Download (if needed) and initialise encoder + decoder ONNX sessions."""
        if self.is_loaded:
            return

        if self._variant not in _ONNX_FILES:
            raise RuntimeError(
                f"Unknown SAM variant {self._variant!r}. "
                f"Valid options: {list(_ONNX_FILES)}"
            )

        enc_file, dec_file = _ONNX_FILES[self._variant]
        _SAM_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

        enc_path = _SAM_WEIGHTS_DIR / enc_file
        dec_path = _SAM_WEIGHTS_DIR / dec_file

        try:
            from huggingface_hub import hf_hub_download

            if not enc_path.exists():
                logger.info("Downloading %s → %s", enc_file, _SAM_WEIGHTS_DIR)
                hf_hub_download(
                    repo_id=_HF_REPO,
                    filename=enc_file,
                    local_dir=str(_SAM_WEIGHTS_DIR),
                )
            if not dec_path.exists():
                logger.info("Downloading %s → %s", dec_file, _SAM_WEIGHTS_DIR)
                hf_hub_download(
                    repo_id=_HF_REPO,
                    filename=dec_file,
                    local_dir=str(_SAM_WEIGHTS_DIR),
                )
        except Exception as exc:
            raise RuntimeError(
                f"SAM ONNX download failed for {self._variant}: {exc}"
            ) from exc

        available = ort.get_available_providers()
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in available
            else ["CPUExecutionProvider"]
        )
        logger.info("Loading SAM2 ONNX variant %s with providers %s", self._variant, providers)

        try:
            self._enc_session = ort.InferenceSession(str(enc_path), providers=providers)
            self._dec_session = ort.InferenceSession(str(dec_path), providers=providers)
        except Exception as exc:
            raise RuntimeError(
                f"SAM ONNX session creation failed ({self._variant}): {exc}"
            ) from exc

        self.is_loaded = True
        logger.info("SAM2 ONNX ready (%s).", self._variant)

    def set_variant(self, variant: str) -> None:
        """Switch to a different model variant; resets loaded state."""
        if variant != self._variant:
            self._variant = variant
            self._enc_session = None
            self._dec_session = None
            self.is_loaded = False

    def predict_bbox(
        self,
        image_bgr: np.ndarray,
        bbox: Tuple[float, float, float, float],
        epsilon: float = 2.0,
    ) -> Tuple[List[Tuple[float, float]], float]:
        """Run SAM inference for a single bounding-box prompt.

        Args:
            image_bgr: Full-resolution BGR image array (original pixels).
            bbox: (x1, y1, x2, y2) in original image coordinates.
            epsilon: Douglas-Peucker simplification tolerance in pixels.

        Returns:
            (polygon_pts, confidence) where polygon_pts is a list of
            (x, y) tuples in original image coordinates, and confidence
            is the SAM IoU quality score in [0, 1].  Returns ([], 0.0)
            when no mask is found.

        Raises:
            RuntimeError: If load() has not been called successfully.
        """
        if not self.is_loaded:
            raise RuntimeError("SAM model not loaded — call load() first.")

        orig_h, orig_w = image_bgr.shape[:2]
        x1, y1, x2, y2 = bbox

        # ── Encode image ────────────────────────────────────────────────────
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(image_rgb, (_ENCODER_SIZE, _ENCODER_SIZE)).astype(np.float32)
        # ImageNet normalisation
        resized = (resized - _IMG_MEAN) / _IMG_STD
        enc_input = resized.transpose(2, 0, 1)[np.newaxis]  # [1, 3, 1024, 1024]

        enc_outputs = self._enc_session.run(None, {
            self._enc_session.get_inputs()[0].name: enc_input
        })
        enc_out_names = [o.name for o in self._enc_session.get_outputs()]
        enc_map = dict(zip(enc_out_names, enc_outputs))

        # ── Scale bbox to encoder space ──────────────────────────────────────
        sx = _ENCODER_SIZE / orig_w
        sy = _ENCODER_SIZE / orig_h
        bx1, by1 = x1 * sx, y1 * sy
        bx2, by2 = x2 * sx, y2 * sy

        # SAM2 ONNX decoder: 2 corner points with labels 2 (top-left) and 3 (bottom-right)
        point_coords = np.array([[[bx1, by1], [bx2, by2]]], dtype=np.float32)  # [1, 2, 2]
        point_labels = np.array([[2, 3]], dtype=np.float32)                     # [1, 2]
        mask_input   = np.zeros((1, 1, 256, 256), dtype=np.float32)
        has_mask_input = np.array([0], dtype=np.float32)

        dec_inputs = {
            "image_embed":     enc_map.get("image_embed", enc_outputs[0]),
            "high_res_feats_0": enc_map.get("high_res_feats_0", enc_outputs[1]),
            "high_res_feats_1": enc_map.get("high_res_feats_1", enc_outputs[2]),
            "point_coords":    point_coords,
            "point_labels":    point_labels,
            "mask_input":      mask_input,
            "has_mask_input":  has_mask_input,
        }

        dec_outputs = self._dec_session.run(None, dec_inputs)
        dec_out_names = [o.name for o in self._dec_session.get_outputs()]
        dec_map = dict(zip(dec_out_names, dec_outputs))

        masks = dec_map.get("masks")          # [1, 3, H, W]
        iou   = dec_map.get("iou_predictions")  # [1, 3]

        if masks is None or iou is None or masks.shape[1] == 0:
            return [], 0.0

        best = int(np.argmax(iou[0]))
        mask = masks[0, best]  # [H, W] — may be at encoder resolution

        # Scale mask back to original image dimensions
        if mask.shape != (orig_h, orig_w):
            mask = cv2.resize(
                mask.astype(np.float32), (orig_w, orig_h), interpolation=cv2.INTER_LINEAR
            )

        pts, _ = self._mask_to_polygon(mask, epsilon)
        return pts, float(iou[0, best])

    # ------------------------------------------------------------------ #
    # Internal helpers — identical to SAMStrategy._mask_to_polygon
    # ------------------------------------------------------------------ #

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
