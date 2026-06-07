"""
SAMStrategy — Qt-free wrapper around Meta SAM 2.

Follows the same interface pattern as anomalib_strategy.py: zero Qt imports,
instantiate → load() → predict_bbox(). On first load() the checkpoint is
downloaded from Hugging Face into <project_root>/sam_weights/; all subsequent
runs load from disk with no network access.
"""

import logging
import pathlib
import sys
from typing import List, Tuple

import cv2
import numpy as np
import torch

from core.utils.geometry import simplify_polygon

logger = logging.getLogger(__name__)

# When frozen by PyInstaller (--onefile), __file__ resolves into the ephemeral
# _MEIPASS temp dir that is deleted on exit. Use the exe's own directory instead
# so weights persist between runs.
if getattr(sys, "frozen", False):
    _SAM_WEIGHTS_DIR = pathlib.Path(sys.executable).parent / "sam_weights"
else:
    _SAM_WEIGHTS_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "sam_weights"

VARIANTS: List[str] = [
    "sam2_t.pt",
    "sam2_s.pt",
    "sam2_b.pt",
    "sam2_l.pt",
]

_HF_IDS = {
    "sam2_t.pt": "facebook/sam2.1-hiera-tiny",
    "sam2_s.pt": "facebook/sam2.1-hiera-small",
    "sam2_b.pt": "facebook/sam2.1-hiera-base-plus",
    "sam2_l.pt": "facebook/sam2.1-hiera-large",
}


def weights_cached(variant: str) -> bool:
    """Return True if the checkpoint for *variant* is already on disk.

    Uses the same HF_MODEL_ID_TO_FILENAMES table as load() so the path check
    always matches what the loader will actually look for.
    """
    try:
        from sam2.build_sam import HF_MODEL_ID_TO_FILENAMES

        hf_id = _HF_IDS.get(variant, "facebook/sam2.1-hiera-tiny")
        _, ckpt_filename = HF_MODEL_ID_TO_FILENAMES[hf_id]
        return (_SAM_WEIGHTS_DIR / ckpt_filename).exists()
    except Exception:
        return False


class SAMStrategy:
    """Wraps Meta SAM 2 for bounding-box-prompted segmentation.

    Usage::

        strategy = SAMStrategy("sam2_t.pt")
        strategy.load()                          # download / init once
        pts, conf = strategy.predict_bbox(bgr, (x1, y1, x2, y2))
    """

    def __init__(self, variant: str = "sam2_t.pt") -> None:
        self._variant = variant
        self._predictor = None
        self.is_loaded: bool = False

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def load(self) -> None:
        """Load the SAM 2 checkpoint, downloading it to sam_weights/ if absent."""
        if self.is_loaded:
            return
        try:
            from sam2.build_sam import build_sam2, HF_MODEL_ID_TO_FILENAMES
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            from huggingface_hub import hf_hub_download

            hf_id = _HF_IDS.get(self._variant, "facebook/sam2.1-hiera-tiny")
            config_name, ckpt_filename = HF_MODEL_ID_TO_FILENAMES[hf_id]

            _SAM_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
            ckpt_path = _SAM_WEIGHTS_DIR / ckpt_filename

            if not ckpt_path.exists():
                logger.info("Downloading %s → %s", ckpt_filename, _SAM_WEIGHTS_DIR)
                hf_hub_download(
                    repo_id=hf_id,
                    filename=ckpt_filename,
                    local_dir=str(_SAM_WEIGHTS_DIR),
                )

            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("Loading SAM 2 variant: %s on %s", self._variant, device)
            self._predictor = SAM2ImagePredictor(
                build_sam2(config_name, str(ckpt_path), device=device)
            )
            self.is_loaded = True
            logger.info("SAM 2 ready.")
        except Exception as exc:
            raise RuntimeError(f"SAM load failed ({self._variant}): {exc}") from exc

    def set_variant(self, variant: str) -> None:
        """Switch to a different model variant; resets loaded state."""
        if variant != self._variant:
            self._variant = variant
            self._predictor = None
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
            is the SAM IoU quality score in [0, 1]. Returns ([], 0.0) when
            no mask is found.

        Raises:
            RuntimeError: If load() has not been called successfully.
        """
        if not self.is_loaded:
            raise RuntimeError("SAM model not loaded — call load() first.")

        x1, y1, x2, y2 = bbox
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        with torch.inference_mode():
            self._predictor.set_image(image_rgb)
            masks, scores, _ = self._predictor.predict(
                box=np.array([x1, y1, x2, y2]),
                multimask_output=True,  # request all 3; pick best by IOU score
            )

        if masks is None or len(masks) == 0:
            return [], 0.0

        best = int(np.argmax(scores))
        mask = masks[best].astype(np.float32)
        pts, _ = self._mask_to_polygon(mask, epsilon)
        return pts, float(scores[best])

    # ------------------------------------------------------------------ #
    # Internal helpers
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
