"""Unit tests for SAMStrategy — no ultralytics or Qt required."""

import numpy as np
import pytest

from ai_strategies.sam_strategy import SAMStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _filled_rect_mask(
    h: int = 100, w: int = 100, x1=20, y1=20, x2=80, y2=80
) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.float32)
    mask[y1:y2, x1:x2] = 1.0
    return mask


def _circle_mask(h: int = 200, w: int = 200, r: int = 80) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.float32)
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    mask[(X - cx) ** 2 + (Y - cy) ** 2 <= r**2] = 1.0
    return mask


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_mask_to_polygon_round_trip():
    """Polygon from a filled rect should have pts within the rect bounds."""
    strategy = SAMStrategy()
    mask = _filled_rect_mask(100, 100, 20, 20, 80, 80)
    pts, conf = strategy._mask_to_polygon(mask, epsilon=1.0)
    assert len(pts) >= 4
    for x, y in pts:
        assert 20 <= x <= 80, f"x={x} outside [20, 80]"
        assert 20 <= y <= 80, f"y={y} outside [20, 80]"
    assert 0.0 < conf < 1.0


def test_simplify_reduces_point_count():
    """Douglas-Peucker simplification should reduce a dense circular contour."""
    strategy = SAMStrategy()
    mask = _circle_mask(200, 200, 80)
    pts_loose, _ = strategy._mask_to_polygon(mask, epsilon=0.5)
    pts_tight, _ = strategy._mask_to_polygon(mask, epsilon=5.0)
    assert len(pts_tight) < len(pts_loose), (
        f"Expected fewer points with epsilon=5.0 ({len(pts_tight)}) "
        f"than epsilon=0.5 ({len(pts_loose)})"
    )


def test_load_guard_raises_before_load():
    """predict_bbox must raise RuntimeError if load() was never called."""
    strategy = SAMStrategy()
    dummy = np.zeros((100, 100, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="not loaded"):
        strategy.predict_bbox(dummy, (0, 0, 50, 50))


def test_set_variant_resets_loaded_state():
    """Changing the variant should mark the strategy as not loaded."""
    strategy = SAMStrategy("sam2_t.pt")
    strategy.is_loaded = True  # simulate a loaded state
    strategy._predictor = object()

    strategy.set_variant("sam2_b.pt")

    assert not strategy.is_loaded
    assert strategy._predictor is None
    assert strategy._variant == "sam2_b.pt"


def test_set_variant_same_variant_no_reset():
    """Setting the same variant should not reset loaded state."""
    strategy = SAMStrategy("sam2_t.pt")
    strategy.is_loaded = True
    strategy.set_variant("sam2_t.pt")
    assert strategy.is_loaded


def test_mask_to_polygon_empty_mask_returns_empty():
    """An all-zeros mask should return an empty polygon and zero confidence."""
    strategy = SAMStrategy()
    mask = np.zeros((100, 100), dtype=np.float32)
    pts, conf = strategy._mask_to_polygon(mask, epsilon=2.0)
    assert pts == []
    assert conf == 0.0


def test_mask_to_polygon_tiny_contour_returns_empty():
    """A contour with area < 10 px should be filtered out."""
    strategy = SAMStrategy()
    mask = np.zeros((100, 100), dtype=np.float32)
    mask[50, 50] = 1.0  # single pixel — area = 0 after CHAIN_APPROX_SIMPLE
    pts, conf = strategy._mask_to_polygon(mask, epsilon=0.0)
    assert pts == []
    assert conf == 0.0
