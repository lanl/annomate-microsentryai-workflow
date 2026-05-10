"""Integration tests for SAM canvas extensions in ImageLabel.

No ML model is required — all tests operate on canvas state and signals only.
"""

import numpy as np
import pytest
from PySide6.QtCore import Qt, QPointF

from views.annomate.image_label import ImageLabel, SAM_BBOX, POLYGON


@pytest.fixture
def canvas(qtbot):
    widget = ImageLabel()
    # Load a small synthetic image so _base_scale and coordinate helpers work
    bgr = np.zeros((100, 100, 3), dtype=np.uint8)
    widget.set_image(bgr)
    qtbot.addWidget(widget)
    widget.show()
    return widget


# ---------------------------------------------------------------------------
# Tool activation
# ---------------------------------------------------------------------------


def test_sam_tool_sets_crosscursor(canvas):
    canvas.set_tool(SAM_BBOX)
    assert canvas.cursor().shape() == Qt.CrossCursor


def test_polygon_tool_does_not_affect_sam_state(canvas):
    canvas.set_tool(SAM_BBOX)
    canvas._sam_bbox_start = QPointF(10, 10)
    canvas._sam_bbox_end = QPointF(50, 50)
    # Switching to polygon should clear SAM state
    canvas.set_tool(POLYGON)
    assert canvas._sam_bbox_start is None
    assert canvas._sam_bbox_end is None
    assert canvas._sam_ghost is None


# ---------------------------------------------------------------------------
# Ghost helpers
# ---------------------------------------------------------------------------


def test_set_sam_ghost_stores_display_pts(canvas):
    pts_orig = [(10.0, 10.0), (50.0, 10.0), (50.0, 50.0), (10.0, 50.0)]
    canvas.set_sam_ghost(pts_orig, 0.85)
    assert canvas._sam_ghost is not None
    display_pts, conf = canvas._sam_ghost
    assert len(display_pts) == 4
    assert abs(conf - 0.85) < 1e-6


def test_set_sam_ghost_empty_pts_clears_ghost(canvas):
    canvas.set_sam_ghost([(0, 0), (1, 0), (1, 1)], 0.5)
    canvas.set_sam_ghost([], 0.0)
    assert canvas._sam_ghost is None


def test_clear_sam_ghost_clears_all_state(canvas):
    canvas.set_sam_ghost([(0, 0), (10, 0), (10, 10)], 0.9)
    canvas._sam_bbox_start = QPointF(5, 5)
    canvas._sam_bbox_end = QPointF(20, 20)
    canvas.clear_sam_ghost()
    assert canvas._sam_ghost is None
    assert canvas._sam_bbox_start is None
    assert canvas._sam_bbox_end is None


# ---------------------------------------------------------------------------
# Accept / reject
# ---------------------------------------------------------------------------


def test_accept_sam_ghost_emits_polygon_finished(canvas, qtbot):
    pts_orig = [(10.0, 20.0), (40.0, 20.0), (40.0, 60.0), (10.0, 60.0)]
    canvas.set_sam_ghost(pts_orig, 0.7)

    received = []
    canvas.polygonFinished.connect(received.append)

    result = canvas.accept_sam_ghost()

    assert result is True
    assert len(received) == 1
    emitted = received[0]
    assert len(emitted) == 4
    # Coordinates should round-trip through display→original within float tolerance
    for (ex, ey), (rx, ry) in zip(pts_orig, emitted):
        assert abs(ex - rx) < 1.0
        assert abs(ey - ry) < 1.0


def test_accept_sam_ghost_clears_state(canvas):
    canvas.set_sam_ghost([(0, 0), (10, 0), (10, 10)], 0.5)
    canvas.accept_sam_ghost()
    assert canvas._sam_ghost is None
    assert canvas._sam_bbox_start is None
    assert canvas._sam_bbox_end is None


def test_accept_sam_ghost_returns_false_when_no_ghost(canvas):
    result = canvas.accept_sam_ghost()
    assert result is False


# ---------------------------------------------------------------------------
# Keyboard
# ---------------------------------------------------------------------------


def test_enter_accepts_ghost(canvas, qtbot):
    canvas.set_tool(SAM_BBOX)
    canvas.set_sam_ghost([(0, 0), (10, 0), (10, 10)], 0.6)

    received = []
    canvas.polygonFinished.connect(received.append)

    qtbot.keyPress(canvas, Qt.Key_Return)

    assert len(received) == 1


def test_escape_clears_ghost_and_emits_tool_canceled(canvas, qtbot):
    canvas.set_tool(SAM_BBOX)
    canvas.set_sam_ghost([(0, 0), (10, 0), (10, 10)], 0.6)

    canceled = []
    canvas.toolCanceled.connect(lambda: canceled.append(True))

    qtbot.keyPress(canvas, Qt.Key_Escape)

    assert canvas._sam_ghost is None
    assert canvas.current_tool is None
    assert len(canceled) == 1
