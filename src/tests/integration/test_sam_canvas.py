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
    """Verify that activating the SAM_BBOX tool changes the cursor to a crosshair.

    The cross cursor provides visual feedback that the user is in bounding-box drawing
    mode. Success means canvas.cursor().shape() equals Qt.CrossCursor after set_tool.
    """
    canvas.set_tool(SAM_BBOX)
    assert canvas.cursor().shape() == Qt.CrossCursor


def test_polygon_tool_does_not_affect_sam_state(canvas):
    """Verify that switching from SAM_BBOX to POLYGON tool clears all SAM-related canvas state.

    Activates SAM mode and manually sets bbox start/end points. Switching to the
    polygon tool should clear these SAM-specific fields. Success means bbox_start,
    bbox_end, and sam_ghost are all None after switching to POLYGON.
    """
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
    """Verify that set_sam_ghost stores the polygon points and confidence score in canvas state.

    Passes a 4-point rectangle ghost polygon and a confidence score of 0.85. Success
    means _sam_ghost is not None, contains 4 display points, and the confidence is
    within floating-point tolerance of 0.85.
    """
    pts_orig = [(10.0, 10.0), (50.0, 10.0), (50.0, 50.0), (10.0, 50.0)]
    canvas.set_sam_ghost(pts_orig, 0.85)
    assert canvas._sam_ghost is not None
    display_pts, conf = canvas._sam_ghost
    assert len(display_pts) == 4
    assert abs(conf - 0.85) < 1e-6


def test_set_sam_ghost_empty_pts_clears_ghost(canvas):
    """Verify that calling set_sam_ghost with an empty point list clears the ghost polygon.

    After setting a valid ghost, passing an empty list should remove it entirely.
    Success means _sam_ghost is None after calling set_sam_ghost with [].
    """
    canvas.set_sam_ghost([(0, 0), (1, 0), (1, 1)], 0.5)
    canvas.set_sam_ghost([], 0.0)
    assert canvas._sam_ghost is None


def test_clear_sam_ghost_clears_all_state(canvas):
    """Verify that clear_sam_ghost removes the ghost polygon and both bbox endpoint fields.

    Sets a ghost polygon and bbox start/end points, then calls clear_sam_ghost(). All
    three SAM-related canvas attributes should be None afterward. Success means
    _sam_ghost, _sam_bbox_start, and _sam_bbox_end are all None.
    """
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
    """Verify that accept_sam_ghost emits polygonFinished with the ghost's original-space coordinates.

    Sets a 4-point ghost polygon and calls accept_sam_ghost(). The polygonFinished
    signal should fire once with the polygon points converted back from display space
    to original image coordinates. Success means the signal fires, the emitted polygon
    has 4 points, and each coordinate is within 1 pixel of the original values.
    """
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
    """Verify that accept_sam_ghost clears all SAM state after emitting the polygon.

    After a successful accept, the ghost polygon and bbox endpoints should all be
    cleared. Success means _sam_ghost, _sam_bbox_start, and _sam_bbox_end are all None.
    """
    canvas.set_sam_ghost([(0, 0), (10, 0), (10, 10)], 0.5)
    canvas.accept_sam_ghost()
    assert canvas._sam_ghost is None
    assert canvas._sam_bbox_start is None
    assert canvas._sam_bbox_end is None


def test_accept_sam_ghost_returns_false_when_no_ghost(canvas):
    """Verify that accept_sam_ghost returns False when there is no active ghost polygon.

    Calling accept on a canvas with no ghost set should return False without raising
    an exception or emitting the signal. Success means the return value is False.
    """
    result = canvas.accept_sam_ghost()
    assert result is False


# ---------------------------------------------------------------------------
# Keyboard
# ---------------------------------------------------------------------------


def test_enter_accepts_ghost(canvas, qtbot):
    """Verify that pressing Enter while in SAM mode with a ghost accepts the polygon.

    With SAM_BBOX tool active and a ghost polygon set, pressing the Return key should
    trigger accept_sam_ghost and emit polygonFinished. Success means the signal fires
    exactly once.
    """
    canvas.set_tool(SAM_BBOX)
    canvas.set_sam_ghost([(0, 0), (10, 0), (10, 10)], 0.6)

    received = []
    canvas.polygonFinished.connect(received.append)

    qtbot.keyPress(canvas, Qt.Key_Return)

    assert len(received) == 1


def test_escape_clears_ghost_and_emits_tool_canceled(canvas, qtbot):
    """Verify that pressing Escape while in SAM mode clears the ghost and emits toolCanceled.

    With SAM_BBOX tool and a ghost polygon active, pressing Escape should clear the
    ghost polygon, deactivate the current tool, and emit the toolCanceled signal.
    Success means _sam_ghost is None, current_tool is None, and the signal fired once.
    """
    canvas.set_tool(SAM_BBOX)
    canvas.set_sam_ghost([(0, 0), (10, 0), (10, 10)], 0.6)

    canceled = []
    canvas.toolCanceled.connect(lambda: canceled.append(True))

    qtbot.keyPress(canvas, Qt.Key_Escape)

    assert canvas._sam_ghost is None
    assert canvas.current_tool is None
    assert len(canceled) == 1
