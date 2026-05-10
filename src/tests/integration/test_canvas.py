"""
Tests for UI Canvas components (ImageLabel, SegPathItem, CanvasPair).
Simulates mouse interactions and geometry modifications.
"""

import pytest
import numpy as np
from PySide6.QtCore import Qt, QPoint, QPointF

from views.annomate.image_label import ImageLabel, POLYGON
from views.microsentry.canvas import SegPathItem


class TestImageLabelInteraction:
    """Test user interactions with the AnnoMate drawing canvas."""

    @pytest.fixture
    def active_canvas(self, qtbot):
        """Fixture providing a canvas pre-loaded with a dummy image and drawing tool active."""
        canvas = ImageLabel()
        qtbot.addWidget(canvas)
        # Create a blank 100x100 BGR image to initialize canvas scales
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        canvas.set_image(dummy_img)
        canvas.set_tool(POLYGON)
        return canvas

    def test_draw_and_finish_polygon(self, qtbot, active_canvas):
        """Verify left clicks add points and double-clicks finish the polygon."""
        # Arrange
        p1 = QPoint(10, 10)
        p2 = QPoint(50, 10)
        p3 = QPoint(50, 50)

        # Act
        qtbot.mouseClick(active_canvas, Qt.MouseButton.LeftButton, pos=p1)
        qtbot.mouseClick(active_canvas, Qt.MouseButton.LeftButton, pos=p2)
        qtbot.mouseClick(active_canvas, Qt.MouseButton.LeftButton, pos=p3)

        # Assert intermediate state
        assert (
            len(active_canvas.current_polygon_points) == 3
        ), "Canvas should have 3 pending points."

        # Act: Double click to finish
        with qtbot.waitSignal(active_canvas.polygonFinished, timeout=1000) as blocker:
            qtbot.mouseDClick(active_canvas, Qt.MouseButton.LeftButton, pos=p3)

        # Assert final state
        emitted_points = blocker.args[0]
        assert len(emitted_points) == 3, "Should emit a 3-point polygon."
        assert (
            len(active_canvas.current_polygon_points) == 0
        ), "Pending points should clear after finishing."

    def test_zoom_in_and_out(self, qtbot, active_canvas):
        """Verify programmatic zooming adjusts internal scale properly."""
        # Arrange
        initial_zoom = active_canvas._zoom

        # Act: Zoom In
        active_canvas.zoom_in()

        # Assert
        assert active_canvas._zoom > initial_zoom, "Zoom factor should increase."

        # Act: Zoom Out
        active_canvas.zoom_out()

        # Assert
        assert active_canvas._zoom == pytest.approx(
            initial_zoom
        ), "Zoom factor should return to base."


class TestSegPathItemGeometry:
    """Test MicroSentry AI visualization components and polygon math."""

    def test_douglas_peucker_simplification(self):
        """Verify highly jagged polygons are smoothed out based on epsilon."""
        # Arrange
        # A 10x10 square with a tiny, unnecessary bump on the top edge
        pts = [
            QPointF(0, 0),
            QPointF(10, 0),
            QPointF(10, 10),
            QPointF(5, 10.1),  # The "jagged" point to be simplified
            QPointF(0, 10),
        ]
        item = SegPathItem(pts)

        # Act
        item.simplify(epsilon=1.0)

        # Assert
        simplified_pts = item._pts
        assert (
            len(simplified_pts) == 4
        ), "The intermediate near-collinear point should be removed."
        assert simplified_pts[0] == QPointF(0, 0)
        assert simplified_pts[1] == QPointF(10, 0)
        assert simplified_pts[2] == QPointF(10, 10)
        assert simplified_pts[3] == QPointF(0, 10)

    def test_scale_about_center(self):
        """Verify polygons can be scaled relative to their own centroid."""
        # Arrange: A 10x10 square from (0,0) to (10,10). Centroid is (5,5).
        pts = [QPointF(0, 0), QPointF(10, 0), QPointF(10, 10), QPointF(0, 10)]
        item = SegPathItem(pts)

        # Act: Scale up by 2x
        item.scale_about_center(2.0)

        # Assert
        # The new bounding box should go from (-5, -5) to (15, 15)
        scaled_pts = item._pts
        assert scaled_pts[0] == QPointF(-5, -5)
        assert scaled_pts[2] == QPointF(15, 15)
