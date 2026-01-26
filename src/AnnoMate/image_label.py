"""
Image Label Widget for AnnoMate.

This module defines the `ImageLabel` class, a custom PyQt5 widget that handles:
1.  Displaying images with high-performance zooming and panning.
2.  Interactive polygon annotation (drawing, canceling, finishing).
3.  Rendering overlays (existing annotations) on top of the image.

Coordinate Systems:
-   **Original:** The actual pixel coordinates of the source image.
-   **Display:** Coordinates scaled to fit the base view (before zoom).
-   **View:** The final widget coordinates (after zoom/pan transformations).
"""

from typing import List, Tuple, Optional
import cv2
import numpy as np

from PyQt5.QtCore import Qt, QPointF, QRect, QSize
from PyQt5.QtGui import (
    QPainter,
    QPen,
    QPolygonF,
    QPixmap,
    QImage,
    QBrush,
    QColor,
    QMouseEvent,
    QWheelEvent,
    QKeyEvent,
    QPaintEvent,
)
from PyQt5.QtWidgets import QLabel, QSizePolicy

# Tool Constants
POLYGON = "polygon"


class ImageLabel(QLabel):
    """
    A custom canvas widget for image annotation.

    Features:
    -   Zooming (Mouse Wheel) and Panning (Right-Click Drag).
    -   Polygon drawing tool with vertex snapping.
    -   Visualizing existing annotations as overlays.
    """

    def __init__(self, parent=None):
        """Initialize the widget and internal state."""
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setScaledContents(False)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # External Dependencies
        self.main_window = None  # Reference to main window for callbacks

        # Tool State
        self.current_tool: Optional[str] = None
        self._active_color = QColor(0, 200, 0)

        # Image Data
        self._orig_image_bgr: Optional[np.ndarray] = None
        self._display_qpix: Optional[QPixmap] = None

        # Transformation State
        self._base_scale = 1.0  # Scale factor to fit image to screen initially
        self._zoom = 1.0        # Dynamic zoom factor (multiplied by base_scale)
        self._pan = QPointF(0, 0)

        # Interaction State
        self._panning = False
        self._last_mouse_pos: Optional[QPointF] = None
        self._mouse_pos: Optional[QPointF] = None

        # Annotation State (Stored in Display Coordinates)
        self.current_polygon_points: List[QPointF] = []
        # Overlays: List of (List[QPointF], QColor)
        self._overlays: List[Tuple[List[QPointF], QColor]] = []

    # =========================================================================
    # Public API (Hooks for Window)
    # =========================================================================

    def set_main_window(self, main_window):
        """Sets the reference to the parent main window."""
        self.main_window = main_window

    def set_tool(self, tool_name: Optional[str]):
        """Sets the active drawing tool (e.g., POLYGON or None)."""
        self.current_tool = tool_name

    def set_active_color(self, color: QColor):
        """Sets the color for the currently active drawing tool."""
        self._active_color = color if isinstance(color, QColor) else QColor(0, 200, 0)

    def set_overlays(self, poly_list: List[Tuple[List[Tuple[float, float]], QColor]]):
        """
        Updates the list of static overlays to draw (existing annotations).

        Args:
            poly_list: A list of tuples, where each tuple contains:
                       1. A list of (x, y) coordinates in ORIGINAL image space.
                       2. A QColor object for filling the polygon.
        """
        self._overlays = []
        for pts_orig, color in poly_list:
            # Transform original coordinates to display coordinates for rendering
            disp_pts = [
                QPointF(x * self._base_scale, y * self._base_scale)
                for (x, y) in pts_orig
            ]
            self._overlays.append((disp_pts, color))
        self.update()

    def clear_current_polygon(self):
        """Aborts the current drawing operation."""
        self.current_polygon_points.clear()
        self.update()

    # =========================================================================
    # Image Loading & Coordinate Math
    # =========================================================================

    def load_image(self, path: str, max_display_dim: int = 1200):
        """
        Loads an image from disk and prepares it for display.

        Args:
            path (str): File path to the image.
            max_display_dim (int): Maximum dimension (width or height) for the
                                   internal display pixmap. Large images are
                                   downscaled for performance.

        Raises:
            RuntimeError: If the image cannot be loaded via OpenCV.
        """
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Failed to load image: {path}")

        self._orig_image_bgr = bgr
        h, w = bgr.shape[:2]

        # Calculate base scale to fit within max_display_dim
        self._base_scale = (
            1.0 if max(h, w) <= max_display_dim
            else max_display_dim / float(max(h, w))
        )

        # Reset View
        self._zoom = 1.0
        self._pan = QPointF(0, 0)

        # Create resized display pixmap
        new_w = int(w * self._base_scale)
        new_h = int(h * self._base_scale)
        
        resized_bgr = cv2.resize(
            bgr, (new_w, new_h), interpolation=cv2.INTER_AREA
        )
        
        # Convert BGR (OpenCV) to RGB (Qt)
        rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
        
        # Create QImage and QPixmap
        qimg = QImage(
            rgb.data,
            rgb.shape[1],
            rgb.shape[0],
            rgb.strides[0],
            QImage.Format_RGB888
        )
        self._display_qpix = QPixmap.fromImage(qimg)

        # Clear state
        self.clear_current_polygon()
        self._overlays = []
        self.update()

    def view_to_display(self, p_view: QPointF) -> QPointF:
        """
        Transforms a point from View coordinates (Widget pixels) to
        Display coordinates (Scaled image pixels).
        """
        return QPointF(
            (p_view.x() - self._pan.x()) / self._zoom,
            (p_view.y() - self._pan.y()) / self._zoom
        )

    def display_to_original(self, p_disp: QPointF) -> Tuple[float, float]:
        """
        Transforms a point from Display coordinates to Original image coordinates.
        """
        return (p_disp.x() / self._base_scale, p_disp.y() / self._base_scale)

    # =========================================================================
    # Polygon Logic
    # =========================================================================

    def maybe_close_on_first_vertex(self, pos_view: QPointF, thresh_px: float = 8.0) -> bool:
        """
        Checks if the mouse click is close enough to the starting vertex to close the polygon.
        """
        if len(self.current_polygon_points) < 3:
            return False
            
        pos_disp = self.view_to_display(pos_view)
        start_pt = self.current_polygon_points[0]
        
        dx = pos_disp.x() - start_pt.x()
        dy = pos_disp.y() - start_pt.y()
        
        distance = (dx * dx + dy * dy) ** 0.5
        return distance <= thresh_px

    def finish_current_polygon(self):
        """
        Finalizes the polygon, converts points to original coordinates, and
        notifies the main window.
        """
        if self.current_polygon_points and self.main_window is not None:
            # Convert to original image coordinates for storage
            pts_orig = [
                self.display_to_original(p) for p in self.current_polygon_points
            ]
            self.main_window.finish_polygon(pts_orig)
            
        self.current_polygon_points.clear()
        self.update()

    # =========================================================================
    # Input Event Handlers
    # =========================================================================

    def keyPressEvent(self, event: QKeyEvent):
        """Handles keyboard shortcuts (Backspace to undo, Esc to cancel)."""
        if self.current_tool == POLYGON and self.current_polygon_points:
            if event.key() == Qt.Key_Backspace:
                self.current_polygon_points.pop()
                self.update()
                return
            if event.key() == Qt.Key_Escape:
                self.current_polygon_points.clear()
                self.update()
                return
        super().keyPressEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        """Double-click closes the polygon."""
        if self.current_tool == POLYGON and self.current_polygon_points:
            self.finish_current_polygon()
            return
        super().mouseDoubleClickEvent(event)

    def mousePressEvent(self, event: QMouseEvent):
        """Handles drawing (Left Click) and Panning (Right Click)."""
        if event.button() == Qt.LeftButton and self.current_tool == POLYGON:
            # Check if clicking near start to close loop
            if self.maybe_close_on_first_vertex(event.pos()):
                self.finish_current_polygon()
            else:
                # Add new vertex
                self.current_polygon_points.append(self.view_to_display(event.pos()))
                self.update()
                
        elif event.button() == Qt.RightButton:
            # Start Pan
            self._panning = True
            self._last_mouse_pos = event.pos()

    def mouseMoveEvent(self, event: QMouseEvent):
        """Handles mouse movement for panning and drawing guides."""
        self._mouse_pos = event.pos()
        
        if self._panning and self._last_mouse_pos is not None:
            delta = event.pos() - self._last_mouse_pos
            self._pan += delta
            self._last_mouse_pos = event.pos()
            self.update()
            return
            
        if self.current_tool == POLYGON and self.current_polygon_points:
            self.update()  # Redraw to show guide line
            return
            
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Ends panning operation."""
        if event.button() == Qt.RightButton:
            self._panning = False
            self._last_mouse_pos = None

    def wheelEvent(self, event: QWheelEvent):
        """Handles zoom in/out."""
        if self._display_qpix is None:
            return

        # Calculate zoom factor
        steps = event.angleDelta().y() / 120.0
        factor = 1.15 ** steps
        
        old_zoom = self._zoom
        self._zoom = max(0.2, min(8.0, old_zoom * factor))

        # Adjust pan to zoom towards mouse cursor
        cursor_pos = event.pos()
        point_in_disp = self.view_to_display(cursor_pos)
        
        # New Pan = Cursor - (PointInDisp * NewZoom)
        self._pan = QPointF(
            cursor_pos.x() - point_in_disp.x() * self._zoom,
            cursor_pos.y() - point_in_disp.y() * self._zoom,
        )
        self.update()

    # =========================================================================
    # View Control Helpers
    # =========================================================================

    def zoom_in(self):
        self._apply_zoom(1.15)

    def zoom_out(self):
        self._apply_zoom(1 / 1.15)

    def reset_view(self):
        self._zoom = 1.0
        self._pan = QPointF(0, 0)
        self.update()

    def _apply_zoom(self, factor: float):
        """Helper to apply zoom relative to the center of the widget."""
        if self._display_qpix is None:
            return
            
        center = QPointF(self.width() / 2, self.height() / 2)
        point_in_disp = self.view_to_display(center)
        
        self._zoom = max(0.2, min(8.0, self._zoom * factor))
        
        self._pan = QPointF(
            center.x() - point_in_disp.x() * self._zoom,
            center.y() - point_in_disp.y() * self._zoom
        )
        self.update()

    # =========================================================================
    # Painting
    # =========================================================================

    def paintEvent(self, event: QPaintEvent):
        """Draws the image, overlays, and active polygon."""
        painter = QPainter(self)
        if self._display_qpix is None:
            return

        # Apply View Transformations
        painter.translate(self._pan)
        painter.scale(self._zoom, self._zoom)
        
        # Draw Background Image
        painter.drawPixmap(0, 0, self._display_qpix)

        # Draw Overlays (Existing Annotations)
        # Format: Faint fill + solid outline
        for pts, color in self._overlays:
            if len(pts) >= 2:
                painter.setPen(QPen(color, 2))
                painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 60)))
                painter.drawPolygon(QPolygonF(pts + [pts[0]]))

        # Draw Active Polygon (Being drawn now)
        # Format: Outline only + Guide line to cursor
        if self.current_tool == POLYGON and self.current_polygon_points:
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(self._active_color, 2))
            
            # Draw existing vertices
            painter.drawPolyline(QPolygonF(self.current_polygon_points))
            
            # Draw "Rubber band" line to mouse cursor
            if self._mouse_pos:
                cursor_in_disp = self.view_to_display(self._mouse_pos)
                painter.drawLine(
                    self.current_polygon_points[-1],
                    cursor_in_disp,
                )