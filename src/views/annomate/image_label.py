"""
Image Label Widget for AnnoMate.

This module defines the `ImageLabel` class, a custom PySide6 widget that handles:
1.  Displaying images with high-performance zooming and panning.
2.  Interactive polygon annotation (drawing, canceling, finishing).
3.  Rendering overlays (existing annotations) on top of the image.
"""

from typing import List, Tuple, Optional
import logging
import cv2
import numpy as np


from PySide6.QtCore import Qt, QPointF, QRect, Signal
from PySide6.QtGui import (
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
from PySide6.QtWidgets import QLabel, QSizePolicy

logger = logging.getLogger("AnnoMate.ImageLabel")

# Tool Constants
POLYGON = "polygon"
SAM_BBOX = "sam_bbox"


class ImageLabel(QLabel):
    """Custom QLabel for image display with zoom, pan, and polygon annotation.

    Handles rendering of a BGR image alongside existing annotation overlays
    and supports interactive polygon drawing, vertex dragging, and whole-polygon
    dragging. All coordinates reported via signals are in the original (unscaled)
    image coordinate system.

    Attributes:
        current_tool (Optional[str]): Active tool name; ``"polygon"`` or ``None``.
        current_polygon_points (List[QPointF]): In-progress polygon vertices in
            display (scaled) coordinates.
        selected_polygon_idx (int): Index of the currently selected overlay polygon,
            or ``-1`` when none is selected.

    Signals:
        polygonFinished (list): Emitted when the user completes a polygon. Carries
            ``List[Tuple[float, float]]`` in original image coordinates.
        polygonEdited (int, list): Emitted after a vertex or polygon drag completes.
            Carries ``(polygon_idx, List[Tuple[float, float]])`` in original coords.
        polygonSelected (int): Emitted when a polygon is clicked. Carries its index,
            or ``-1`` on deselect.
        toolCanceled (): Emitted when ``Escape`` is pressed while a tool is active.
    """

    polygonFinished = Signal(list)  # pts: List[Tuple[float, float]] in original coords
    polygonEdited = Signal(int, list)  # (polygon_idx, pts in original coords)
    polygonSelected = Signal(int)  # polygon index (-1 for deselect)
    toolCanceled = Signal()  # Escape pressed while a tool is active
    draw_attempted = Signal()  # left-click while a drawing tool is active
    zoom_changed = Signal(float)  # emitted whenever _zoom changes
    image_loaded = Signal(int, int)  # (orig_w, orig_h) emitted when a new image is set
    ai_polygon_clicked = Signal(int, QPointF)  # (ai_idx, view_pos); -1 = deselect
    samBboxDrawn = Signal(
        float, float, float, float
    )  # x1,y1,x2,y2 in original image coords

    def __init__(self, parent: object = None) -> None:
        """Initialize ImageLabel with default zoom, pan, and annotation state.

        Args:
            parent (object): Optional Qt parent widget. Defaults to ``None``.
        """
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setScaledContents(False)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setFocusPolicy(Qt.StrongFocus)  # Allow widget to catch Key_Escape

        self.current_tool: Optional[str] = None
        self._active_color = QColor(0, 200, 0)
        self._line_thickness = 2.0

        self._orig_image_bgr: Optional[np.ndarray] = None
        self._display_qpix: Optional[QPixmap] = None
        self._heatmap_pix: Optional[QPixmap] = None
        self._heatmap_alpha: float = 0.0

        self._base_scale = 1.0
        self._zoom = 1.0
        self._pan = QPointF(0, 0)

        self._panning = False
        self._last_mouse_pos: Optional[QPointF] = None
        self._mouse_pos: Optional[QPointF] = None

        self.current_polygon_points: List[QPointF] = []
        self._overlays: List[Tuple[List[QPointF], QColor]] = []
        self._ai_overlays: List[List[QPointF]] = []

        # --- UI State Trackers ---
        self.selected_polygon_idx: int = -1
        self._dragging_polygon: bool = False
        self._dragging_vertex_idx: int = -1
        self._dragging_vertex_poly: int = -1
        self._selected_ai_idx: int = -1

        # --- SAM tool state (display coords) ---
        self._sam_bbox_start: Optional[QPointF] = None
        self._sam_bbox_end: Optional[QPointF] = None
        self._sam_ghost: Optional[Tuple[List[QPointF], float]] = (
            None  # (display_pts, confidence)
        )

    def set_image(self, bgr: np.ndarray, max_display_dim: int = 1200) -> None:
        """Load a BGR ndarray and prepare it for display.

        Computes a ``_base_scale`` that fits the image within *max_display_dim*
        pixels on its longest edge, converts to RGB, and stores a
        :class:`~PySide6.QtGui.QPixmap` for painting. Resets all in-progress
        annotation and view state (zoom, pan, overlays).

        The caller (controller) is responsible for reading the file from disk;
        this widget only handles rendering.

        Args:
            bgr (np.ndarray): Source image in BGR format as returned by
                ``cv2.imread``.
            max_display_dim (int): Maximum pixel length of the longer image
                edge at ``zoom == 1``. Defaults to ``1200``.
        """
        self._orig_image_bgr = bgr
        h, w = bgr.shape[:2]

        self._base_scale = (
            1.0 if max(h, w) <= max_display_dim else max_display_dim / float(max(h, w))
        )

        self._zoom = 1.0
        self._pan = QPointF(0, 0)

        new_w = int(w * self._base_scale)
        new_h = int(h * self._base_scale)

        resized_bgr = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

        rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)

        qimg = QImage(
            rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888
        )
        self._display_qpix = QPixmap.fromImage(qimg)
        self.image_loaded.emit(w, h)

        # --- Reset all tracking states on new image ---
        self.clear_current_polygon()
        self._overlays = []
        self._ai_overlays = []
        self._heatmap_pix = None
        self._heatmap_alpha = 0.0
        self.selected_polygon_idx = -1
        self._dragging_polygon = False
        self._dragging_vertex_idx = -1
        self._dragging_vertex_poly = -1
        self._selected_ai_idx = -1
        self._sam_bbox_start = None
        self._sam_bbox_end = None
        self._sam_ghost = None

        self.update()

    def set_heatmap_layer(
        self, score_map: np.ndarray, alpha: float, heat_min_pct: int = 0
    ) -> None:
        """Overlay a heatmap on the canvas without resetting zoom or pan.

        Resizes *score_map* to match the stored display pixmap, applies the
        COLORMAP_JET colormap, and stores the result as a semi-transparent
        layer drawn at *alpha* opacity during paintEvent.

        Args:
            score_map: 2-D float array of anomaly scores (any resolution).
            alpha: Opacity 0.0–1.0.
            heat_min_pct: Suppress scores below this percentile (0 = show all).
        """
        if self._display_qpix is None or score_map is None:
            return
        self._heatmap_alpha = max(0.0, min(1.0, alpha))
        s = score_map.astype(np.float32)
        if heat_min_pct > 0:
            thr = np.percentile(s, heat_min_pct)
            s = np.clip(s - thr, 0.0, None)
        s_min, s_max = float(s.min()), float(s.max())
        if s_max <= s_min:
            self._heatmap_pix = None
            self.update()
            return
        s_norm = ((s - s_min) / (s_max - s_min) * 255.0).astype(np.uint8)
        colored_bgr = cv2.applyColorMap(s_norm, cv2.COLORMAP_JET)
        colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
        pix_w, pix_h = self._display_qpix.width(), self._display_qpix.height()
        resized = cv2.resize(
            colored_rgb, (pix_w, pix_h), interpolation=cv2.INTER_LINEAR
        )
        resized = np.ascontiguousarray(resized)
        qimg = QImage(
            resized.data,
            resized.shape[1],
            resized.shape[0],
            resized.strides[0],
            QImage.Format_RGB888,
        )
        self._heatmap_pix = QPixmap.fromImage(qimg.copy())
        self.update()

    def clear_heatmap_layer(self) -> None:
        """Remove the heatmap overlay and repaint."""
        self._heatmap_pix = None
        self._heatmap_alpha = 0.0
        self.update()

    def clear_image(self) -> None:
        """Clear the displayed image and reset all canvas state to blank."""
        self._display_qpix = None
        self._orig_image_bgr = None
        self._heatmap_pix = None
        self._heatmap_alpha = 0.0
        self._zoom = 1.0
        self._pan = QPointF(0, 0)
        self.current_polygon_points.clear()
        self._overlays = []
        self._ai_overlays = []
        self.selected_polygon_idx = -1
        self._dragging_polygon = False
        self._dragging_vertex_idx = -1
        self._dragging_vertex_poly = -1
        self._selected_ai_idx = -1
        self._sam_bbox_start = None
        self._sam_bbox_end = None
        self._sam_ghost = None
        self.update()

    def set_tool(self, tool_name: Optional[str]) -> None:
        """Set the active interaction tool.

        Args:
            tool_name (Optional[str]): Tool identifier — ``"polygon"`` to
                enable polygon drawing, ``"sam_bbox"`` for SAM-assisted
                segmentation, or ``None`` to deactivate all tools.
        """
        if self.current_tool == SAM_BBOX and tool_name != SAM_BBOX:
            self._sam_bbox_start = None
            self._sam_bbox_end = None
            self._sam_ghost = None
        self.current_tool = tool_name
        if tool_name == SAM_BBOX:
            self.setCursor(Qt.CrossCursor)
        elif tool_name is None:
            self.setCursor(Qt.ArrowCursor)

    def set_active_color(self, color: QColor) -> None:
        """Set the stroke color used when drawing a new polygon.

        Args:
            color (QColor): Desired color. Falls back to ``QColor(0, 200, 0)``
                if *color* is not a valid :class:`~PySide6.QtGui.QColor`.
        """
        self._active_color = color if isinstance(color, QColor) else QColor(0, 200, 0)

    @property
    def line_thickness(self) -> float:
        """Current line thickness for drawing polygons and overlays."""
        return self._line_thickness

    def set_line_thickness(self, thickness: float) -> None:
        """Set the line thickness for drawing polygons and overlays.

        Args: thickness (float): The brush width in pixels.
        """
        self._line_thickness = thickness
        self.update()

    def set_overlays(
        self, poly_list: List[Tuple[List[Tuple[float, float]], QColor]]
    ) -> None:
        """Replace all rendered overlay polygons.

        Converts each polygon from original image coordinates to display
        (base-scaled) coordinates and triggers a repaint.

        Args:
            poly_list (List[Tuple[List[Tuple[float, float]], QColor]]): List of
                ``(points, color)`` pairs where *points* is a list of
                ``(x, y)`` tuples in original image coordinates.
        """
        self._overlays = []
        for pts_orig, color, thick in poly_list:
            disp_pts = [
                QPointF(x * self._base_scale, y * self._base_scale)
                for (x, y) in pts_orig
            ]
            self._overlays.append((disp_pts, color, thick))

        n = len(self._overlays)
        if self.selected_polygon_idx >= n:
            logger.debug(
                "selected_polygon_idx %d out of range after overlay update (new size %d) — resetting",
                self.selected_polygon_idx,
                n,
            )
            self.selected_polygon_idx = -1

        self.update()

    def set_ai_overlays(self, contours: List[List[Tuple[float, float]]]) -> None:
        """Set AI segmentation polygons rendered as dashed ghost lines.

        Args:
            contours: List of polygons, each a list of (x, y) in original image coords.
        """
        self._ai_overlays = [
            [QPointF(x * self._base_scale, y * self._base_scale) for (x, y) in pts]
            for pts in contours
        ]
        self._selected_ai_idx = -1
        self.update()

    def clear_ai_overlays(self) -> None:
        """Remove all AI polygon overlays and repaint."""
        self._ai_overlays = []
        self._selected_ai_idx = -1
        self.update()

    def get_ai_polygon_view_rect(self, idx: int) -> QRect:
        """Return the bounding rect of AI polygon *idx* in widget (view) coordinates."""
        if idx < 0 or idx >= len(self._ai_overlays):
            return QRect()
        pts = self._ai_overlays[idx]
        if not pts:
            return QRect()
        xs = [p.x() * self._zoom + self._pan.x() for p in pts]
        ys = [p.y() * self._zoom + self._pan.y() for p in pts]
        x0, x1 = int(min(xs)), int(max(xs))
        y0, y1 = int(min(ys)), int(max(ys))
        return QRect(x0, y0, x1 - x0, y1 - y0)

    def clear_current_polygon(self) -> None:
        """Discard all in-progress polygon vertices and repaint."""
        self.current_polygon_points.clear()
        self.update()

    # ------------------------------------------------------------------ #
    # SAM ghost helpers
    # ------------------------------------------------------------------ #

    def set_sam_ghost(
        self, pts_orig: List[Tuple[float, float]], confidence: float
    ) -> None:
        """Store a SAM ghost polygon for preview rendering.

        Args:
            pts_orig: Polygon vertices in original image coordinates.
            confidence: Area-ratio confidence score in [0, 1].
        """
        if not pts_orig:
            self._sam_ghost = None
        else:
            disp_pts = [
                QPointF(x * self._base_scale, y * self._base_scale)
                for (x, y) in pts_orig
            ]
            self._sam_ghost = (disp_pts, confidence)
        self.update()

    def clear_sam_ghost(self) -> None:
        """Discard the SAM ghost polygon, bbox, and repaint."""
        self._sam_ghost = None
        self._sam_bbox_start = None
        self._sam_bbox_end = None
        self.update()

    def accept_sam_ghost(self) -> bool:
        """Convert the SAM ghost polygon into a finished annotation.

        Emits :attr:`polygonFinished` with the ghost's vertices in original
        image coordinates, which feeds into the normal data flow.

        Returns:
            bool: ``True`` if a ghost existed and was accepted, ``False``
                if there was no ghost to accept.
        """
        if self._sam_ghost is None:
            return False
        pts_disp, _ = self._sam_ghost
        pts_orig = [self.display_to_original(p) for p in pts_disp]
        self._sam_ghost = None
        self._sam_bbox_start = None
        self._sam_bbox_end = None
        self.polygonFinished.emit(pts_orig)
        self.update()
        return True

    def is_dragging(self) -> bool:
        """Return whether a vertex or polygon drag is currently in progress.

        Returns:
            bool: ``True`` if a vertex drag or whole-polygon drag is active.
        """
        return self._dragging_polygon or self._dragging_vertex_idx != -1

    def view_to_display(self, p_view: QPointF) -> QPointF:
        """Convert a point from view (widget) coordinates to display coordinates.

        Display coordinates are the pixel space of the stored
        :class:`~PySide6.QtGui.QPixmap` before zoom is applied.

        Args:
            p_view (QPointF): Point in widget pixel coordinates.

        Returns:
            QPointF: Corresponding point in display (pixmap) coordinates.
        """
        return QPointF(
            (p_view.x() - self._pan.x()) / self._zoom,
            (p_view.y() - self._pan.y()) / self._zoom,
        )

    def display_to_original(self, p_disp: QPointF) -> Tuple[float, float]:
        """Convert a point from display coordinates to original image coordinates.

        Args:
            p_disp (QPointF): Point in display (pixmap) coordinates.

        Returns:
            Tuple[float, float]: ``(x, y)`` in the original unscaled image.
        """
        return (p_disp.x() / self._base_scale, p_disp.y() / self._base_scale)

    def _near_polygon_start(self, pos_view: QPointF, threshold: float = 10.0) -> bool:
        """Return True if pos_view is within threshold screen pixels of the first in-progress vertex.

        Requires at least 3 placed vertices so the resulting polygon is valid.
        """
        if len(self.current_polygon_points) < 3:
            return False
        first = self.current_polygon_points[0]
        first_view = QPointF(
            first.x() * self._zoom + self._pan.x(),
            first.y() * self._zoom + self._pan.y(),
        )
        dx = pos_view.x() - first_view.x()
        dy = pos_view.y() - first_view.y()
        return (dx * dx + dy * dy) ** 0.5 < threshold

    def _find_nearest_vertex(self, pos_view: QPointF, threshold: float = 10.0):
        """Return (poly_idx, vertex_idx) of the nearest overlay vertex within threshold screen pixels, or (-1, -1)."""
        best_dist = threshold
        best_poly = -1
        best_vert = -1
        for poly_i, (pts, _, _) in enumerate(self._overlays):
            for vert_i, p_disp in enumerate(pts):
                p_view = QPointF(
                    p_disp.x() * self._zoom + self._pan.x(),
                    p_disp.y() * self._zoom + self._pan.y(),
                )
                dx = pos_view.x() - p_view.x()
                dy = pos_view.y() - p_view.y()
                dist = (dx * dx + dy * dy) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_poly = poly_i
                    best_vert = vert_i
        return best_poly, best_vert

    def finish_current_polygon(self) -> None:
        """Emit :attr:`polygonFinished` with the current polygon and clear it.

        Converts all in-progress display-coordinate vertices to original image
        coordinates before emitting. Does nothing if no points have been added.
        """
        if self.current_polygon_points:
            pts_orig = [
                self.display_to_original(p) for p in self.current_polygon_points
            ]
            self.polygonFinished.emit(pts_orig)

        self.current_polygon_points.clear()
        self.update()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Handle keyboard shortcuts for polygon drawing.

        - ``Escape``: cancel the active tool and clear in-progress points.
        - ``Backspace`` (polygon tool): remove the last placed vertex.

        Args:
            event (QKeyEvent): The key press event.
        """
        if self.current_tool == SAM_BBOX:
            if event.key() in (Qt.Key_Return, Qt.Key_Enter):
                self.accept_sam_ghost()
                return
            if event.key() == Qt.Key_Escape:
                self.clear_sam_ghost()
                self.set_tool(None)
                self.toolCanceled.emit()
                return

        if event.key() == Qt.Key_Escape:
            self.clear_current_polygon()
            self.set_tool(None)
            self.toolCanceled.emit()
            return

        if self.current_tool == POLYGON and self.current_polygon_points:
            if event.key() == Qt.Key_Backspace:
                self.current_polygon_points.pop()
                self.update()
                return
        super().keyPressEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        """Finish the current polygon on double-click while drawing.

        Args:
            event (QMouseEvent): The mouse double-click event.
        """
        if self.current_tool == POLYGON and self.current_polygon_points:
            self.finish_current_polygon()
            return

        super().mouseDoubleClickEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Handle left-click (add vertex / select / drag) and right-click (pan).

        Left button behaviour depends on active mode:

        1. Edit mode — nearest vertex within 10 px begins a vertex drag;
           clicking inside the polygon body begins a polygon drag; clicking
           outside exits edit mode.
        2. Polygon tool — appends a point to the in-progress polygon.
        3. Normal — selects the top-most polygon under the cursor and begins
           a polygon drag if one was found.

        Right button initiates panning.

        Args:
            event (QMouseEvent): The mouse press event.
        """
        self.setFocus()

        if event.button() == Qt.LeftButton:
            if self.current_tool in (SAM_BBOX, POLYGON):
                self.draw_attempted.emit()
                if (
                    self.current_tool is None
                ):  # handler cleared the tool (e.g. no class)
                    return

            pos_view = QPointF(event.pos())

            # --- SAM bbox tool: start rubber-band on press ---
            if self.current_tool == SAM_BBOX:
                self._sam_bbox_start = self.view_to_display(pos_view)
                self._sam_bbox_end = self._sam_bbox_start
                self._sam_ghost = None
                self.update()
                return

            pos_disp = self.view_to_display(pos_view)

            # --- PRE-CHECK: Identify if we clicked inside any existing polygon ---
            found_idx = -1
            for i, (pts, _, _) in enumerate(reversed(self._overlays)):
                if QPolygonF(pts).containsPoint(pos_disp, Qt.OddEvenFill):
                    found_idx = len(self._overlays) - 1 - i
                    break

            # Check AI polygons when no tool is active
            if self.current_tool != POLYGON and self._ai_overlays:
                ai_hit = -1
                for i, pts in enumerate(reversed(self._ai_overlays)):
                    if len(pts) >= 3 and QPolygonF(pts).containsPoint(
                        pos_disp, Qt.OddEvenFill
                    ):
                        ai_hit = len(self._ai_overlays) - 1 - i
                        break
                if ai_hit != -1:
                    self._selected_ai_idx = ai_hit
                    self.update()
                    self.ai_polygon_clicked.emit(ai_hit, pos_view)
                    return
                elif self._selected_ai_idx != -1:
                    self._selected_ai_idx = -1
                    self.update()
                    self.ai_polygon_clicked.emit(-1, pos_view)

            # 1. Normal Polygon Drawing
            if self.current_tool == POLYGON:
                # Visually highlight the polygon we clicked on, even while drawing
                if not self.current_polygon_points:
                    self.selected_polygon_idx = found_idx
                    self.polygonSelected.emit(found_idx)

                if self._near_polygon_start(pos_view):
                    self.setCursor(Qt.ArrowCursor)
                    self.finish_current_polygon()
                    return

                self.current_polygon_points.append(
                    self.view_to_display(QPointF(event.pos()))
                )
                self.update()
                return

            # 2. Normal Selection & Polygon Dragging (When Polygon Tool is OFF)
            poly_i, vert_i = self._find_nearest_vertex(pos_view)
            if poly_i != -1:
                self._dragging_vertex_poly = poly_i
                self._dragging_vertex_idx = vert_i
                self.selected_polygon_idx = poly_i
                self.polygonSelected.emit(poly_i)
                self.update()
                return

            self.selected_polygon_idx = found_idx
            self.update()
            self.polygonSelected.emit(found_idx)

            if found_idx != -1:
                self._dragging_polygon = True
                self._last_mouse_pos = pos_view

        elif event.button() == Qt.RightButton:
            self._panning = True
            self._last_mouse_pos = QPointF(event.pos())

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Update vertex drag, polygon drag, pan, or crosshair cursor on move.

        Args:
            event (QMouseEvent): The mouse move event.
        """
        self._mouse_pos = QPointF(event.pos())

        # --- SAM bbox rubber-band update ---
        if self.current_tool == SAM_BBOX and self._sam_bbox_start is not None:
            self._sam_bbox_end = self.view_to_display(self._mouse_pos)
            self.update()
            return

        # --- Drag vertex ---
        if self._dragging_vertex_idx != -1:
            new_pos = self.view_to_display(self._mouse_pos)
            pts, _, _ = self._overlays[self._dragging_vertex_poly]
            pts[self._dragging_vertex_idx] = new_pos
            self.update()
            return

        # --- Drag entire polygon ---
        if (
            self._dragging_polygon
            and self.selected_polygon_idx != -1
            and self._last_mouse_pos is not None
        ):
            delta_view = self._mouse_pos - self._last_mouse_pos
            delta_disp = QPointF(
                delta_view.x() / self._zoom, delta_view.y() / self._zoom
            )

            pts, _, _ = self._overlays[self.selected_polygon_idx]
            for i in range(len(pts)):
                pts[i] = QPointF(
                    pts[i].x() + delta_disp.x(), pts[i].y() + delta_disp.y()
                )

            self._last_mouse_pos = self._mouse_pos
            self.update()
            return

        if self._panning and self._last_mouse_pos is not None:
            delta = self._mouse_pos - self._last_mouse_pos
            self._pan += delta
            self._last_mouse_pos = self._mouse_pos
            self.update()
            return

        if self.current_tool == POLYGON and self.current_polygon_points:
            if self._near_polygon_start(self._mouse_pos):
                self.setCursor(Qt.CrossCursor)
            else:
                self.setCursor(Qt.ArrowCursor)
            self.update()
            return

        if self.current_tool is None and self._overlays:
            poly_i, _ = self._find_nearest_vertex(self._mouse_pos)
            self.setCursor(Qt.CrossCursor if poly_i != -1 else Qt.ArrowCursor)

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """Commit a vertex or polygon drag and emit :attr:`polygonEdited`.

        Resets drag flags before emitting so that ``is_dragging()`` returns
        ``False`` by the time connected slots react.

        Args:
            event (QMouseEvent): The mouse release event.
        """
        if event.button() == Qt.LeftButton:
            # --- SAM bbox complete: clear rubber-band and emit original-coords bbox ---
            if self.current_tool == SAM_BBOX and self._sam_bbox_start is not None:
                end = self.view_to_display(QPointF(event.pos()))
                x1 = min(self._sam_bbox_start.x(), end.x())
                y1 = min(self._sam_bbox_start.y(), end.y())
                x2 = max(self._sam_bbox_start.x(), end.x())
                y2 = max(self._sam_bbox_start.y(), end.y())
                # Clear rubber-band immediately so it disappears on release
                self._sam_bbox_start = None
                self._sam_bbox_end = None
                self.update()
                if (x2 - x1) > 4 and (y2 - y1) > 4:
                    ox1, oy1 = self.display_to_original(QPointF(x1, y1))
                    ox2, oy2 = self.display_to_original(QPointF(x2, y2))
                    self.samBboxDrawn.emit(ox1, oy1, ox2, oy2)
                return

            if self._dragging_vertex_idx != -1:
                poly_i = self._dragging_vertex_poly
                pts, _, _ = self._overlays[poly_i]
                pts_orig = [self.display_to_original(p) for p in pts]
                self._dragging_vertex_idx = -1
                self._dragging_vertex_poly = -1
                self.polygonEdited.emit(poly_i, pts_orig)
                return

            if self._dragging_polygon:
                idx = self.selected_polygon_idx
                pts, _, _ = self._overlays[idx]
                pts_orig = [self.display_to_original(p) for p in pts]
                self._dragging_polygon = False
                self._last_mouse_pos = None
                self.polygonEdited.emit(idx, pts_orig)
                return

        if event.button() == Qt.RightButton:
            self._panning = False
            self._last_mouse_pos = None

    def wheelEvent(self, event: QWheelEvent) -> None:
        """Perform cursor-anchored zoom in response to the scroll wheel.

        Clamps the scroll delta to ±5 steps and applies a ``1.15``-per-step
        zoom factor, keeping the point under the cursor stationary.

        Args:
            event (QWheelEvent): The wheel event carrying angle delta.
        """
        if self._display_qpix is None:
            return

        delta = event.angleDelta().y()
        if delta == 0:
            return

        steps = delta / 120.0
        steps = max(-5.0, min(5.0, steps))  # Clamp extreme inputs
        factor = 1.15**steps

        cursor_pos = event.position()
        point_in_disp = self.view_to_display(cursor_pos)

        old_zoom = self._zoom
        self._zoom = max(0.2, min(8.0, old_zoom * factor))
        self.zoom_changed.emit(self._zoom)

        self._pan = QPointF(
            cursor_pos.x() - point_in_disp.x() * self._zoom,
            cursor_pos.y() - point_in_disp.y() * self._zoom,
        )
        self.update()

    def zoom_in(self) -> None:
        """Zoom in by one step (factor ``1.15``), anchored at the widget center."""
        self._apply_zoom(1.15)

    def zoom_out(self) -> None:
        """Zoom out by one step (factor ``1 / 1.15``), anchored at the widget center."""
        self._apply_zoom(1 / 1.15)

    def reset_view(self) -> None:
        """Reset zoom to ``1.0`` and pan to the origin, then repaint."""
        self._zoom = 1.0
        self._pan = QPointF(0, 0)
        self.zoom_changed.emit(self._zoom)
        self.update()

    def _apply_zoom(self, factor: float) -> None:
        """Apply a zoom *factor* anchored at the center of the widget.

        Adjusts :attr:`_pan` so the center point stays fixed after the zoom.
        Clamps zoom to the range ``[0.2, 8.0]``.

        Args:
            factor (float): Multiplicative zoom change (e.g. ``1.15`` to
                zoom in, ``1 / 1.15`` to zoom out).
        """
        if self._display_qpix is None:
            return

        center = QPointF(self.width() / 2, self.height() / 2)
        point_in_disp = self.view_to_display(center)

        self._zoom = max(0.2, min(8.0, self._zoom * factor))
        self.zoom_changed.emit(self._zoom)

        self._pan = QPointF(
            center.x() - point_in_disp.x() * self._zoom,
            center.y() - point_in_disp.y() * self._zoom,
        )
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        """Paint the image, annotation overlays, and the in-progress polygon.

        Applies pan and zoom transforms before drawing the pixmap, then draws
        each overlay polygon with selection/edit highlighting. Finally draws
        the rubber-band line from the last placed vertex to the cursor.

        Args:
            event (QPaintEvent): The paint event (used by the Qt framework).
        """
        painter = QPainter(self)
        if self._display_qpix is None:
            return

        painter.translate(self._pan)
        painter.scale(self._zoom, self._zoom)
        painter.drawPixmap(0, 0, self._display_qpix)

        if self._heatmap_pix is not None and self._heatmap_alpha > 0:
            painter.setOpacity(self._heatmap_alpha)
            painter.drawPixmap(0, 0, self._heatmap_pix)
            painter.setOpacity(1.0)

        # Draw SAM bounding-box rubber-band while the user is dragging
        if (
            self.current_tool == SAM_BBOX
            and self._sam_bbox_start is not None
            and self._sam_bbox_end is not None
        ):
            from PySide6.QtCore import QRectF

            x1 = min(self._sam_bbox_start.x(), self._sam_bbox_end.x())
            y1 = min(self._sam_bbox_start.y(), self._sam_bbox_end.y())
            x2 = max(self._sam_bbox_start.x(), self._sam_bbox_end.x())
            y2 = max(self._sam_bbox_start.y(), self._sam_bbox_end.y())
            bbox_color = self._active_color
            bbox_pen = QPen(bbox_color, 1.5 / self._zoom, Qt.SolidLine)
            painter.setPen(bbox_pen)
            fill = QColor(bbox_color.red(), bbox_color.green(), bbox_color.blue(), 30)
            painter.setBrush(QBrush(fill))
            painter.drawRect(QRectF(x1, y1, x2 - x1, y2 - y1))

        # Draw Overlays with Selection Highlighting
        for i, (pts, color, thick) in enumerate(self._overlays):
            if len(pts) >= 2:
                is_selected = i == self.selected_polygon_idx
                screen_thick = ((thick * 2.0) if is_selected else thick) / self._zoom
                pen = QPen(color, screen_thick)
                alpha = 150 if is_selected else 60  # Darker fill if selected

                painter.setPen(pen)
                painter.setBrush(
                    QBrush(QColor(color.red(), color.green(), color.blue(), alpha))
                )
                painter.drawPolygon(QPolygonF(pts + [pts[0]]))

                r = 4.0 / self._zoom
                painter.setBrush(QBrush(color))
                painter.setPen(QPen(QColor(20, 20, 20), 1.0 / self._zoom))
                for p in pts:
                    painter.drawEllipse(p, r, r)

        # Draw AI segmentation polygons as dashed ghost outlines
        for i, pts in enumerate(self._ai_overlays):
            if len(pts) < 3:
                continue
            is_selected = i == self._selected_ai_idx
            pen = QPen(
                QColor(255, 80, 80),
                (2.5 if is_selected else 1.5) / self._zoom,
                Qt.DashLine,
            )
            pen.setDashPattern([6, 4])
            painter.setPen(pen)
            fill_alpha = 60 if is_selected else 20
            painter.setBrush(QBrush(QColor(255, 80, 80, fill_alpha)))
            painter.drawPolygon(QPolygonF(pts + [pts[0]]))

        # Draw SAM ghost polygon (pending accept/reject)
        if self._sam_ghost is not None:
            ghost_pts, confidence = self._sam_ghost
            if len(ghost_pts) >= 3:
                ghost_color = QColor(self._active_color)
                ghost_pen = QPen(ghost_color, 2.0 / self._zoom, Qt.DashLine)
                ghost_pen.setDashPattern([8, 4])
                painter.setPen(ghost_pen)
                ghost_color.setAlpha(45)
                painter.setBrush(QBrush(ghost_color))
                painter.drawPolygon(QPolygonF(ghost_pts + [ghost_pts[0]]))
                # xs = [p.x() for p in ghost_pts]
                # ys = [p.y() for p in ghost_pts]
                painter.setPen(QPen(self._active_color))
                f = painter.font()
                f.setPointSizeF(max(8.0, 9.0 / self._zoom))
                # painter.setFont(f)
                # painter.drawText(label_pos, f"SAM  {confidence:.2f}")

        if self.current_tool == POLYGON and self.current_polygon_points:
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(self._active_color, self._line_thickness / self._zoom))
            painter.drawPolyline(QPolygonF(self.current_polygon_points))

            if self._mouse_pos:
                cursor_in_disp = self.view_to_display(self._mouse_pos)
                painter.drawLine(
                    self.current_polygon_points[-1],
                    cursor_in_disp,
                )

            # Draw vertex dots — screen-constant radii by dividing by zoom
            r_mid = 4.0 / self._zoom
            r_first = 8.0 / self._zoom
            outline_pen = QPen(QColor(20, 20, 20), 1.0 / self._zoom)
            for j, pt in enumerate(self.current_polygon_points):
                r = r_first if j == 0 else r_mid
                painter.setBrush(QBrush(self._active_color))
                painter.setPen(outline_pen)
                painter.drawEllipse(pt, r, r)
