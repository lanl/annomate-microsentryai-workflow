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


from PySide6.QtCore import Qt, QPointF, QRect, QRectF, Signal
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
    QPainterPath,
    QFontMetricsF,
)
from PySide6.QtWidgets import QLabel, QSizePolicy

logger = logging.getLogger("AnnoMate.ImageLabel")

# Tool Constants
POLYGON = "polygon"
SAM_BBOX = "sam_bbox"
CALIBRATE = "calibrate"
MEASURE = "measure"


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
    calibrationPointsPlaced = Signal(tuple, tuple)  # (p1_orig, p2_orig)
    centerCropChanged = Signal(dict)

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
        self._center_crop_enabled: bool = False
        self._center_crop_shape: str = "circle"
        self._center_crop_width: Optional[int] = None
        self._center_crop_height: Optional[int] = None
        self._center_crop_opacity: float = 0.37
        self._center_crop_dot_visible: bool = False
        self._center_crop_center_x: Optional[float] = None
        self._center_crop_center_y: Optional[float] = None
        self._center_crop_calibrating: bool = False
        self._dragging_center_crop: bool = False
        self._center_crop_color: Optional[tuple] = None  # None = auto-contrast

        self._base_scale = 1.0
        self._zoom = 1.0
        self._pan = QPointF(0, 0)
        self._view_is_fit = False

        self._panning = False
        self._last_mouse_pos: Optional[QPointF] = None
        self._mouse_pos: Optional[QPointF] = None

        self.current_polygon_points: List[QPointF] = []
        self._overlays: List[Tuple[List[QPointF], QColor, float, bool]] = []
        self._ai_overlays: List[List[QPointF]] = []
        self._anomaly_area_violations: set = set()
        self._anomaly_distance_pairs: set = set()
        self._anomaly_dist_values: dict = {}
        self._anomaly_dist_unit: str = "px"
        self._anomaly_area_color: tuple = (255, 165, 0)
        self._anomaly_dist_color: tuple = (220, 50, 50)

        # --- UI State Trackers ---
        self.selected_polygon_idx: int = -1
        self._dragging_polygon: bool = False
        self._polygon_drag_moved: bool = False
        self._dragging_vertex_idx: int = -1
        self._dragging_vertex_poly: int = -1
        self._selected_ai_idx: int = -1

        # --- SAM tool state (display coords) ---
        self._sam_bbox_start: Optional[QPointF] = None
        self._sam_bbox_end: Optional[QPointF] = None
        self._sam_ghost: Optional[Tuple[List[QPointF], float]] = (
            None  # (display_pts, confidence)
        )

        # --- Calibration / measure tool state ---
        self._calib_model = None  # CalibrationModel; set via set_calibration_model()
        self._pending_calib_pts: list = []  # accumulates up to 2 original-coord tuples
        self._watermark_bar_y: int | None = None

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

        new_w = int(w * self._base_scale)
        new_h = int(h * self._base_scale)

        resized_bgr = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

        rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)

        qimg = QImage(
            rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888
        )
        self._display_qpix = QPixmap.fromImage(qimg)
        self.image_loaded.emit(w, h)
        self.reset_view()

        # --- Reset all tracking states on new image ---
        self.clear_current_polygon()
        self._overlays = []
        self._ai_overlays = []
        self._anomaly_area_violations = set()
        self._anomaly_distance_pairs = set()
        self._anomaly_dist_values = {}
        self._heatmap_pix = None
        self._heatmap_alpha = 0.0
        self.selected_polygon_idx = -1
        self._dragging_polygon = False
        self._polygon_drag_moved = False
        self._dragging_vertex_idx = -1
        self._dragging_vertex_poly = -1
        self._selected_ai_idx = -1
        self._sam_bbox_start = None
        self._sam_bbox_end = None
        self._sam_ghost = None
        self._pending_calib_pts = []
        self._center_crop_center_x = None
        self._center_crop_center_y = None
        self._dragging_center_crop = False
        if self._calib_model is not None:
            self._calib_model.clear_measurement()
        self._ensure_center_crop_defaults(w, h)

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
            s = np.clip(s, thr, None)
        s_min, s_max = float(s.min()), float(s.max())
        if s_max <= s_min:
            self._heatmap_pix = None
            self.update()
            return
        s_norm = ((s - s_min) / (s_max - s_min) * 255.0).astype(np.uint8)
        colored_bgr = cv2.applyColorMap(s_norm, cv2.COLORMAP_TURBO)
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
        self._view_is_fit = False
        self._center_crop_enabled = False
        self._center_crop_width = None
        self._center_crop_height = None
        self._center_crop_center_x = None
        self._center_crop_center_y = None
        self._center_crop_calibrating = False
        self._dragging_center_crop = False
        self.current_polygon_points.clear()
        self._overlays = []
        self._ai_overlays = []
        self.selected_polygon_idx = -1
        self._dragging_polygon = False
        self._polygon_drag_moved = False
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
            tool_name (Optional[str]): Tool identifier — ``"polygon"``,
                ``"sam_bbox"``, ``"calibrate"``, ``"measure"``, or ``None``.
        """
        if self.current_tool == SAM_BBOX and tool_name != SAM_BBOX:
            self._sam_bbox_start = None
            self._sam_bbox_end = None
            self._sam_ghost = None
        if self.current_tool in (CALIBRATE, MEASURE) and tool_name not in (
            CALIBRATE,
            MEASURE,
        ):
            self._pending_calib_pts = []
            if self._calib_model is not None:
                self._calib_model.clear_measurement()
        self.current_tool = tool_name
        if tool_name in (SAM_BBOX, CALIBRATE, MEASURE):
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

    _UNSET = object()

    def set_center_crop(
        self,
        enabled: Optional[bool] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        shape: Optional[str] = None,
        opacity: Optional[float] = None,
        center_dot: Optional[bool] = None,
        center_x: Optional[float] = None,
        center_y: Optional[float] = None,
        calibrating: Optional[bool] = None,
        border_color=_UNSET,
    ) -> None:
        """Set the centered crop preview mask drawn over the image.

        Dimensions are in original image pixels. This is a view overlay only;
        it does not mutate image data or annotations.
        """
        if self._orig_image_bgr is not None:
            img_h, img_w = self._orig_image_bgr.shape[:2]
            self._ensure_center_crop_defaults(img_w, img_h)
        else:
            img_w = img_h = None

        if enabled is not None:
            self._center_crop_enabled = bool(enabled)
        if shape is not None:
            self._center_crop_shape = (
                shape if shape in ("rectangle", "circle") else "rectangle"
            )
        if width is not None:
            self._center_crop_width = max(1, int(width))
        if height is not None:
            self._center_crop_height = max(1, int(height))
        if opacity is not None:
            self._center_crop_opacity = max(0.0, min(1.0, float(opacity)))
        if center_dot is not None:
            self._center_crop_dot_visible = bool(center_dot)
        if center_x is not None and img_w is not None:
            self._center_crop_center_x = max(0.0, min(float(center_x), float(img_w)))
        if center_y is not None and img_h is not None:
            self._center_crop_center_y = max(0.0, min(float(center_y), float(img_h)))
        if calibrating is not None:
            self._center_crop_calibrating = bool(calibrating)
            self._dragging_center_crop = False
            if self._center_crop_calibrating:
                self.setCursor(Qt.SizeAllCursor)
                self.setFocus()
            elif self.current_tool is None:
                self.setCursor(Qt.ArrowCursor)
        if border_color is not self._UNSET:
            self._center_crop_color = border_color  # None resets to auto
        self.update()
        self.centerCropChanged.emit(self.center_crop_settings())

    def center_crop_settings(self) -> dict:
        """Return the current center crop preview settings."""
        return {
            "enabled": self._center_crop_enabled,
            "shape": self._center_crop_shape,
            "width": self._center_crop_width,
            "height": self._center_crop_height,
            "opacity": self._center_crop_opacity,
            "center_dot": self._center_crop_dot_visible,
            "center_x": self._center_crop_center_x,
            "center_y": self._center_crop_center_y,
            "calibrating": self._center_crop_calibrating,
            "border_color": self._center_crop_color,
        }

    def _ensure_center_crop_defaults(self, img_w: int, img_h: int) -> None:
        if self._center_crop_width is None:
            self._center_crop_width = 1210
        else:
            self._center_crop_width = max(1, self._center_crop_width)
        if self._center_crop_height is None:
            self._center_crop_height = 1210
        else:
            self._center_crop_height = max(1, self._center_crop_height)
        if self._center_crop_center_x is None:
            self._center_crop_center_x = img_w / 2.0
        else:
            self._center_crop_center_x = max(
                0.0, min(float(self._center_crop_center_x), float(img_w))
            )
        if self._center_crop_center_y is None:
            self._center_crop_center_y = img_h / 2.0
        else:
            self._center_crop_center_y = max(
                0.0, min(float(self._center_crop_center_y), float(img_h))
            )

    def _move_center_crop_to_view(self, pos_view: QPointF) -> None:
        if self._orig_image_bgr is None:
            return
        img_h, img_w = self._orig_image_bgr.shape[:2]
        x, y = self.display_to_original(self.view_to_display(pos_view))
        self._center_crop_center_x = max(0.0, min(float(x), float(img_w)))
        self._center_crop_center_y = max(0.0, min(float(y), float(img_h)))
        self.update()
        self.centerCropChanged.emit(self.center_crop_settings())

    def _nudge_center_crop(self, dx: float, dy: float) -> None:
        if self._orig_image_bgr is None:
            return
        img_h, img_w = self._orig_image_bgr.shape[:2]
        cx = (self._center_crop_center_x if self._center_crop_center_x is not None else img_w / 2.0) + dx
        cy = (self._center_crop_center_y if self._center_crop_center_y is not None else img_h / 2.0) + dy
        self._center_crop_center_x = max(0.0, min(cx, float(img_w)))
        self._center_crop_center_y = max(0.0, min(cy, float(img_h)))
        self.update()
        self.centerCropChanged.emit(self.center_crop_settings())

    def set_overlays(self, poly_list: list) -> None:
        """Replace all rendered overlay polygons.

        Converts each polygon from original image coordinates to display
        (base-scaled) coordinates and triggers a repaint.

        Args:
            poly_list: List of ``(points, color, thickness[, visible])`` tuples
                where *points* is a list of ``(x, y)`` tuples in original image
                coordinates. Hidden overlays keep their index but are not drawn
                or hit-tested.
        """
        self._overlays = []
        for item in poly_list:
            pts_orig, color, thick = item[:3]
            visible = bool(item[3]) if len(item) > 3 else True
            disp_pts = [
                QPointF(x * self._base_scale, y * self._base_scale)
                for (x, y) in pts_orig
            ]
            self._overlays.append((disp_pts, color, thick, visible))

        n = len(self._overlays)
        selected_hidden = (
            0 <= self.selected_polygon_idx < n
            and not self._overlays[self.selected_polygon_idx][3]
        )
        if self.selected_polygon_idx >= n or selected_hidden:
            logger.debug(
                "selected_polygon_idx %d unavailable after overlay update "
                "(new size %d) — resetting",
                self.selected_polygon_idx,
                n,
            )
            self.selected_polygon_idx = -1

        self.update()

    def set_violation_highlights(
        self,
        area_violations: set,
        distance_pairs: set,
        dist_values: dict | None = None,
    ) -> None:
        """Update which annotations to highlight as constraint violations.

        Args:
            area_violations: Set of annotation indices whose area exceeds the threshold.
            distance_pairs: Set of frozensets ``{i, j}`` for pairs that are too close.
            dist_values: Optional mapping of each pair frozenset to its world-unit distance.
        """
        self._anomaly_area_violations = area_violations
        self._anomaly_distance_pairs = distance_pairs
        self._anomaly_dist_values = dist_values or {}
        self.update()

    def set_violation_unit(self, unit: str) -> None:
        """Set the distance unit label shown on proximity violation lines."""
        self._anomaly_dist_unit = unit
        self.update()

    def set_violation_colors(self, area_color: tuple, distance_color: tuple) -> None:
        """Update the colors used to render violation highlights."""
        self._anomaly_area_color = area_color
        self._anomaly_dist_color = distance_color
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

    def set_calibration_model(self, model) -> None:
        """Bind a CalibrationModel so the canvas can render the grid and dots."""
        self._calib_model = model
        if model is not None:
            model.calibration_changed.connect(self.update)
            model.grid_changed.connect(self.update)
            model.measurement_updated.connect(self.update)

    def set_watermark_bar_y(self, y: int) -> None:
        """Tell the canvas where the floating action bar's top edge sits (canvas coords)."""
        self._watermark_bar_y = y

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
        for poly_i, (pts, _, _, visible) in enumerate(self._overlays):
            if not visible:
                continue
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
        if self._center_crop_calibrating:
            step = 10.0 if event.modifiers() & Qt.ShiftModifier else 1.0
            _arrow_delta = {
                Qt.Key_Left: (-step, 0.0),
                Qt.Key_Right: (step, 0.0),
                Qt.Key_Up: (0.0, -step),
                Qt.Key_Down: (0.0, step),
            }
            if event.key() in _arrow_delta:
                dx, dy = _arrow_delta[event.key()]
                self._nudge_center_crop(dx, dy)
                return

        if self.current_tool == SAM_BBOX:
            if event.key() in (Qt.Key_Return, Qt.Key_Enter):
                self.accept_sam_ghost()
                return
            if event.key() == Qt.Key_Escape:
                self.clear_sam_ghost()
                self.set_tool(None)
                self.toolCanceled.emit()
                return

        if self.current_tool in (CALIBRATE, MEASURE):
            if event.key() == Qt.Key_Escape:
                self._pending_calib_pts = []
                if self._calib_model is not None:
                    self._calib_model.clear_measurement()
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

            # --- Calibrate tool ---
            if self.current_tool == CALIBRATE:
                pos_view = QPointF(event.pos())
                pos_disp = self.view_to_display(pos_view)
                orig_pt = self.display_to_original(pos_disp)
                self._pending_calib_pts.append(orig_pt)
                self.update()
                if len(self._pending_calib_pts) == 2:
                    p1, p2 = self._pending_calib_pts
                    # Don't clear here — keep pts visible while the dialog is open.
                    # window.py clears them via canvas.set_tool(None) after dialog closes.
                    self.calibrationPointsPlaced.emit(p1, p2)
                return

            # --- Measure tool ---
            if self.current_tool == MEASURE:
                if self._calib_model is None or not self._calib_model.has_scale():
                    return
                pos_view = QPointF(event.pos())
                pos_disp = self.view_to_display(pos_view)
                orig_pt = self.display_to_original(pos_disp)
                p1, p2 = self._calib_model.meas_points()
                if p1 is None:
                    self._calib_model.set_meas_p1(orig_pt)
                elif p2 is None:
                    self._calib_model.set_meas_p2(orig_pt)
                else:
                    # Start a new measurement
                    self._calib_model.set_meas_p1(orig_pt)
                return

            pos_view = QPointF(event.pos())

            if (
                self._center_crop_calibrating
                and self._center_crop_enabled
                and self.current_tool is None
            ):
                self._dragging_center_crop = True
                self._move_center_crop_to_view(pos_view)
                event.accept()
                return

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
            for i, (pts, _, _, visible) in enumerate(reversed(self._overlays)):
                if not visible:
                    continue
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
                self._polygon_drag_moved = False
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

        if self._dragging_center_crop:
            self._move_center_crop_to_view(self._mouse_pos)
            event.accept()
            return

        # --- SAM bbox rubber-band update ---
        if self.current_tool == SAM_BBOX and self._sam_bbox_start is not None:
            self._sam_bbox_end = self.view_to_display(self._mouse_pos)
            self.update()
            return

        # --- Drag vertex ---
        if self._dragging_vertex_idx != -1:
            new_pos = self.view_to_display(self._mouse_pos)
            pts, _, _, _ = self._overlays[self._dragging_vertex_poly]
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

            pts, _, _, _ = self._overlays[self.selected_polygon_idx]
            for i in range(len(pts)):
                pts[i] = QPointF(
                    pts[i].x() + delta_disp.x(), pts[i].y() + delta_disp.y()
                )

            self._polygon_drag_moved = True
            self._last_mouse_pos = self._mouse_pos
            self.update()
            return

        if self._panning and self._last_mouse_pos is not None:
            delta = self._mouse_pos - self._last_mouse_pos
            self._pan += delta
            self._view_is_fit = False
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

        if self._center_crop_calibrating and self.current_tool is None:
            self.setCursor(Qt.SizeAllCursor)
        elif self.current_tool is None and self._overlays:
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
            if self._dragging_center_crop:
                self._dragging_center_crop = False
                event.accept()
                return

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
                pts, _, _, _ = self._overlays[poly_i]
                pts_orig = [self.display_to_original(p) for p in pts]
                self._dragging_vertex_idx = -1
                self._dragging_vertex_poly = -1
                self.polygonEdited.emit(poly_i, pts_orig)
                return

            if self._dragging_polygon:
                moved = self._polygon_drag_moved
                self._dragging_polygon = False
                self._polygon_drag_moved = False
                self._last_mouse_pos = None
                if moved:
                    idx = self.selected_polygon_idx
                    pts, _, _, _ = self._overlays[idx]
                    pts_orig = [self.display_to_original(p) for p in pts]
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
        self._zoom = max(0.2, min(12.0, old_zoom * factor))
        self._view_is_fit = False
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
        """Fit the image into the current viewport and center it."""
        self._fit_image_to_view()
        self._view_is_fit = True
        self.zoom_changed.emit(self._zoom)
        self.update()

    def _apply_zoom(self, factor: float) -> None:
        """Apply a zoom *factor* anchored at the center of the widget.

        Adjusts :attr:`_pan` so the center point stays fixed after the zoom.
        Clamps zoom to the range ``[0.2, 12.0]``.

        Args:
            factor (float): Multiplicative zoom change (e.g. ``1.15`` to
                zoom in, ``1 / 1.15`` to zoom out).
        """
        if self._display_qpix is None:
            return

        center = QPointF(self.width() / 2, self.height() / 2)
        point_in_disp = self.view_to_display(center)

        self._zoom = max(0.2, min(12.0, self._zoom * factor))
        self._view_is_fit = False
        self.zoom_changed.emit(self._zoom)

        self._pan = QPointF(
            center.x() - point_in_disp.x() * self._zoom,
            center.y() - point_in_disp.y() * self._zoom,
        )
        self.update()

    def _fit_image_to_view(self) -> None:
        """Set zoom/pan so the full display pixmap is centered in the widget."""
        if self._display_qpix is None:
            self._zoom = 1.0
            self._pan = QPointF(0, 0)
            return

        pix_w = max(1, self._display_qpix.width())
        pix_h = max(1, self._display_qpix.height())
        view_w = max(1, self.width())
        view_h = max(1, self.height())

        self._zoom = min(view_w / pix_w, view_h / pix_h)
        self._pan = QPointF(
            (view_w - pix_w * self._zoom) / 2.0,
            (view_h - pix_h * self._zoom) / 2.0,
        )

    def resizeEvent(self, event) -> None:
        """Keep an already reset/newly loaded image fitted as the viewport changes."""
        super().resizeEvent(event)
        if self._view_is_fit and self._display_qpix is not None:
            self._fit_image_to_view()
            self.zoom_changed.emit(self._zoom)
            self.update()

    # ------------------------------------------------------------------ #
    # Calibration rendering helpers
    # ------------------------------------------------------------------ #

    def _paint_grid(self, painter: QPainter) -> None:
        """Draw grid lines fixed to the viewport in screen coordinates."""
        m = self._calib_model
        if m is None or not m.grid_visible() or not m.has_scale():
            return
        step_world = m.grid_spacing_world()
        if step_world <= 0:
            return
        step_screen = (step_world / m.scale()) * self._base_scale * self._zoom
        if step_screen < 8:
            return

        r, g, b = m.grid_color()
        alpha = int(m.grid_opacity() * 255)
        color = QColor(r, g, b, alpha)
        painter.setPen(QPen(color, 1.0))
        painter.setBrush(Qt.NoBrush)

        w = float(self.width())
        h = float(self.height())

        x = 0.0
        while x <= w:
            painter.drawLine(QPointF(x, 0), QPointF(x, h))
            x += step_screen

        y = 0.0
        while y <= h:
            painter.drawLine(QPointF(0, y), QPointF(w, y))
            y += step_screen

        self._paint_grid_watermark(
            painter, step_world, m.px_count(), m.world_val(), m.unit(), w, h
        )

    def _paint_grid_watermark(
        self,
        painter: QPainter,
        step_world: float,
        px_count: float,
        world_val: float,
        unit: str,
        viewport_w: float,
        viewport_h: float,
    ) -> None:
        line1 = f"{px_count:g}px:{world_val:g}{unit}"
        line2 = f"Grid: {step_world:g} {unit}"
        font = painter.font()
        font.setPointSizeF(11.0)
        painter.setFont(font)
        fm = QFontMetricsF(font)
        margin = 12.0
        line_h = fm.height()

        w1 = fm.horizontalAdvance(line1)
        w2 = fm.horizontalAdvance(line2)

        if self._watermark_bar_y is not None:
            top_y = float(self._watermark_bar_y)
        else:
            top_y = viewport_h - 2 * line_h - margin

        baseline1 = top_y + fm.ascent()
        baseline2 = baseline1 + line_h

        x1 = viewport_w - w1 - margin
        x2 = viewport_w - w2 - margin

        painter.setPen(QColor(0, 0, 0, 120))
        painter.drawText(QPointF(x1 + 1, baseline1 + 1), line1)
        painter.drawText(QPointF(x2 + 1, baseline2 + 1), line2)
        painter.setPen(QColor(255, 255, 255, 200))
        painter.drawText(QPointF(x1, baseline1), line1)
        painter.drawText(QPointF(x2, baseline2), line2)

    def _paint_violation_highlights(self, painter: QPainter) -> None:
        """Draw anomaly constraint violation highlights in display coords.

        Called inside the pan/zoom-transformed painter so coordinates match
        the normal overlay layer.  Area violations get an amber outline;
        distance violations get a dashed red line connecting polygon centroids.
        """
        if not self._anomaly_area_violations and not self._anomaly_distance_pairs:
            return

        painter.setBrush(Qt.NoBrush)

        if self._anomaly_area_violations:
            ar, ag, ab = self._anomaly_area_color
            color = QColor(ar, ag, ab)
            f = painter.font()
            f.setBold(True)
            font_pts = max(6.0, 20.0 / self._zoom)
            f.setPointSizeF(font_pts)
            painter.setFont(f)
            painter.setPen(QPen(color))
            # Box must expand with the font when zoom clips it below its natural size
            half = max(15.0 / self._zoom, font_pts * 2.0)
            for idx in self._anomaly_area_violations:
                if idx < len(self._overlays):
                    pts, _color, _thick, visible = self._overlays[idx]
                    if visible and pts:
                        cx = sum(p.x() for p in pts) / len(pts)
                        cy = sum(p.y() for p in pts) / len(pts)
                        painter.drawText(
                            QRectF(cx - half, cy - half, half * 2, half * 2),
                            Qt.AlignCenter,
                            "!",
                        )

        if self._anomaly_distance_pairs:
            dr, dg, db = self._anomaly_dist_color
            dist_color = QColor(dr, dg, db, 200)
            dist_pen = QPen(dist_color, 2.0 / self._zoom, Qt.DashLine)
            painter.setPen(dist_pen)
            for pair in self._anomaly_distance_pairs:
                idxs = list(pair)
                if len(idxs) != 2:
                    continue
                i, j = idxs[0], idxs[1]
                if i >= len(self._overlays) or j >= len(self._overlays):
                    continue
                pts_i = self._overlays[i][0]
                pts_j = self._overlays[j][0]
                if not pts_i or not pts_j:
                    continue
                cx_i = sum(p.x() for p in pts_i) / len(pts_i)
                cy_i = sum(p.y() for p in pts_i) / len(pts_i)
                cx_j = sum(p.x() for p in pts_j) / len(pts_j)
                cy_j = sum(p.y() for p in pts_j) / len(pts_j)
                painter.drawLine(QPointF(cx_i, cy_i), QPointF(cx_j, cy_j))

                dist_val = self._anomaly_dist_values.get(pair)
                if dist_val is None:
                    continue
                label = f"{dist_val:.4g}{self._anomaly_dist_unit}"
                lf = painter.font()
                lf.setBold(True)
                lfont_pts = max(5.0, 11.0 / self._zoom)
                lf.setPointSizeF(lfont_pts)
                painter.setFont(lf)
                painter.setPen(QPen(QColor(dr, dg, db)))
                mx = (cx_i + cx_j) / 2
                my = (cy_i + cy_j) / 2
                dx = cx_j - cx_i
                dy = cy_j - cy_i
                # Box grows with the font when zoom pushes font past its natural size
                box_w = max(55.0 / self._zoom, lfont_pts * 8.0)
                box_h = max(14.0 / self._zoom, lfont_pts * 2.0)
                # Near-vertical line: shift label right so it doesn't overlap
                if abs(dy) > abs(dx):
                    x0 = mx + 4.0 / self._zoom
                else:
                    x0 = mx - box_w / 2
                painter.drawText(
                    QRectF(x0, my - box_h / 2, box_w, box_h),
                    Qt.AlignCenter,
                    label,
                )
                painter.setPen(dist_pen)

    def _paint_calib_dots(self, painter: QPainter) -> None:
        """Draw calibration and measure dots in display coords (inside scaled painter)."""
        m = self._calib_model

        # Pending calibration clicks (local state, not yet in model)
        dot_color = QColor(255, 107, 107)
        r = 6.0 / self._zoom
        QPen(dot_color, 1.5 / self._zoom)
        QPen(dot_color)
        f = painter.font()
        f.setPointSizeF(max(7.0, 9.0 / self._zoom))
        painter.setFont(f)

        pts_to_draw = []
        for i, (ox, oy) in enumerate(self._pending_calib_pts):
            pts_to_draw.append((ox, oy, str(i + 1), dot_color))

        for ox, oy, label, color in pts_to_draw:
            cx = ox * self._base_scale
            cy = oy * self._base_scale
            painter.setPen(QPen(color, 1.5 / self._zoom))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(QPointF(cx, cy), r, r)
            painter.drawEllipse(QPointF(cx, cy), 1.5 / self._zoom, 1.5 / self._zoom)
            painter.setPen(QPen(color))
            painter.drawText(QPointF(cx + r + 2.0 / self._zoom, cy - r), label)

        if len(pts_to_draw) == 2:
            p1d = QPointF(
                pts_to_draw[0][0] * self._base_scale,
                pts_to_draw[0][1] * self._base_scale,
            )
            p2d = QPointF(
                pts_to_draw[1][0] * self._base_scale,
                pts_to_draw[1][1] * self._base_scale,
            )
            dash_pen = QPen(dot_color, 1.5 / self._zoom, Qt.DashLine)
            dash_pen.setDashPattern([6, 4])
            painter.setPen(dash_pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawLine(p1d, p2d)

        # Measure points
        if m is None:
            return
        meas_color = QColor(255, 216, 102)
        mp1, mp2 = m.meas_points()
        meas_pts = []
        if mp1:
            meas_pts.append((mp1[0], mp1[1]))
        if mp2:
            meas_pts.append((mp2[0], mp2[1]))

        for ox, oy in meas_pts:
            cx = ox * self._base_scale
            cy = oy * self._base_scale
            painter.setPen(QPen(meas_color, 1.5 / self._zoom))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(QPointF(cx, cy), r, r)
            painter.drawEllipse(QPointF(cx, cy), 1.5 / self._zoom, 1.5 / self._zoom)

        if len(meas_pts) == 2:
            p1d = QPointF(
                meas_pts[0][0] * self._base_scale, meas_pts[0][1] * self._base_scale
            )
            p2d = QPointF(
                meas_pts[1][0] * self._base_scale, meas_pts[1][1] * self._base_scale
            )
            painter.setPen(QPen(meas_color, 1.5 / self._zoom))
            painter.setBrush(Qt.NoBrush)
            painter.drawLine(p1d, p2d)
            dist = m.measured_distance()
            if dist is not None:
                mx = (p1d.x() + p2d.x()) / 2
                my = (p1d.y() + p2d.y()) / 2 - 10.0 / self._zoom
                f2 = painter.font()
                f2.setPointSizeF(max(8.0, 10.0 / self._zoom))
                f2.setBold(True)
                painter.setFont(f2)
                painter.setPen(QPen(meas_color))
                painter.drawText(QPointF(mx, my), f"{dist:.1f} {m.unit()}")
        elif len(meas_pts) == 1 and self.current_tool == MEASURE and self._mouse_pos:
            p1d = QPointF(
                meas_pts[0][0] * self._base_scale, meas_pts[0][1] * self._base_scale
            )
            cursor_disp = self.view_to_display(self._mouse_pos)
            dash_pen = QPen(meas_color, 1.0 / self._zoom, Qt.DashLine)
            dash_pen.setDashPattern([4, 4])
            painter.setPen(dash_pen)
            painter.drawLine(p1d, cursor_disp)

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

        self._paint_center_crop(painter)

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
        for i, (pts, color, thick, visible) in enumerate(self._overlays):
            if visible and len(pts) >= 2:
                is_selected = i == self.selected_polygon_idx
                screen_thick = ((thick * 2.0) if is_selected else thick) / self._zoom
                pen = QPen(color, screen_thick)
                alpha = 150 if is_selected else 60  # Darker fill if selected

                painter.setPen(pen)
                painter.setBrush(
                    QBrush(QColor(color.red(), color.green(), color.blue(), alpha))
                )
                painter.drawPolygon(QPolygonF(pts + [pts[0]]))

                screen_r = max(1.75, min(4.0, self._zoom * 2.0))
                r = screen_r / self._zoom
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

        self._paint_violation_highlights(painter)
        self._paint_calib_dots(painter)

        # Switch to screen coordinates for the viewport-fixed grid
        painter.resetTransform()
        self._paint_grid(painter)

    def _crop_border_qcolor(self) -> QColor:
        """Return the border color: user-set color or auto-contrast from the crop region."""
        if self._center_crop_color is not None:
            r, g, b = self._center_crop_color
            return QColor(r, g, b)
        img = self._orig_image_bgr
        if img is None:
            return QColor(255, 255, 255)
        img_h, img_w = img.shape[:2]
        crop_w = self._center_crop_width or img_w
        crop_h = self._center_crop_height or img_h
        if self._center_crop_shape == "circle":
            d = min(crop_w, crop_h)
            crop_w = crop_h = d
        cx = self._center_crop_center_x if self._center_crop_center_x is not None else img_w / 2.0
        cy = self._center_crop_center_y if self._center_crop_center_y is not None else img_h / 2.0
        x1 = max(0, int(cx - crop_w / 2))
        y1 = max(0, int(cy - crop_h / 2))
        x2 = min(img_w, int(cx + crop_w / 2))
        y2 = min(img_h, int(cy + crop_h / 2))
        if x1 >= x2 or y1 >= y2:
            return QColor(255, 255, 255)
        region = img[y1:y2:4, x1:x2:4]
        mean = region.mean(axis=(0, 1))
        avg_b, avg_g, avg_r = float(mean[0]), float(mean[1]), float(mean[2])
        return QColor(int(255 - avg_r), int(255 - avg_g), int(255 - avg_b))

    def _paint_center_crop(self, painter: QPainter) -> None:
        """Dim everything outside the configured center crop preview."""
        if (
            not self._center_crop_enabled
            or self._display_qpix is None
            or self._orig_image_bgr is None
        ):
            return
        img_h, img_w = self._orig_image_bgr.shape[:2]
        self._ensure_center_crop_defaults(img_w, img_h)
        crop_w = self._center_crop_width or img_w
        crop_h = self._center_crop_height or img_h
        if self._center_crop_shape == "circle":
            diameter = min(crop_w, crop_h)
            crop_w = crop_h = diameter

        center_x = (
            self._center_crop_center_x
            if self._center_crop_center_x is not None
            else img_w / 2.0
        )
        center_y = (
            self._center_crop_center_y
            if self._center_crop_center_y is not None
            else img_h / 2.0
        )
        x = (center_x - crop_w / 2.0) * self._base_scale
        y = (center_y - crop_h / 2.0) * self._base_scale
        w = crop_w * self._base_scale
        h = crop_h * self._base_scale
        crop_rect = QRectF(x, y, w, h)
        image_rect = QRectF(
            0, 0, self._display_qpix.width(), self._display_qpix.height()
        )

        outside = QPainterPath()
        outside.setFillRule(Qt.OddEvenFill)
        outside.addRect(image_rect)
        if self._center_crop_shape == "circle":
            outside.addEllipse(crop_rect)
        else:
            outside.addRect(crop_rect)

        border_color = self._crop_border_qcolor()
        painter.save()
        painter.fillPath(outside, QColor(0, 0, 0, int(self._center_crop_opacity * 255)))
        pen = QPen(border_color, 1.5 / self._zoom, Qt.SolidLine)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        if self._center_crop_shape == "circle":
            painter.drawEllipse(crop_rect)
            if self._center_crop_calibrating:
                self._paint_center_calibration_grid(painter, crop_rect, border_color)
        else:
            painter.drawRect(crop_rect)
        if self._center_crop_dot_visible:
            cx = center_x * self._base_scale
            cy = center_y * self._base_scale
            radius = 4.0 / self._zoom
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(255, 0, 0, 170)))
            painter.drawEllipse(QPointF(cx, cy), radius, radius)
        painter.restore()

    def _paint_center_calibration_grid(
        self, painter: QPainter, crop_rect: QRectF, border_color: QColor
    ) -> None:
        """Draw a reference grid clipped to the active circular center crop."""
        diameter_px = crop_rect.width() / max(self._base_scale, 0.0001)
        divisions = max(4, min(16, int(round(diameter_px / 150.0))))
        if divisions <= 1:
            return

        clip_path = QPainterPath()
        clip_path.addEllipse(crop_rect)
        painter.save()
        painter.setClipPath(clip_path)

        spacing_x = crop_rect.width() / divisions
        spacing_y = crop_rect.height() / divisions
        center_index = divisions / 2.0

        for i in range(1, divisions):
            is_center_line = abs(i - center_index) < 0.001
            alpha = 150 if is_center_line else 85
            width = (1.25 if is_center_line else 0.75) / self._zoom
            c = QColor(border_color.red(), border_color.green(), border_color.blue(), alpha)
            pen = QPen(c, width, Qt.SolidLine)
            painter.setPen(pen)

            x = crop_rect.left() + i * spacing_x
            y = crop_rect.top() + i * spacing_y
            painter.drawLine(
                QPointF(x, crop_rect.top()),
                QPointF(x, crop_rect.bottom()),
            )
            painter.drawLine(
                QPointF(crop_rect.left(), y),
                QPointF(crop_rect.right(), y),
            )

        painter.restore()
