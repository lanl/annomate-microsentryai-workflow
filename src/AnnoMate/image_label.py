"""
Image Label Widget for AnnoMate.

This module defines the `ImageLabel` class, a custom PySide6 widget that handles:
1.  Displaying images with high-performance zooming and panning.
2.  Interactive polygon annotation (drawing, canceling, finishing).
3.  Rendering overlays (existing annotations) on top of the image.
"""

from typing import List, Tuple, Optional
import cv2
import numpy as np

from PySide6.QtCore import Qt, QPointF, QRect, QSize
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

# Tool Constants
POLYGON = "polygon"


class ImageLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setScaledContents(False)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setFocusPolicy(Qt.StrongFocus)  # Allow widget to catch Key_Escape

        self.main_window = None 
        self.current_tool: Optional[str] = None
        self._active_color = QColor(0, 200, 0)

        self._orig_image_bgr: Optional[np.ndarray] = None
        self._display_qpix: Optional[QPixmap] = None

        self._base_scale = 1.0  
        self._zoom = 1.0        
        self._pan = QPointF(0, 0)

        self._panning = False
        self._last_mouse_pos: Optional[QPointF] = None
        self._mouse_pos: Optional[QPointF] = None

        self.line_thickness = 2.0  # Default brush thickness

        self.current_polygon_points: List[QPointF] = []
        self._overlays: List[Tuple[List[QPointF], QColor]] = []

        
        # --- UI State Trackers ---
        self.selected_polygon_idx: int = -1  
        self.editing_polygon_idx: int = -1   
        self.dragging_vertex_idx: int = -1   
        self._dragging_polygon: bool = False 

    def load_image(self, path: str, max_display_dim: int = 1200):
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Failed to load image: {path}")

        self._orig_image_bgr = bgr
        h, w = bgr.shape[:2]

        self._base_scale = (
            1.0 if max(h, w) <= max_display_dim
            else max_display_dim / float(max(h, w))
        )

        self._zoom = 1.0
        self._pan = QPointF(0, 0)

        new_w = int(w * self._base_scale)
        new_h = int(h * self._base_scale)
        
        resized_bgr = cv2.resize(
            bgr, (new_w, new_h), interpolation=cv2.INTER_AREA
        )
        
        rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
        
        qimg = QImage(
            rgb.data,
            rgb.shape[1],
            rgb.shape[0],
            rgb.strides[0],
            QImage.Format_RGB888
        )
        self._display_qpix = QPixmap.fromImage(qimg)

        # --- Reset all tracking states on new image ---
        self.clear_current_polygon()
        self._overlays = []
        self.selected_polygon_idx = -1
        self.editing_polygon_idx = -1   
        self.dragging_vertex_idx = -1   
        self._dragging_polygon = False  
        
        self.update()

    def set_main_window(self, main_window):
        self.main_window = main_window

    def set_tool(self, tool_name: Optional[str]):
        self.current_tool = tool_name

    def set_active_color(self, color: QColor):
        self._active_color = color if isinstance(color, QColor) else QColor(0, 200, 0)

    def set_overlays(self, poly_list: List[Tuple[List[Tuple[float, float]], QColor, float]]):
        self._overlays = []
        for pts_orig, color, thickness in poly_list:
            disp_pts = [
                QPointF(x * self._base_scale, y * self._base_scale)
                for (x, y) in pts_orig
            ]
            self._overlays.append((disp_pts, color, thickness))
        self.update()

    def clear_current_polygon(self):
        self.current_polygon_points.clear()
        self.update()

    def load_image(self, path: str, max_display_dim: int = 1200):
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Failed to load image: {path}")

        self._orig_image_bgr = bgr
        h, w = bgr.shape[:2]

        self._base_scale = (
            1.0 if max(h, w) <= max_display_dim
            else max_display_dim / float(max(h, w))
        )

        self._zoom = 1.0
        self._pan = QPointF(0, 0)

        new_w = int(w * self._base_scale)
        new_h = int(h * self._base_scale)
        
        resized_bgr = cv2.resize(
            bgr, (new_w, new_h), interpolation=cv2.INTER_AREA
        )
        
        rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
        
        qimg = QImage(
            rgb.data,
            rgb.shape[1],
            rgb.shape[0],
            rgb.strides[0],
            QImage.Format_RGB888
        )
        self._display_qpix = QPixmap.fromImage(qimg)

        self.clear_current_polygon()
        self._overlays = []
        self.selected_polygon_idx = -1
        self.editing_polygon_idx = -1   
        self.dragging_vertex_idx = -1  
        self.update()

    def view_to_display(self, p_view: QPointF) -> QPointF:
        return QPointF(
            (p_view.x() - self._pan.x()) / self._zoom,
            (p_view.y() - self._pan.y()) / self._zoom
        )

    def display_to_original(self, p_disp: QPointF) -> Tuple[float, float]:
        return (p_disp.x() / self._base_scale, p_disp.y() / self._base_scale)

    def finish_current_polygon(self):
        if self.current_polygon_points and self.main_window is not None:
            pts_orig = [
                self.display_to_original(p) for p in self.current_polygon_points
            ]
            self.main_window.finish_polygon(pts_orig)
            
        self.current_polygon_points.clear()
        self.update()

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_Escape:
            self.clear_current_polygon()
            return

        if self.current_tool == POLYGON and self.current_polygon_points:
            if event.key() == Qt.Key_Backspace:
                self.current_polygon_points.pop()
                self.update()
                return
        super().keyPressEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        if self.current_tool == POLYGON and self.current_polygon_points:
            self.finish_current_polygon()
            return
        
         # --- NEW: Double Click to Edit ---
        if event.button() == Qt.LeftButton:
            pos_disp = self.view_to_display(QPointF(event.pos()))
            found_idx = -1
            for i, (pts, *_) in enumerate(reversed(self._overlays)):
                if QPolygonF(pts).containsPoint(pos_disp, Qt.OddEvenFill):
                    found_idx = len(self._overlays) - 1 - i
                    break

            if found_idx != -1:
                self.editing_polygon_idx = found_idx
                self.selected_polygon_idx = found_idx
                if self.main_window and hasattr(self.main_window, 'on_polygon_selected'):
                    self.main_window.on_polygon_selected(found_idx)
                self.update()
                return
            
        super().mouseDoubleClickEvent(event)

    def mousePressEvent(self, event: QMouseEvent):
        self.setFocus()  
        
        if event.button() == Qt.LeftButton:
            pos_view = QPointF(event.pos())
            pos_disp = self.view_to_display(pos_view)

            # --- PRE-CHECK: Identify if we clicked inside any existing polygon ---
            found_idx = -1
            for i, (pts, *_) in enumerate(reversed(self._overlays)):
                if QPolygonF(pts).containsPoint(pos_disp, Qt.OddEvenFill):
                    found_idx = len(self._overlays) - 1 - i
                    break

            # 1. Edit Mode: Vertex Dragging & Polygon Dragging
            if self.editing_polygon_idx != -1:
                pts, *_ = self._overlays[self.editing_polygon_idx]
                closest_idx = -1
                min_dist = float('inf')
                
                for i, p_disp in enumerate(pts):
                    p_view = QPointF(
                        p_disp.x() * self._zoom + self._pan.x(),
                        p_disp.y() * self._zoom + self._pan.y()
                    )
                    dist = ((p_view.x() - pos_view.x())**2 + (p_view.y() - pos_view.y())**2)**0.5
                    if dist < 10.0 and dist < min_dist:
                        min_dist = dist
                        closest_idx = i
                
                if closest_idx != -1:
                    self.dragging_vertex_idx = closest_idx
                    return

                if QPolygonF(pts).containsPoint(pos_disp, Qt.OddEvenFill):
                    self._dragging_polygon = True
                    self._last_mouse_pos = pos_view
                    return
                else:
                    self.editing_polygon_idx = -1
                    self.setCursor(Qt.ArrowCursor)
                    self.update()
                    
            # 2. Normal Polygon Drawing
            if self.current_tool == POLYGON:
                # Visually highlight the polygon we clicked on, even while drawing
                if not self.current_polygon_points:
                    self.selected_polygon_idx = found_idx
                    if self.main_window and hasattr(self.main_window, 'on_polygon_selected'):
                        self.main_window.on_polygon_selected(found_idx)
                
                # Always append point on single click
                self.current_polygon_points.append(self.view_to_display(QPointF(event.pos())))
                self.update()
                return

            # 3. Normal Selection & Polygon Dragging (When Polygon Tool is OFF)
            self.selected_polygon_idx = found_idx
            self.update()
            if self.main_window and hasattr(self.main_window, 'on_polygon_selected'):
                self.main_window.on_polygon_selected(found_idx)
            
            if found_idx != -1:
                self._dragging_polygon = True
                self._last_mouse_pos = pos_view
                
        elif event.button() == Qt.RightButton:
            self._panning = True
            self._last_mouse_pos = QPointF(event.pos())

    def mouseMoveEvent(self, event: QMouseEvent):
        self._mouse_pos = QPointF(event.pos())
        
        # --- NEW: Live update vertex while dragging ---
        if self.dragging_vertex_idx != -1 and self.editing_polygon_idx != -1:
            new_pos = self.view_to_display(self._mouse_pos)
            pts, *_ = self._overlays[self.editing_polygon_idx]
            pts[self.dragging_vertex_idx] = new_pos
            self.update()
            return

        # --- NEW: Drag entire polygon ---
        if self._dragging_polygon and self.selected_polygon_idx != -1 and self._last_mouse_pos is not None:
            delta_view = self._mouse_pos - self._last_mouse_pos
            delta_disp = QPointF(delta_view.x() / self._zoom, delta_view.y() / self._zoom)
            
            pts, *_ = self._overlays[self.selected_polygon_idx]
            for i in range(len(pts)):
                pts[i] = QPointF(pts[i].x() + delta_disp.x(), pts[i].y() + delta_disp.y())
            
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
            self.update()  
            return
            
        # --- NEW: Hover detection for precision cursor ---
        if self.editing_polygon_idx != -1 and not self._dragging_polygon and not self._panning:
            pts, *_ = self._overlays[self.editing_polygon_idx]
            hovering = False
            for p_disp in pts:
                p_view = QPointF(
                    p_disp.x() * self._zoom + self._pan.x(),
                    p_disp.y() * self._zoom + self._pan.y()
                )
                dist = ((p_view.x() - self._mouse_pos.x())**2 + (p_view.y() - self._mouse_pos.y())**2)**0.5
                if dist < 10.0:
                    hovering = True
                    break
            
            # Switch to thin crosshair if hovering, otherwise standard arrow
            self.setCursor(Qt.CrossCursor if hovering else Qt.ArrowCursor)
            
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            # Commit Vertex Drag
            if self.dragging_vertex_idx != -1:
                if self.main_window and hasattr(self.main_window, 'update_polygon_points'):
                    pts, *_ = self._overlays[self.editing_polygon_idx]
                    pts_orig = [self.display_to_original(p) for p in pts]
                    self.main_window.update_polygon_points(self.editing_polygon_idx, pts_orig)
                self.dragging_vertex_idx = -1
                return
            
            # Commit Polygon Drag
            if self._dragging_polygon:
                if self.main_window and hasattr(self.main_window, 'update_polygon_points'):
                    pts, *_ = self._overlays[self.selected_polygon_idx]
                    pts_orig = [self.display_to_original(p) for p in pts]
                    self.main_window.update_polygon_points(self.selected_polygon_idx, pts_orig)
                self._dragging_polygon = False
                self._last_mouse_pos = None
                return

        if event.button() == Qt.RightButton:
            self._panning = False
            self._last_mouse_pos = None

    def wheelEvent(self, event: QWheelEvent):
        """Mathematically precise cursor-anchored zoom."""
        if self._display_qpix is None:
            return

        delta = event.angleDelta().y()
        if delta == 0:
            return

        steps = delta / 120.0
        steps = max(-5.0, min(5.0, steps)) # Clamp extreme inputs
        factor = 1.15 ** steps

        cursor_pos = event.position()
        point_in_disp = self.view_to_display(cursor_pos)
        
        old_zoom = self._zoom
        self._zoom = max(0.2, min(8.0, old_zoom * factor))

        self._pan = QPointF(
            cursor_pos.x() - point_in_disp.x() * self._zoom,
            cursor_pos.y() - point_in_disp.y() * self._zoom,
        )
        self.update()

    def zoom_in(self):
        self._apply_zoom(1.15)

    def zoom_out(self):
        self._apply_zoom(1 / 1.15)

    def reset_view(self):
        self._zoom = 1.0
        self._pan = QPointF(0, 0)
        self.update()

    def _apply_zoom(self, factor: float):
        """Button-based zoom targeted precisely at the center of the widget."""
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

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing) # NEW: Smooth line rendering
        
        if self._display_qpix is None:
            return

        painter.translate(self._pan)
        painter.scale(self._zoom, self._zoom)
        painter.drawPixmap(0, 0, self._display_qpix)

        # Draw Overlays with Selection Highlighting
        for i, (pts, color, poly_thickness) in enumerate(self._overlays):
            if len(pts) >= 2:
                is_selected = (i == self.selected_polygon_idx)
                
                # Adjusted opacity for lines and fills
                line_alpha = 100  
                fill_alpha = 90 if is_selected else 35  
                
                pen_color = QColor(color.red(), color.green(), color.blue(), line_alpha)
                brush_color = QColor(color.red(), color.green(), color.blue(), fill_alpha)
                
                # Use the POLYGON'S thickness, scale slightly if selected
                drawn_thickness = poly_thickness * 1.5 if is_selected else poly_thickness
                
                pen = QPen(pen_color, drawn_thickness)
                pen.setJoinStyle(Qt.RoundJoin) # Rounded corners on thick lines
                
                painter.setPen(pen)
                painter.setBrush(QBrush(brush_color))
                painter.drawPolygon(QPolygonF(pts + [pts[0]]))
                
                # --- Draw Editing Handles ---
                if i == self.editing_polygon_idx:
                    r = max(4.0 / self._zoom, 1.0) 
                    painter.setBrush(QBrush(QColor("#FFEB3B"))) 
                    painter.setPen(QPen(QColor(20, 20, 20, 90), 1))
                    for p in pts:
                        painter.drawEllipse(p, r, r)

        # --- Draw the actively drawing, in-progress polygon ---
        if self.current_tool == POLYGON and self.current_polygon_points:
            
            if self._mouse_pos:
                cursor_in_disp = self.view_to_display(self._mouse_pos)
                painter.drawLine(
                    self.current_polygon_points[-1],
                    cursor_in_disp,
                )
