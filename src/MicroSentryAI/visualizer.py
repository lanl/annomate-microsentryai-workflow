"""
MicroSentryAI Visualizer Module.

This module provides a high-fidelity Graphical User Interface (GUI) for 
AI-assisted defect detection and validation. It utilizes the PyQt5 framework 
to synchronize dual viewports for side-by-side analysis of raw data and 
statistical anomaly heatmaps.

Key Components:
    * MicroSentryWindow: Primary application controller and interface.
    * SegPathItem & VertexHandle: Graphics primitives for polygon manipulation.
    * SyncedGraphicsView: Viewport with spatial synchronization logic.
    * InferenceWorker: QThread implementation for non-blocking model execution.
"""

import os
import glob
import sys
import numpy as np
import cv2
import torch
from PIL import Image
from pathlib import Path
from typing import List, Optional, Tuple, Any, Dict

from scipy.ndimage import gaussian_filter
from matplotlib import colormaps as mpl_cmaps

from PyQt5.QtCore import (
    Qt, QPointF, QRectF, pyqtSignal, QTimer, QThread, QSize, QObject
)
from PyQt5.QtGui import (
    QPixmap, QImage, QPen, QBrush, QColor, QPainterPath,
    QKeySequence, QIcon, QMouseEvent
)
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QFileDialog,
    QHBoxLayout, QVBoxLayout, QMessageBox, QSlider, QStatusBar, QGridLayout,
    QSpinBox, QDoubleSpinBox, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QGraphicsPathItem, QGraphicsEllipseItem, QShortcut,
    QProgressBar, QTableWidgetItem, QTableWidget, QHeaderView, QAbstractItemView, QSplitter,
    QInputDialog
)

# Configuration for relative module discovery
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from .strategies.anomalib_strategy import AnomalibStrategy

# Attempt to integrate with AnnoMate ecosystem; implement fallbacks for standalone execution
try:
    from AnnoMate.widgets import WrappingTableWidget, CustomSplitter
    from AnnoMate.styles import SPLITTER_STYLE, MAIN_STYLESHEET
except ImportError as e:
    print(f"CRITICAL: Integration modules not found: {e}")
    print("Initializing internal fallback components.")
    
    class WrappingTableWidget(QTableWidget):
        """Internal implementation of table widget behavior."""
        def keyPressEvent(self, event):
            super().keyPressEvent(event)

    class CustomSplitter(QSplitter):
        """Internal implementation of splitter behavior."""
        pass

    SPLITTER_STYLE = ""
    MAIN_STYLESHEET = ""


# =========================================================================
# GRAPHICS PRIMITIVES
# =========================================================================

HANDLE_RADIUS = 4.0


class VertexHandle(QGraphicsEllipseItem):
    """
    Interactive handle for controlling individual vertices within a SegPathItem.
    
    Attributes:
        parent_item (SegPathItem): The parent polygon object.
        idx (int): The index of the vertex in the parent's coordinate list.
    """

    def __init__(self, parent: 'SegPathItem', idx: int, pos: QPointF):
        super().__init__(-HANDLE_RADIUS, -HANDLE_RADIUS, HANDLE_RADIUS * 2, HANDLE_RADIUS * 2, parent)
        self.setAcceptHoverEvents(True)
        self.setBrush(QBrush(QColor("#FFEB3B")))
        self.setPen(QPen(QColor(20, 20, 20), 1))
        self.setFlag(QGraphicsEllipseItem.ItemIsMovable, True)
        self.setFlag(QGraphicsEllipseItem.ItemSendsGeometryChanges, True)
        self.setZValue(100)
        
        self.parent_item = parent
        self.idx = idx
        self.setPos(pos)

    def hoverEnterEvent(self, event):
        self.setBrush(QBrush(QColor("#00BCD4")))
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.setBrush(QBrush(QColor("#FFEB3B")))
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event):
        self.parent_item.lock_move = True
        self.parent_item.setFlag(QGraphicsPathItem.ItemIsMovable, False)
        if self.parent_item.on_any_edit:
            self.parent_item.on_any_edit('vertex_drag_begin')
        super().mousePressEvent(event)
        event.accept()

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        self.parent_item.update_vertex(self.idx, self.pos())
        event.accept()

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        self.parent_item.setFlag(QGraphicsPathItem.ItemIsMovable, True)
        self.parent_item.lock_move = False
        if self.parent_item.on_any_edit:
            self.parent_item.on_any_edit('vertex_drag_end')
        event.accept()

    def cleanup(self):
        """Safely removes handle from scene and parent."""
        self.setParentItem(None)
        if self.scene():
            self.scene().removeItem(self)


class SegPathItem(QGraphicsPathItem):
    """
    An editable polygon primitive representing segmented regions of interest.
    
    Supports Douglas-Peucker simplification, affine scaling, and vertex-level 
    spatial transformations.
    """

    def __init__(self, pts: List[QPointF], on_any_edit=None):
        super().__init__()
        self.setFlags(QGraphicsPathItem.ItemIsMovable | QGraphicsPathItem.ItemIsSelectable)
        self.pen_normal = QPen(QColor(0, 255, 0), 2)
        self.pen_selected = QPen(QColor(255, 235, 59), 2)
        
        self.handles: List[VertexHandle] = []
        self._pts = pts[:]
        self.lock_move = False
        self.on_any_edit = on_any_edit
        
        self.setZValue(10)
        self._rebuild_path()

    def _rebuild_path(self):
        """Constructs a QPainterPath from the current vertex sequence."""
        path = QPainterPath()
        if self._pts:
            path.moveTo(self._pts[0])
            for p in self._pts[1:]:
                path.lineTo(p)
            path.closeSubpath()
        self.setPath(path)

    def paint(self, painter, option, widget=None):
        """Overrides default paint behavior to update pens based on selection state."""
        self.setPen(self.pen_selected if self.isSelected() else self.pen_normal)
        super().paint(painter, option, widget)

    def set_selected(self, selected: bool):
        """Updates selection state and toggles visibility of control handles."""
        self.setSelected(selected)
        if selected and not self.handles:
            for i, p in enumerate(self._pts):
                h = VertexHandle(parent=self, idx=i, pos=p)
                self.handles.append(h)
        elif not selected and self.handles:
            for h in self.handles:
                h.cleanup()
            self.handles = []

    def update_vertex(self, idx: int, newpos: QPointF):
        """Updates internal coordinates and redraws path path geometry."""
        if 0 <= idx < len(self._pts):
            self._pts[idx] = newpos
            self._rebuild_path()
            if self.on_any_edit:
                self.on_any_edit('vertex_drag')

    def simplify(self, epsilon: float):
        """
        Reduces vertex count using the Douglas-Peucker algorithm.
        
        Args:
            epsilon (float): Distance tolerance for simplification.
        """
        if len(self._pts) < 3:
            return
            
        cnt = np.array([[[p.x(), p.y()]] for p in self._pts], dtype=np.float32)
        approx = cv2.approxPolyDP(cnt, epsilon=epsilon, closed=True)
        
        if approx is None or len(approx) < 3:
            return

        self._pts = [QPointF(float(p[0][0]), float(p[0][1])) for p in approx]
        self._rebuild_path()
        
        if self.handles:
            for h in self.handles:
                h.cleanup()
            self.handles = [
                VertexHandle(parent=self, idx=i, pos=p) 
                for i, p in enumerate(self._pts)
            ]
            
        if self.on_any_edit:
            self.on_any_edit('polygon_simplify')

    def itemChange(self, change, value):
        """Locks item transformation if a vertex drag operation is in progress."""
        if change == QGraphicsPathItem.ItemPositionChange and self.lock_move:
            return self.pos()
        return super().itemChange(change, value)

    def mousePressEvent(self, event):
        self._start_pos = self.pos()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        if not self.lock_move and (self.pos() != getattr(self, '_start_pos', self.pos())):
            if self.on_any_edit:
                self.on_any_edit('polygon_move')

    def scale_about_center(self, factor: float):
        """Applies a uniform scale transformation relative to the polygon centroid."""
        if not self._pts:
            return
        cx = sum(p.x() for p in self._pts) / len(self._pts)
        cy = sum(p.y() for p in self._pts) / len(self._pts)
        
        new_pts = []
        for p in self._pts:
            nx = cx + (p.x() - cx) * factor
            ny = cy + (p.y() - cy) * factor
            new_pts.append(QPointF(nx, ny))
            
        self._pts = new_pts
        for i, h in enumerate(self.handles):
            h.setPos(self._pts[i])
            
        self._rebuild_path()
        if self.on_any_edit:
            self.on_any_edit('polygon_scale')


class SyncedGraphicsView(QGraphicsView):
    """
    A viewport implementation that facilitates synchronized multi-view rendering.
    
    Signals:
        viewChanged: Emits normalized center ratios (x, y) and absolute scale.
    """
    viewChanged = pyqtSignal(float, float, float)

    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        self._is_syncing = False
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        
        self.horizontalScrollBar().valueChanged.connect(self._emit_view)
        self.verticalScrollBar().valueChanged.connect(self._emit_view)

    def _emit_view(self):
        """Calculates and emits the current spatial state of the viewport."""
        if self._is_syncing:
            return
        if self.sceneRect().width() <= 0 or self.sceneRect().height() <= 0:
            return

        center_pt = self.mapToScene(self.viewport().rect().center())
        w, h = self.sceneRect().width(), self.sceneRect().height()
        rx = center_pt.x() / w
        ry = center_pt.y() / h
        scale = self.transform().m11()
        
        self.viewChanged.emit(rx, ry, scale)

    def wheelEvent(self, event):
        """Implements exponential zoom anchored at the cursor position."""
        zoom_in_factor = 1.15
        zoom_out_factor = 1 / zoom_in_factor

        if event.angleDelta().y() > 0:
            zoom_factor = zoom_in_factor
        else:
            zoom_factor = zoom_out_factor

        self.scale(zoom_factor, zoom_factor)
        self._emit_view()
        event.accept()

    def set_view_state(self, rx: float, ry: float, scale: float):
        """Updates viewport position and scale based on external signals."""
        if self.sceneRect().width() <= 0:
            return
            
        self._is_syncing = True
        self.resetTransform()
        self.scale(scale, scale)
        
        w, h = self.sceneRect().width(), self.sceneRect().height()
        target_pt = QPointF(rx * w, ry * h)
        self.centerOn(target_pt)
        
        self._is_syncing = False


# =========================================================================
# IMAGE PROCESSING & THREADING
# =========================================================================

def reconstruct_display_crop(orig: Image.Image, target_size: int) -> Tuple[Image.Image, Tuple[float, float], Tuple[int, int]]:
    """
    Resizes the image to fit within target_size x target_size while maintaining aspect ratio.
    Does NOT crop the image, allowing full-image segmentation visualization.
    
    Returns:
        Tuple containing the processed Image, (scale_x, scale_y), and (offset_x, offset_y).
        Note: Offsets are (0, 0) as no cropping occurs.
    """
    w, h = orig.size
    # Scale based on the largest dimension to fit within the box
    scale = target_size / max(w, h)
    
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
        
    img = orig.resize((new_w, new_h), Image.LANCZOS)
    
    # Return (img, scale_factor, offset)
    return img, (scale, scale), (0, 0)


def pil_to_qpixmap(pil_img: Image.Image) -> QPixmap:
    """Utility to bridge PIL Image objects with Qt QPixmap representations."""
    rgb = pil_img.convert("RGB")
    w, h = rgb.size
    data = rgb.tobytes("raw", "RGB")
    qimage = QImage(data, w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimage)


class InferenceWorker(QThread):
    """
    Concurrent execution engine for anomaly model inference.
    
    Signals:
        resultReady: Emits file path and the raw anomaly score map.
        progress: Emits integer percentage or step count.
        finished: Emits on completion of the input file list.
    """
    resultReady = pyqtSignal(str, object) 
    progress = pyqtSignal(int)
    finished = pyqtSignal()

    def __init__(self, strategy, file_list: List[str]):
        super().__init__()
        self.strategy = strategy
        self.file_list = file_list
        self.is_running = True

    def run(self):
        """Iterates through the file list and executes strategy prediction."""
        for i, path in enumerate(self.file_list):
            if not self.is_running:
                break
            try:
                _, score_map = self.strategy.predict(path)
                self.resultReady.emit(path, score_map)
            except Exception:
                pass
            
            self.progress.emit(i + 1)
        
        self.finished.emit()

    def stop(self):
        """Terminates thread execution after the current iteration."""
        self.is_running = False


# =========================================================================
# APPLICATION CONTROLLER
# =========================================================================

class MicroSentryWindow(QMainWindow):
    """
    Main controller for the MicroSentryAI ecosystem.
    
    Manages state synchronization between the model strategy, local dataset 
    management, and dual-viewport rendering.
    """
    polygonsSent = pyqtSignal(list, str)
    imageIndexChanged = pyqtSignal(int)
    viewChanged = pyqtSignal(float, float, float)
    folderLoaded = pyqtSignal(str, list)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("MicroSentryAI (Modular)")
        self.resize(1400, 900)
        
        if MAIN_STYLESHEET:
            self.setStyleSheet(MAIN_STYLESHEET)

        # Application State Management
        self.active_strategy = None
        self.image_files: List[str] = []
        self.idx = 0
        self.score_map = None
        self.orig_full: Optional[Image.Image] = None
        self.last_scale = (1.0, 1.0)
        self.last_offset = (0, 0)
        
        # Visualization Hyperparameters
        self.display_target = 600
        self.alpha = 0.45
        self.sigma = 4.0
        
        # Resource Caching & History
        self.undo_stack = []
        self.redo_stack = []
        self._block_history = False
        self.inference_cache = {}
        self.worker: Optional[InferenceWorker] = None

        self._init_ui()

    def _init_ui(self):
        """Orchestrates layout construction and component registration."""
        main_splitter = CustomSplitter(Qt.Horizontal, self)
        if SPLITTER_STYLE:
            main_splitter.setStyleSheet(SPLITTER_STYLE)
        main_splitter.setHandleWidth(3)
        self.setCentralWidget(main_splitter)

        left_widget = QWidget()
        self.left_layout = QVBoxLayout(left_widget)
        
        self._setup_toolbar()
        self._setup_canvas_area()
        self._setup_bottom_bar()

        right_widget = QWidget()
        self.right_layout = QVBoxLayout(right_widget)
        self._setup_sidebar()

        main_splitter.addWidget(left_widget)
        main_splitter.addWidget(right_widget)
        main_splitter.setStretchFactor(0, 2) 
        main_splitter.setStretchFactor(1, 1)

        self.status = QStatusBar()
        self.setStatusBar(self.status)

        self._connect_signals()
        self._setup_shortcuts()
        self._set_window_icon()

    def _set_window_icon(self):
        """Resolves filesystem paths to locate application branding."""
        d = os.path.dirname(os.path.abspath(__file__))
        self.icon_path = None
        for _ in range(5):
            possible_logos = os.path.join(d, "logos")
            if os.path.isdir(possible_logos):
                for name in ["MicroSentryAI.png", "MicroSentryAI.ico"]:
                    full_path = os.path.join(possible_logos, name)
                    if os.path.exists(full_path):
                        self.icon_path = full_path
                        self.setWindowIcon(QIcon(full_path))
                        return
            d = os.path.dirname(d)

    def _setup_toolbar(self):
        """Constructs the primary control rows for parameter tuning and file I/O."""
        row1 = QWidget()
        row1_layout = QHBoxLayout(row1)
        row1_layout.setContentsMargins(5, 5, 5, 0)
        
        row2 = QWidget()
        row2_layout = QHBoxLayout(row2)
        row2_layout.setContentsMargins(5, 0, 5, 5)

        self.model_label = QLabel("No Model Loaded")
        self.model_label.setStyleSheet("font-weight: bold; color: #913333;")
        
        self.display_spin = QSpinBox()
        self.display_spin.setRange(256, 2048)
        self.display_spin.setValue(self.display_target)
        self.display_spin.setSuffix(" px")
        
        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0.0, 1.0)
        self.alpha_spin.setSingleStep(0.05)
        self.alpha_spin.setValue(self.alpha)

        self.heat_thresh_spin = QSpinBox()
        self.heat_thresh_spin.setRange(0, 100)
        self.heat_thresh_spin.setValue(00)
        self.heat_thresh_spin.setSuffix("%")
        
        self.sigma_spin = QSpinBox()
        self.sigma_spin.setRange(0, 16)
        self.sigma_spin.setValue(int(self.sigma))
        self.sigma_label = QLabel(f"σ: {int(self.sigma)}")
        
        self.eps_spin = QDoubleSpinBox()
        self.eps_spin.setRange(0.0, 20.0)
        self.eps_spin.setSingleStep(0.5)
        self.eps_spin.setDecimals(1)
        self.eps_spin.setValue(1.5)

        self.btn_simpl_sel = QPushButton("Simplify Selected")
        self.btn_simpl_all = QPushButton("Simplify All")
        self.btn_load_model = QPushButton("Load Model Folder") 
        self.btn_load_images = QPushButton("Load Image Folder")
        self.btn_send_annot = QPushButton("Send to AnnoMate")
        self.btn_send_annot.setStyleSheet("background-color: #d1f7d1; font-weight: bold; padding: 5px;")

        logo_label = QLabel()
        if hasattr(self, 'icon_path') and self.icon_path:
            pm = QPixmap(self.icon_path)
            if not pm.isNull():
                logo_label.setPixmap(pm.scaled(80, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation))

        row1_layout.addWidget(logo_label)
        row1_layout.addStretch()
        row1_layout.addWidget(QLabel("Current Model:"))
        row1_layout.addWidget(self.model_label)
        row1_layout.addStretch()
        row1_layout.addWidget(QLabel("Display:"))
        row1_layout.addWidget(self.display_spin)
        row1_layout.addSpacing(10)
        row1_layout.addWidget(QLabel("Heat α:"))
        row1_layout.addWidget(self.alpha_spin)
        row1_layout.addSpacing(10)
        row1_layout.addWidget(QLabel("Heat Min:"))
        row1_layout.addWidget(self.heat_thresh_spin)

        row2_layout.addWidget(QLabel("Smooth (σ):"))
        row2_layout.addWidget(self.sigma_spin)
        row2_layout.addWidget(self.sigma_label)
        row2_layout.addSpacing(15)
        row2_layout.addWidget(QLabel("Simplify ε:"))
        row2_layout.addWidget(self.eps_spin)
        row2_layout.addWidget(self.btn_simpl_sel)
        row2_layout.addWidget(self.btn_simpl_all)
        row2_layout.addStretch()
        row2_layout.addWidget(self.btn_load_model)
        row2_layout.addWidget(self.btn_load_images)
        row2_layout.addWidget(self.btn_send_annot)

        self.left_layout.addWidget(row1)
        self.left_layout.addWidget(row2)

    def _setup_canvas_area(self):
        """Initializes the dual QGraphicsView system and connects synchronization signals."""
        center = QWidget()
        grid = QGridLayout(center)
        
        grid.addWidget(QLabel("Segmentation"), 0, 0, alignment=Qt.AlignHCenter)
        grid.addWidget(QLabel("Heatmap Overlay"), 0, 1, alignment=Qt.AlignHCenter)
        
        self.scene_left = QGraphicsScene()
        self.view_left = SyncedGraphicsView(self.scene_left)
        self.view_left.setBackgroundBrush(QBrush(QColor(0, 0, 0)))
        
        self.scene_right = QGraphicsScene()
        self.view_right = SyncedGraphicsView(self.scene_right)
        self.view_right.setBackgroundBrush(QBrush(QColor(0, 0, 0)))
        
        grid.addWidget(self.view_left, 1, 0)
        grid.addWidget(self.view_right, 1, 1)
        
        self.view_left.viewChanged.connect(
            lambda rx, ry, s: self._sync_views(self.view_right, rx, ry, s)
        )
        self.view_right.viewChanged.connect(
            lambda rx, ry, s: self._sync_views(self.view_left, rx, ry, s)
        )

        self.left_layout.addWidget(center, stretch=1)

    def _setup_bottom_bar(self):
        """Constructs navigation controls and status indicators."""
        bottom = QWidget()
        bottom_layout = QHBoxLayout(bottom)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #00BCD4; }")
        
        self.btn_prev = QPushButton("< Previous")
        self.btn_next = QPushButton("Next >")
        
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(95)
        self.slider_label = QLabel("Percentile Threshold: 95.0")
        
        bottom_layout.addWidget(self.progress_bar)
        bottom_layout.addWidget(self.btn_prev)
        bottom_layout.addWidget(self.slider)
        bottom_layout.addWidget(self.btn_next)

        self.left_layout.addWidget(self.slider_label)
        self.left_layout.addWidget(bottom)

    def on_heat_threshold_change(self, v: int):
        """Updates threshold label and triggers heatmap re-rendering."""
        self.render_current_images(push_undo=False)

    def _setup_sidebar(self):
        """Configures the wrapping table widget for dataset navigation."""
        self.right_layout.setContentsMargins(0, 0, 0, 0)
        self.right_layout.addWidget(QLabel("Dataset"))
        
        self.table = WrappingTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Image Name", "Status"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        
        self.table.currentCellChanged.connect(lambda r, c, pr, pc: self.goto_index(r))
        self.right_layout.addWidget(self.table)

    def _connect_signals(self):
        """Maps interactive UI elements to application logic methods."""
        self.btn_load_model.clicked.connect(self.load_model_clicked)
        self.btn_load_images.clicked.connect(self.load_images_clicked)
        self.btn_send_annot.clicked.connect(self.send_annotations)
        self.btn_prev.clicked.connect(self.prev_image)
        self.btn_next.clicked.connect(self.next_image)
        
        self.slider.valueChanged.connect(self.on_threshold_change)
        self.display_spin.valueChanged.connect(self.on_display_change)
        
        self.alpha_spin.valueChanged.connect(self.on_alpha_change)
        self.heat_thresh_spin.valueChanged.connect(self.on_heat_threshold_change)
        self.sigma_spin.valueChanged.connect(self.on_sigma_change)
        
        self.btn_simpl_sel.clicked.connect(self.simplify_selected)
        self.btn_simpl_all.clicked.connect(self.simplify_all)

    def _setup_shortcuts(self):
        """Registers standard keyboard sequences for common operations."""
        QShortcut(QKeySequence('Ctrl+Z'), self, activated=self.undo)
        QShortcut(QKeySequence('Ctrl+Y'), self, activated=self.redo)
        QShortcut(QKeySequence('S'), self, activated=self.simplify_selected_shortcut)

    # --- DATASET MANAGEMENT ---

    def _is_processed(self, idx: int) -> bool:
        """Determines if the image at index has cached inference results."""
        if 0 <= idx < len(self.image_files):
            return self.image_files[idx] in self.inference_cache
        return False

    def _status_text(self, i: int) -> str:
        return "Processed" if self._is_processed(i) else "Pending"

    def _status_brush(self, i: int) -> QBrush:
        """Returns color theme for table status based on processing state."""
        if self._is_processed(i):
            return QBrush(QColor(210, 245, 210))
        return QBrush(QColor(255, 235, 210))

    def _build_table(self):
        """Populates the dataset table based on the current image file list."""
        n = len(self.image_files)
        self.table.setRowCount(n)
        for i in range(n):
            self._update_table_row(i)
        self.table.resizeRowsToContents()

    def _update_table_row(self, i: int):
        """Updates individual row metadata and styling."""
        if 0 <= i < self.table.rowCount():
            stem = Path(self.image_files[i]).stem
            
            idx_item = QTableWidgetItem(stem)
            idx_item.setTextAlignment(Qt.AlignCenter)
            idx_item.setFlags(idx_item.flags() ^ Qt.ItemIsEditable)
            
            status_item = QTableWidgetItem(self._status_text(i))
            status_item.setFlags(status_item.flags() ^ Qt.ItemIsEditable)
            status_item.setBackground(self._status_brush(i))
            
            self.table.setItem(i, 0, idx_item)
            self.table.setItem(i, 1, status_item)

    def goto_index(self, idx: int):
        """Transitions application state to focus on a specific image index."""
        if not self.image_files:
            return
        
        # If we are strictly already at this index (and it wasn't a forced -1 reset), skip
        if idx == self.idx and idx != -1:
            return

        idx = max(0, min(len(self.image_files) - 1, idx))
        self.idx = idx
        self.process_image()
        
        self.table.blockSignals(True)
        self.table.selectRow(idx)
        self.table.blockSignals(False)
        self.imageIndexChanged.emit(idx)

    # --- FILE I/O ---

    def load_model_clicked(self):
        """
        Loads exported weights via AnomalibStrategy. 
        Supports PyTorch (.pt) and OpenVINO (.xml) formats.
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select Exported Model", 
            "", 
            "Anomalib Models (*.pt *.xml)"
        )
        
        if not file_path:
            return
        
        try:
            if file_path.endswith(".xml"):
                devices = ["CPU", "GPU", "AUTO"]
            else:
                devices = ["auto", "cpu", "cuda", "mps"]
                
            device, ok = QInputDialog.getItem(
                self, "Select Device", "Inference Hardware:", devices, 0, False
            )
            if not ok:
                return

            self.status.showMessage(f"Loading {os.path.basename(file_path)}...")
            QApplication.processEvents()
            
            strategy = AnomalibStrategy()
            strategy.set_device(device.lower())
            strategy.load_from_file(file_path)
            
            self.active_strategy = strategy
            self.model_label.setText(f"Active: {strategy.model_name}")
            self.model_label.setStyleSheet("font-weight: bold; color: #538A3F;")
            self.status.showMessage("Model Loaded Successfully")
            
            if self.image_files:
                self.inference_cache = {}
                self.start_background_inference()
                self.process_image()
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model:\n{str(e)}")
            self.status.showMessage("Error loading model")

    def load_images_clicked(self):
        """Discovers compatible image assets in a user-selected directory."""
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder", "")
        if not folder:
            return
            
        files = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
            files.extend(glob.glob(os.path.join(folder, ext)))
        
        self.image_files = sorted(files)
        if not self.image_files:
            QMessageBox.information(self, "No Images", "No compatible images found.")
            return
        
        self._build_table()
        
        # --- FIXED: Robustly trigger first-image loading ---
        self.idx = -1  # Force reset so goto_index(0) always triggers
        self.goto_index(0)
        # ---------------------------------------------------
        
        self.folderLoaded.emit(folder, self.image_files)
        
        if self.active_strategy:
            self.start_background_inference()

    # --- INFERENCE PIPELINE ---

    def start_background_inference(self):
        """Initializes a background worker for batch prediction."""
        if not self.active_strategy or not self.image_files:
            return
        
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()

        pending_files = [f for f in self.image_files if f not in self.inference_cache]
        if not pending_files: 
            self.status.showMessage("All images cached.")
            return

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, len(pending_files))
        self.progress_bar.setValue(0)
        self.status.showMessage(f"Batching {len(pending_files)} images in background...")

        self.worker = InferenceWorker(self.active_strategy, pending_files)
        self.worker.resultReady.connect(self.on_worker_result)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.start()

    def on_worker_result(self, path: str, score_map: Any):
        """Callback for individual image processing completion."""
        self.inference_cache[path] = score_map
        
        try:
            idx = self.image_files.index(path)
            self._update_table_row(idx)
        except ValueError:
            pass

        if self.image_files and path == self.image_files[self.idx]:
            self.score_map = score_map
            self.render_current_images(push_undo=False)

    def on_worker_finished(self):
        self.progress_bar.setVisible(False)
        self.status.showMessage("Batch inference complete.")

    # --- RENDERING & VISUALIZATION ---

    def process_image(self):
        """Prepares a single image for display, executing model logic if not cached."""
        if not self.image_files:
            return
            
        path = self.image_files[self.idx]
        try:
            self.orig_full = Image.open(path).convert('RGB')
            
            if path in self.inference_cache:
                self.score_map = self.inference_cache[path]
            elif self.active_strategy:
                # Synchronous fallback for immediate viewing
                _, self.score_map = self.active_strategy.predict(path)
                self.inference_cache[path] = self.score_map
                self._update_table_row(self.idx)
            else:
                self.score_map = None
            
            self.undo_stack.clear()
            self.redo_stack.clear()
            
            self.render_current_images(push_undo=False)
            self.status.showMessage(f"Image: {os.path.basename(path)} ({self.idx + 1}/{len(self.image_files)})")
            
        except Exception as e:
            QMessageBox.critical(self, "Inference Error", f"Failed to process {path}:\n{e}")

    def render_current_images(self, push_undo: bool = False):
        """Unified rendering entry point for scene updates."""
        if self.orig_full is None:
            return

        self.scene_left.clear()
        self.scene_right.clear()

        target = int(self.display_target)
        disp, scale, offset = reconstruct_display_crop(self.orig_full, target)
        self.last_scale = scale
        self.last_offset = offset

        w_disp, h_disp = disp.size

        left_bg = QGraphicsPixmapItem(pil_to_qpixmap(disp))
        left_bg.setZValue(-10)
        self.scene_left.addItem(left_bg)
        
        rect = QRectF(0, 0, w_disp, h_disp)
        self.view_left.setSceneRect(rect)
        self.scene_left.setSceneRect(rect)
        self.view_right.setSceneRect(rect)
        self.scene_right.setSceneRect(rect)

        if self.score_map is not None:
            self._render_heatmap_and_polygons(disp, target)

        self.refresh_view()

    def _render_heatmap_and_polygons(self, disp_image: Image.Image, target_size: int):
        """Generates heatmap overlays and derived polygons from raw anomaly scores."""
        s = self.score_map
        if self.sigma > 0:
            s = gaussian_filter(s, sigma=self.sigma)
            
        # Normalization and visualization clipping
        v_percentile = float(self.heat_thresh_spin.value())
        v_min_thr = np.percentile(s, v_percentile)
        
        s_clipped = np.clip(s, v_min_thr, s.max())
        mx, mn = s_clipped.max(), s_clipped.min()
        s_norm = (s_clipped - mn) / (mx - mn + 1e-12) if mx > mn else (s_clipped - mn)
        
        w_disp, h_disp = disp_image.size
        s_norm_resized = cv2.resize(s_norm, (w_disp, h_disp), interpolation=cv2.INTER_LINEAR)
        
        # Colorize
        heat_img_arr = (s_norm_resized * 255).astype(np.uint8)
        colored = (mpl_cmaps["jet"](heat_img_arr / 255.0) * 255).astype(np.uint8)
        
        overlay = Image.fromarray(colored, mode="RGBA")
        r, g, b, a = overlay.split()
        a = a.point(lambda p: int(p * self.alpha))
        overlay = Image.merge("RGBA", (r, g, b, a))
        
        comp = disp_image.convert("RGBA")
        final_right = Image.alpha_composite(comp, overlay).convert("RGB")
        self.scene_right.addItem(QGraphicsPixmapItem(pil_to_qpixmap(final_right)))
        
        # Segmentation logic
        percentile = float(self.slider.value())
        self.slider_label.setText(f"Percentile Threshold: {percentile:.1f}")
        seg_thr = np.percentile(s, percentile)

        mask = (s > seg_thr).astype(np.uint8) * 255
        mask = cv2.resize(mask, (w_disp, h_disp), interpolation=cv2.INTER_NEAREST)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        eps = float(self.eps_spin.value())
        
        for cnt in contours:
            if len(cnt) < 3:
                continue
            approx = cv2.approxPolyDP(cnt, epsilon=eps, closed=True)
            pts = [
                QPointF(float(pt[0][0]), float(pt[0][1])) 
                for pt in (approx if approx is not None else cnt)
            ]
            item = SegPathItem(pts, on_any_edit=self.on_any_edit)
            self.scene_left.addItem(item)

    # --- VIEWPORT NAVIGATION ---

    def prev_image(self):
        self.goto_index(self.idx - 1)

    def next_image(self):
        self.goto_index(self.idx + 1)

    def on_threshold_change(self, _):
        self.render_current_images(push_undo=False)

    def on_display_change(self, v):
        self.display_target = int(v)
        self.render_current_images(push_undo=False)

    def on_alpha_change(self, v):
        self.alpha = float(v)
        self.render_current_images(push_undo=False)

    def on_sigma_change(self, v):
        self.sigma = float(v)
        self.sigma_label.setText(f"σ: {int(self.sigma)}")
        self.render_current_images(push_undo=False)

    def _sync_views(self, target_view, rx, ry, scale):
        """Propagates spatial changes to a target view."""
        target_view.set_view_state(rx, ry, scale)
        self.viewChanged.emit(rx, ry, scale)

    def refresh_view(self):
        """Force-synchronizes canvas zoom levels and centered alignment."""
        if self.scene_left.itemsBoundingRect().width() > 0:
            self.view_left.resetTransform()
            self.view_left.fitInView(self.scene_left.sceneRect(), Qt.KeepAspectRatio)
        if self.scene_right.itemsBoundingRect().width() > 0:
            self.view_right.resetTransform()
            self.view_right.fitInView(self.scene_right.sceneRect(), Qt.KeepAspectRatio)
        QTimer.singleShot(50, self._force_fit_delayed)

    def _force_fit_delayed(self):
        if self.scene_left.itemsBoundingRect().width() > 0:
            self.view_left.fitInView(self.scene_left.sceneRect(), Qt.KeepAspectRatio)
        if self.scene_right.itemsBoundingRect().width() > 0:
            self.view_right.fitInView(self.scene_right.sceneRect(), Qt.KeepAspectRatio)

    def set_view_state(self, rx, ry, scale):
        self.view_left.set_view_state(rx, ry, scale)
        self.view_right.set_view_state(rx, ry, scale)

    def showEvent(self, ev):
        super().showEvent(ev)
        self.refresh_view()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self.refresh_view()

    # --- MEMENTO PATTERN (UNDO/REDO) ---

    def serialize_polygons(self):
        """Converts scene items into a JSON-serializable state representation."""
        polys = []
        for item in self.scene_left.items():
            if isinstance(item, SegPathItem):
                pts = [(p.x(), p.y()) for p in item._pts]
                pos = (item.pos().x(), item.pos().y())
                polys.append({'pts': pts, 'pos': pos})
        return polys

    def restore_polygons(self, polys: List[Dict]):
        """Reconstructs scene state from a serialized representation."""
        self.scene_left.clear()
        if self.orig_full:
            target = int(self.display_target)
            disp, _, _ = reconstruct_display_crop(self.orig_full, target)
            left_bg = QGraphicsPixmapItem(pil_to_qpixmap(disp))
            left_bg.setZValue(-10)
            self.scene_left.addItem(left_bg)
            
            w_disp, h_disp = disp.size
            rect = QRectF(0, 0, w_disp, h_disp)
            self.view_left.setSceneRect(rect)
            self.scene_left.setSceneRect(rect)
            
        for poly in polys:
            pts = [QPointF(x, y) for (x, y) in poly['pts']]
            item = SegPathItem(pts, on_any_edit=self.on_any_edit)
            item.setPos(poly['pos'][0], poly['pos'][1])
            self.scene_left.addItem(item)

    def push_undo_state(self):
        """Saves current state to the undo stack."""
        if self._block_history:
            return
        self.undo_stack.append(self.serialize_polygons())
        self.redo_stack.clear()

    def on_any_edit(self, kind: str):
        if kind in ('vertex_drag_begin', 'polygon_move'):
            self.push_undo_state()

    def undo(self):
        """Reverts to the previous state in the history stack."""
        if not self.undo_stack:
            return
        self._block_history = True
        try:
            current = self.serialize_polygons()
            prior = self.undo_stack.pop()
            self.redo_stack.append(current)
            self.restore_polygons(prior)
        finally:
            self._block_history = False

    def redo(self):
        """Advances to the next state in the history stack."""
        if not self.redo_stack:
            return
        self._block_history = True
        try:
            current = self.serialize_polygons()
            nxt = self.redo_stack.pop()
            self.undo_stack.append(current)
            self.restore_polygons(nxt)
        finally:
            self._block_history = False

    def keyPressEvent(self, event):
        """Handles keyboard-based transformations and item deletions."""
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            self.push_undo_state()
            for item in list(self.scene_left.selectedItems()):
                self.scene_left.removeItem(item)
        elif event.key() == Qt.Key_BracketLeft:
            self.push_undo_state()
            for item in self.scene_left.selectedItems():
                if isinstance(item, SegPathItem):
                    item.scale_about_center(0.9)
        elif event.key() == Qt.Key_BracketRight:
            self.push_undo_state()
            for item in self.scene_left.selectedItems():
                if isinstance(item, SegPathItem):
                    item.scale_about_center(1.1)
        super().keyPressEvent(event)

    # --- GEOMETRIC EDITING ---

    def simplify_selected_shortcut(self):
        self.simplify_selected()

    def simplify_selected(self):
        """Reduces complexity for currently selected SegPathItems."""
        eps = float(self.eps_spin.value())
        self.push_undo_state()
        for item in self.scene_left.selectedItems():
            if isinstance(item, SegPathItem):
                item.simplify(eps)

    def simplify_all(self):
        """Reduces complexity for all SegPathItems in the scene."""
        eps = float(self.eps_spin.value())
        self.push_undo_state()
        for item in self.scene_left.items():
            if isinstance(item, SegPathItem):
                item.simplify(eps)

    def send_annotations(self):
        """
        Calculates and emits original-coordinate polygons.
        
        Applies inverse spatial transformations to convert canvas-space points 
        back to high-resolution source image coordinates.
        """
        selected_only = len(self.scene_left.selectedItems()) > 0
        items_to_send = self.scene_left.selectedItems() if selected_only else self.scene_left.items()
        
        polys_to_send = []
        scale_x, scale_y = self.last_scale
        off_x, off_y = self.last_offset
        
        for item in items_to_send:
            if isinstance(item, SegPathItem):
                orig_pts = []
                mx, my = item.pos().x(), item.pos().y()
                
                for p in item._pts:
                    disp_x = p.x() + mx
                    disp_y = p.y() + my
                    uncropped_x = disp_x + off_x
                    uncropped_y = disp_y + off_y
                    final_x = uncropped_x / scale_x
                    final_y = uncropped_y / scale_y
                    orig_pts.append((final_x, final_y))
                polys_to_send.append(orig_pts)
                
        if not polys_to_send:
            QMessageBox.information(self, "Info", "No polygons to send.")
            return
            
        self.polygonsSent.emit(polys_to_send, "Anomaly")

    def mousePressEvent(self, event: QMouseEvent):
        super().mousePressEvent(event)
        for item in self.scene_left.items():
            if isinstance(item, SegPathItem):
                item.set_selected(item.isSelected())


if __name__ == "__main__":
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    w = MicroSentryWindow()
    w.show()
    sys.exit(app.exec_())