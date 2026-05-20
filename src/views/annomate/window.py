"""AnnoMateWindow — unified annotation and inference main window.

See MVC.md § Architecture Rules for the full layer contract.
Color scheme: no explicit stylesheet colors — Qt platform palette only.
"""

import logging
import os

import numpy as np
from PySide6.QtCore import Qt, QEvent, QPoint, QPointF, QTimer, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QWidget,
    QFrame,
    QVBoxLayout,
    QLabel,
    QSizePolicy,
    QToolButton,
    QPushButton,
    QFileDialog,
    QMessageBox,
    QHBoxLayout,
    QComboBox,
    QApplication,
)

from views.annomate._splitter import StyledSplitter

from views.annomate.image_label import ImageLabel, SAM_BBOX, CALIBRATE, MEASURE
from views.annomate.right_panel import RightPanel
from views.annomate.tool_palette import ToolPalette
from views.annomate.status_bar import AnnoMateStatusBar
from views.annomate.viewport_actions import ViewportActionsBar
from controllers.sam_controller import SAMController

logger = logging.getLogger(__name__)


class _AIAcceptPopup(QFrame):
    """Floating accept button + class selector for a selected AI polygon."""

    accepted = Signal()

    _BTN_SIZE = 28

    def __init__(self, canvas: QWidget, parent: QWidget = None) -> None:
        super().__init__(parent or canvas)
        self._canvas = canvas
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        self.setAutoFillBackground(True)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        btn_accept = QToolButton()
        btn_accept.setText("✓")
        btn_accept.setToolTip("Accept polygon into selected class")
        btn_accept.setFixedSize(self._BTN_SIZE, self._BTN_SIZE)
        btn_accept.clicked.connect(self.accepted)
        layout.addWidget(btn_accept)

        self._combo = QComboBox()
        self._combo.setToolTip("Class to assign polygon to")
        layout.addWidget(self._combo)

        self.adjustSize()
        self.setVisible(False)

    def set_classes(self, names: list, active: str) -> None:
        self._combo.blockSignals(True)
        self._combo.clear()
        self._combo.addItems(names)
        if active in names:
            self._combo.setCurrentIndex(names.index(active))
        self._combo.blockSignals(False)

    def current_class(self) -> str:
        return self._combo.currentText()

    def show_at_polygon(self, bbox) -> None:
        """Position just outside the right edge of *bbox* (a QRect in widget coords)."""
        x = bbox.right() + 8
        y = bbox.top()
        # If too close to the right edge, flip to the left side
        if x + self.width() > self._canvas.width():
            x = bbox.left() - self.width() - 8
        x = max(0, x)
        y = max(0, min(y, self._canvas.height() - self.height()))
        self.move(x, y)
        self.setVisible(True)
        self.raise_()


class _ReviewBar(QFrame):
    """Floating Accept/Reject bar positioned at the top-right of the canvas.

    Emits decision_changed("accept"), decision_changed("reject"), or
    decision_changed(None) when the user toggles a button.
    """

    decision_changed = Signal(object)

    _MARGIN = 10
    _BTN_H = 28

    def __init__(self, canvas: QWidget, parent: QWidget = None) -> None:
        super().__init__(parent)
        self._canvas = canvas
        self._dragging = False
        self._drag_start = QPoint()
        self._custom_pos_ratio = None
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        self.setAutoFillBackground(True)
        self.setObjectName("reviewBar")
        self.setStyleSheet(
            """
            QFrame#reviewBar {
                background: palette(window);
                border: 1px solid palette(mid);
                border-radius: 8px;
            }
            """
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(6)

        self._drag_handle = QLabel("⋮")
        self._drag_handle.setToolTip("Drag to reposition")
        self._drag_handle.setCursor(Qt.SizeAllCursor)
        self._drag_handle.setFixedWidth(14)
        self._drag_handle.setAlignment(Qt.AlignCenter)
        handle_font = self._drag_handle.font()
        handle_font.setPointSize(handle_font.pointSize() + 3)
        handle_font.setBold(True)
        self._drag_handle.setFont(handle_font)
        layout.addWidget(self._drag_handle)

        self._btn_accept = QToolButton()
        self._btn_accept.setText("✓ Accept")
        self._btn_accept.setToolTip("Mark this image as accepted")
        self._btn_accept.setCheckable(True)
        self._btn_accept.setCursor(Qt.PointingHandCursor)
        self._btn_accept.setFixedHeight(self._BTN_H)
        self._btn_accept.clicked.connect(
            lambda checked: self._on_clicked("accept", checked)
        )
        layout.addWidget(self._btn_accept)

        self._btn_reject = QToolButton()
        self._btn_reject.setText("✗ Reject")
        self._btn_reject.setToolTip("Mark this image as rejected")
        self._btn_reject.setCheckable(True)
        self._btn_reject.setCursor(Qt.PointingHandCursor)
        self._btn_reject.setFixedHeight(self._BTN_H)
        self._btn_reject.clicked.connect(
            lambda checked: self._on_clicked("reject", checked)
        )
        layout.addWidget(self._btn_reject)

        self.adjustSize()
        self.setVisible(False)

    def _on_clicked(self, which: str, checked: bool) -> None:
        if checked:
            other = self._btn_reject if which == "accept" else self._btn_accept
            other.setChecked(False)
            self._update_styles()
            self.decision_changed.emit(which)
        else:
            self._update_styles()
            self.decision_changed.emit(None)

    def set_decision(self, decision) -> None:
        """Silently update button states to reflect *decision* without emitting."""
        self._btn_accept.blockSignals(True)
        self._btn_reject.blockSignals(True)
        self._btn_accept.setChecked(decision == "accept")
        self._btn_reject.setChecked(decision == "reject")
        self._btn_accept.blockSignals(False)
        self._btn_reject.blockSignals(False)
        self._update_styles()

    def _update_styles(self) -> None:
        self._btn_accept.setStyleSheet(
            "background-color: #4caf50; color: white;"
            if self._btn_accept.isChecked()
            else ""
        )
        self._btn_reject.setStyleSheet(
            "background-color: #f44336; color: white;"
            if self._btn_reject.isChecked()
            else ""
        )

    def reposition(self, canvas_size) -> None:
        if self._custom_pos_ratio is not None:
            rx, ry = self._custom_pos_ratio
            x = int(rx * max(1, canvas_size.width() - self.width()))
            y = int(ry * max(1, canvas_size.height() - self.height()))
        else:
            x = canvas_size.width() - self.width() - self._MARGIN
            y = self._MARGIN
        self.move(self._clamped_pos(x, y))

    def mousePressEvent(self, event) -> None:
        handle_rect = self._drag_handle.geometry()
        if event.button() == Qt.LeftButton and handle_rect.contains(
            event.position().toPoint()
        ):
            self._dragging = True
            self._drag_start = event.position().toPoint()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._dragging:
            pos = self.mapToParent(event.position().toPoint() - self._drag_start)
            self.move(self._clamped_pos(pos.x(), pos.y()))
            self._store_custom_pos_ratio()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self._dragging:
            self._dragging = False
            self._store_custom_pos_ratio()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _clamped_pos(self, x: int, y: int) -> QPoint:
        max_x = max(0, self._canvas.width() - self.width())
        max_y = max(0, self._canvas.height() - self.height())
        return QPoint(max(0, min(x, max_x)), max(0, min(y, max_y)))

    def _store_custom_pos_ratio(self) -> None:
        max_x = max(1, self._canvas.width() - self.width())
        max_y = max(1, self._canvas.height() - self.height())
        self._custom_pos_ratio = (self.x() / max_x, self.y() / max_y)


class _ProjectStartScreen(QFrame):
    """Centered empty-state panel shown before a project or image folder is loaded."""

    new_project_requested = Signal()
    open_project_requested = Signal()
    open_image_folder_requested = Signal()
    open_recent_project_requested = Signal(str)
    open_recent_image_folder_requested = Signal(str)

    def __init__(self, parent: QWidget = None) -> None:
        super().__init__(parent)
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        self.setAutoFillBackground(True)
        self.setObjectName("ProjectStartScreen")
        self.setStyleSheet(
            """
            QFrame#ProjectStartScreen {
                border-radius: 10px;
            }
            QLabel#ProjectStartTitle {
                font-size: 22px;
                font-weight: bold;
            }
            QLabel#ProjectStartBody {
                font-size: 13px;
            }
            QPushButton {
                min-height: 30px;
                padding-left: 12px;
                padding-right: 12px;
            }
            """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 22, 24, 22)
        layout.setSpacing(12)

        title = QLabel("Start a project")
        title.setObjectName("ProjectStartTitle")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        body = QLabel(
            "Open an existing .annoproj project, load a folder of images, "
            "or start a fresh annotation session."
        )
        body.setObjectName("ProjectStartBody")
        body.setAlignment(Qt.AlignCenter)
        body.setWordWrap(True)
        layout.addWidget(body)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        btn_open_project = QPushButton("Open Project")
        btn_open_project.setToolTip("Open an existing .annoproj project")
        btn_open_project.clicked.connect(
            lambda: QTimer.singleShot(0, self.open_project_requested.emit)
        )
        btn_row.addWidget(btn_open_project)

        btn_open_images = QPushButton("Open Image Folder")
        btn_open_images.setToolTip("Load images from a local folder")
        btn_open_images.clicked.connect(
            lambda: QTimer.singleShot(0, self.open_image_folder_requested.emit)
        )
        btn_row.addWidget(btn_open_images)

        layout.addLayout(btn_row)

        self._recent_label = QLabel("Recent")
        self._recent_label.setObjectName("ProjectStartBody")
        self._recent_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self._recent_label)

        recent_row = QHBoxLayout()
        recent_row.setSpacing(8)

        self._recent_project_path = ""
        self._btn_recent_project = QPushButton("Open Last Project")
        self._btn_recent_project.clicked.connect(self._open_recent_project)
        recent_row.addWidget(self._btn_recent_project)

        self._recent_image_dir = ""
        self._btn_recent_images = QPushButton("Open Last Image Folder")
        self._btn_recent_images.clicked.connect(self._open_recent_images)
        recent_row.addWidget(self._btn_recent_images)

        layout.addLayout(recent_row)
        self.set_recent_actions("", "")
        self.adjustSize()

    def set_recent_actions(self, project_path: str, image_dir: str) -> None:
        """Show shortcuts to the last opened project and image folder."""
        self._recent_project_path = project_path or ""
        self._recent_image_dir = image_dir or ""

        has_project = bool(self._recent_project_path)
        has_images = bool(self._recent_image_dir)
        self._recent_label.setVisible(True)
        self._btn_recent_project.setVisible(True)
        self._btn_recent_images.setVisible(True)
        self._btn_recent_project.setEnabled(has_project)
        self._btn_recent_images.setEnabled(has_images)

        if has_project:
            name = os.path.basename(self._recent_project_path)
            self._btn_recent_project.setText(f"Last Project: {name}")
            self._btn_recent_project.setToolTip(self._recent_project_path)
        else:
            self._btn_recent_project.setText("No recent project")
            self._btn_recent_project.setToolTip(
                "Open or save a project to show it here."
            )

        if has_images:
            name = os.path.basename(os.path.normpath(self._recent_image_dir))
            self._btn_recent_images.setText(f"Last Image Folder: {name}")
            self._btn_recent_images.setToolTip(self._recent_image_dir)
        else:
            self._btn_recent_images.setText("No recent image folder")
            self._btn_recent_images.setToolTip("Open an image folder to show it here.")

        self.adjustSize()

    def _open_recent_project(self) -> None:
        if self._recent_project_path:
            QTimer.singleShot(
                0,
                lambda: self.open_recent_project_requested.emit(
                    self._recent_project_path
                ),
            )

    def _open_recent_images(self) -> None:
        if self._recent_image_dir:
            QTimer.singleShot(
                0,
                lambda: self.open_recent_image_folder_requested.emit(
                    self._recent_image_dir
                ),
            )


class AnnoMateWindow(QWidget):
    """Experimental Photoshop-style layout tab.

    Receives the dataset model and IO controller so later phases can wire up
    the canvas, navigator, and class panel without touching AppWindow.

    Args:
        dataset_model: DatasetTableModel instance.
        io_controller: IOController instance.
        parent: Optional Qt parent widget.
    """

    new_project_requested = Signal()
    open_project_requested = Signal()
    open_image_folder_requested = Signal()
    open_recent_project_requested = Signal(str)
    open_recent_image_folder_requested = Signal(str)

    def __init__(
        self,
        dataset_model,
        io_controller,
        inference_model=None,
        inference_controller=None,
        calibration_model=None,
        parent: QWidget = None,
    ) -> None:
        super().__init__(parent)
        self.dataset_model = dataset_model
        self.io_controller = io_controller
        self.inference_model = inference_model
        self.inference_controller = inference_controller
        self._calib_model = calibration_model
        self._current_row: int = -1
        self._active_class: str = ""
        self._active_tool: str = ""
        self._microsentry_enabled: bool = False
        self._current_bgr = None
        self._current_ai_contours: list = []
        self._selected_ai_idx: int = -1
        self._saved_model_path: str = ""
        self._sam_controller = SAMController(parent=self)
        self._sam_loading: bool = False
        self._init_ui()
        if calibration_model is not None:
            self.canvas.set_calibration_model(calibration_model)

        # Dataset changes
        self.dataset_model.modelReset.connect(self._on_model_reset)

        # Canvas → navigation / annotation
        self.canvas.polygonFinished.connect(self._on_polygon_finished)
        self.canvas.polygonEdited.connect(self._on_polygon_edited)
        self.canvas.toolCanceled.connect(self._on_tool_canceled)
        self.canvas.polygonSelected.connect(self._on_canvas_polygon_selected)
        self.canvas.ai_polygon_clicked.connect(self._on_ai_polygon_clicked)

        # Canvas → status bar (live feedback)
        self.canvas.zoom_changed.connect(self.status_bar.set_zoom)
        self.canvas.image_loaded.connect(self.status_bar.set_dimensions)

        # Right panel
        self.right_panel.image_selected.connect(self._load_row)
        self.right_panel.class_selected.connect(self._set_active_class)
        self.right_panel.prev_requested.connect(self._prev_image)
        self.right_panel.next_requested.connect(self._next_image)
        self.right_panel.annotation_selected.connect(self._on_annotation_selected)
        self.right_panel.load_model_requested.connect(self._on_load_model_requested)
        self.right_panel.load_previous_model_requested.connect(
            self._on_load_previous_model_requested
        )
        self.right_panel.microsentry_settings_changed.connect(
            self._refresh_canvas_render
        )
        self.right_panel.accept_polygons_requested.connect(self._on_accept_ai_polygons)

        # Keep canvas in sync when annotations change outside the canvas
        self.dataset_model.dataChanged.connect(self._on_dataset_data_changed)

        # Tool palette
        self.tool_palette.tool_selected.connect(self._on_tool_selected)
        self.viewport_actions.tool_selected.connect(self._on_tool_selected)
        self.canvas.draw_attempted.connect(self._on_draw_attempted)

        # Route thickness signal directly to canvas setter
        self.tool_palette.thickness_changed.connect(self._on_thickness_changed)

        # SAM tool
        self.canvas.samBboxDrawn.connect(self._on_sam_bbox_drawn)
        self.tool_palette.sam_variant_changed.connect(self._on_sam_variant_changed)

        # Calibration tool
        self.canvas.calibrationPointsPlaced.connect(self._on_calibration_points_placed)
        self._sam_controller.result_ready.connect(self._on_sam_result_ready)
        self._sam_controller.inference_failed.connect(self._on_sam_inference_failed)
        self._sam_controller.loading_done.connect(self._on_sam_loading_done)
        self._sam_controller.loading_failed.connect(self._on_sam_loading_failed)

        # Auto-load SAM silently if the checkpoint is already on disk
        variant = self.tool_palette.current_sam_variant()
        logger.info(
            "AnnoMateWindow startup: checking for cached SAM weights (%s)", variant
        )
        if self._sam_controller.try_autoload(variant):
            self._sam_loading = True
            logger.info("AnnoMateWindow startup: SAM autoload initiated")

        # Inference controller signals
        if self.inference_controller is not None:
            self.inference_controller.result_ready.connect(self._on_inference_result)
            self.inference_controller.progress.connect(self._on_inference_progress)
            self.inference_controller.batch_done.connect(self._on_inference_batch_done)

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #

    def _init_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_workspace(), stretch=1)

        self.status_bar = AnnoMateStatusBar(self)
        root.addWidget(self.status_bar)
        self._set_start_screen_visible(self.dataset_model.rowCount() == 0)

    def _build_workspace(self) -> QWidget:
        workspace = QWidget()
        workspace.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        h_layout = QHBoxLayout(workspace)
        h_layout.setContentsMargins(0, 0, 0, 0)
        h_layout.setSpacing(0)

        self.tool_palette = ToolPalette(self)
        h_layout.addWidget(self.tool_palette)

        splitter = StyledSplitter(Qt.Horizontal, margin=0)
        splitter.setHandleWidth(8)
        splitter.setChildrenCollapsible(False)

        self.canvas = ImageLabel(self)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        splitter.addWidget(self.canvas)

        self.viewport_actions = ViewportActionsBar(
            self.canvas, self._calib_model, self.canvas
        )
        self.viewport_actions.raise_()

        self._start_screen = _ProjectStartScreen(self.canvas)
        self._start_screen.new_project_requested.connect(
            self.new_project_requested.emit
        )
        self._start_screen.open_project_requested.connect(
            self.open_project_requested.emit
        )
        self._start_screen.open_image_folder_requested.connect(
            self.open_image_folder_requested.emit
        )
        self._start_screen.open_recent_project_requested.connect(
            self.open_recent_project_requested.emit
        )
        self._start_screen.open_recent_image_folder_requested.connect(
            self.open_recent_image_folder_requested.emit
        )
        self._start_screen.raise_()

        self._review_bar = _ReviewBar(self.canvas, self.canvas)
        self._review_bar.decision_changed.connect(self._on_review_decision)
        self._review_bar.raise_()

        self._ai_popup = _AIAcceptPopup(self.canvas, self.canvas)
        self._ai_popup.accepted.connect(self._on_accept_single_ai)

        self.canvas.installEventFilter(self)

        self.right_panel = RightPanel(
            self.dataset_model, self.inference_model, self._calib_model, self
        )
        self.right_panel.setMinimumWidth(160)
        splitter.addWidget(self.right_panel)

        splitter.setSizes([700, 280])
        h_layout.addWidget(splitter, stretch=1)

        return workspace

    # ------------------------------------------------------------------ #
    # Floating canvas controls
    # ------------------------------------------------------------------ #

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.viewport_actions.reposition(self.canvas.size())
        self._review_bar.reposition(self.canvas.size())
        self._reposition_start_screen()

    def eventFilter(self, obj, event) -> bool:
        if obj is self.canvas and event.type() == QEvent.Resize:
            self.viewport_actions.reposition(event.size())
            self._review_bar.reposition(event.size())
            self._reposition_start_screen()
        return super().eventFilter(obj, event)

    def _set_start_screen_visible(self, visible: bool) -> None:
        """Show the project start panel only while no dataset is loaded."""
        self._start_screen.setVisible(visible)
        if visible:
            self._start_screen.raise_()
            self._reposition_start_screen()

    def set_project_start_state(self, last_project: str, last_image_dir: str) -> None:
        """Update start-screen recent actions."""
        self._start_screen.set_recent_actions(last_project, last_image_dir)
        self._reposition_start_screen()

    def _reposition_start_screen(self) -> None:
        """Keep the empty-state panel centered in the canvas area."""
        if not hasattr(self, "_start_screen"):
            return
        width = min(520, max(360, self.canvas.width() - 80))
        self._start_screen.setFixedWidth(width)
        self._start_screen.adjustSize()
        x = max(0, (self.canvas.width() - self._start_screen.width()) // 2)
        y = max(24, (self.canvas.height() - self._start_screen.height()) // 2)
        self._start_screen.move(x, y)

    # ------------------------------------------------------------------ #
    # Navigation slots
    # ------------------------------------------------------------------ #

    def _on_model_reset(self) -> None:
        if self.dataset_model.rowCount() > 0:
            self._load_row(0)
        else:
            self._current_row = -1
            self._current_bgr = None
            self._current_ai_contours = []
            self._selected_ai_idx = -1
            self._active_class = ""
            self._review_bar.setVisible(False)
            self._ai_popup.setVisible(False)
            self.viewport_actions.set_image_loaded(False)
            self.viewport_actions.set_active_tool("")
            self.canvas.clear_image()
            self.right_panel.set_current_row(-1)
            self.status_bar.set_class("")
            self._set_start_screen_visible(True)

    def _load_row(self, row: int) -> None:
        bgr = self.io_controller.load_image_for_display(row)
        if bgr is None:
            return
        self._set_start_screen_visible(False)
        self._current_bgr = bgr
        self._current_row = row
        self._review_bar.set_decision(self.dataset_model.get_review_decision(row))
        self._review_bar.setVisible(True)
        self._review_bar.reposition(self.canvas.size())
        self.viewport_actions.set_image_loaded(True)
        self.viewport_actions.reposition(self.canvas.size())
        self._ai_popup.setVisible(False)
        self._selected_ai_idx = -1
        self.canvas.set_image(
            bgr
        )  # always set the original; resets zoom (expected on new image)
        self.status_bar.set_zoom(
            1.0
        )  # set_image resets zoom without emitting zoom_changed
        self._refresh_canvas_render()  # apply heatmap / overlay layer without resetting zoom
        total = self.dataset_model.rowCount()
        self.right_panel.set_counter(row, total)
        self.right_panel.select_row(row)
        self.right_panel.set_current_row(row)

    def _prev_image(self) -> None:
        row = self.right_panel.navigator_adjacent_source_row(self._current_row, -1)
        if row >= 0:
            self._load_row(row)

    def _next_image(self) -> None:
        row = self.right_panel.navigator_adjacent_source_row(self._current_row, 1)
        if row >= 0:
            self._load_row(row)

    # ------------------------------------------------------------------ #
    # Tool slots
    # ------------------------------------------------------------------ #

    def _on_tool_selected(self, tool_name: str) -> None:
        if tool_name == "sam_bbox":
            self._active_tool = "sam_bbox"
            self.viewport_actions.set_active_tool("")
            self.canvas.set_tool(SAM_BBOX)
            self.status_bar.set_tool("sam_bbox")
            if not self._sam_loading:
                self._sam_loading = True
                variant = self.tool_palette.current_sam_variant()
                self._sam_controller.set_variant(variant)
                self.status_bar.set_sam_hint("Loading SAM model…")
                self._sam_controller.ensure_loaded_async()
            return

        if tool_name == "calibrate":
            self._active_tool = "calibrate"
            self.tool_palette.deselect_all()
            self.viewport_actions.set_active_tool("calibrate")
            self.canvas.set_tool(CALIBRATE)
            self.status_bar.set_tool("calibrate")
            return

        if tool_name == "measure":
            self._active_tool = "measure"
            self.tool_palette.deselect_all()
            self.viewport_actions.set_active_tool("measure")
            self.canvas.set_tool(MEASURE)
            self.status_bar.set_tool("measure")
            return

        self._active_tool = tool_name
        self.viewport_actions.set_active_tool("")
        self.canvas.set_tool("polygon" if tool_name == "polygon" else None)
        self.status_bar.set_tool(tool_name)

    def _on_tool_canceled(self) -> None:
        self.tool_palette.deselect_all()
        self.viewport_actions.set_active_tool("")
        self._active_tool = ""
        self.status_bar.set_tool("")
        self.status_bar.set_sam_hint("")

    def _on_calibration_points_placed(self, p1: tuple, p2: tuple) -> None:
        import math
        from views.annomate.calibration_dialog import CalibrationDialog
        from PySide6.QtWidgets import QDialog

        if self._calib_model is None:
            return
        pixel_dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        dlg = CalibrationDialog(pixel_dist, parent=self)
        if dlg.exec() == QDialog.Accepted:
            real_dist, unit = dlg.get_result()
            self._calib_model.set_calib_points(p1, p2)
            ok = self._calib_model.apply_calibration(real_dist, unit)
            if not ok:
                QMessageBox.warning(
                    self, "Calibration", "The two points are too close together."
                )
        self.canvas.set_tool(None)  # clears _pending_calib_pts, resets cursor
        self.tool_palette.deselect_all()
        self.viewport_actions.set_active_tool("")
        self._active_tool = ""
        self.status_bar.set_tool("")

    def _on_draw_attempted(self) -> None:
        """Guard against drawing without a valid class; cancels the tool if missing."""
        class_names = self.dataset_model.get_class_names()
        if not class_names:
            self.canvas.set_tool(None)
            self.tool_palette.deselect_all()
            self.viewport_actions.set_active_tool("")
            self._active_tool = ""
            self.status_bar.set_tool("")
            QMessageBox.warning(
                self, "No Classes Defined", "Add an annotation class before drawing."
            )
            return
        if not self._active_class or self._active_class not in class_names:
            self.canvas.set_tool(None)
            self.tool_palette.deselect_all()
            self.viewport_actions.set_active_tool("")
            self._active_tool = ""
            self.status_bar.set_tool("")
            QMessageBox.warning(
                self,
                "No Class Selected",
                "Select an annotation class in the panel before drawing.",
            )

    # ------------------------------------------------------------------ #
    # Annotation slots
    # ------------------------------------------------------------------ #

    def _set_active_class(self, name: str) -> None:
        self._active_class = name
        r, g, b = self.dataset_model.get_class_color(name)
        self.canvas.set_active_color(QColor(r, g, b))
        self.status_bar.set_class(name)

    def _on_polygon_finished(self, pts: list) -> None:
        if self._current_row < 0 or not pts:
            return
        class_names = self.dataset_model.get_class_names()
        if not class_names:
            return
        target = (
            self._active_class if self._active_class in class_names else class_names[0]
        )
        self.dataset_model.add_annotation(
            self._current_row, target, pts, self.canvas.line_thickness
        )
        self._refresh_canvas_render()

    def _on_polygon_edited(self, idx: int, pts: list) -> None:
        if self._current_row < 0 or self.canvas.is_dragging():
            return
        self.dataset_model.update_annotation_points(self._current_row, idx, pts)

    def _on_dataset_data_changed(self, top_left, bottom_right, *_) -> None:
        if self._current_row >= 0:
            if top_left.row() <= self._current_row <= bottom_right.row():
                self._review_bar.set_decision(
                    self.dataset_model.get_review_decision(self._current_row)
                )
            self._refresh_canvas_render()

    def _on_canvas_polygon_selected(self, idx: int) -> None:
        """Sync the right panel list and slider when a polygon is clicked on the canvas."""
        self.right_panel.annotations.select_annotation(idx)
        self._on_annotation_selected(idx)

    def _on_review_decision(self, decision) -> None:
        if self._current_row >= 0:
            self.dataset_model.set_review_decision(self._current_row, decision)
            if decision in ("accept", "reject"):
                self._next_image()

    def _on_annotation_selected(self, idx: int) -> None:
        """Apply selection to the canvas and sync the UI slider to match the polygon's thickness."""
        self.canvas.selected_polygon_idx = idx
        self.canvas.update()

        # Read the polygon's specific thickness and update the slider
        if idx != -1 and self._current_row >= 0:
            annos = self.dataset_model.get_annotations(self._current_row)
            if 0 <= idx < len(annos):
                thick = annos[idx].get("thickness", 2.0)

                # Block signals so setting the slider doesn't accidentally trigger a drawing update
                self.tool_palette.slider_thickness.blockSignals(True)
                self.tool_palette.slider_thickness.setValue(
                    int(thick * 4)
                )  # slider is 1-40
                self.tool_palette.lbl_thickness.setText(f"{thick:.2f} px")
                self.tool_palette.slider_thickness.blockSignals(False)

                self.canvas.set_line_thickness(thick)

    def _refresh_overlays(self) -> None:
        """Rebuild canvas overlays from annotations only (no AI polygons)."""
        current_sel = self.canvas.selected_polygon_idx
        annos = self.dataset_model.get_annotations(self._current_row)
        overlays = [
            (
                a["polygon"],
                QColor(*self.dataset_model.get_class_color(a["category_name"])),
                a.get("thickness", 2.0),
            )
            for a in annos
        ]
        self.canvas.selected_polygon_idx = -1
        self.canvas.set_overlays(overlays)
        self.canvas.clear_ai_overlays()
        if 0 <= current_sel < len(overlays):
            self.canvas.selected_polygon_idx = current_sel
        else:
            self.canvas.selected_polygon_idx = -1

    def _update_canvas_overlays(self, anno_overlays: list) -> None:
        """Push annotation overlays and current AI contours to the canvas."""
        self.canvas.set_overlays(anno_overlays)
        self.canvas.set_ai_overlays(self._current_ai_contours)

    def _on_thickness_changed(self, thickness: float) -> None:
        """Apply thickness based on current tool mode."""
        # Always update the canvas so future drawing uses this thickness
        self.canvas.set_line_thickness(thickness)

        # If we are NOT actively drawing, and a polygon is selected, mutate its data
        if self._active_tool != "polygon":
            idx = self.canvas.selected_polygon_idx
            if idx != -1 and self._current_row >= 0:
                self.dataset_model.update_annotation_thickness(
                    self._current_row, idx, thickness
                )

    # ------------------------------------------------------------------ #
    # Microsentry toggle & rendering
    # ------------------------------------------------------------------ #

    def set_saved_model_path(self, path: str) -> None:
        """Called by AppWindow after opening a project to record the saved model path."""
        self._saved_model_path = path

    # ------------------------------------------------------------------ #
    # Microsentry toggle & rendering
    # ------------------------------------------------------------------ #

    def _on_microsentry_toggled(self, checked: bool) -> None:
        self._microsentry_enabled = checked
        if checked:
            if (
                self.inference_controller is not None
                and self.inference_controller.has_model()
            ):
                self.right_panel.set_model_loaded(
                    self.inference_controller.get_model_name(),
                    self.inference_controller.get_model_path(),
                )
                self._start_pending_inference()
            self.right_panel.show_microsentry_section()
        else:
            self.right_panel.hide_microsentry_section()
        self.right_panel.navigator_set_microsentry_mode(checked)
        if checked:
            for row in range(self.dataset_model.rowCount()):
                path = self.dataset_model.get_image_path(row)
                score = self.inference_model.get_score(path)
                label = self.inference_model.get_label(path)
                if score is not None:
                    self.right_panel.navigator_set_inference(row, score, label)
        self._refresh_canvas_render()

    def _on_load_previous_model_requested(self) -> None:
        if self.inference_controller is None:
            return
        if not self._saved_model_path:
            QMessageBox.information(
                self,
                "Load Previous Model",
                "No model path found in the current project.\n"
                "Open a project that was saved with a model loaded, or use 'Load New'.",
            )
            return
        if not os.path.isfile(self._saved_model_path):
            QMessageBox.warning(
                self,
                "Load Previous Model",
                f"The saved model file no longer exists:\n{self._saved_model_path}\n\nUse 'Load New' to browse for it.",
            )
            return
        self._load_model_from_path(self._saved_model_path)

    def _on_load_model_requested(self) -> None:
        if self.inference_controller is None:
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load AI Model",
            os.getcwd(),
            "PyTorch Model (*.pt *.pth);;All Files (*)",
        )
        if not path:
            return
        self._load_model_from_path(path)

    def _load_model_from_path(self, path: str) -> None:
        self.status_bar.set_model_loading(True)
        QApplication.processEvents()
        try:
            name = self.inference_controller.load_model(path)
        except Exception as exc:
            self.status_bar.set_model_loading(False)
            QMessageBox.critical(self, "Load Model", f"Could not load model:\n{exc}")
            return
        self.status_bar.set_model_loading(False)
        # Clear score maps from any previous model so the new model re-processes
        # everything rather than reusing stale heatmaps.
        self.inference_model.clear()
        self._refresh_canvas_render()
        self.right_panel.set_model_loaded(name, path)
        self._start_pending_inference()

    def _start_pending_inference(self) -> None:
        if (
            self.inference_controller is None
            or not self.inference_controller.has_model()
        ):
            return
        total = self.dataset_model.rowCount()
        paths = [
            self.dataset_model.get_image_path(i)
            for i in range(total)
            if not self.inference_model.is_processed(
                self.dataset_model.get_image_path(i)
            )
        ]
        if paths:
            self.inference_controller.start_batch_inference(paths)

    def _refresh_canvas_render(self) -> None:
        """Update heatmap layer and polygon overlays without resetting zoom or pan.

        Never calls canvas.set_image — that lives exclusively in _load_row so
        that zoom only resets when the user navigates to a new image.
        """
        if self._current_row < 0 or self._current_bgr is None:
            return

        ms_active = (
            self._microsentry_enabled
            and self.inference_controller is not None
            and self.inference_controller.has_model()
        )

        if not ms_active:
            self.canvas.clear_heatmap_layer()
            self._refresh_overlays()
            return

        path = self.dataset_model.get_image_path(self._current_row)
        if not self.inference_model.is_processed(path):
            self.canvas.clear_heatmap_layer()
            self._refresh_overlays()
            return

        ms = self.right_panel.get_microsentry_settings()
        score_map = self.inference_model.get_score_map(path)
        s = score_map.astype(np.float32)

        # Heatmap layer — drawn as a semi-transparent QPixmap over the original
        if ms["heatmap_enabled"]:
            self.canvas.set_heatmap_layer(s, ms["alpha"], ms["heat_min"])
        else:
            self.canvas.clear_heatmap_layer()

        # Annotation overlays are always shown
        annos = self.dataset_model.get_annotations(self._current_row)
        anno_overlays = [
            (
                a["polygon"],
                QColor(*self.dataset_model.get_class_color(a["category_name"])),
                a.get("thickness", 2.0),
            )
            for a in annos
        ]

        # AI segmentation overlays — computed fresh, stored for per-polygon accept/reject
        if ms["seg_enabled"]:
            orig_h, orig_w = self._current_bgr.shape[:2]
            self._current_ai_contours = self.inference_controller.compute_segmentation(
                s, ms["seg_pct"], ms["epsilon"], orig_w, orig_h
            )
        else:
            self._current_ai_contours = []

        self._ai_popup.setVisible(False)
        self._selected_ai_idx = -1
        self.canvas.selected_polygon_idx = -1
        self._update_canvas_overlays(anno_overlays)

    # ------------------------------------------------------------------ #
    # Inference signal slots
    # ------------------------------------------------------------------ #

    def _on_inference_result(self, path: str, score: float, score_map) -> None:
        self.inference_model.set_score_map(path, score, score_map)
        row = self._row_for_path(path)
        if row >= 0:
            label = self.inference_model.get_label(path)
            self.right_panel.navigator_set_inference(row, score, label)
        if row == self._current_row and self._microsentry_enabled:
            self._refresh_canvas_render()

    def _on_inference_progress(self, done: int) -> None:
        total = self.dataset_model.rowCount()
        self.status_bar.set_inference_progress(done, total)

    def _on_inference_batch_done(self) -> None:
        self.status_bar.clear_inference_progress()

    def _on_ai_polygon_clicked(self, idx: int, view_pos: QPointF) -> None:
        self._selected_ai_idx = idx
        if idx == -1:
            self._ai_popup.setVisible(False)
            return
        class_names = self.dataset_model.get_class_names()
        if not class_names:
            self._ai_popup.setVisible(False)
            return
        self._ai_popup.set_classes(class_names, self._active_class)
        bbox = self.canvas.get_ai_polygon_view_rect(idx)
        self._ai_popup.show_at_polygon(bbox)

    def _on_accept_single_ai(self) -> None:
        idx = self._selected_ai_idx
        if idx < 0 or idx >= len(self._current_ai_contours):
            return
        pts = self._current_ai_contours[idx]
        if len(pts) >= 3:
            target = self._ai_popup.current_class()
            if not target:
                class_names = self.dataset_model.get_class_names()
                target = (
                    self._active_class
                    if self._active_class in class_names
                    else (class_names[0] if class_names else "")
                )
            if target:
                self.dataset_model.add_annotation(self._current_row, target, pts)
        del self._current_ai_contours[idx]
        self._selected_ai_idx = -1
        self._ai_popup.setVisible(False)
        self._push_overlays_after_edit()

    def _push_overlays_after_edit(self) -> None:
        """Refresh canvas overlays from current annotations + remaining AI contours."""
        if self._current_row < 0:
            return
        annos = self.dataset_model.get_annotations(self._current_row)
        anno_overlays = [
            (
                a["polygon"],
                QColor(*self.dataset_model.get_class_color(a["category_name"])),
                a.get("thickness", 2.0),
            )
            for a in annos
        ]
        self._update_canvas_overlays(anno_overlays)

    def _on_accept_ai_polygons(self) -> None:
        """Add all current AI contours as annotations on the active class."""
        if self._current_row < 0 or not self._current_ai_contours:
            return
        class_names = self.dataset_model.get_class_names()
        if not class_names:
            return
        target = (
            self._active_class if self._active_class in class_names else class_names[0]
        )
        for pts in self._current_ai_contours:
            if len(pts) >= 3:
                self.dataset_model.add_annotation(self._current_row, target, pts)
        self._current_ai_contours = []
        self._selected_ai_idx = -1
        self._ai_popup.setVisible(False)
        self._refresh_canvas_render()

    def _row_for_path(self, path: str) -> int:
        for i in range(self.dataset_model.rowCount()):
            if self.dataset_model.get_image_path(i) == path:
                return i
        return -1

    # ------------------------------------------------------------------ #
    # Hotkeys
    # ------------------------------------------------------------------ #

    def keyPressEvent(self, event) -> None:
        """Handle annotation hotkeys.

        - ``P``: toggle polygon tool
        - ``S``: toggle SAM segment tool
        - ``C``: toggle calibration tool
        - ``M``: toggle measure tool
        - ``Delete``: delete the selected annotation

        Args:
            event: The key press event.
        """
        if event.key() == Qt.Key_P:
            self.tool_palette.toggle_polygon()
        elif event.key() == Qt.Key_S:
            self.tool_palette.toggle_sam()
        elif event.key() == Qt.Key_C:
            self.viewport_actions.toggle_calibrate()
        elif event.key() == Qt.Key_M:
            self.viewport_actions.toggle_measure()
        elif event.key() == Qt.Key_Delete:
            self._delete_selected_annotation()
        super().keyPressEvent(event)

    def _delete_selected_annotation(self) -> None:
        idx = self.canvas.selected_polygon_idx
        if idx != -1 and self._current_row >= 0:
            self.dataset_model.delete_annotation(self._current_row, idx)

    # ------------------------------------------------------------------ #
    # SAM tool slots
    # ------------------------------------------------------------------ #

    def _on_sam_variant_changed(self, variant: str) -> None:
        self._sam_controller.set_variant(variant)
        self._sam_loading = False
        self.tool_palette.sam_status_lbl.setText("Model: not loaded")
        self.tool_palette.sam_status_lbl.setStyleSheet(
            "color: grey; font-style: italic;"
        )

    def _on_sam_loading_done(self) -> None:
        self._sam_loading = False
        display_name = self.tool_palette.sam_variant_combo.currentText()
        self.tool_palette.sam_status_lbl.setText(f"Ready: {display_name}")
        self.tool_palette.sam_status_lbl.setStyleSheet(
            "color: green; font-style: normal;"
        )
        self.status_bar.set_sam_hint(f"Ready: {display_name}  ·  draw bbox to segment")

    def _on_sam_loading_failed(self, msg: str) -> None:
        self._sam_loading = False
        self.tool_palette.deselect_all()
        self._active_tool = ""
        self.canvas.set_tool(None)
        self.status_bar.set_tool("")
        self.status_bar.set_sam_hint("")
        self.tool_palette.sam_status_lbl.setText("Load failed")
        self.tool_palette.sam_status_lbl.setStyleSheet(
            "color: red; font-style: normal;"
        )
        QMessageBox.critical(self, "SAM Load Error", f"Could not load model:\n{msg}")

    def _on_sam_bbox_drawn(self, x1: float, y1: float, x2: float, y2: float) -> None:
        if self._current_row < 0 or self._current_bgr is None:
            return
        self.canvas.setCursor(Qt.WaitCursor)
        self.status_bar.set_sam_hint("Running SAM…")
        self._sam_controller.run_inference(self._current_bgr, (x1, y1, x2, y2))

    def _on_sam_result_ready(self, pts: list, confidence: float) -> None:
        self.canvas.setCursor(Qt.CrossCursor)
        if not pts:
            self.status_bar.set_sam_hint("No mask found — try a larger bbox")
            return
        self.canvas.set_sam_ghost(pts, confidence)
        self.status_bar.set_sam_hint(
            f"conf={confidence:.2f}  ·  Enter=accept  ·  Esc=cancel"
        )

    def _on_sam_inference_failed(self, msg: str) -> None:
        self.canvas.setCursor(Qt.CrossCursor)
        self.status_bar.set_sam_hint(f"Inference error — {msg}")
