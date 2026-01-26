"""
Main Window Module for AnnoMate.

This module defines the `ImageAnnotator` class, which serves as the primary
GUI for the annotation tool. It handles image loading, polygon annotation,
class management, and synchronization with external tools (MicroSentry).
"""

import os
import csv
import json
import traceback
from typing import Optional, List, Tuple, Dict, Any, Set
from pathlib import Path
from datetime import datetime

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import (
    QMainWindow,
    QWidget,
    QFileDialog,
    QMessageBox,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QListWidget,
    QListWidgetItem,
    QComboBox,
    QLineEdit,
    QTextEdit,
    QTableWidgetItem,
    QAbstractItemView,
    QScrollArea,
    QColorDialog,
    QHeaderView,
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QBrush, QIcon, QPixmap

# --- Local Imports ---
# Using a try-except block to handle both package-relative and standalone execution
try:
    from .image_label import ImageLabel, POLYGON
    from .utils import polygon_area
    from .constants import DEFAULT_CLASS_COLORS
    from .styles import MAIN_STYLESHEET, SPLITTER_STYLE
    from .widgets import CustomSplitter, WrappingTableWidget
except ImportError:
    from image_label import ImageLabel, POLYGON
    from utils import polygon_area
    from constants import DEFAULT_CLASS_COLORS
    from styles import MAIN_STYLESHEET, SPLITTER_STYLE
    from widgets import CustomSplitter, WrappingTableWidget


class ImageAnnotator(QMainWindow):
    """
    The main application window for the AnnoMate annotation tool.

    Signals:
        viewChanged (float, float, float): Emitted when zoom/pan changes (rx, ry, scale).
        folderLoaded (str, list): Emitted when a new folder is loaded (path, file_list).
    """

    # Signals for external sync
    viewChanged = pyqtSignal(float, float, float)
    folderLoaded = pyqtSignal(str, list)

    def __init__(self):
        """Initialize the main window, UI components, and internal state."""
        super().__init__()
        self.setWindowTitle("AnnoMate")
        self.setStyleSheet(MAIN_STYLESHEET)
        self._set_window_icon()

        # --- Dataset State ---
        self.image_dir: Optional[str] = None
        self.image_files: List[str] = []
        self.current_idx: int = -1

        # --- Class/Annotation State ---
        self.class_names: List[str] = []
        self.class_colors: Dict[str, QColor] = {}
        self._next_color_idx: int = 0

        # --- Per-Image Metadata ---
        self.annotations: Dict[str, List[Dict[str, Any]]] = {}
        self.inspectors: Dict[str, str] = {}
        self.notes: Dict[str, str] = {}
        self._global_inspector: str = ""
        self._sized_once: bool = False

        # --- UI Initialization ---
        self._init_ui()

    def _set_window_icon(self):
        """Attempts to locate and set the window icon."""
        here = os.path.dirname(__file__)
        project_root = os.path.abspath(os.path.join(here, "..", ".."))
        self._icon_file = None
        
        for p in ("logos/AnnoMate.png", "logos/AnnoMate.ico"):
            fp = os.path.join(project_root, p)
            if os.path.exists(fp):
                self._icon_file = fp
                self.setWindowIcon(QIcon(fp))
                break

    def _init_ui(self):
        """Orchestrates the creation of the user interface."""
        # Main Layout using CustomSplitter
        self._splitter = CustomSplitter(Qt.Horizontal, self)
        self._splitter.setHandleWidth(3)
        self._splitter.setStyleSheet(SPLITTER_STYLE)
        self.setCentralWidget(self._splitter)

        # 1. Canvas Area (Left)
        self._setup_canvas()

        # 2. Control Area (Right)
        self._setup_sidebar()

        # Splitter Configuration
        self._splitter.setCollapsible(0, False)
        self._splitter.setCollapsible(1, True)
        self._splitter.setStretchFactor(0, 2)
        self._splitter.setStretchFactor(1, 1)

        # Default Window Size
        self.resize(1400, 900)

    def _setup_canvas(self):
        """Initializes the image display canvas."""
        self.canvas = ImageLabel()
        self.canvas.set_main_window(self)
        self._splitter.addWidget(self.canvas)

        # Hook scrollbars for view sync
        if hasattr(self.canvas, "horizontalScrollBar"):
            try:
                self.canvas.horizontalScrollBar().valueChanged.connect(self._emit_view_sync)
                self.canvas.verticalScrollBar().valueChanged.connect(self._emit_view_sync)
            except Exception:
                pass

    def _setup_sidebar(self):
        """Initializes the right-hand sidebar with controls."""
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(380)
        self._splitter.addWidget(scroll)

        side_widget = QWidget()
        scroll.setWidget(side_widget)
        
        self.side_layout = QVBoxLayout(side_widget)
        self.side_layout.setContentsMargins(10, 10, 10, 10)
        self.side_layout.setSpacing(10)

        # Add UI Sections
        self._create_tray_header()
        self._create_nav_controls()
        self._create_zoom_controls()
        self._create_class_controls()
        self._create_meta_inputs()
        self._create_dataset_table()
        self._create_annotation_list()
        self._create_ops_controls()
        self._create_export_controls()

    def _create_tray_header(self):
        """Creates the header section with logo and folder open button."""
        row = QHBoxLayout()
        
        self.logo = QLabel()
        if self._icon_file:
            pm = QPixmap(self._icon_file)
            if not pm.isNull():
                self.logo.setPixmap(pm.scaled(75, 75, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.logo.setMinimumSize(75, 75)
        
        row.addWidget(self.logo)
        row.addWidget(QLabel("Tray:"))
        
        self.lbl_dir = QLabel("—")
        self.lbl_dir.setStyleSheet("font-weight: bold;")
        
        self.btn_open = QPushButton("Open Folder")
        self.btn_open.clicked.connect(self.open_folder_dialog)
        
        row.addWidget(self.lbl_dir, 1)
        row.addWidget(self.btn_open)
        self.side_layout.addLayout(row)

    def _create_nav_controls(self):
        """Creates previous/next navigation buttons."""
        nav_layout = QHBoxLayout()
        
        self.btn_prev = QPushButton("Prev")
        self.btn_prev.clicked.connect(lambda: self.goto_index(self.current_idx - 1))
        
        self.btn_next = QPushButton("Next")
        self.btn_next.clicked.connect(lambda: self.goto_index(self.current_idx + 1))
        
        self.lbl_img = QLabel("0 / 0")
        self.lbl_img.setAlignment(Qt.AlignCenter)
        
        nav_layout.addWidget(self.btn_prev)
        nav_layout.addWidget(self.btn_next)
        nav_layout.addWidget(self.lbl_img)
        self.side_layout.addLayout(nav_layout)

    def _create_zoom_controls(self):
        """Creates zoom, reset, and tool selection buttons."""
        bar = QHBoxLayout()
        
        self.btn_zoom_in = QPushButton("Zoom +")
        self.btn_zoom_in.clicked.connect(self.zoom_in_sync)
        
        self.btn_zoom_out = QPushButton("Zoom -")
        self.btn_zoom_out.clicked.connect(self.zoom_out_sync)
        
        self.btn_reset = QPushButton("Reset View")
        self.btn_reset.clicked.connect(self.reset_view_sync)
        
        self.btn_poly = QPushButton("Polygon")
        self.btn_poly.setCheckable(True)
        self.btn_poly.clicked.connect(lambda: self._set_tool_from_button(POLYGON, self.btn_poly))

        for w in (self.btn_zoom_in, self.btn_zoom_out, self.btn_reset, self.btn_poly):
            bar.addWidget(w)
        self.side_layout.addLayout(bar)

    def _create_class_controls(self):
        """Creates the class management section (Add, Color, Delete)."""
        class_layout = QHBoxLayout()
        
        self.class_combo = QComboBox()
        self.class_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.class_combo.currentTextChanged.connect(self.on_class_changed)
        
        self.class_name_edit = QLineEdit()
        self.class_name_edit.setPlaceholderText("Enter class...")
        
        self.btn_add_class = QPushButton("Add Class")
        self.btn_add_class.clicked.connect(self.add_class_from_edit)
        
        self.btn_color = QPushButton("Change Color")
        self.btn_color.clicked.connect(self.change_class_color)
        
        self.btn_del_class = QPushButton("Delete Class")
        self.btn_del_class.clicked.connect(self.delete_current_class)
        
        class_layout.addWidget(self.class_combo, 2)
        class_layout.addWidget(self.class_name_edit, 2)
        class_layout.addWidget(self.btn_add_class, 0)
        class_layout.addWidget(self.btn_color, 0)
        class_layout.addWidget(self.btn_del_class, 0)
        self.side_layout.addLayout(class_layout)

    def _create_meta_inputs(self):
        """Creates inspector name and note input fields."""
        self.side_layout.addWidget(QLabel("Inspector"))
        
        self.inspector_edit = QLineEdit()
        self.inspector_edit.editingFinished.connect(self._store_inspector)
        self.side_layout.addWidget(self.inspector_edit)
        
        self.side_layout.addWidget(QLabel("Image note"))
        
        self.note_edit = QTextEdit()
        self.note_edit.textChanged.connect(self._store_note)
        self.side_layout.addWidget(self.note_edit, 1)

    def _create_dataset_table(self):
        """Creates the table widget showing file list and review status."""
        self.side_layout.addWidget(QLabel("Dataset"))
        
        self.table = WrappingTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Image Name", "Status"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        
        # Sync selection with main view
        self.table.currentCellChanged.connect(lambda r, c, pr, pc: self.goto_index(r))
        
        self.side_layout.addWidget(self.table, 1)

    def _create_annotation_list(self):
        """Creates the list widget for annotations on the current image."""
        self.side_layout.addWidget(QLabel("Annotations (current image)"))
        self.ann_list = QListWidget()
        self.side_layout.addWidget(self.ann_list, 1)

    def _create_ops_controls(self):
        """Creates operations buttons (Delete Selected, Sort)."""
        ops_layout = QHBoxLayout()
        
        self.btn_delete = QPushButton("Delete Selected")
        self.btn_delete.clicked.connect(self.delete_selected)
        
        self.btn_sort = QPushButton("Sort by Area")
        self.btn_sort.clicked.connect(self.sort_by_area)
        
        ops_layout.addWidget(self.btn_delete)
        ops_layout.addWidget(self.btn_sort)
        self.side_layout.addLayout(ops_layout)

    def _create_export_controls(self):
        """Creates export and import buttons."""
        exp_layout = QHBoxLayout()
        
        self.btn_export_polys = QPushButton("Export Polygons + Data")
        self.btn_export_polys.clicked.connect(self.export_polygons_and_data)
        
        self.btn_export_csv = QPushButton("Export CSV (metadata)")
        self.btn_export_csv.clicked.connect(self.export_csv)
        
        exp_layout.addWidget(self.btn_export_polys)
        exp_layout.addWidget(self.btn_export_csv)
        self.side_layout.addLayout(exp_layout)
        
        self.btn_import_data = QPushButton("Import Data JSON")
        self.btn_import_data.clicked.connect(self.import_data_json)
        self.side_layout.addWidget(self.btn_import_data)

    # --- Loading & Navigation ---

    def load_folder_programmatically(self, folder_path: str, files: List[str]):
        """
        Loads a specific folder and file list directly (used by MicroSentry sync).
        CRITICAL: Converts absolute paths (from sync) to simple filenames to avoid
        path duplication errors.
        
        Args:
            folder_path (str): The absolute path to the directory.
            files (List[str]): List of filenames or absolute paths.
        """
        if not folder_path or not files:
            return
        
        self.image_dir = folder_path
        
        # Ensure we only store filenames (e.g. "img.png"), not full paths.
        self.image_files = [Path(f).name for f in files]
        
        self.lbl_dir.setText(Path(folder_path).name)
        self._build_table()
        self.goto_index(0)

    def open_folder_dialog(self):
        """Opens a QFileDialog for the user to select an image directory."""
        directory = QFileDialog.getExistingDirectory(self, "Open image folder", os.getcwd())
        if not directory:
            return
        
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        files = [f for f in os.listdir(directory) if Path(f).suffix.lower() in exts]
        files = sorted(files)
        
        if not files:
            return
        
        # Load locally
        self.load_folder_programmatically(directory, files)
        
        # Emit signal to sync external tabs (sending files as is; adapter handles them)
        self.folderLoaded.emit(directory, files)

    def open_folder(self):
        """Backward compatibility wrapper for open_folder_dialog."""
        self.open_folder_dialog()

    def goto_index(self, idx: int):
        """
        Navigates to the specified image index.
        """
        if not self.image_files:
            return
        
        # Clamp index
        idx = max(0, min(len(self.image_files) - 1, idx))
        if idx == self.current_idx:
            return
        
        self.current_idx = idx
        self.lbl_img.setText(f"{idx + 1} / {len(self.image_files)}")
        
        # Construct path safely using pathlib
        # This handles OS-specific separators automatically
        full_path = Path(self.image_dir) / self.image_files[idx]
        
        try:
            # Explicit check to help debugging
            if not full_path.exists():
                raise FileNotFoundError(f"File does not exist: {full_path}")

            # FIX: Updated parameter name to match new ImageLabel.load_image() definition
            self.canvas.load_image(str(full_path), max_display_dim=1000)
            
        except Exception as e:
            # Shows error to user instead of failing silently
            QMessageBox.warning(self, "Load Error", f"Failed to load image:\n{full_path}\n\nError: {e}")
            return
            
        self.refresh_ann_list()
        self.refresh_overlays()
        self.refresh_meta_fields()
        
        # Sync inspector if global inspector is set
        if self._global_inspector and not self.inspectors.get(self.image_files[idx]):
            self.inspector_edit.blockSignals(True)
            self.inspector_edit.setText(self._global_inspector)
            self.inspector_edit.blockSignals(False)

        # Sync table selection (blocking signals to avoid recursion)
        self.table.blockSignals(True)
        self.table.selectRow(idx)
        self.table.blockSignals(False)

    # --- View Synchronization ---

    def zoom_in_sync(self):
        self.canvas.zoom_in()
        self._emit_view_sync()
    
    def zoom_out_sync(self):
        self.canvas.zoom_out()
        self._emit_view_sync()
        
    def reset_view_sync(self):
        self.canvas.reset_view()
        self._emit_view_sync()

    def _emit_view_sync(self):
        """Emits the viewChanged signal with current viewport state."""
        try:
            scale = getattr(self.canvas, 'scale', 1.0)
            if hasattr(self.canvas, "horizontalScrollBar"):
                h_bar = self.canvas.horizontalScrollBar()
                v_bar = self.canvas.verticalScrollBar()
                
                # Calculate relative center positions (0.0 to 1.0)
                h_max = h_bar.maximum() + h_bar.pageStep() or 1
                v_max = v_bar.maximum() + v_bar.pageStep() or 1
                
                rx = (h_bar.value() + h_bar.pageStep() / 2) / h_max
                ry = (v_bar.value() + v_bar.pageStep() / 2) / v_max
                
                self.viewChanged.emit(rx, ry, scale)
        except Exception:
            pass

    def set_view_state(self, rx: float, ry: float, scale: float):
        """
        Updates the view based on external signal (e.g. from MicroSentry).
        """
        try:
            if hasattr(self.canvas, "horizontalScrollBar"):
                h_bar = self.canvas.horizontalScrollBar()
                v_bar = self.canvas.verticalScrollBar()
                
                h_max = h_bar.maximum() + h_bar.pageStep()
                v_max = v_bar.maximum() + v_bar.pageStep()
                
                h_target = int(rx * h_max - h_bar.pageStep() / 2)
                v_target = int(ry * v_max - v_bar.pageStep() / 2)
                
                h_bar.setValue(h_target)
                v_bar.setValue(v_target)
        except Exception:
            pass

    # --- Annotation Logic ---

    def finish_polygon(self, points: List[Tuple[float, float]]):
        """Callback when a polygon is completed on the canvas."""
        if self.current_idx < 0 or not self.image_files:
            return
        
        current_class = self.class_combo.currentText()
        if not current_class:
            QMessageBox.warning(self, "No class", "Please create/select a class first.")
            return
        
        self.add_polygon_external(points, current_class)

    def add_polygon_external(self, points: List[Tuple[float, float]], class_name: str, color: Optional[QColor] = None):
        """
        Adds a polygon to the data structure and updates the UI.
        """
        if self.current_idx < 0 or not self.image_files:
            return
        
        # Register class if new
        if class_name not in self.class_names:
            self.class_names.append(class_name)
            if color and color.isValid():
                self.class_colors[class_name] = color
            else:
                self.class_colors[class_name] = self._pick_next_unique_color()
            
            self.class_combo.blockSignals(True)
            self.class_combo.addItem(class_name)
            self.class_combo.blockSignals(False)

        # Sync combo box
        if self.class_combo.currentText() != class_name:
            self.class_combo.blockSignals(True)
            self.class_combo.setCurrentText(class_name)
            self.class_combo.blockSignals(False)

        # Add Data
        img_name = self.image_files[self.current_idx]
        self.annotations.setdefault(img_name, []).append(
            {"category_name": class_name, "polygon": points}
        )
        
        # Propagate inspector
        if self._global_inspector and not self.inspectors.get(img_name):
            self.inspectors[img_name] = self._global_inspector

        self.refresh_ann_list()
        self.refresh_overlays()
        self._update_table_row(self.current_idx)

    # --- Class Management ---

    def on_class_changed(self, name: str):
        """Updates the active color when the class dropdown changes."""
        color = self.class_colors.get(name, QColor(0, 200, 0))
        self.canvas.set_active_color(color)

    def add_class_from_edit(self):
        """Adds a new class from the text input."""
        name = self.class_name_edit.text().strip()
        if not name or name in self.class_names:
            return
        
        self.class_names.append(name)
        self.class_colors[name] = self._pick_next_unique_color()
        self.class_combo.addItem(name)
        self.class_combo.setCurrentText(name)
        self.class_name_edit.clear()
        
        self.btn_poly.setChecked(True)
        self.canvas.set_tool(POLYGON)
        self.on_class_changed(name)

    def change_class_color(self):
        """Opens a dialog to change the color of the current class."""
        name = self.class_combo.currentText().strip()
        if not name:
            return
        
        current = self.class_colors.get(name, QColor(0, 200, 0))
        col = QColorDialog.getColor(current, self)
        
        if col.isValid():
            self.class_colors[name] = col
            self.refresh_overlays()
            self.on_class_changed(name)

    def delete_current_class(self):
        """Deletes the currently selected class and removes associated annotations."""
        name = self.class_combo.currentText().strip()
        if not name:
            return
        
        self.class_names.remove(name)
        self.class_colors.pop(name, None)
        
        # Remove annotations of this class from all images
        for img, ann_list in list(self.annotations.items()):
            self.annotations[img] = [a for a in ann_list if a.get("category_name") != name]
        
        # Update UI
        idx = self.class_combo.currentIndex()
        if idx >= 0:
            self.class_combo.removeItem(idx)
        
        if self.class_combo.count() > 0:
            self.class_combo.setCurrentIndex(0)
            self.on_class_changed(self.class_combo.currentText())
        else:
            self.canvas.set_active_color(QColor(0, 200, 0))
        
        if self.current_idx >= 0:
            self.refresh_ann_list()
            self.refresh_overlays()

    # --- Helper Utilities ---

    def _set_tool_from_button(self, tool_name: str, btn: QPushButton):
        if btn.isChecked():
            if tool_name == POLYGON:
                self.btn_poly.setChecked(True)
            self.canvas.set_tool(tool_name)
        else:
            self.canvas.set_tool(None)

    def set_tool(self, tool_name: Optional[str]):
        """Sets the active tool on the canvas programmatically."""
        self.btn_poly.setChecked(tool_name == POLYGON)
        self.canvas.set_tool(tool_name)

    def _pick_next_unique_color(self) -> QColor:
        """Selects a color that hasn't been used recently."""
        used = {
            (c.red(), c.green(), c.blue()) for c in self.class_colors.values()
        }
        n = len(DEFAULT_CLASS_COLORS)
        
        for off in range(n):
            cand = DEFAULT_CLASS_COLORS[(self._next_color_idx + off) % n]
            if (cand.red(), cand.green(), cand.blue()) not in used:
                self._next_color_idx = (self._next_color_idx + off + 1) % n
                return cand
        
        # Fallback if all colors used
        c = DEFAULT_CLASS_COLORS[self._next_color_idx % n]
        self._next_color_idx = (self._next_color_idx + 1) % n
        return c

    # --- Table / Data Management ---

    def _is_reviewed(self, img_name: str) -> bool:
        """Determines if an image has been reviewed (has annotation, inspector, or note)."""
        has_anno = bool(self.annotations.get(img_name))
        has_meta = bool(self.inspectors.get(img_name) or self.notes.get(img_name))
        return has_anno or has_meta

    def _build_table(self):
        """Rebuilds the entire file table."""
        n = len(self.image_files)
        self.table.setRowCount(n)
        for i in range(n):
            self._populate_table_row(i)
        self.table.resizeRowsToContents()

    def _populate_table_row(self, i: int):
        filename = self.image_files[i]
        stem = Path(filename).stem
        
        idx_item = QTableWidgetItem(stem)
        idx_item.setTextAlignment(Qt.AlignCenter)
        idx_item.setFlags(idx_item.flags() ^ Qt.ItemIsEditable)
        
        status_item = QTableWidgetItem(self._status_text(i))
        status_item.setFlags(status_item.flags() ^ Qt.ItemIsEditable)
        status_item.setBackground(self._status_brush(i))
        
        self.table.setItem(i, 0, idx_item)
        self.table.setItem(i, 1, status_item)

    def _update_table_row(self, i: int):
        """Updates the status column for a specific row."""
        if 0 <= i < self.table.rowCount():
            self.table.item(i, 1).setText(self._status_text(i))
            self.table.item(i, 1).setBackground(self._status_brush(i))

    def _status_text(self, i: int) -> str:
        return "Reviewed" if self._is_reviewed(self.image_files[i]) else "Not reviewed"

    def _status_brush(self, i: int) -> QBrush:
        """Returns green background for reviewed, beige for unreviewed."""
        is_rev = self._is_reviewed(self.image_files[i])
        return QBrush(QColor(210, 245, 210) if is_rev else QColor(255, 235, 210))

    def _store_inspector(self):
        if self.current_idx < 0:
            return
        img = self.image_files[self.current_idx]
        val = self.inspector_edit.text().strip()
        self.inspectors[img] = val
        self._global_inspector = val
        self._update_table_row(self.current_idx)

    def _store_note(self):
        if self.current_idx < 0:
            return
        img = self.image_files[self.current_idx]
        self.notes[img] = self.note_edit.toPlainText().strip()
        self._update_table_row(self.current_idx)

    # --- Selection / Sort Operations ---

    def selected_indices(self) -> List[int]:
        return [i.row() for i in self.ann_list.selectedIndexes()]

    def delete_selected(self):
        """Deletes selected annotations from the current image."""
        idxs = sorted(self.selected_indices())
        if not idxs:
            return
        
        img = self.image_files[self.current_idx]
        lst = self.annotations.get(img, [])
        
        for i in reversed(idxs):
            if 0 <= i < len(lst):
                lst.pop(i)
        
        self.refresh_ann_list()
        self.refresh_overlays()
        self._update_table_row(self.current_idx)

    def sort_by_area(self):
        """Sorts annotations on the current image by polygon area (descending)."""
        if self.current_idx < 0:
            return
        
        img = self.image_files[self.current_idx]
        lst = self.annotations.get(img, [])
        lst.sort(key=lambda a: polygon_area(a["polygon"]), reverse=True)
        
        self.refresh_ann_list()
        self.refresh_overlays()

    def refresh_ann_list(self):
        """Refreshes the sidebar list widget."""
        self.ann_list.clear()
        if self.current_idx < 0:
            return
        
        img = self.image_files[self.current_idx]
        for a in self.annotations.get(img, []):
            it = QListWidgetItem(f"{a['category_name']} - {len(a['polygon'])} pts")
            it.setData(Qt.UserRole, a)
            self.ann_list.addItem(it)

    def refresh_overlays(self):
        """Pushes current annotations to the canvas for drawing."""
        if self.current_idx < 0:
            self.canvas.set_overlays([])
            return
        
        img = self.image_files[self.current_idx]
        overlays = []
        for a in self.annotations.get(img, []):
            color = self.class_colors.get(a["category_name"], QColor(255, 255, 255))
            overlays.append((a["polygon"], color))
        self.canvas.set_overlays(overlays)

    def refresh_meta_fields(self):
        """Updates the inspector and note fields from stored data."""
        if self.current_idx < 0:
            self.inspector_edit.setText("")
            self.note_edit.setPlainText("")
            return
        
        img = self.image_files[self.current_idx]
        self.inspector_edit.blockSignals(True)
        self.note_edit.blockSignals(True)
        
        self.inspector_edit.setText(self.inspectors.get(img, ""))
        self.note_edit.setPlainText(self.notes.get(img, ""))
        
        self.inspector_edit.blockSignals(False)
        self.note_edit.blockSignals(False)
        self._update_table_row(self.current_idx)

    # --- Export / Import ---

    def export_polygons_and_data(self):
        """Exports annotations as overlay images and a JSON data file."""
        if not self.image_files:
            return
        
        out_dir = QFileDialog.getExistingDirectory(self, "Choose output folder", os.getcwd())
        if not out_dir:
            return
        
        out_dir = Path(out_dir)
        tray_name = Path(self.image_dir).name if self.image_dir else "tray"
        timestamp = datetime.now().strftime("%m-%d-%y-%H-%M-%S")
        
        # Prepare Data Payload
        payload = {
            "meta": {"tray": tray_name, "exported_at": timestamp},
            "classes": list(self.class_names),
            "class_colors": {
                name: (c.red(), c.green(), c.blue())
                for name, c in self.class_colors.items()
            },
            "images": {}
        }
        
        from PIL import Image, ImageDraw  # Local import to avoid global dependency if unused
        
        saved_count = 0
        for name in self.image_files:
            anns = self.annotations.get(name, [])
            is_reviewed = self._is_reviewed(name)
            
            # Serialize metadata
            payload["images"][name] = {
                "inspector": self.inspectors.get(name, "") if is_reviewed else "",
                "note": self.notes.get(name, "") if is_reviewed else "",
                "annotations": [
                    {
                        "class": a["category_name"],
                        "polygon": [(float(x), float(y)) for (x, y) in a["polygon"]]
                    }
                    for a in anns
                ]
            }
            
            if not anns:
                continue
            
            # Generate Overlay Image
            src = Path(self.image_dir) / name
            if not src.exists():
                continue
            
            try:
                base = Image.open(src).convert("RGBA")
                overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
                draw = ImageDraw.Draw(overlay, "RGBA")
                
                for a in anns:
                    pts = [(float(x), float(y)) for (x, y) in a["polygon"]]
                    if len(pts) < 2:
                        continue
                    
                    qcolor = self.class_colors.get(a["category_name"], QColor(255, 255, 255))
                    rgb = (qcolor.red(), qcolor.green(), qcolor.blue())
                    
                    # Fill with semi-transparent; Outline with opaque
                    draw.polygon(pts, fill=(*rgb, 80), outline=(*rgb, 255))
                    draw.line(pts + [pts[0]], fill=(*rgb, 255), width=3)
                
                composed = Image.alpha_composite(base, overlay).convert("RGB")
                out_name = f"{tray_name}_{Path(name).stem}_{timestamp}_poly.jpg"
                composed.save(out_dir / out_name, "JPEG", quality=95)
                saved_count += 1
            except Exception:
                continue
            
        data_path = out_dir / f"{tray_name}_{timestamp}_data.json"
        with open(data_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            
        QMessageBox.information(
            self, "Export", f"Saved {saved_count} image(s) + data JSON:\n{data_path}"
        )

    def export_csv(self):
        """Exports dataset metadata (review status, inspector, notes) to CSV."""
        if not self.image_files:
            return
        
        out_path, _ = QFileDialog.getSaveFileName(self, "Save CSV", "metadata.csv", "CSV (*.csv)")
        if not out_path:
            return
        
        tray_name = Path(self.image_dir).name if self.image_dir else ""
        rows = []
        
        for name in self.image_files:
            annos = self.annotations.get(name, [])
            unique_classes = sorted({a["category_name"] for a in annos})
            reviewed = self._is_reviewed(name)
            
            rows.append({
                "tray": tray_name,
                "image_name": name,
                "inspector": self.inspectors.get(name, "") if reviewed else "",
                "note": self.notes.get(name, "") if reviewed else "",
                "classes": (", ".join(unique_classes) if unique_classes else "good") if reviewed else ""
            })
        
        try:
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["tray", "image_name", "inspector", "note", "classes"])
                writer.writeheader()
                writer.writerows(rows)
            QMessageBox.information(self, "Export", f"CSV saved to:\n{out_path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def import_data_json(self):
        """Imports annotations from a custom JSON or COCO-format JSON."""
        path, _ = QFileDialog.getOpenFileName(self, "Open Data JSON", "", "JSON (*.json)")
        if not path:
            return
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to read JSON file:\n{e}")
            return

        try:
            # Clear existing data
            self.annotations.clear()
            self.inspectors.clear()
            self.notes.clear()
            
            images_node = data.get("images")
            
            if isinstance(images_node, dict):
                self._import_custom_format(data, images_node)
            elif isinstance(images_node, list):
                self._import_coco_format(data, images_node)

            # Refresh UI
            if self.image_dir and self.image_files:
                self._build_table()
                if self.current_idx >= 0:
                    self.refresh_ann_list()
                    self.refresh_overlays()
                    self.refresh_meta_fields()
                    
            QMessageBox.information(self, "Import", "Data JSON imported successfully.")
            
        except Exception as e:
            QMessageBox.critical(self, "Import Crash", f"An error occurred during import:\n{e}")

    def _import_custom_format(self, data: dict, images_node: dict):
        """Helper to parse the internal/custom JSON format."""
        # Restore classes
        classes = data.get("classes", [])
        if classes:
            self.class_names = list(classes)
            saved_colors = data.get("class_colors", {})
            self.class_colors = {}
            for i, name in enumerate(self.class_names):
                tup = saved_colors.get(name)
                if isinstance(tup, (list, tuple)) and len(tup) == 3:
                    self.class_colors[name] = QColor(int(tup[0]), int(tup[1]), int(tup[2]))
                else:
                    self.class_colors[name] = DEFAULT_CLASS_COLORS[i % len(DEFAULT_CLASS_COLORS)]
            self._update_class_combo()

        # Restore annotations
        for name, info in images_node.items():
            self.inspectors[name] = info.get("inspector", "")
            self.notes[name] = info.get("note", "")
            
            recs = []
            for a in info.get("annotations", []):
                recs.append({
                    "category_name": a.get("class", ""),
                    "polygon": a.get("polygon", [])
                })
            
            if recs:
                self.annotations[name] = recs

    def _import_coco_format(self, data: dict, images_node: list):
        """Helper to parse COCO standard JSON format."""
        # 1. Map Categories
        cat_map = {}  # id -> name
        if "categories" in data:
            for c in data["categories"]:
                name = c["name"]
                cat_map[c["id"]] = name
                if name not in self.class_names:
                    self.class_names.append(name)
                    self.class_colors[name] = self._pick_next_unique_color()
            self._update_class_combo()
        
        # 2. Map Image IDs to Filenames
        img_id_map = {img["id"]: img["file_name"] for img in images_node}
        
        # 3. Parse Annotations
        if "annotations" in data:
            for ann in data["annotations"]:
                img_id = ann["image_id"]
                if img_id not in img_id_map:
                    continue
                
                filename = img_id_map[img_id]
                cat_name = cat_map.get(ann["category_id"], "Unknown")
                
                # Convert COCO segmentation to Polygon
                seg = ann.get("segmentation", [])
                final_poly = []
                
                if isinstance(seg, list) and len(seg) > 0:
                    # Handle nested list [[x,y,x,y...]] or flat [x,y,x,y...]
                    pts_list = seg[0] if isinstance(seg[0], list) else seg
                    
                    # Convert [x1, y1, x2, y2] -> [(x1,y1), (x2,y2)]
                    for i in range(0, len(pts_list), 2):
                        if i + 1 < len(pts_list):
                            final_poly.append((float(pts_list[i]), float(pts_list[i + 1])))
                
                if final_poly:
                    self.annotations.setdefault(filename, []).append({
                        "category_name": cat_name,
                        "polygon": final_poly
                    })

    def _update_class_combo(self):
        """Updates the dropdown with the current class list."""
        self.class_combo.blockSignals(True)
        self.class_combo.clear()
        self.class_combo.addItems(self.class_names)
        self.class_combo.blockSignals(False)
        self._next_color_idx = len(self.class_names) % len(DEFAULT_CLASS_COLORS)

    def showEvent(self, ev):
        """Override to set initial splitter sizes upon first display."""
        super().showEvent(ev)
        if not self._sized_once:
            self._sized_once = True
            total_width = max(600, self.width())
            # 2/3 Canvas, 1/3 Sidebar
            self._splitter.setSizes([int(total_width * 2 / 3), int(total_width * 1 / 3)])
            try:
                if not self.isMaximized():
                    self.showMaximized()
            except Exception:
                pass