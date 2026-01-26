"""
Adapter Module for AnnoMate Integration.

This module provides the `AnnotatorTab` class, which wraps the standalone
`ImageAnnotator` window into a QWidget suitable for embedding within a
QTabWidget. It handles event forwarding and signal synchronization between
the AnnoMate logic and the parent application (main.py).
"""

import os
from pathlib import Path
from typing import List, Optional

from PyQt5.QtCore import pyqtSignal, Qt, QObject, QEvent
from PyQt5.QtGui import QKeyEvent
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QApplication,
    QMainWindow,
    QLineEdit,
    QTextEdit,
    QPlainTextEdit,
)

# Relative import to find window.py in the same package
from .window import ImageAnnotator


class _KeyForwarder(QObject):
    """
    Event filter to forward key presses from the container to the embedded window.
    """

    def __init__(self, target: QObject):
        super().__init__()
        self._target = target
        self._in_forward = False  # Recursion guard

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if isinstance(event, QKeyEvent):
            # 1. Let text input widgets handle their own typing
            if isinstance(obj, (QLineEdit, QTextEdit, QPlainTextEdit)):
                return False

            # 2. Prevent infinite recursion
            if self._in_forward:
                return False

            self._in_forward = True
            try:
                # Forward the event to the hosted main window
                QApplication.sendEvent(self._target, event)
            finally:
                self._in_forward = False

            # Return False to allow the original widget to process it as well if needed
            return False

        return False


class AnnotatorTab(QWidget):
    """
    A wrapper widget that hosts the AnnoMate `ImageAnnotator`.

    Signals:
        folderChanged (str, list): Emitted when a new folder is loaded.
                                   Args: (folder_path, list_of_absolute_file_paths)
        indexChanged (int, str): Emitted when the current image index changes.
                                 Args: (index, absolute_file_path)
    """

    folderChanged = pyqtSignal(str, list)
    indexChanged = pyqtSignal(int, str)

    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize the tab and embed the ImageAnnotator."""
        super().__init__(parent)
        self._host = ImageAnnotator()

        # Configure Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Handle embedding QMainWindow inside QWidget
        if isinstance(self._host, QMainWindow):
            self._host.setParent(self)
            self._host.setWindowFlags(self._host.windowFlags() | Qt.Widget)
            layout.addWidget(self._host)
        else:
            layout.addWidget(self._host)

        self._host.show()
        self._host.raise_()

        # Install Key Forwarder
        self._forwarder = _KeyForwarder(self._host)
        self.installEventFilter(self._forwarder)

        # Initialize Signal Hooks
        self._setup_synchronization()

    def _setup_synchronization(self):
        """
        Connects internal signals and overrides methods to ensure state changes
        propagate to the parent application.
        """
        # 1. Folder Synchronization
        # The window emits 'folderLoaded' with RELATIVE filenames.
        # We must convert them to ABSOLUTE paths before re-emitting.
        self._host.folderLoaded.connect(self._on_folder_loaded_internal)

        # 2. Index Synchronization
        # Intercept navigation calls to emit the `indexChanged` signal.
        self._setup_index_hooks()

    def _on_folder_loaded_internal(self, folder_path: str, files: List[str]):
        """
        Intercepts the window's signal, converts relative filenames to absolute paths,
        and emits the public folderChanged signal.
        """
        try:
            abs_files = [str(Path(folder_path) / f) for f in files]
            self.folderChanged.emit(folder_path, abs_files)
        except Exception:
            # Fallback in case of path errors
            pass

    def _setup_index_hooks(self):
        """
        Intercepts navigation calls to emit the `indexChanged` signal.
        """
        original_goto = self._host.goto_index

        def wrapped_goto(idx: int):
            # Execute original logic
            original_goto(idx)

            # Retrieve new state
            d = getattr(self._host, "image_dir", None)
            files = getattr(self._host, "image_files", [])
            cur = getattr(self._host, "current_idx", -1)

            # Emit signal if valid
            if d and files and 0 <= cur < len(files):
                # Ensure we emit an absolute path
                abs_path = str(Path(d) / files[cur])
                self.indexChanged.emit(cur, abs_path)

        # Override the method on the instance
        self._host.goto_index = wrapped_goto

        # Reconnect navigation buttons to use the wrapped method.
        # This ensures clicking 'Next'/'Prev' triggers the synchronization logic.
        self._reconnect_button(self._host.btn_prev, lambda: wrapped_goto(self._host.current_idx - 1))
        self._reconnect_button(self._host.btn_next, lambda: wrapped_goto(self._host.current_idx + 1))

    def _reconnect_button(self, button, new_slot):
        """Safe helper to disconnect old slots and connect a new one."""
        try:
            button.clicked.disconnect()
        except TypeError:
            pass
        except Exception:
            pass
        button.clicked.connect(new_slot)

    # --- Public API for Parent App ---

    def programmatic_open_folder(self, folder_path: str):
        """
        Opens a folder programmatically (e.g., triggered by the other tab).

        Args:
            folder_path (str): The absolute path to the directory.
        """
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        
        try:
            # Filter for valid images
            files = sorted([
                f for f in os.listdir(folder_path) 
                if Path(f).suffix.lower() in exts
            ])
            
            if not files:
                return

            # Update Host State
            self._host.load_folder_programmatically(folder_path, files)
            
            # Emit Signal (Convert to Absolute)
            abs_files = [str(Path(folder_path) / f) for f in files]
            self.folderChanged.emit(folder_path, abs_files)

        except Exception:
            pass

    def set_index(self, idx: int):
        """Sets the current image index programmatically."""
        self._host.goto_index(idx)