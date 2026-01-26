"""
Adapter Module for MicroSentryAI Integration.

This module provides the `MicroSentryTab` class, which wraps the standalone
`MicroSentryWindow` into a QWidget suitable for embedding within the main
application's QTabWidget. It manages signal forwarding and external control
commands (opening folders, navigation).
"""

import os
import glob
from typing import List, Optional

from PyQt5.QtCore import Qt, QObject, pyqtSignal, QEvent
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

# Relative import to find visualizer.py in the same package
from .visualizer import MicroSentryWindow


class _KeyForwarder(QObject):
    """
    Event filter to forward key presses from the container to the embedded window.
    This ensures keyboard shortcuts work even when the QMainWindow is embedded.
    """

    def __init__(self, target: QObject):
        super().__init__()
        self._target = target
        self._in_forward = False

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if isinstance(event, QKeyEvent):
            # 1. Let text inputs handle their own typing
            if isinstance(obj, (QLineEdit, QTextEdit, QPlainTextEdit)):
                return False

            # 2. Prevent infinite recursion
            if self._in_forward:
                return False

            self._in_forward = True
            try:
                QApplication.sendEvent(self._target, event)
            finally:
                self._in_forward = False

            return False

        return False


class MicroSentryTab(QWidget):
    """
    A wrapper widget that hosts the `MicroSentryWindow`.

    Signals:
        folderLoaded (str, list): Emitted when a new folder is loaded within this tab.
                                  Args: (folder_path, list_of_absolute_file_paths)
    """

    folderLoaded = pyqtSignal(str, list)

    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize the tab and embed the MicroSentryWindow."""
        super().__init__(parent)
        self._host = MicroSentryWindow()

        # Connect internal signal to public wrapper signal
        self._host.folderLoaded.connect(self.folderLoaded.emit)

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

    def open_image_folder(self, folder_path: str, absolute_files: Optional[List[str]] = None):
        """
        Opens an image folder programmatically (e.g., via sync from AnnoMate).

        Args:
            folder_path (str): The absolute path to the directory.
            absolute_files (List[str], optional): Explicit list of file paths.
                                                  If None, scans directory for images.
        """
        files = []
        
        if absolute_files is not None:
            files = absolute_files
        else:
            # Scan for images if no list provided
            patterns = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
            for pat in patterns:
                files.extend(glob.glob(os.path.join(folder_path, pat)))
            files.sort()

        self._host.image_files = files
        self._host._build_table()
        self._host.idx = 0
        
        if files:
            self._host.process_image()

    def set_index(self, idx: int):
        """
        Sets the current image index programmatically.

        Args:
            idx (int): The target index.
        """
        if not getattr(self._host, 'image_files', None):
            return
        
        # Ensure index is within bounds before calling host
        if 0 <= idx < len(self._host.image_files):
            self._host.goto_index(idx)