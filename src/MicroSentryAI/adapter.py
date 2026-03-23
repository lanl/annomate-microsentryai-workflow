"""
Adapter Module for MicroSentryAI Integration.
"""

import os
import glob
from typing import List, Optional

from PySide6.QtCore import Qt, QObject, Signal, QEvent
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QApplication, QMainWindow,
    QLineEdit, QTextEdit, QPlainTextEdit,
)

from .visualizer import MicroSentryWindow


class _KeyForwarder(QObject):
    def __init__(self, target: QObject):
        super().__init__()
        self._target = target
        self._in_forward = False

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if isinstance(event, QKeyEvent):
            if isinstance(obj, (QLineEdit, QTextEdit, QPlainTextEdit)):
                return False
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
    folderLoaded = Signal(str, list)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._host = MicroSentryWindow()
        self._host.folderLoaded.connect(self.folderLoaded.emit)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        if isinstance(self._host, QMainWindow):
            self._host.setParent(self)
            self._host.setWindowFlags(self._host.windowFlags() | Qt.Widget)
        
        layout.addWidget(self._host)
        self._host.show()
        self._host.raise_()

        self._forwarder = _KeyForwarder(self._host)
        self.installEventFilter(self._forwarder)

    def open_image_folder(self, folder_path: str, absolute_files: Optional[List[str]] = None):
        files = []
        if absolute_files is not None:
            files = absolute_files
        else:
            patterns = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
            for pat in patterns:
                files.extend(glob.glob(os.path.join(folder_path, pat)))
            files.sort()

        self._host.image_files = files
        self._host._build_table()
        
        # Ensure resetting forces the view update
        self._host.idx = -1
        self._host.goto_index(0)
        
        # --- FIX: Start background inference so the table updates dynamically! ---
        if files and getattr(self._host, 'active_strategy', None):
            self._host.start_background_inference()

    def set_index(self, idx: int):
        if not getattr(self._host, 'image_files', None):
            return
        if 0 <= idx < len(self._host.image_files):
            self._host.goto_index(idx)