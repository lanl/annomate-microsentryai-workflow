"""
Application Entry Point for AnnoMate and MicroSentryAI.

This module initializes the PySide6 application, configures the main window with
integrated tabs (AnnoMate, MicroSentryAI, and Validation), and establishes
bidirectional synchronization for navigation, data transfer, and view state.
"""

import sys
from typing import List

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QTabWidget,
    QInputDialog,
    QColorDialog,
)
from PySide6.QtCore import Qt

# Local Application Imports
from AnnoMate.adapter import AnnotatorTab
from MicroSentryAI.adapter import MicroSentryTab
from Validation.adapter import ValidationTab


class AppWindow(QMainWindow):
    """
    Main application window hosting the AnnoMate, MicroSentryAI, and Validation tabs.
    Handles all cross-tab synchronization.
    """
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AnnoMate with MicroSentryAI + Validation")
        
        # Initialize UI Components
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.annotator_tab = AnnotatorTab()
        self.micro_sentry_tab = MicroSentryTab()
        self.validation_tab = ValidationTab()

        self.tabs.addTab(self.annotator_tab, "AnnoMate")
        self.tabs.addTab(self.micro_sentry_tab, "MicroSentryAI")
        self.tabs.addTab(self.validation_tab, "Validation")

        # Establish bidirectional signal/slot connections
        self._setup_connections()

    def _setup_connections(self):
        """Hooks up all synchronization signals between tabs."""
        # Folder Navigation
        self.annotator_tab.folderChanged.connect(self.sync_annotator_to_sentry_folder)
        self.micro_sentry_tab.folderLoaded.connect(self.sync_sentry_to_annotator_folder)

        # Index Navigation
        self.annotator_tab.indexChanged.connect(self.sync_annotator_to_sentry_index)
        self.micro_sentry_tab._host.imageIndexChanged.connect(self.sync_sentry_to_annotator_index)

        # Polygon Transfer
        self.micro_sentry_tab._host.polygonsSent.connect(self.handle_polygon_transfer)

        # Viewport Sync
        self.micro_sentry_tab._host.viewChanged.connect(self.sync_view_sentry_to_annotator)
        if hasattr(self.annotator_tab._host, "viewChanged"):
            self.annotator_tab._host.viewChanged.connect(self.sync_view_annotator_to_sentry)

    # =========================================================================
    # Synchronization Slots
    # =========================================================================

    def sync_annotator_to_sentry_folder(self, folder: str, abs_files: List[str]):
        """Updates MicroSentry when the Annotator folder changes."""
        self.micro_sentry_tab.open_image_folder(folder, absolute_files=abs_files)
        self.micro_sentry_tab.set_index(0)

    def sync_sentry_to_annotator_folder(self, folder: str, abs_files: List[str]):
        """Updates Annotator when the MicroSentry folder changes."""
        if hasattr(self.annotator_tab, "_host"):
            self.annotator_tab._host.load_folder_programmatically(folder, abs_files)

    def sync_annotator_to_sentry_index(self, idx: int, abs_path: str):
        """Syncs MicroSentry index to match Annotator."""
        self.micro_sentry_tab.set_index(idx)

    def sync_sentry_to_annotator_index(self, idx: int):
        """Syncs Annotator index to match MicroSentry."""
        if self.annotator_tab._host.current_idx != idx:
            self.annotator_tab._host.goto_index(idx)

    def handle_polygon_transfer(self, polys: list, default_name: str):
        """
        Receives polygons from MicroSentry and prompts user to add them to AnnoMate.
        """
        host = self.annotator_tab._host
        existing_classes = host.class_names
        
        items = existing_classes + ["New Class..."] if existing_classes else ["Anomaly", "New Class..."]

        item, ok = QInputDialog.getItem(
            self, 
            "Export Polygons", 
            "Select or Type Class Name:", 
            items, 
            0, 
            True
        )

        if not ok or not item:
            return

        class_name = item
        chosen_color = None

        if item == "New Class...":
            text, ok_new = QInputDialog.getText(self, "New Class", "Enter new class name:")
            if ok_new and text:
                class_name = text
                color_dialog = QColorDialog.getColor(Qt.white, self, "Choose Class Color")
                if color_dialog.isValid():
                    chosen_color = color_dialog
            else:
                return  

        for poly in polys:
            host.add_polygon_external(poly, class_name, color=chosen_color)

    def sync_view_annotator_to_sentry(self, rx, ry, scale):
        """Syncs MicroSentry view if it is not currently the active focus."""
        if not self.micro_sentry_tab.isVisible():
            self.micro_sentry_tab._host.set_view_state(rx, ry, scale)

    def sync_view_sentry_to_annotator(self, rx, ry, scale):
        """Syncs Annotator view to match MicroSentry."""
        if hasattr(self.annotator_tab._host, "set_view_state"):
            self.annotator_tab._host.set_view_state(rx, ry, scale)


def main():
    """
    Initializes the PySide6 application and event loop.
    """
    # NOTE: In Qt6/PySide6, High DPI scaling is enabled by default. 
    # The Qt.AA_EnableHighDpiScaling and Qt.AA_UseHighDpiPixmaps attributes 
    # were completely removed from the framework, so these are commented out.
    # QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    # QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    
    main_window = AppWindow()
    main_window.showMaximized()

    sys.exit(app.exec()) # Changed from exec_() to exec()


if __name__ == "__main__":
    main()