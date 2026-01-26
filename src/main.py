"""
Application Entry Point for AnnoMate and MicroSentryAI.

This module initializes the PyQt5 application, configures the main window with
integrated tabs (AnnoMate, MicroSentryAI, and Validation), and establishes
bidirectional synchronization for navigation, data transfer, and view state.
"""

import sys
from typing import List

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QTabWidget,
    QInputDialog,
    QColorDialog,
)
from PyQt5.QtCore import Qt

# Local Application Imports
from AnnoMate.adapter import AnnotatorTab
from MicroSentryAI.adapter import MicroSentryTab
from Validation.adapter import ValidationTab


def main():
    """
    Initializes the main application window and event loop.

    Sets up:
        1. High-DPI scaling attributes.
        2. The primary QMainWindow and QTabWidget.
        3. Signal-slot connections between the AnnoMate and MicroSentryAI tabs
           to sync folders, indices, and viewports.
    """
    # Configure High-DPI scaling for modern displays
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    main_window = QMainWindow()
    tabs = QTabWidget()
    main_window.setCentralWidget(tabs)

    # Initialize Modules
    annotator_tab = AnnotatorTab()
    micro_sentry_tab = MicroSentryTab()
    validation_tab = ValidationTab()

    # Compose Tabs
    tabs.addTab(annotator_tab, "AnnoMate")
    tabs.addTab(micro_sentry_tab, "MicroSentryAI")
    tabs.addTab(validation_tab, "Validation")

    # =========================================================================
    # Synchronization Logic: Navigation (Bidirectional)
    # =========================================================================

    def sync_annotator_to_sentry_folder(folder: str, abs_files: List[str]):
        """Updates MicroSentry when the Annotator folder changes."""
        micro_sentry_tab.open_image_folder(folder, absolute_files=abs_files)
        micro_sentry_tab.set_index(0)

    def sync_sentry_to_annotator_folder(folder: str, abs_files: List[str]):
        """Updates Annotator when the MicroSentry folder changes."""
        if hasattr(annotator_tab, "_host"):
            annotator_tab._host.load_folder_programmatically(folder, abs_files)

    # Connect Navigation Signals
    annotator_tab.folderChanged.connect(sync_annotator_to_sentry_folder)
    micro_sentry_tab.folderLoaded.connect(sync_sentry_to_annotator_folder)

    # =========================================================================
    # Synchronization Logic: Image Index
    # =========================================================================

    def sync_annotator_to_sentry_index(idx: int, abs_path: str):
        """Syncs MicroSentry index to match Annotator."""
        micro_sentry_tab.set_index(idx)

    def sync_sentry_to_annotator_index(idx: int):
        """Syncs Annotator index to match MicroSentry."""
        # Accessing internal host to sync state
        if annotator_tab._host.current_idx != idx:
            annotator_tab._host.goto_index(idx)

    # Connect Index Signals
    annotator_tab.indexChanged.connect(sync_annotator_to_sentry_index)
    micro_sentry_tab._host.imageIndexChanged.connect(sync_sentry_to_annotator_index)

    # =========================================================================
    # Synchronization Logic: Data Transfer (Polygons)
    # =========================================================================

    def handle_polygon_transfer(polys: list, default_name: str):
        """
        Receives polygons from MicroSentry and prompts user to add them to AnnoMate.
        """
        host = annotator_tab._host
        existing_classes = host.class_names
        
        # Determine dropdown options
        items = existing_classes + ["New Class..."] if existing_classes else ["Anomaly", "New Class..."]

        item, ok = QInputDialog.getItem(
            main_window, 
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

        # Handle creation of a new class type
        if item == "New Class...":
            text, ok_new = QInputDialog.getText(main_window, "New Class", "Enter new class name:")
            if ok_new and text:
                class_name = text
                color_dialog = QColorDialog.getColor(Qt.white, main_window, "Choose Class Color")
                if color_dialog.isValid():
                    chosen_color = color_dialog
            else:
                return  # User cancelled new class creation

        # Add polygons to the annotator host
        for poly in polys:
            host.add_polygon_external(poly, class_name, color=chosen_color)

    micro_sentry_tab._host.polygonsSent.connect(handle_polygon_transfer)

    # =========================================================================
    # Synchronization Logic: View State (Zoom/Pan)
    # =========================================================================

    def sync_view_annotator_to_sentry(rx, ry, scale):
        """Syncs MicroSentry view if it is not currently the active focus."""
        if not micro_sentry_tab.isVisible():
            micro_sentry_tab._host.set_view_state(rx, ry, scale)

    def sync_view_sentry_to_annotator(rx, ry, scale):
        """Syncs Annotator view to match MicroSentry."""
        if hasattr(annotator_tab._host, "set_view_state"):
            annotator_tab._host.set_view_state(rx, ry, scale)

    micro_sentry_tab._host.viewChanged.connect(sync_view_sentry_to_annotator)
    
    # Defensive check: ensure the annotator host supports view signals
    if hasattr(annotator_tab._host, "viewChanged"):
        annotator_tab._host.viewChanged.connect(sync_view_annotator_to_sentry)

    # =========================================================================
    # Final Window Configuration
    # =========================================================================
    
    main_window.setWindowTitle("AnnoMate with MicroSentryAI + Validation")
    main_window.showMaximized()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()