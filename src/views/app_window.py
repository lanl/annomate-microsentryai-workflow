"""AppWindow — top-level application shell view.

See MVC.md § Architecture Rules for the full layer contract.
"""

import os

from PySide6.QtWidgets import (
    QMainWindow,
    QInputDialog,
    QFileDialog,
    QMessageBox,
)
from PySide6.QtGui import QAction, QKeySequence

from views.validation.window import ValidationWindow
from views.annomate.window import AnnoMateWindow

_APP_TITLE = "AnnoMate & MicroSentryAI"


class AppWindow(QMainWindow):
    """Top-level application shell.

    Owns the tab widget and the File menu. Receives all models and
    controllers as constructor arguments — creates nothing itself.
    All dialogs (QFileDialog, QMessageBox, QInputDialog) live here.

    Args:
        dataset_model: DatasetTableModel instance.
        inference_model: InferenceModel instance.
        validation_model: ValidationModel instance.
        io_controller: IOController instance.
        inference_controller: InferenceController instance.
        validation_controller: ValidationController instance.
        project_controller: ProjectController instance.
    """

    def __init__(
        self,
        dataset_model,
        inference_model,
        validation_model,
        io_controller,
        inference_controller,
        validation_controller,
        project_controller,
    ) -> None:
        super().__init__()
        self.setWindowTitle(_APP_TITLE)
        self.resize(1400, 900)

        self.dataset_model = dataset_model
        self.inference_model = inference_model
        self.validation_model = validation_model
        self.io_controller = io_controller
        self.inference_controller = inference_controller
        self.validation_controller = validation_controller
        self.project_controller = project_controller

        # Sub-views
        self.validation_view = ValidationWindow(validation_model, validation_controller)
        self.validation_view.setWindowTitle("Validation")
        self.validation_view.resize(900, 650)
        self.annomate_view = AnnoMateWindow(
            dataset_model, io_controller, inference_model, inference_controller
        )

        self.setCentralWidget(self.annomate_view)

        # React to ProjectController signals
        self.project_controller.dirty_changed.connect(self._update_title)
        self.project_controller.project_saved.connect(
            lambda path: self.statusBar().showMessage(f"Saved: {path}", 4000)
        )
        self.project_controller.autosave_written.connect(
            lambda _: self.statusBar().showMessage("Autosaved", 3000)
        )
        self.project_controller.autosave_failed.connect(
            lambda msg: self.statusBar().showMessage(f"Autosave failed: {msg}", 5000)
        )

        self._build_menu()

    # ================================================================== #
    # Menu bar
    # ================================================================== #

    def _build_menu(self) -> None:
        def add(menu, label, shortcut, slot):
            act = QAction(label, self)
            if shortcut:
                act.setShortcut(QKeySequence(shortcut))
            act.triggered.connect(slot)
            menu.addAction(act)

        file_menu = self.menuBar().addMenu("&File")
        add(file_menu, "New Project", "Ctrl+N", self._new_project)
        add(file_menu, "Open Project…", "Ctrl+O", self._open_project)
        add(file_menu, "Save Project", "Ctrl+S", self._save_project)
        add(file_menu, "Save Project As…", "Ctrl+Shift+S", self._save_project_as)
        file_menu.addSeparator()
        add(file_menu, "Open Image Folder…", "", self._open_image_folder)
        add(file_menu, "Relocate Images…", "", self._relocate_images)
        file_menu.addSeparator()
        add(file_menu, "Preferences…", "", self._open_preferences)
        file_menu.addSeparator()
        add(file_menu, "Exit", "Ctrl+Q", self.close)

        data_menu = self.menuBar().addMenu("&Data")
        add(data_menu, "Import JSON Data…", "", self._import_coco)
        data_menu.addSeparator()
        add(data_menu, "Export Polygons + Data…", "", self._export_polygons)
        add(data_menu, "Export Binary Masks…", "", self._export_binary_masks)
        add(data_menu, "Export CSV…", "", self._export_csv)

        validation_menu = self.menuBar().addMenu("&Validation")
        add(validation_menu, "Open Validation…", "", self._open_validation)

        view_menu = self.menuBar().addMenu("&Microsentry")
        self._ms_action = QAction("Enable MicroSentryAI", self)
        self._ms_action.setCheckable(True)
        self._ms_action.setToolTip("Toggle MicroSentryAI heatmap and segmentation")
        self._ms_action.toggled.connect(self.annomate_view._on_microsentry_toggled)
        view_menu.addAction(self._ms_action)

    # ================================================================== #
    # Project slots
    # ================================================================== #

    def _new_project(self) -> None:
        if self.project_controller.is_dirty and not self._confirm_discard():
            return
        self.project_controller.new_project()

    def _open_project(self) -> None:
        if self.project_controller.is_dirty and not self._confirm_discard():
            return

        path, _ = QFileDialog.getOpenFileName(
            self, "Open Project", os.getcwd(), "AnnoMate Project (*.annoproj)"
        )
        if not path:
            return

        try:
            project_data, warnings = self.project_controller.open_project(path)
        except Exception as exc:
            QMessageBox.critical(
                self, "Open Project", f"Could not read project:\n{exc}"
            )
            return

        if warnings:
            QMessageBox.warning(self, "Open Project", "\n\n".join(warnings))

        model_path = project_data.get("inference", {}).get("model_path", "")
        self.annomate_view.set_saved_model_path(model_path)
        if model_path and not self.inference_controller.has_model():
            self.statusBar().showMessage(
                f"Previous model saved: {os.path.basename(model_path)} — use 'Load Previous' in the MicroSentryAI panel.",
                8000,
            )

    def _save_project(self) -> None:
        if not self.project_controller.has_project:
            self._save_project_as()
            return
        self._check_orphans_then_save(self.project_controller.save_project)

    def _save_project_as(self) -> None:
        parent_dir = QFileDialog.getExistingDirectory(
            self, "Choose Project Folder", os.getcwd()
        )
        if not parent_dir:
            return

        image_dir = self.dataset_model.get_image_dir()
        from pathlib import Path

        default_name = Path(image_dir).name if image_dir else "project"

        name, ok = QInputDialog.getText(
            self, "Project Name", "Enter project name:", text=default_name
        )
        if not ok or not name.strip():
            return

        name = name.strip()
        project_dir = os.path.join(parent_dir, name)
        self._check_orphans_then_save(
            lambda: self.project_controller.save_project_as(project_dir, name)
        )

    def _check_orphans_then_save(self, save_fn) -> None:
        """Show orphaned-annotation warning if needed, then call save_fn."""
        warning = self.project_controller.orphaned_annotations_warning()
        if warning:
            reply = QMessageBox.warning(
                self,
                "Orphaned Annotations",
                warning,
                QMessageBox.Ok | QMessageBox.Cancel,
                QMessageBox.Cancel,
            )
            if reply != QMessageBox.Ok:
                return
        try:
            save_fn()
        except Exception as exc:
            QMessageBox.critical(self, "Save Project", f"Could not save:\n{exc}")

    def _open_image_folder(self) -> None:
        """Scan a folder for images and load them as the current dataset."""
        directory = QFileDialog.getExistingDirectory(
            self, "Open Image Folder", os.getcwd()
        )
        if not directory:
            return
        self.io_controller.load_folder(directory)

    def _relocate_images(self) -> None:
        """Point to a new image directory without clearing annotations."""
        new_dir = QFileDialog.getExistingDirectory(
            self, "Select New Image Folder", os.getcwd()
        )
        if not new_dir:
            return
        try:
            self.project_controller.relocate_images(new_dir)
        except Exception as exc:
            QMessageBox.critical(
                self, "Relocate Images", f"Could not scan folder:\n{exc}"
            )
            return

        orphan_msg = self.project_controller.orphaned_annotations_warning()
        if orphan_msg:
            QMessageBox.information(
                self,
                "Annotations After Relocation",
                orphan_msg.replace(
                    "Continue?", "They will be dropped on the next save."
                ),
            )

    def _open_preferences(self) -> None:
        QMessageBox.information(self, "Preferences", "Preferences coming soon.")

    # ================================================================== #
    # Data menu handlers
    # ================================================================== #

    def _import_coco(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Import JSON Data", "", "JSON (*.json)"
        )
        if not path:
            return
        try:
            self.io_controller.import_data_json(path)
        except Exception as exc:
            QMessageBox.critical(self, "Import JSON Data", f"Import failed:\n{exc}")
            return

    def _export_polygons(self) -> None:
        out_dir = QFileDialog.getExistingDirectory(
            self, "Choose output folder", os.getcwd()
        )
        if not out_dir:
            return
        try:
            msg = self.io_controller.export_polygons_and_data(out_dir)
            QMessageBox.information(self, "Export", msg)
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))

    def _export_binary_masks(self) -> None:
        out_dir = QFileDialog.getExistingDirectory(
            self, "Choose ground truth output folder", os.getcwd()
        )
        if not out_dir:
            return
        try:
            msg = self.io_controller.export_binary_masks(out_dir)
            QMessageBox.information(self, "Export Binary Masks", msg)
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))

    def _export_csv(self) -> None:
        out_path, _ = QFileDialog.getSaveFileName(
            self, "Save CSV", "metadata.csv", "CSV (*.csv)"
        )
        if not out_path:
            return
        try:
            msg = self.io_controller.export_csv(out_path)
            QMessageBox.information(self, "Export", msg)
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))

    # ================================================================== #
    # Title, close, unsaved-changes guard
    # ================================================================== #

    def _update_title(self, is_dirty: bool = None) -> None:
        name = self.project_controller.project_name
        if name:
            dirty = "*" if self.project_controller.is_dirty else ""
            self.setWindowTitle(f"{name}{dirty} — {_APP_TITLE}")
        else:
            self.setWindowTitle(_APP_TITLE)

    def _confirm_discard(self) -> bool:
        """Prompt save/discard for unsaved changes. Returns True to proceed."""
        reply = QMessageBox.question(
            self,
            "Unsaved Changes",
            "You have unsaved changes. Save before continuing?",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            QMessageBox.Save,
        )
        if reply == QMessageBox.Save:
            self._save_project()
            return True
        return reply == QMessageBox.Discard

    def closeEvent(self, event) -> None:
        if self.project_controller.is_dirty:
            reply = QMessageBox.question(
                self,
                "Unsaved Changes",
                "You have unsaved changes. Save before closing?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                QMessageBox.Save,
            )
            if reply == QMessageBox.Save:
                self._save_project()
                event.accept()
            elif reply == QMessageBox.Discard:
                event.accept()
            else:
                event.ignore()
                return
        self.project_controller.autosave_manager.stop()
        super().closeEvent(event)

    def _open_validation(self) -> None:
        self.validation_view.show()
        self.validation_view.raise_()
        self.validation_view.activateWindow()
