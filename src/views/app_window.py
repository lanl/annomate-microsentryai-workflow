"""AppWindow — top-level application shell view.

See MVC.md § Architecture Rules for the full layer contract.
"""

import os

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import (
    QMainWindow,
    QInputDialog,
    QFileDialog,
    QMessageBox,
)
from PySide6.QtGui import QAction, QKeySequence

from views.annomate.window import AnnoMateWindow

_APP_TITLE = "AnnoMate & MicroSentryAI"
_LAST_IMAGE_DIR_KEY = "recent/last_image_dir"
_LAST_PROJECT_KEY = "recent/last_project"


class AppWindow(QMainWindow):
    """Top-level application shell.

    Owns the tab widget and the File menu. Receives all models and
    controllers as constructor arguments — creates nothing itself.
    All dialogs (QFileDialog, QMessageBox, QInputDialog) live here.

    Args:
        dataset_model: DatasetTableModel instance.
        inference_model: InferenceModel instance.
        io_controller: IOController instance.
        inference_controller: InferenceController instance.
        project_controller: ProjectController instance.
    """

    def __init__(
        self,
        dataset_model,
        inference_model,
        io_controller,
        inference_controller,
        project_controller,
        calibration_model=None,
        center_template_model=None,
        center_template_controller=None,
        anomaly_constraint_model=None,
    ) -> None:
        super().__init__()
        self.setWindowTitle(_APP_TITLE)
        self.resize(1400, 900)

        self.dataset_model = dataset_model
        self.inference_model = inference_model
        self.io_controller = io_controller
        self.inference_controller = inference_controller
        self.project_controller = project_controller
        self.calibration_model = calibration_model
        self.center_template_model = center_template_model
        self.center_template_controller = center_template_controller
        self._settings = QSettings("LANL", "AnnoMateMicroSentryAI")

        # Sub-views
        self.annomate_view = AnnoMateWindow(
            dataset_model,
            io_controller,
            inference_model,
            inference_controller,
            calibration_model=calibration_model,
            center_template_model=center_template_model,
            center_template_controller=center_template_controller,
            project_controller=project_controller,
            anomaly_constraint_model=anomaly_constraint_model,
        )

        self.annomate_view.new_project_requested.connect(self._new_project)
        self.annomate_view.open_project_requested.connect(self._open_project)
        self.annomate_view.open_image_folder_requested.connect(self._open_image_folder)
        self.annomate_view.open_recent_project_requested.connect(
            self._open_recent_project
        )
        self.annomate_view.open_recent_image_folder_requested.connect(
            self._open_recent_image_folder
        )
        self._refresh_project_start_state()

        self.setCentralWidget(self.annomate_view)

        # React to ProjectController signals
        self.project_controller.dirty_changed.connect(self._update_title)
        self.project_controller.project_opened.connect(lambda _: self._update_title())
        self.project_controller.project_saved.connect(
            lambda path: self.statusBar().showMessage(f"Saved: {path}", 4000)
        )
        self.project_controller.project_saved.connect(lambda _: self._update_title())
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
        add(
            data_menu,
            "Import Annotation Classes…",
            "",
            self._import_annotation_classes,
        )
        data_menu.addSeparator()
        add(
            data_menu,
            "Export Annotation Classes",
            "",
            self._export_annotation_classes,
        )
        add(data_menu, "Export Binary Masks…", "", self._export_binary_masks)
        add(data_menu, "Export CSV…", "", self._export_csv)
        add(data_menu, "Export Train Structure…", "", self._export_train_structure)

    def _refresh_project_start_state(self) -> None:
        """Refresh recent-action shortcuts on the empty project start screen."""
        self.annomate_view.set_project_start_state(
            self._last_project(), self._last_image_dir()
        )

    def _remember_recent_project(self, path: str) -> None:
        if path:
            self._settings.setValue(_LAST_PROJECT_KEY, path)

    def _remember_recent_image_dir(self, path: str) -> None:
        if path:
            self._settings.setValue(_LAST_IMAGE_DIR_KEY, path)

    def _last_project(self) -> str:
        return self._settings.value(_LAST_PROJECT_KEY, "", type=str) or ""

    def _last_image_dir(self) -> str:
        return self._settings.value(_LAST_IMAGE_DIR_KEY, "", type=str) or ""

    def _open_project_path(self, path: str) -> None:
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

        self._remember_recent_project(path)
        image_dir = project_data.get("dataset", {}).get("image_dir", "")
        if image_dir:
            self._remember_recent_image_dir(image_dir)
        self._refresh_project_start_state()

    def _open_recent_project(self, path: str) -> None:
        if self.project_controller.is_dirty and not self._confirm_discard():
            return
        if not os.path.exists(path):
            self._settings.remove(_LAST_PROJECT_KEY)
            self._refresh_project_start_state()
            QMessageBox.warning(
                self,
                "Open Project",
                f"Recent project no longer exists:\n{path}",
            )
            return
        self._open_project_path(path)

    def _open_recent_image_folder(self, directory: str) -> None:
        if not os.path.isdir(directory):
            self._settings.remove(_LAST_IMAGE_DIR_KEY)
            self._refresh_project_start_state()
            QMessageBox.warning(
                self,
                "Open Image Folder",
                f"Recent image folder no longer exists:\n{directory}",
            )
            return
        self.io_controller.load_folder(directory)
        self._remember_recent_image_dir(directory)
        self._refresh_project_start_state()

    # ================================================================== #
    # Project slots
    # ================================================================== #

    def _new_project(self) -> None:
        if self.project_controller.is_dirty and not self._confirm_discard():
            return
        self.project_controller.new_project()
        self._refresh_project_start_state()

    def _open_project(self) -> None:
        if self.project_controller.is_dirty and not self._confirm_discard():
            return

        path, _ = QFileDialog.getOpenFileName(
            self, "Open Project", os.getcwd(), "AnnoMate Project (*.annoproj)"
        )
        if not path:
            return

        self._open_project_path(path)

    def _save_project(self) -> None:
        if not self.project_controller.has_project:
            self._save_project_as()
            return
        self._check_orphans_then_save(self.project_controller.save_project)

    def _save_project_as(self) -> None:
        project_dir = QFileDialog.getExistingDirectory(
            self, "Choose Project Folder", os.getcwd()
        )
        if not project_dir:
            return

        image_dir = self.dataset_model.get_image_dir()
        from pathlib import Path

        default_name = Path(image_dir).name if image_dir else Path(project_dir).name

        name, ok = QInputDialog.getText(
            self, "Project Name", "Enter project name:", text=default_name
        )
        if not ok or not name.strip():
            return

        self._check_orphans_then_save(
            lambda: self.project_controller.save_project_as(project_dir, name.strip())
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
            saved_path = save_fn()
        except Exception as exc:
            QMessageBox.critical(self, "Save Project", f"Could not save:\n{exc}")
            return
        self._remember_recent_project(saved_path)
        image_dir = self.dataset_model.get_image_dir()
        if image_dir:
            self._remember_recent_image_dir(image_dir)
        self._refresh_project_start_state()

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
            self._remember_recent_image_dir(new_dir)
            self._refresh_project_start_state()
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

    def _import_annotation_classes(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Annotation Classes", "", "Text Files (*.txt)"
        )
        if not path:
            return
        try:
            msg = self.io_controller.import_annotation_classes(path)
            QMessageBox.information(self, "Import Annotation Classes", msg)
        except Exception as exc:
            QMessageBox.critical(
                self, "Import Annotation Classes", f"Import failed:\n{exc}"
            )
            return

    def _export_start_dir(self) -> str:
        return self.project_controller.project_dir or os.getcwd()

    def _export_annotation_classes(self) -> None:
        if not self.project_controller.has_project:
            self._save_project_as()
            project_dir = self.project_controller.project_dir
            if not project_dir or not os.path.isdir(project_dir):
                return

        try:
            msg = self.io_controller.export_annotation_classes(
                self.project_controller.project_dir
            )
            QMessageBox.information(self, "Export Annotation Classes", msg)
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))

    def _export_binary_masks(self) -> None:
        chosen = QFileDialog.getExistingDirectory(
            self, "Choose ground truth output folder", self._export_start_dir()
        )
        if not chosen:
            return
        try:
            msg = self.io_controller.export_binary_masks(
                os.path.join(chosen, "binary_masks")
            )
            QMessageBox.information(self, "Export Binary Masks", msg)
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))

    def _export_csv(self) -> None:
        out_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save CSV",
            os.path.join(self._export_start_dir(), "metadata.csv"),
            "CSV (*.csv)",
        )
        if not out_path:
            return
        try:
            msg = self.io_controller.export_csv(out_path)
            QMessageBox.information(self, "Export", msg)
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))

    def _export_train_structure(self) -> None:
        parent_dir = QFileDialog.getExistingDirectory(
            self, "Choose Train Structure Output Folder", self._export_start_dir()
        )
        if not parent_dir:
            return

        default_name = self.project_controller.project_name or "dataset"
        name, ok = QInputDialog.getText(
            self, "Dataset Folder Name", "Enter dataset folder name:", text=default_name
        )
        if not ok or not name.strip():
            return

        out_dir = os.path.join(parent_dir, name.strip())
        try:
            msg = self.io_controller.export_train_structure(out_dir)
            QMessageBox.information(self, "Export Train Structure", msg)
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
        if self.center_template_controller is not None:
            self.center_template_controller.shutdown()
        self.inference_controller.shutdown()
        self.annomate_view.shutdown()
        super().closeEvent(event)
