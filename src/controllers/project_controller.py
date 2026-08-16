"""
ProjectController — owns the project save/load lifecycle.

Rules (consistent with existing controllers):
  - No Qt GUI types (no QFileDialog, QMessageBox). All dialogs live in views.
  - Accepts plain Python values; callers handle user-facing display.
  - QObject/Signal is permitted as infrastructure, not UI.
  - State is mutated directly for bulk operations (same pattern as IOController),
    but a single model reset is emitted at the end so views update atomically.
"""

import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

from PySide6.QtCore import QObject, Signal

from controllers.autosave import AutosaveManager
from core.persistence.project_io import ProjectIO
from core.utils.image_scan import scan_images

logger = logging.getLogger("AnnoMate.ProjectController")


class ProjectController(QObject):
    """Own the project save/load lifecycle and autosave timer.

    Encapsulates ProjectIO and AutosaveManager. Accepts all three models
    so it can read state directly for saves and mutate state for loads —
    the same bulk-mutation pattern used by IOController. Emits a single
    model reset after load so views refresh atomically with complete data.

    Dirty tracking is managed here rather than in AppWindow. The _loading
    guard suppresses dirty signals that fire during the load sequence itself.

    Attributes:
        dataset_model: DatasetTableModel.
        inference_model: InferenceModel.
        io_controller: IOController (used for image-dir scans).
        inference_controller: InferenceController (used for switching the
            active cached model).

    Signals:
        dirty_changed (bool): Emitted when the dirty flag changes state.
        project_opened (str): Emitted with the project name after a successful open.
        project_saved (str): Emitted with the .annoproj path after a successful save.
        autosave_written (str): Emitted with the autosave path on success.
        autosave_failed (str): Emitted with an error message on autosave failure.
    """

    dirty_changed = Signal(bool)
    project_opened = Signal(str)  # project_name
    project_saved = Signal(str)  # annoproj_path
    autosave_written = Signal(str)  # autosave annoproj path
    autosave_failed = Signal(str)  # error message

    def __init__(
        self,
        dataset_model,
        inference_model,
        io_controller,
        inference_controller=None,
        calibration_model=None,
        center_template_model=None,
        anomaly_constraint_model=None,
        parent: QObject = None,
    ) -> None:
        super().__init__(parent)
        self._dataset_model = dataset_model
        self._inference_model = inference_model
        self._io_controller = io_controller
        self._inference_controller = inference_controller
        self._calibration_model = calibration_model
        self._center_template_model = center_template_model
        self._anomaly_constraint_model = anomaly_constraint_model

        self._project_io = ProjectIO()
        self._autosave_manager = AutosaveManager(interval_minutes=5, parent=self)
        self._autosave_manager.save_requested.connect(self._do_autosave)

        self._project_dir: Optional[str] = None
        self._project_name: str = ""
        self._created_at: Optional[str] = None
        self._is_dirty: bool = False
        self._loading: bool = False
        self._accumulated_seconds: float = 0.0
        self._session_start: Optional[float] = None

        # Connect model signals for dirty tracking. The _loading guard
        # suppresses spurious dirty events fired during our own load sequence.
        self._dataset_model.dataChanged.connect(self._on_model_changed)
        self._dataset_model.modelReset.connect(self._on_model_changed)
        if self._calibration_model is not None:
            self._calibration_model.calibration_changed.connect(self._on_model_changed)
            self._calibration_model.grid_changed.connect(self._on_model_changed)
        if self._center_template_model is not None:
            self._center_template_model.template_changed.connect(self._on_model_changed)
        if self._anomaly_constraint_model is not None:
            self._anomaly_constraint_model.constraints_changed.connect(
                self._on_model_changed
            )

    # ------------------------------------------------------------------ #
    # Properties (read-only for AppWindow)
    # ------------------------------------------------------------------ #

    @property
    def is_dirty(self) -> bool:
        return self._is_dirty

    @property
    def project_dir(self) -> Optional[str]:
        return self._project_dir

    @property
    def project_name(self) -> str:
        return self._project_name

    @property
    def has_project(self) -> bool:
        return self._project_dir is not None

    @property
    def autosave_manager(self) -> AutosaveManager:
        return self._autosave_manager

    def get_session_seconds(self) -> float:
        """Return cumulative seconds spent in this project, including the current session."""
        if self._session_start is None:
            return self._accumulated_seconds
        return self._accumulated_seconds + (time.monotonic() - self._session_start)

    # ------------------------------------------------------------------ #
    # Dirty tracking
    # ------------------------------------------------------------------ #

    def mark_dirty(self) -> None:
        """Mark the project as having unsaved changes."""
        if self._loading:
            return
        if not self._is_dirty:
            self._is_dirty = True
            self.dirty_changed.emit(True)

    def clear_dirty(self) -> None:
        """Mark the project as clean (all changes saved)."""
        if self._is_dirty:
            self._is_dirty = False
            self.dirty_changed.emit(False)

    def _on_model_changed(self, *args) -> None:
        self.mark_dirty()

    # ------------------------------------------------------------------ #
    # New / Open / Save
    # ------------------------------------------------------------------ #

    def new_project(self) -> None:
        """Clear all state and reset project tracking.

        Callers should check is_dirty and prompt the user before calling this.
        """
        state = self._dataset_model.state
        state.clear()
        state.reset_classes()
        self._inference_model.state.clear()
        if self._center_template_model is not None:
            self._center_template_model.clear_template()

        self._project_dir = None
        self._project_name = ""
        self._created_at = None
        self._accumulated_seconds = 0.0
        self._session_start = None
        self._autosave_manager.stop()

        self._loading = True
        try:
            if self._calibration_model is not None:
                self._calibration_model.clear_calibration()
            self._dataset_model.beginResetModel()
            self._dataset_model.endResetModel()
        finally:
            self._loading = False

        self.clear_dirty()

    def open_project(self, annoproj_path: str) -> Tuple[dict, List[str]]:
        """Load and apply a project file.

        Performs a single atomic model reset after all state is populated,
        avoiding the double-reset flicker of calling load_folder first.
        Returns (project_data, warnings). Raises on file read errors.

        Callers should check is_dirty and prompt the user before calling this.

        Args:
            annoproj_path: Absolute path to the .annoproj file.

        Returns:
            Tuple of (project_data dict, list of warning strings).
            project_data contains 'inference.model_path' if a model was previously saved.

        Raises:
            FileNotFoundError, json.JSONDecodeError: On unreadable project file.
        """
        project_data = self._project_io.load_project(annoproj_path)
        image_dir = project_data.get("dataset", {}).get("image_dir", "")
        is_template = project_data.get("is_template", False)
        warnings: List[str] = []

        self._loading = True
        try:
            ds = self._dataset_model.state
            ds.clear()

            if is_template:
                ds.image_dir = ""
                ds.image_files = []
                self._inference_model.state.clear()
            elif image_dir and os.path.isdir(image_dir):
                ds.image_dir = image_dir
                ds.image_files = scan_images(image_dir)
            # A missing image_dir is not warned here: the view detects it
            # from project_data and offers relocation directly.

            calib_state = (
                self._calibration_model._state if self._calibration_model else None
            )
            center_template_state = (
                self._center_template_model._state
                if self._center_template_model
                else None
            )
            anomaly_constraint_state = (
                self._anomaly_constraint_model._state
                if self._anomaly_constraint_model
                else None
            )
            self._project_io.apply_project_to_states(
                project_data,
                ds,
                self._inference_model.state,
                calibration_state=calib_state,
                center_template_state=center_template_state,
                anomaly_constraint_state=anomaly_constraint_state,
            )
            if self._calibration_model is not None:
                self._calibration_model.calibration_changed.emit()
            if self._anomaly_constraint_model is not None:
                self._anomaly_constraint_model.constraints_changed.emit()
                self._calibration_model.grid_changed.emit()
            if self._center_template_model is not None:
                self._center_template_model.template_changed.emit()
                logger.debug(
                    "Center template state applied from project: enabled=%s path=%s",
                    self._center_template_model.enabled(),
                    self._center_template_model.template_path(),
                )

            # Single reset — views see the complete state on first refresh
            self._dataset_model.beginResetModel()
            self._dataset_model.endResetModel()
        finally:
            self._loading = False

        orphaned = self._orphaned_filenames()
        if orphaned:
            sample = sorted(orphaned)[:5]
            tail = f" and {len(orphaned) - 5} more" if len(orphaned) > 5 else ""
            warnings.append(
                f"{len(orphaned)} annotation(s) exist for files not in the image "
                f"directory ({', '.join(sample)}{tail}). "
                "They will be dropped on the next save."
            )

        if is_template:
            self._project_dir = None
            self._autosave_manager.stop()
        else:
            self._project_dir = str(Path(annoproj_path).parent)
            self._autosave_manager.set_project_dir(self._project_dir)
        self._project_name = project_data.get("project_name", Path(annoproj_path).stem)
        self._created_at = project_data.get("created_at")
        self._accumulated_seconds = project_data.get("session_seconds", 0.0)
        self._session_start = time.monotonic()
        self.clear_dirty()
        self.project_opened.emit(self._project_name)

        return project_data, warnings

    def save_project(self) -> str:
        """Save to the current project directory.

        Returns the absolute path to the written .annoproj file.
        Raises ValueError if no project directory is set (call save_project_as first).
        """
        if not self._project_dir:
            raise ValueError("No project directory set. Use save_project_as first.")
        return self._write_project(self._project_dir, self._project_name)

    def save_project_as(self, project_dir: str, project_name: str) -> str:
        """Save to a new location and update the current project tracking.

        Returns the absolute path to the written .annoproj file.
        """
        is_new_session = self._session_start is None
        self._project_dir = project_dir
        self._project_name = project_name
        if is_new_session:
            self._session_start = time.monotonic()
        path = self._write_project(project_dir, project_name)
        self._autosave_manager.set_project_dir(project_dir)
        if is_new_session:
            self.project_opened.emit(self._project_name)
        return path

    def switch_active_model(self, key: str) -> None:
        """Switch which cached model's results are active.

        Performs the in-memory swap, then loads the newly-active model's
        cached heatmaps from disk if this project has a saved cache for it.
        Does not check for or prompt about unsaved changes on the outgoing
        model — callers (the view) must check
        ``inference_model.is_score_maps_dirty()`` and handle that before
        calling this, since showing a dialog isn't this layer's job.

        Args:
            key (str): Model key to activate. Must already be registered via
                ``inference_model.register_model()``.
        """
        if self._inference_controller is not None:
            self._inference_controller.switch_model(key)
        else:
            self._inference_model.switch_active_model(key)

        if not self._project_dir:
            return
        score_maps_file = self._inference_model.get_score_maps_file(key)
        if not score_maps_file:
            return
        npz_path = os.path.join(self._project_dir, score_maps_file)
        score_maps = self._project_io.load_model_scoremaps(npz_path)
        if score_maps:
            self._inference_model.load_score_maps_into_active(score_maps)

    def _write_project(self, project_dir: str, project_name: str) -> str:
        orphaned = self._orphaned_filenames()
        if orphaned:
            logger.warning(
                "Saving with %d orphaned annotation(s) — they will be dropped: %s",
                len(orphaned),
                sorted(orphaned),
            )

        calib_state = (
            self._calibration_model._state if self._calibration_model else None
        )
        center_template_state = (
            self._center_template_model._state if self._center_template_model else None
        )
        anomaly_constraint_state = (
            self._anomaly_constraint_model._state
            if self._anomaly_constraint_model
            else None
        )
        path = self._project_io.save_project(
            project_dir=project_dir,
            project_name=project_name,
            dataset_state=self._dataset_model.state,
            inference_state=self._inference_model.state,
            created_at=self._created_at,
            save_score_maps=True,
            calibration_state=calib_state,
            center_template_state=center_template_state,
            anomaly_constraint_state=anomaly_constraint_state,
            session_seconds=self.get_session_seconds(),
        )
        if self._created_at is None:
            self._created_at = datetime.now(timezone.utc).isoformat()
        self.clear_dirty()
        self.project_saved.emit(path)
        return path

    # ------------------------------------------------------------------ #
    # Autosave
    # ------------------------------------------------------------------ #

    def _do_autosave(self, project_dir: str) -> None:
        """Write an autosave snapshot. Score maps are excluded — manual save only."""
        if not self._is_dirty and self._session_start is None:
            return
        autosave_dir = os.path.join(project_dir, "autosave")
        try:
            calib_state = (
                self._calibration_model._state if self._calibration_model else None
            )
            center_template_state = (
                self._center_template_model._state
                if self._center_template_model
                else None
            )
            anomaly_constraint_state = (
                self._anomaly_constraint_model._state
                if self._anomaly_constraint_model
                else None
            )
            path = self._project_io.save_project(
                project_dir=autosave_dir,
                project_name=f"{self._project_name}.autosave",
                dataset_state=self._dataset_model.state,
                inference_state=self._inference_model.state,
                created_at=self._created_at,
                save_score_maps=False,
                calibration_state=calib_state,
                center_template_state=center_template_state,
                anomaly_constraint_state=anomaly_constraint_state,
                session_seconds=self.get_session_seconds(),
            )
            main_annoproj = os.path.join(project_dir, f"{self._project_name}.annoproj")
            if os.path.exists(main_annoproj):
                self._project_io.patch_session_seconds(
                    main_annoproj, self.get_session_seconds()
                )
            self.autosave_written.emit(path)
        except Exception as exc:
            self.autosave_failed.emit(str(exc))

    # ------------------------------------------------------------------ #
    # Relocate images
    # ------------------------------------------------------------------ #

    def relocate_images(self, new_dir: str) -> None:
        """Point to a new image directory without clearing annotations.

        Updates image_dir and image_files in state, then emits a model
        reset so views refresh. Annotations are preserved — only the
        images list changes. Files in the new directory that don't have
        annotations will appear with none; annotations for filenames not
        present in the new directory become orphaned.

        Args:
            new_dir: Absolute path to the new image directory.

        Raises:
            OSError: If new_dir cannot be listed.
        """
        state = self._dataset_model.state
        state.image_dir = new_dir
        state.image_files = scan_images(new_dir)

        self._loading = True
        try:
            self._dataset_model.beginResetModel()
            self._dataset_model.endResetModel()
        finally:
            self._loading = False

        self.mark_dirty()

    # ------------------------------------------------------------------ #
    # Template export
    # ------------------------------------------------------------------ #

    def export_template(self) -> str:
        """Export a settings-only .annoproj to <project_dir>/template/template.annoproj.

        Also copies the center template image into the subfolder if one is set.
        Returns the absolute path to the written .annoproj file.

        Raises:
            ValueError: If no project has been saved yet (no project directory).
        """
        if not self._project_dir:
            raise ValueError(
                "No project directory set. Save the project before exporting a template."
            )
        template_dir = os.path.join(self._project_dir, "template")
        output_path = os.path.join(template_dir, "template.annoproj")
        calib_state = (
            self._calibration_model._state if self._calibration_model else None
        )
        center_template_state = (
            self._center_template_model._state if self._center_template_model else None
        )
        anomaly_constraint_state = (
            self._anomaly_constraint_model._state
            if self._anomaly_constraint_model
            else None
        )
        self._project_io.export_template(
            template_path=output_path,
            project_name=self._project_name or "template",
            dataset_state=self._dataset_model.state,
            calibration_state=calib_state,
            center_template_state=center_template_state,
            anomaly_constraint_state=anomaly_constraint_state,
        )
        return output_path

    # ------------------------------------------------------------------ #
    # COCO standalone export
    # ------------------------------------------------------------------ #

    def export_coco(self, coco_path: str) -> None:
        """Export current annotations as a standalone COCO JSON file."""
        self._project_io.export_coco(coco_path, self._dataset_model.state)

    # ------------------------------------------------------------------ #
    # Orphaned annotation detection
    # ------------------------------------------------------------------ #

    def orphaned_annotations_warning(self) -> str:
        """Return a user-facing warning string if orphaned annotations exist.

        Orphaned annotations are those keyed to filenames not present in the
        current image_files list. They survive in state but are silently
        dropped from the COCO file on the next save.
        """
        orphaned = self._orphaned_filenames()
        if not orphaned:
            return ""
        sample = sorted(orphaned)[:5]
        tail = f" and {len(orphaned) - 5} more" if len(orphaned) > 5 else ""
        return (
            f"{len(orphaned)} annotation(s) exist for files not in the current "
            f"image directory ({', '.join(sample)}{tail}). "
            "These will be permanently dropped on save. Continue?"
        )

    def _orphaned_filenames(self) -> List[str]:
        state = self._dataset_model.state
        image_set = set(state.image_files)
        return [f for f in state.annotations if f not in image_set]
