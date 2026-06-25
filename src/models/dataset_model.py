import os
import logging
from pathlib import Path

from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex, Signal
from PySide6.QtGui import QColor, QBrush

from core.states.dataset_state import DatasetState
from core.utils.geometry import polygon_area

logger = logging.getLogger("AnnoMate.DatasetModel")


class DatasetTableModel(QAbstractTableModel):
    """Qt model layer for the dataset image list.

    Owns the :class:`~PySide6.QtCore.QAbstractTableModel` interface and
    exposes a typed query/command API so Views never touch
    :class:`~core.states.dataset_state.DatasetState` directly.

    Color rule: colors are stored as ``(r, g, b)`` tuples in
    :class:`~core.states.dataset_state.DatasetState` and in all
    controller/domain code. :class:`~PySide6.QtGui.QColor` is only
    constructed here, at the Qt boundary, for display roles.

    Attributes:
        state (DatasetState): The underlying domain state this model wraps.
        headers (list[str]): Column header labels — ``["Filename", "Status"]``.
    """

    classVisibilityChanged = Signal(str, bool)

    def __init__(self, state: DatasetState, parent: object = None) -> None:
        """Initialize DatasetTableModel with a domain state object.

        Args:
            state (DatasetState): The dataset state instance to wrap.
            parent (object): Optional Qt parent object. Defaults to ``None``.
        """
        super().__init__(parent)
        self.state = state
        self.headers = ["Filename", "Status"]

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        """Return the number of rows (images) in the model.

        Args:
            parent (QModelIndex): Unused; present to satisfy the Qt interface.
                Defaults to an invalid index.

        Returns:
            int: Number of image files currently loaded.
        """
        return len(self.state.image_files)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        """Return the number of columns in the model.

        Args:
            parent (QModelIndex): Unused; present to satisfy the Qt interface.
                Defaults to an invalid index.

        Returns:
            int: Number of header columns (always ``2``).
        """
        return len(self.headers)

    def headerData(
        self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole
    ) -> object:
        """Return header label for the given section and orientation.

        Args:
            section (int): Column index.
            orientation (Qt.Orientation): ``Qt.Horizontal`` returns column
                labels; ``Qt.Vertical`` returns ``None``.
            role (int): Qt item data role. Only ``Qt.DisplayRole`` is handled.
                Defaults to ``Qt.DisplayRole``.

        Returns:
            object: Column label string, or ``None`` for unsupported roles or
                orientations.
        """
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return self.headers[section]
        return None

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> object:
        """Return display or decoration data for a given cell.

        Handles two roles:

        - ``Qt.DisplayRole`` — column 0 returns the image stem; column 1
          returns ``"Reviewed"`` or ``"Pending"``.
        - ``Qt.BackgroundRole`` — column 1 returns a green brush for reviewed
          images and an orange brush for pending ones.

        Args:
            index (QModelIndex): Cell index being queried.
            role (int): Qt item data role. Defaults to ``Qt.DisplayRole``.

        Returns:
            object: The requested data value, or ``None`` for invalid indices
                or unhandled roles.
        """
        if not index.isValid() or not (0 <= index.row() < self.rowCount()):
            return None

        filename = self.state.image_files[index.row()]
        col = index.column()

        if role == Qt.DisplayRole:
            if col == 0:
                return Path(filename).stem
            elif col == 1:
                return "Reviewed" if self.state.is_reviewed(filename) else "Pending"

        elif role == Qt.BackgroundRole:
            if col == 1:
                if self.state.is_reviewed(filename):
                    return QBrush(QColor(210, 245, 210))
                else:
                    return QBrush(QColor(255, 235, 210))

        return None

    def load_folder(self, directory: str, files: list) -> None:
        """Replace the current image list with a new folder's contents.

        Clears all existing per-folder state via
        :meth:`~core.states.dataset_state.DatasetState.clear`, then sets the
        new directory and file list. Wraps the mutation in
        ``beginResetModel`` / ``endResetModel`` so all attached views refresh.

        Args:
            directory (str): Absolute path to the folder being loaded.
            files (list): Sorted list of image filenames within *directory*.
        """
        self.beginResetModel()
        self.state.clear()
        self.state.image_dir = directory
        self.state.image_files = files
        self.endResetModel()

    def add_annotation(
        self, row: int, category: str, polygon: list, thickness: float = 2.0
    ) -> None:
        """Append a polygon annotation to the image at *row*.

        Out-of-bounds *row* values are logged as errors and silently ignored.
        Emits ``dataChanged`` for the affected row on success.

        Args:
            row (int): Zero-based row index of the target image.
            category (str): Class name to assign to the annotation.
            polygon (list): Sequence of ``(x, y)`` coordinate pairs defining
                the polygon boundary.
            thickness (float): Line thickness for the annotation (default: 2.0).
        """
        if not (0 <= row < self.rowCount()):
            logger.error("Failed to add annotation: Row %d is out of bounds.", row)
            return

        filename = self.state.image_files[row]
        logger.debug(
            "Adding '%s' annotation to '%s' (%d points)",
            category,
            filename,
            len(polygon),
        )

        self.state.add_annotation(filename, category.lower(), polygon, thickness)
        self._emit_row(row)

    def update_annotation_thickness(
        self, row: int, annotation_idx: int, thickness: float
    ) -> None:
        """Update the line thickness of an existing annotation."""
        if not (0 <= row < self.rowCount()):
            return
        self.state.update_annotation_thickness(
            self.state.image_files[row], annotation_idx, thickness
        )
        self._emit_row(row)

    def update_annotation_class(
        self, row: int, annotation_idx: int, new_class: str
    ) -> None:
        """Change the class of an existing annotation."""
        if not (0 <= row < self.rowCount()):
            return
        self.state.update_annotation_class(
            self.state.image_files[row], annotation_idx, new_class
        )
        self._emit_row(row)

    def delete_annotation(self, row: int, annotation_idx: int) -> None:
        """Remove a specific annotation from the image at *row*.

        Out-of-bounds *row* values are silently ignored. Emits
        ``dataChanged`` for the affected row on success.

        Args:
            row (int): Zero-based row index of the target image.
            annotation_idx (int): Zero-based index of the annotation to
                remove within that image's annotation list.
        """
        if not (0 <= row < self.rowCount()):
            return

        filename = self.state.image_files[row]
        logger.debug(
            "Deleted annotation at index %d from '%s'", annotation_idx, filename
        )

        self.state.delete_annotation(self.state.image_files[row], annotation_idx)
        self._emit_row(row)

    def update_annotation_points(
        self, row: int, annotation_idx: int, points: list
    ) -> None:
        """Replace the polygon vertices of an existing annotation.

        Out-of-bounds *row* values are silently ignored. Emits
        ``dataChanged`` for the affected row on success.

        Args:
            row (int): Zero-based row index of the target image.
            annotation_idx (int): Zero-based index of the annotation to
                update within that image's annotation list.
            points (list): New sequence of ``(x, y)`` coordinate pairs.
        """
        if not (0 <= row < self.rowCount()):
            return
        self.state.update_annotation_points(
            self.state.image_files[row], annotation_idx, points
        )
        self._emit_row(row)

    def set_annotation_visible(
        self, row: int, annotation_idx: int, visible: bool
    ) -> None:
        """Set whether a specific annotation renders in the viewport."""
        if not (0 <= row < self.rowCount()):
            return
        self.state.set_annotation_visible(
            self.state.image_files[row], annotation_idx, visible
        )
        self._emit_row(row)

    def toggle_annotation_visibility(self, row: int, annotation_idx: int) -> bool:
        """Toggle annotation viewport visibility and return the new state."""
        visible = not self.is_annotation_visible(row, annotation_idx)
        self.set_annotation_visible(row, annotation_idx, visible)
        return visible

    def set_inspector(self, row: int, value: str) -> None:
        """Assign an inspector name to the image at *row*.

        Out-of-bounds *row* values are silently ignored. Emits
        ``dataChanged`` for the affected row on success.

        Args:
            row (int): Zero-based row index of the target image.
            value (str): Inspector's name or identifier.
        """
        if not (0 <= row < self.rowCount()):
            return
        self.state.set_inspector(self.state.image_files[row], value)
        self._emit_row(row)

    def set_note(self, row: int, value: str) -> None:
        """Attach a free-text note to the image at *row*.

        Out-of-bounds *row* values are silently ignored. Emits
        ``dataChanged`` for the affected row on success.

        Args:
            row (int): Zero-based row index of the target image.
            value (str): Note content to store.
        """
        if not (0 <= row < self.rowCount()):
            return
        self.state.set_note(self.state.image_files[row], value)
        self._emit_row(row)

    def add_class(self, name: str, color: tuple) -> bool:
        """Register a new class in the global class registry.

        Args:
            name (str): Class label to register.
            color (tuple): RGB color tuple to associate with the class.

        Returns:
            bool: ``True`` if the class was added; ``False`` if *name* was
                already registered.
        """
        name = name.lower()
        if name in self.state.class_names:
            return False
        self.state.add_class(name, color)
        return True

    def set_class_color(self, name: str, color: tuple) -> None:
        """Update an existing class's display color.

        Emits ``dataChanged`` for the entire table so all rows repaint with
        the new color. Does nothing if the model contains no rows.

        Args:
            name (str): Class label whose color should be updated.
            color (tuple): New RGB color tuple.
        """
        self.state.class_colors[name] = color
        if self.rowCount() > 0:
            self.dataChanged.emit(
                self.index(0, 0),
                self.index(self.rowCount() - 1, self.columnCount() - 1),
            )

    def set_class_visible(self, name: str, visible: bool) -> None:
        """Set whether annotations for *name* render in the viewport."""
        if name not in self.state.class_names:
            return
        visible = bool(visible)
        if self.state.is_class_visible(name) == visible:
            return
        self.state.set_class_visible(name, visible)
        self.classVisibilityChanged.emit(name, visible)

    def toggle_class_visibility(self, name: str) -> bool:
        """Toggle viewport visibility for *name* and return the new state."""
        visible = not self.is_class_visible(name)
        self.set_class_visible(name, visible)
        return visible

    def delete_class(self, name: str) -> None:
        """Remove a class and all annotations that reference it.

        Emits ``dataChanged`` for the entire table after deletion so all
        rows repaint. Does nothing to the view if no rows are loaded.

        Args:
            name (str): Class label to remove.
        """
        self.state.delete_class(name)
        if self.rowCount() > 0:
            self.dataChanged.emit(
                self.index(0, 0),
                self.index(self.rowCount() - 1, self.columnCount() - 1),
            )

    def sort_annotations(self, row: int) -> None:
        """Sort the annotations of the image at *row* by polygon area descending.

        Largest polygons appear first after sorting. Out-of-bounds *row*
        values are silently ignored. Emits ``dataChanged`` for the affected
        row on success.

        Args:
            row (int): Zero-based row index of the target image.
        """
        if not (0 <= row < self.rowCount()):
            return
        filename = self.state.image_files[row]
        annos = self.state.annotations.get(filename, [])
        annos.sort(key=lambda a: polygon_area(a["polygon"]), reverse=True)
        self._emit_row(row)

    # ------------------------------------------------------------------ #
    # Query API — Views must use these instead of accessing .state
    # ------------------------------------------------------------------ #

    def get_image_dir(self) -> str:
        """Return the path of the currently loaded image directory.

        Returns:
            str: Absolute path to the image folder, or an empty string if
                no folder has been loaded.
        """
        return self.state.image_dir

    def get_image_path(self, row: int) -> str:
        """Return the absolute file path for the image at *row*.

        Args:
            row (int): Zero-based row index of the target image.

        Returns:
            str: Absolute path constructed from the image directory and the
                filename at *row*.
        """
        return os.path.join(self.state.image_dir, self.state.image_files[row])

    def get_annotations(self, row: int) -> list:
        """Return the annotation list for the image at *row*.

        Args:
            row (int): Zero-based row index of the target image.

        Returns:
            list: List of annotation dicts (each with ``category_name`` and
                ``polygon`` keys), or an empty list for out-of-bounds *row*.
        """
        if not (0 <= row < self.rowCount()):
            return []
        return self.state.annotations.get(self.state.image_files[row], [])

    def is_annotation_visible(self, row: int, annotation_idx: int) -> bool:
        """Return whether a specific annotation should render."""
        if not (0 <= row < self.rowCount()):
            return True
        return self.state.is_annotation_visible(
            self.state.image_files[row], annotation_idx
        )

    def get_class_annotation_count(self, class_name: str) -> int:
        """Return the total number of annotations assigned to *class_name*."""
        total = 0
        for annotations in self.state.annotations.values():
            total += sum(
                1
                for annotation in annotations
                if annotation.get("category_name") == class_name
            )
        return total

    def get_class_names(self) -> list:
        """Return a copy of the ordered class name registry.

        Returns:
            list: List of class label strings in registration order.
        """
        return list(self.state.class_names)

    def get_class_color(self, class_name: str) -> tuple:
        """Return the RGB color tuple for *class_name*.

        Callers are responsible for converting the tuple to
        :class:`~PySide6.QtGui.QColor` at draw time.

        Args:
            class_name (str): Class label to look up.

        Returns:
            tuple: ``(r, g, b)`` color tuple, defaulting to
                ``(255, 255, 255)`` for unregistered class names.
        """
        return self.state.class_colors.get(class_name, (255, 255, 255))

    def is_class_visible(self, class_name: str) -> bool:
        """Return whether annotations for *class_name* should render."""
        return self.state.is_class_visible(class_name)

    def get_used_class_colors(self) -> list:
        """Return all currently assigned RGB color tuples.

        Returns:
            list: List of ``(r, g, b)`` tuples for every registered class,
                in registration order.
        """
        return list(self.state.class_colors.values())

    def get_inspector(self, row: int) -> str:
        """Return the inspector name assigned to the image at *row*.

        Args:
            row (int): Zero-based row index of the target image.

        Returns:
            str: Inspector name string, or an empty string if none is assigned
                or *row* is out of bounds.
        """
        if not (0 <= row < self.rowCount()):
            return ""
        return self.state.inspectors.get(self.state.image_files[row], "")

    def get_note(self, row: int) -> str:
        """Return the free-text note attached to the image at *row*.

        Args:
            row (int): Zero-based row index of the target image.

        Returns:
            str: Note string, or an empty string if none is set or *row* is
                out of bounds.
        """
        if not (0 <= row < self.rowCount()):
            return ""
        return self.state.notes.get(self.state.image_files[row], "")

    def is_reviewed(self, row: int) -> bool:
        """Return whether the image at *row* has been reviewed.

        Args:
            row (int): Zero-based row index of the target image.

        Returns:
            bool: ``True`` if the image has annotations, an inspector, or a
                note; ``False`` for out-of-bounds rows or unreviewed images.
        """
        if not (0 <= row < self.rowCount()):
            return False
        return self.state.is_reviewed(self.state.image_files[row])

    def set_review_decision(
        self, row: int, decision, omit_reason: str | None = None
    ) -> None:
        """Set the image-level review decision for the image at *row*.

        Args:
            row (int): Zero-based row index of the target image.
            decision (str | None): ``"accept"``, ``"reject"``, ``"omitted"``, or
                ``None`` to clear.
            omit_reason (str | None): Reason key when decision is ``"omitted"``.
        """
        if not (0 <= row < self.rowCount()):
            return
        self.state.set_review_decision(
            self.state.image_files[row], decision, omit_reason
        )
        self._emit_row(row)

    def get_review_decision(self, row: int):
        """Return the image-level review decision for the image at *row*, or None.

        Args:
            row (int): Zero-based row index of the target image.
        """
        if not (0 <= row < self.rowCount()):
            return None
        return self.state.get_review_decision(self.state.image_files[row])

    def get_omit_reason(self, row: int) -> str | None:
        """Return the omit reason key for the image at *row*, or None."""
        if not (0 <= row < self.rowCount()):
            return None
        return self.state.get_omit_reason(self.state.image_files[row])

    def get_navigation_warning(self, row: int) -> str | None:
        """Return the omit reason key if navigating away from *row* should warn the user.

        Returns ``"no_decision"`` when the image has annotations but no decision,
        ``"no_annotation"`` when it is rejected without annotations, or ``None``
        when no warning is needed.
        """
        if not (0 <= row < self.rowCount()):
            return None
        decision = self.get_review_decision(row)
        count = self.get_annotation_count(row)
        if count > 0 and not decision:
            return "no_decision"
        if decision == "reject" and count == 0:
            return "no_annotation"
        return None

    def get_annotation_count(self, row: int) -> int:
        """Return the number of polygon annotations for the image at *row*.

        Args:
            row (int): Zero-based row index of the target image.
        """
        if not (0 <= row < self.rowCount()):
            return 0
        return len(self.state.annotations.get(self.state.image_files[row], []))

    def get_image_filename(self, row: int) -> str:
        """Return the raw filename (basename) for the image at *row*.

        Args:
            row (int): Zero-based row index of the target image.

        Returns:
            str: Filename string, or an empty string for out-of-bounds rows.
        """
        if not (0 <= row < self.rowCount()):
            return ""
        return self.state.image_files[row]

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _emit_row(self, row: int) -> None:
        """Emit ``dataChanged`` for all columns of *row*.

        Args:
            row (int): Zero-based row index whose data has changed.
        """
        self.dataChanged.emit(
            self.index(row, 0),
            self.index(row, self.columnCount() - 1),
        )
