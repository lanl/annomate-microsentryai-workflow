from PySide6.QtCore import QAbstractTableModel, QModelIndex, QSortFilterProxyModel, Qt

from core.utils.geometry import polygon_area


class AnnotationColumns:
    COLOR = 0
    CLASS = 1
    VERTICES = 2
    AREA = 3
    VISIBILITY = 4
    DELETE = 5


ANNOTATION_INDEX_ROLE = Qt.UserRole + 11
SORT_ROLE = Qt.UserRole + 12
COLOR_ROLE = Qt.UserRole + 13
VISIBLE_ROLE = Qt.UserRole + 14

_HEADERS = ["", "Class", "Points", "Area", "", ""]
_TOOLTIPS = {
    AnnotationColumns.COLOR: "Annotation class color",
    AnnotationColumns.CLASS: "Annotation class",
    AnnotationColumns.VERTICES: "Point count",
    AnnotationColumns.AREA: "Polygon area in current calibration units",
    AnnotationColumns.VISIBILITY: "Show or hide this annotation",
    AnnotationColumns.DELETE: "Delete annotation",
}


class AnnotationTableModel(QAbstractTableModel):
    """Read/edit table model for annotations on the current image."""

    def __init__(self, dataset_model, calibration_model=None, parent=None) -> None:
        super().__init__(parent)
        self._dataset_model = dataset_model
        self._calibration_model = calibration_model
        self._current_row = -1
        self._annotations = []
        self._dataset_model.dataChanged.connect(self._on_dataset_data_changed)
        if self._calibration_model is not None:
            self._calibration_model.calibration_changed.connect(
                self._on_calibration_changed
            )

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.isValid():
            return 0
        return len(self._annotations)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.isValid():
            return 0
        return len(_HEADERS)

    def flags(self, index: QModelIndex) -> Qt.ItemFlag:
        if not index.isValid():
            return Qt.NoItemFlags
        flags = Qt.ItemIsEnabled | Qt.ItemIsSelectable
        if index.column() == AnnotationColumns.CLASS:
            flags |= Qt.ItemIsEditable
        return flags

    def headerData(
        self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole
    ) -> object:
        if orientation != Qt.Horizontal or not (0 <= section < len(_HEADERS)):
            return None
        if role == Qt.DisplayRole:
            if section == AnnotationColumns.AREA:
                return f"Area ({self._area_unit()})"
            return _HEADERS[section]
        if role == Qt.ToolTipRole:
            return _TOOLTIPS.get(section)
        return None

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> object:
        if not index.isValid() or not (0 <= index.row() < self.rowCount()):
            return None

        row = index.row()
        col = index.column()
        annotation = self._annotations[row]
        name = annotation["category_name"]

        if role == ANNOTATION_INDEX_ROLE:
            return row
        if role == SORT_ROLE:
            return self.sort_value(row, col)
        if role == COLOR_ROLE and col == AnnotationColumns.COLOR:
            return self._dataset_model.get_class_color(name)
        if role == VISIBLE_ROLE:
            return annotation.get("visible", True)
        if role in (Qt.DisplayRole, Qt.EditRole):
            return self._display(annotation, col)
        if role == Qt.ToolTipRole:
            return self._tooltip(annotation, col)
        if role == Qt.TextAlignmentRole:
            return self._alignment(col)

        return None

    def setData(
        self, index: QModelIndex, value: object, role: int = Qt.EditRole
    ) -> bool:
        if (
            role != Qt.EditRole
            or not index.isValid()
            or index.column() != AnnotationColumns.CLASS
        ):
            return False
        name = str(value).strip().lower()
        if not name or name not in self._dataset_model.get_class_names():
            return False
        annotation_idx = index.row()
        if not (0 <= annotation_idx < self.rowCount()):
            return False
        if self._annotations[annotation_idx]["category_name"] == name:
            return False
        self._dataset_model.update_annotation_class(
            self._current_row, annotation_idx, name
        )
        return True

    def set_current_row(self, row: int) -> None:
        self._current_row = row
        self.refresh()

    def refresh(self) -> None:
        self.beginResetModel()
        self._annotations = (
            list(self._dataset_model.get_annotations(self._current_row))
            if self._current_row >= 0
            else []
        )
        self.endResetModel()

    def annotation_index(self, row: int) -> int:
        if not (0 <= row < self.rowCount()):
            return -1
        return row

    def sort_value(self, row: int, col: int):
        if not (0 <= row < self.rowCount()):
            return None
        annotation = self._annotations[row]
        if col == AnnotationColumns.COLOR:
            return tuple(
                self._dataset_model.get_class_color(annotation["category_name"])
            )
        if col == AnnotationColumns.CLASS:
            return annotation["category_name"].casefold()
        if col == AnnotationColumns.VERTICES:
            return len(annotation.get("polygon", []))
        if col == AnnotationColumns.AREA:
            return self._area_value(annotation)
        return row

    def tie_break_value(self, row: int) -> int:
        return row

    def class_names(self) -> list:
        return self._dataset_model.get_class_names()

    def _on_dataset_data_changed(self, top_left, bottom_right, roles=None) -> None:
        if self._current_row < 0:
            return
        if top_left.row() <= self._current_row <= bottom_right.row():
            self.refresh()

    def _display(self, annotation: dict, col: int) -> str:
        if col == AnnotationColumns.COLOR:
            return ""
        if col == AnnotationColumns.CLASS:
            return annotation["category_name"]
        if col == AnnotationColumns.VERTICES:
            return str(len(annotation.get("polygon", [])))
        if col == AnnotationColumns.AREA:
            return self._format_area_value(self._area_value(annotation))
        if col == AnnotationColumns.VISIBILITY:
            return "Hide" if annotation.get("visible", True) else "Show"
        if col == AnnotationColumns.DELETE:
            return "Delete"
        return ""

    def _tooltip(self, annotation: dict, col: int) -> str:
        name = annotation["category_name"]
        if col == AnnotationColumns.COLOR:
            r, g, b = self._dataset_model.get_class_color(name)
            return f"{name}: rgb({r}, {g}, {b})"
        if col == AnnotationColumns.DELETE:
            return "Delete annotation"
        if col == AnnotationColumns.VISIBILITY:
            action = "Hide" if annotation.get("visible", True) else "Show"
            return f"{action} annotation"
        return self._display(annotation, col) or (_TOOLTIPS.get(col) or "")

    def _alignment(self, col: int) -> Qt.AlignmentFlag:
        if col in (AnnotationColumns.VERTICES, AnnotationColumns.AREA):
            return Qt.AlignRight | Qt.AlignVCenter
        if col in (
            AnnotationColumns.COLOR,
            AnnotationColumns.VISIBILITY,
            AnnotationColumns.DELETE,
        ):
            return Qt.AlignCenter
        return Qt.AlignLeft | Qt.AlignVCenter

    def _area_value(self, annotation: dict) -> float:
        scale = 1.0
        if self._calibration_model is not None and self._calibration_model.has_scale():
            scale = float(self._calibration_model.scale())
        return polygon_area(annotation.get("polygon", [])) * scale * scale

    def _format_area_value(self, area: float) -> str:
        if area == 0:
            return "0"
        if abs(area) < 1:
            return f"{area:.6g}"
        rounded = round(area)
        if abs(rounded) < 1_000_000:
            return str(rounded)
        return f"{rounded:.6g}"

    def _area_unit(self) -> str:
        if self._calibration_model is None or not self._calibration_model.has_scale():
            return "px"
        return self._calibration_model.unit()

    def _on_calibration_changed(self) -> None:
        self.headerDataChanged.emit(
            Qt.Horizontal, AnnotationColumns.AREA, AnnotationColumns.AREA
        )
        if self.rowCount() == 0:
            return
        self.dataChanged.emit(
            self.index(0, AnnotationColumns.AREA),
            self.index(self.rowCount() - 1, AnnotationColumns.AREA),
            [Qt.DisplayRole, SORT_ROLE, Qt.ToolTipRole],
        )


class AnnotationSortProxyModel(QSortFilterProxyModel):
    """Type-aware proxy for current-image annotation sorting."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setDynamicSortFilter(True)
        self.setSortCaseSensitivity(Qt.CaseInsensitive)

    def lessThan(self, left: QModelIndex, right: QModelIndex) -> bool:
        model = self.sourceModel()
        if model is None:
            return super().lessThan(left, right)

        col = left.column()
        left_value = model.sort_value(left.row(), col)
        right_value = model.sort_value(right.row(), col)

        if left_value == right_value:
            return model.tie_break_value(left.row()) < model.tie_break_value(
                right.row()
            )
        return left_value < right_value
