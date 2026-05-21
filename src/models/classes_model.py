from PySide6.QtCore import QAbstractTableModel, QModelIndex, QSortFilterProxyModel, Qt


class ClassColumns:
    COLOR = 0
    CLASS = 1
    IMAGE = 2
    TOTAL = 3
    VISIBILITY = 4
    DELETE = 5


CLASS_NAME_ROLE = Qt.UserRole + 1
SORT_ROLE = Qt.UserRole + 2
COLOR_ROLE = Qt.UserRole + 3
VISIBLE_ROLE = Qt.UserRole + 4

_HEADERS = ["", "Class", "Img", "Tot", "", ""]
_TOOLTIPS = {
    ClassColumns.COLOR: "Annotation class color",
    ClassColumns.CLASS: "Annotation class name",
    ClassColumns.IMAGE: "Class count for this image",
    ClassColumns.TOTAL: "Class count for the whole dataset",
    ClassColumns.VISIBILITY: "Show or hide this class's annotations",
    ClassColumns.DELETE: "Delete annotation class",
}


class ClassTableModel(QAbstractTableModel):
    """Read-only table model for the annotation class registry."""

    def __init__(self, dataset_model, parent=None) -> None:
        super().__init__(parent)
        self._dataset_model = dataset_model
        self._current_row = -1
        self._class_names = self._dataset_model.get_class_names()

        self._dataset_model.modelReset.connect(self.refresh_classes)
        self._dataset_model.dataChanged.connect(self._on_source_data_changed)
        self._dataset_model.classVisibilityChanged.connect(
            self._on_class_visibility_changed
        )

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.isValid():
            return 0
        return len(self._class_names)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.isValid():
            return 0
        return len(_HEADERS)

    def flags(self, index: QModelIndex) -> Qt.ItemFlag:
        if not index.isValid():
            return Qt.NoItemFlags
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable

    def headerData(
        self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole
    ) -> object:
        if orientation != Qt.Horizontal or not (0 <= section < len(_HEADERS)):
            return None
        if role == Qt.DisplayRole:
            return _HEADERS[section]
        if role == Qt.ToolTipRole:
            return _TOOLTIPS.get(section)
        return None

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> object:
        if not index.isValid() or not (0 <= index.row() < self.rowCount()):
            return None

        row = index.row()
        col = index.column()
        name = self._class_names[row]

        if role == CLASS_NAME_ROLE:
            return name
        if role == SORT_ROLE:
            return self.sort_value(row, col)
        if role == COLOR_ROLE and col == ClassColumns.COLOR:
            return self._dataset_model.get_class_color(name)
        if role == VISIBLE_ROLE:
            return self._dataset_model.is_class_visible(name)
        if role == Qt.ToolTipRole:
            return self._tooltip(name, col)
        if role == Qt.TextAlignmentRole:
            return self._alignment(col)
        if role == Qt.DisplayRole:
            return self._display(name, col)

        return None

    def set_current_row(self, row: int) -> None:
        if row == self._current_row:
            return
        self._current_row = row
        self.refresh_counts()

    def refresh_classes(self) -> None:
        self.beginResetModel()
        self._class_names = self._dataset_model.get_class_names()
        self.endResetModel()

    def refresh_counts(self) -> None:
        if self.rowCount() == 0:
            return
        self.dataChanged.emit(
            self.index(0, ClassColumns.IMAGE),
            self.index(self.rowCount() - 1, ClassColumns.TOTAL),
            [Qt.DisplayRole, SORT_ROLE, Qt.ToolTipRole],
        )

    def class_name(self, row: int) -> str:
        if not (0 <= row < self.rowCount()):
            return ""
        return self._class_names[row]

    def sort_value(self, row: int, col: int):
        name = self.class_name(row)
        if col == ClassColumns.COLOR:
            return tuple(self._dataset_model.get_class_color(name))
        if col == ClassColumns.CLASS:
            return name.casefold()
        if col == ClassColumns.IMAGE:
            return self._count_image_annotations(name)
        if col == ClassColumns.TOTAL:
            return self._count_total_annotations(name)
        return row

    def tie_break_value(self, row: int) -> str:
        return self.class_name(row).casefold()

    def _on_source_data_changed(self, top_left, bottom_right, roles=None) -> None:
        class_names = self._dataset_model.get_class_names()
        if class_names != self._class_names:
            self.refresh_classes()
            return
        if self.rowCount() == 0:
            return
        self.dataChanged.emit(
            self.index(0, 0),
            self.index(self.rowCount() - 1, self.columnCount() - 1),
            [Qt.DisplayRole, SORT_ROLE, COLOR_ROLE, VISIBLE_ROLE, Qt.ToolTipRole],
        )

    def _on_class_visibility_changed(self, name: str, visible: bool) -> None:
        row = self._class_names.index(name) if name in self._class_names else -1
        if row < 0:
            return
        self.dataChanged.emit(
            self.index(row, ClassColumns.VISIBILITY),
            self.index(row, ClassColumns.VISIBILITY),
            [Qt.DisplayRole, VISIBLE_ROLE, Qt.ToolTipRole],
        )

    def _display(self, name: str, col: int) -> str:
        if col == ClassColumns.COLOR:
            return ""
        if col == ClassColumns.CLASS:
            return name
        if col == ClassColumns.IMAGE:
            return str(self._count_image_annotations(name))
        if col == ClassColumns.TOTAL:
            return str(self._count_total_annotations(name))
        if col == ClassColumns.VISIBILITY:
            return "Hide" if self._dataset_model.is_class_visible(name) else "Show"
        if col == ClassColumns.DELETE:
            return "Delete"
        return ""

    def _tooltip(self, name: str, col: int) -> str:
        if col == ClassColumns.COLOR:
            r, g, b = self._dataset_model.get_class_color(name)
            return f"{name}: rgb({r}, {g}, {b})"
        if col == ClassColumns.DELETE:
            return f'Delete "{name}"'
        if col == ClassColumns.VISIBILITY:
            action = "Hide" if self._dataset_model.is_class_visible(name) else "Show"
            return f'{action} annotations for "{name}"'
        return self._display(name, col) or (_TOOLTIPS.get(col) or "")

    def _alignment(self, col: int) -> Qt.AlignmentFlag:
        if col in (ClassColumns.IMAGE, ClassColumns.TOTAL):
            return Qt.AlignRight | Qt.AlignVCenter
        if col in (ClassColumns.COLOR, ClassColumns.VISIBILITY, ClassColumns.DELETE):
            return Qt.AlignCenter
        return Qt.AlignLeft | Qt.AlignVCenter

    def _count_image_annotations(self, class_name: str) -> int:
        if self._current_row < 0:
            return 0
        return sum(
            1
            for annotation in self._dataset_model.get_annotations(self._current_row)
            if annotation["category_name"] == class_name
        )

    def _count_total_annotations(self, class_name: str) -> int:
        total = 0
        for row in range(self._dataset_model.rowCount()):
            for annotation in self._dataset_model.get_annotations(row):
                if annotation["category_name"] == class_name:
                    total += 1
        return total


class ClassSortProxyModel(QSortFilterProxyModel):
    """Type-aware proxy for annotation class sorting."""

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
