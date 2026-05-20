from pathlib import Path

from PySide6.QtCore import QAbstractTableModel, QModelIndex, QSortFilterProxyModel, Qt
from PySide6.QtGui import QColor, QBrush, QFont


class NavigatorColumns:
    STATUS = 0
    IMG_ID = 1
    ANNOTS = 2
    DECISION = 3
    SCORE = 4
    CLASS = 5


SOURCE_ROW_ROLE = Qt.UserRole + 1
SORT_ROLE = Qt.UserRole + 2
STATUS_COLOR_ROLE = Qt.UserRole + 3


_HEADERS = ["", "Img ID", "Annots", "Decision", "Score", "Class"]
_TOOLTIPS = {
    NavigatorColumns.STATUS: "Review status",
    NavigatorColumns.IMG_ID: "Image identifier",
    NavigatorColumns.ANNOTS: "Annotation count",
    NavigatorColumns.DECISION: "Review decision",
    NavigatorColumns.SCORE: "MicroSentry anomaly score",
    NavigatorColumns.CLASS: "MicroSentry class",
}
_DECISION_LABELS = {"accept": "Accept", "reject": "Reject"}
_DECISION_SORT = {None: 0, "": 0, "accept": 1, "reject": 2}
_CLASS_SORT = {"": 0, "ANOMALY": 1, "NORMAL": 2}


class NavigatorTableModel(QAbstractTableModel):
    """Read-only table model for the Dataset Navigator view."""

    def __init__(self, dataset_model, inference_model=None, parent=None) -> None:
        super().__init__(parent)
        self._dataset_model = dataset_model
        self._inference_model = inference_model

        self._dataset_model.modelReset.connect(self._on_source_reset)
        self._dataset_model.dataChanged.connect(self._on_source_data_changed)

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.isValid():
            return 0
        return self._dataset_model.rowCount()

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

        if role == SOURCE_ROW_ROLE:
            return row
        if role == SORT_ROLE:
            return self.sort_value(row, col)
        if role == STATUS_COLOR_ROLE and col == NavigatorColumns.STATUS:
            return "#4caf50" if self._dataset_model.is_reviewed(row) else "#ff9800"
        if role == Qt.ToolTipRole:
            return self._tooltip(row, col)
        if role == Qt.TextAlignmentRole:
            return self._alignment(col)
        if role == Qt.FontRole:
            return self._font(row, col)
        if role == Qt.ForegroundRole:
            return self._foreground(row, col)
        if role == Qt.DisplayRole:
            return self._display(row, col)

        return None

    def source_row(self, index: QModelIndex) -> int:
        if not index.isValid():
            return -1
        return index.row()

    def sort_value(self, row: int, col: int):
        if col == NavigatorColumns.STATUS:
            return 1 if self._dataset_model.is_reviewed(row) else 0
        if col == NavigatorColumns.IMG_ID:
            return self._image_stem(row).casefold()
        if col == NavigatorColumns.ANNOTS:
            return self._dataset_model.get_annotation_count(row)
        if col == NavigatorColumns.DECISION:
            return _DECISION_SORT.get(self._dataset_model.get_review_decision(row), 0)
        if col == NavigatorColumns.SCORE:
            score = self._score(row)
            return None if score is None else float(score)
        if col == NavigatorColumns.CLASS:
            label = self._label(row) or ""
            return _CLASS_SORT.get(label, label.casefold())
        return ""

    def tie_break_value(self, row: int) -> str:
        return self._image_stem(row).casefold()

    def notify_inference_changed(self, row: int) -> None:
        if not (0 <= row < self.rowCount()):
            return
        self.dataChanged.emit(
            self.index(row, NavigatorColumns.SCORE),
            self.index(row, NavigatorColumns.CLASS),
            [Qt.DisplayRole, Qt.ToolTipRole, SORT_ROLE, Qt.ForegroundRole, Qt.FontRole],
        )

    def refresh_inference(self) -> None:
        if self.rowCount() == 0:
            return
        self.dataChanged.emit(
            self.index(0, NavigatorColumns.SCORE),
            self.index(self.rowCount() - 1, NavigatorColumns.CLASS),
            [Qt.DisplayRole, Qt.ToolTipRole, SORT_ROLE, Qt.ForegroundRole, Qt.FontRole],
        )

    def _on_source_reset(self) -> None:
        self.beginResetModel()
        self.endResetModel()

    def _on_source_data_changed(self, top_left, bottom_right, roles=None) -> None:
        if self.rowCount() == 0:
            return
        top = max(0, top_left.row())
        bottom = min(self.rowCount() - 1, bottom_right.row())
        if top > bottom:
            return
        self.dataChanged.emit(
            self.index(top, 0),
            self.index(bottom, self.columnCount() - 1),
            [Qt.DisplayRole, Qt.ToolTipRole, SORT_ROLE, STATUS_COLOR_ROLE],
        )

    def _display(self, row: int, col: int) -> str:
        if col == NavigatorColumns.STATUS:
            return ""
        if col == NavigatorColumns.IMG_ID:
            return self._image_stem(row)
        if col == NavigatorColumns.ANNOTS:
            count = self._dataset_model.get_annotation_count(row)
            return str(count) if count > 0 else ""
        if col == NavigatorColumns.DECISION:
            return _DECISION_LABELS.get(
                self._dataset_model.get_review_decision(row), ""
            )
        if col == NavigatorColumns.SCORE:
            score = self._score(row)
            return "" if score is None else f"{score:.2f}"
        if col == NavigatorColumns.CLASS:
            return self._label(row) or ""
        return ""

    def _tooltip(self, row: int, col: int) -> str:
        if col == NavigatorColumns.STATUS:
            return "Reviewed" if self._dataset_model.is_reviewed(row) else "In Review"
        value = self._display(row, col)
        return value or (_TOOLTIPS.get(col) or "")

    def _alignment(self, col: int) -> Qt.AlignmentFlag:
        if col in (NavigatorColumns.ANNOTS, NavigatorColumns.SCORE):
            return Qt.AlignRight | Qt.AlignVCenter
        if col in (
            NavigatorColumns.STATUS,
            NavigatorColumns.DECISION,
            NavigatorColumns.CLASS,
        ):
            return Qt.AlignCenter
        return Qt.AlignLeft | Qt.AlignVCenter

    def _font(self, row: int, col: int) -> QFont | None:
        if col == NavigatorColumns.DECISION and self._dataset_model.get_review_decision(
            row
        ):
            font = QFont()
            font.setBold(True)
            return font
        if col == NavigatorColumns.CLASS and self._label(row):
            font = QFont()
            font.setBold(True)
            return font
        return None

    def _foreground(self, row: int, col: int) -> QBrush | None:
        if col == NavigatorColumns.DECISION:
            decision = self._dataset_model.get_review_decision(row)
            if decision == "accept":
                return QBrush(QColor("#4caf50"))
            if decision == "reject":
                return QBrush(QColor("#f44336"))
        if col == NavigatorColumns.CLASS:
            label = self._label(row)
            if label == "ANOMALY":
                return QBrush(QColor("#f44336"))
            if label == "NORMAL":
                return QBrush(QColor("#4caf50"))
        return None

    def _image_stem(self, row: int) -> str:
        return Path(self._dataset_model.get_image_filename(row)).stem

    def _image_path(self, row: int) -> str:
        return self._dataset_model.get_image_path(row)

    def _score(self, row: int) -> float | None:
        if self._inference_model is None:
            return None
        return self._inference_model.get_score(self._image_path(row))

    def _label(self, row: int) -> str | None:
        if self._inference_model is None:
            return None
        return self._inference_model.get_label(self._image_path(row))


class NavigatorSortProxyModel(QSortFilterProxyModel):
    """Type-aware proxy for navigator column sorting."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setDynamicSortFilter(True)
        self.setSortCaseSensitivity(Qt.CaseInsensitive)

    def lessThan(self, left: QModelIndex, right: QModelIndex) -> bool:
        model = self.sourceModel()
        if model is None:
            return super().lessThan(left, right)

        col = left.column()
        left_row = left.row()
        right_row = right.row()
        left_value = model.sort_value(left_row, col)
        right_value = model.sort_value(right_row, col)

        if col == NavigatorColumns.SCORE:
            left_missing = left_value is None
            right_missing = right_value is None
            if left_missing != right_missing:
                if self.sortOrder() == Qt.DescendingOrder:
                    return left_missing
                return right_missing

        if left_value == right_value:
            return model.tie_break_value(left_row) < model.tie_break_value(right_row)
        return left_value < right_value
