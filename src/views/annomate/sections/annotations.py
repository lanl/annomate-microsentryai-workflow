from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QFontMetrics
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from models.annotations_model import (
    ANNOTATION_INDEX_ROLE,
    COLOR_ROLE,
    VISIBLE_ROLE,
    AnnotationColumns,
    AnnotationSortProxyModel,
    AnnotationTableModel,
)

from views.icons import material_icon

from ._shared import _ClickableFrame

_DOT_W = 16
_ICON_BTN_SIZE = 16
_ICON_BTN_W = 28
_HEADER_FONT_PX = 10
_CELL_FONT_PX = 11
_COLUMN_TEXT_PADDING = 4
_VERTICES_HEADER_TEXT = "Pts"
_AREA_HEADER_TEXT = "Area"
_HEADER_STYLE = f"font-size: {_HEADER_FONT_PX}px; font-weight: bold; color: black;"


def _header_label_width(text: str) -> int:
    """Width needed to show *text* in the header font, plus a little breathing room.

    Points/Nodes and Area columns size themselves to their own header label
    instead of a hand-picked pixel number, so the class combo box (the only
    stretchy element in each row) always gets whatever space is left over.
    """
    font = QFont()
    font.setPixelSize(_HEADER_FONT_PX)
    font.setBold(True)
    return QFontMetrics(font).horizontalAdvance(text) + _COLUMN_TEXT_PADDING


def _cell_text_width(text: str) -> int:
    """Width needed for a numeric cell value without clipping."""
    font = QFont()
    font.setPixelSize(_CELL_FONT_PX)
    return QFontMetrics(font).horizontalAdvance(text) + _COLUMN_TEXT_PADDING


class _NoWheelComboBox(QComboBox):
    """Combo box that does not change value from accidental mouse-wheel scroll."""

    def wheelEvent(self, event) -> None:
        if self.view().isVisible():
            super().wheelEvent(event)
        else:
            event.ignore()


class _AnnotationRow(_ClickableFrame):
    """One annotation as a plain widget row (no table/delegates involved).

    Signals:
        activated (int): This row's annotation index, on click anywhere
            outside the combo box / eye / delete controls.
        visibility_toggled (int): Annotation index whose eye button was clicked.
        delete_requested (int): Annotation index whose trash button was clicked.
        class_changed (int, str): Annotation index + newly chosen class name.
    """

    activated = Signal(int)
    visibility_toggled = Signal(int)
    delete_requested = Signal(int)
    class_changed = Signal(int, str)

    def __init__(
        self,
        idx: int,
        table_model: AnnotationTableModel,
        vertices_col_w: int,
        area_col_w: int,
        parent: QWidget = None,
    ) -> None:
        super().__init__(parent)
        self._idx = idx
        self.setCursor(Qt.PointingHandCursor)
        self.clicked.connect(lambda: self.activated.emit(self._idx))

        h = QHBoxLayout(self)
        h.setContentsMargins(4, 3, 4, 3)
        h.setSpacing(6)

        rgb = table_model.index(idx, AnnotationColumns.COLOR).data(COLOR_ROLE)
        self._dot = QLabel()
        self._dot.setFixedSize(_DOT_W, _DOT_W)
        if rgb:
            self._dot.setStyleSheet(
                f"QLabel {{ background-color: rgb{tuple(rgb)}; border: 1px solid "
                f"rgba(120,120,120,150); border-radius: {_DOT_W // 2}px; }}"
            )
        h.addWidget(self._dot)

        self._combo = _NoWheelComboBox()
        self._combo.addItems(table_model.class_names())
        current_name = table_model.index(idx, AnnotationColumns.CLASS).data(
            Qt.DisplayRole
        )
        pos = self._combo.findText(current_name or "")
        self._combo.setCurrentIndex(max(0, pos))
        self._combo.activated.connect(
            lambda: self.class_changed.emit(self._idx, self._combo.currentText())
        )
        h.addWidget(self._combo, 1)

        vertices = table_model.index(idx, AnnotationColumns.VERTICES).data(
            Qt.DisplayRole
        )
        vertices_lbl = QLabel(str(vertices or ""))
        vertices_lbl.setFixedWidth(vertices_col_w)
        vertices_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        vertices_lbl.setStyleSheet("color: black; font-size: 11px;")
        h.addWidget(vertices_lbl)

        area = table_model.index(idx, AnnotationColumns.AREA).data(Qt.DisplayRole)
        area_lbl = QLabel(str(area or ""))
        area_lbl.setFixedWidth(area_col_w)
        area_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        area_lbl.setStyleSheet("color: black; font-size: 11px;")
        h.addWidget(area_lbl)

        visible = bool(
            table_model.index(idx, AnnotationColumns.VISIBILITY).data(VISIBLE_ROLE)
        )
        self._eye_btn = QToolButton()
        self._eye_btn.setFixedSize(_ICON_BTN_W, _ICON_BTN_W)
        self._eye_btn.setAutoRaise(True)
        self._eye_btn.setToolTip("Show or hide this annotation")
        self._set_eye_icon(visible)
        self._eye_btn.clicked.connect(lambda: self.visibility_toggled.emit(self._idx))
        h.addWidget(self._eye_btn)

        self._delete_btn = QToolButton()
        self._delete_btn.setFixedSize(_ICON_BTN_W, _ICON_BTN_W)
        self._delete_btn.setAutoRaise(True)
        self._delete_btn.setToolTip("Delete annotation")
        self._delete_btn.setIcon(material_icon("delete", size=_ICON_BTN_SIZE, color="black"))
        self._delete_btn.clicked.connect(lambda: self.delete_requested.emit(self._idx))
        h.addWidget(self._delete_btn)

        tooltip = table_model.index(idx, AnnotationColumns.CLASS).data(
            Qt.ToolTipRole
        )
        self.setToolTip(tooltip or "")

    def _set_eye_icon(self, visible: bool) -> None:
        name = "visibility" if visible else "visibility_off"
        self._eye_btn.setIcon(material_icon(name, size=_ICON_BTN_SIZE, color="black"))

    def set_selected(self, selected: bool) -> None:
        self.setStyleSheet(
            "background-color: palette(highlight);" if selected else ""
        )


class AnnotationsSection(QWidget):
    """Plain-widget list of annotations for the currently displayed image.

    Sorted alphabetically by class name (fixed, no user-facing sort control).
    Each annotation is a `_AnnotationRow`, rebuilt whenever the underlying
    model resets (add/delete/class-change/visibility-toggle all reset it).

    Signals:
        annotation_selected (int): Annotation index within the current image.
    """

    annotation_selected = Signal(int)

    def __init__(
        self, dataset_model, calibration_model=None, parent: QWidget = None
    ) -> None:
        super().__init__(parent)
        self.dataset_model = dataset_model
        self._current_row: int = -1
        self._selected_idx: int = -1
        self._rows: dict[int, _AnnotationRow] = {}

        self._table_model = AnnotationTableModel(dataset_model, calibration_model, self)
        self._proxy = AnnotationSortProxyModel(self)
        self._proxy.setSourceModel(self._table_model)
        self._proxy.sort(AnnotationColumns.CLASS, Qt.AscendingOrder)

        self._init_ui()
        self._table_model.modelReset.connect(self._on_model_reset)
        self._table_model.headerDataChanged.connect(self._on_header_data_changed)
        self._table_model.dataChanged.connect(self._on_table_data_changed)

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self._header_row = self._build_header_row()
        layout.addWidget(self._header_row)

        self._rows_container = QWidget()
        self._rows_layout = QVBoxLayout(self._rows_container)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(2)
        layout.addWidget(self._rows_container)

        self._empty_lbl = QLabel("No annotations")
        self._empty_lbl.setStyleSheet("color: black; font-size: 11px;")
        self._empty_lbl.setContentsMargins(6, 4, 6, 4)
        layout.addWidget(self._empty_lbl)
        self._sync_empty_label()

    def _build_header_row(self) -> QWidget:
        """Column labels above the rows -- blank over the swatch/eye/delete slots."""
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(4, 2, 4, 2)
        h.setSpacing(6)

        swatch_spacer = QLabel()
        swatch_spacer.setFixedWidth(_DOT_W)
        h.addWidget(swatch_spacer)

        class_lbl = QLabel(
            self._table_model.headerData(AnnotationColumns.CLASS, Qt.Horizontal)
        )
        class_lbl.setStyleSheet(_HEADER_STYLE)
        h.addWidget(class_lbl, 1)

        vertices_tooltip = self._table_model.headerData(
            AnnotationColumns.VERTICES, Qt.Horizontal, Qt.ToolTipRole
        )
        self._vertices_col_w = _header_label_width(_VERTICES_HEADER_TEXT)
        self._vertices_header_lbl = QLabel(_VERTICES_HEADER_TEXT)
        self._vertices_header_lbl.setToolTip(vertices_tooltip)
        self._vertices_header_lbl.setFixedWidth(self._vertices_col_w)
        self._vertices_header_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._vertices_header_lbl.setStyleSheet(_HEADER_STYLE)
        h.addWidget(self._vertices_header_lbl)

        area_tooltip = self._table_model.headerData(
            AnnotationColumns.AREA, Qt.Horizontal
        )
        self._area_col_w = _header_label_width(_AREA_HEADER_TEXT)
        self._area_header_lbl = QLabel(_AREA_HEADER_TEXT)
        self._area_header_lbl.setToolTip(area_tooltip)
        self._area_header_lbl.setFixedWidth(self._area_col_w)
        self._area_header_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._area_header_lbl.setStyleSheet(_HEADER_STYLE)
        h.addWidget(self._area_header_lbl)

        eye_spacer = QLabel()
        eye_spacer.setFixedWidth(_ICON_BTN_W)
        h.addWidget(eye_spacer)

        delete_spacer = QLabel()
        delete_spacer.setFixedWidth(_ICON_BTN_W)
        h.addWidget(delete_spacer)

        return row

    def set_current_row(self, row: int) -> None:
        self._current_row = row
        self._selected_idx = -1
        self._table_model.set_current_row(row)

    def select_annotation(self, idx: int) -> None:
        """Silently highlight *idx* without emitting annotation_selected."""
        self._selected_idx = idx
        self._sync_selection()

    def _on_model_reset(self) -> None:
        self._refresh_area_header()
        self._refresh_numeric_column_widths()
        self._rebuild_rows()
        self._sync_empty_label()
        if self._selected_idx not in self._rows:
            self._selected_idx = -1
        self._sync_selection()

    def _on_header_data_changed(self, orientation, first: int, last: int) -> None:
        if (
            orientation == Qt.Horizontal
            and first <= AnnotationColumns.AREA <= last
        ):
            self._refresh_area_header()

    def _on_table_data_changed(self, top_left, bottom_right, roles=None) -> None:
        if (
            top_left.column() <= AnnotationColumns.AREA <= bottom_right.column()
        ):
            self._refresh_numeric_column_widths()
            self._rebuild_rows()
            self._sync_selection()

    def _refresh_area_header(self) -> None:
        """Keep the current calibration unit available without widening the column."""
        area_tooltip = self._table_model.headerData(
            AnnotationColumns.AREA, Qt.Horizontal
        )
        self._area_header_lbl.setToolTip(area_tooltip)

    def _refresh_numeric_column_widths(self) -> None:
        """Fit numeric columns to their header or widest visible value."""
        self._vertices_col_w = self._widest_column_text(
            AnnotationColumns.VERTICES, self._vertices_header_lbl.text()
        )
        self._area_col_w = self._widest_column_text(
            AnnotationColumns.AREA, self._area_header_lbl.text()
        )
        self._vertices_header_lbl.setFixedWidth(self._vertices_col_w)
        self._area_header_lbl.setFixedWidth(self._area_col_w)

    def _widest_column_text(self, column: int, header_text: str) -> int:
        width = _header_label_width(header_text)
        for row in range(self._table_model.rowCount()):
            value = self._table_model.index(row, column).data(Qt.DisplayRole)
            width = max(width, _cell_text_width(str(value or "")))
        return width

    def _rebuild_rows(self) -> None:
        while self._rows_layout.count():
            item = self._rows_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._rows.clear()

        for proxy_row in range(self._proxy.rowCount()):
            proxy_index = self._proxy.index(proxy_row, AnnotationColumns.CLASS)
            idx = self._annotation_index_from_proxy(proxy_index)
            if idx < 0:
                continue
            row_widget = _AnnotationRow(
                idx, self._table_model, self._vertices_col_w, self._area_col_w
            )
            row_widget.activated.connect(self._on_row_activated)
            row_widget.visibility_toggled.connect(self._toggle_visibility)
            row_widget.delete_requested.connect(self._delete_annotation)
            row_widget.class_changed.connect(self._on_class_changed)
            self._rows_layout.addWidget(row_widget)
            self._rows[idx] = row_widget

    def _annotation_index_from_proxy(self, proxy_index) -> int:
        value = proxy_index.data(ANNOTATION_INDEX_ROLE)
        return int(value) if value is not None else -1

    def _on_row_activated(self, idx: int) -> None:
        self.select_annotation(idx)
        self.annotation_selected.emit(idx)

    def _sync_selection(self) -> None:
        for idx, row_widget in self._rows.items():
            row_widget.set_selected(idx == self._selected_idx)

    def _delete_annotation(self, idx: int) -> None:
        if self._selected_idx == idx:
            self._selected_idx = -1
        elif self._selected_idx > idx:
            self._selected_idx -= 1
        self.dataset_model.delete_annotation(self._current_row, idx)

    def _toggle_visibility(self, idx: int) -> None:
        self.dataset_model.toggle_annotation_visibility(self._current_row, idx)

    def _on_class_changed(self, idx: int, name: str) -> None:
        self.dataset_model.update_annotation_class(self._current_row, idx, name)

    def _sync_empty_label(self) -> None:
        has_rows = self._table_model.rowCount() > 0
        self._empty_lbl.setVisible(not has_rows)
        self._header_row.setVisible(has_rows)
