from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics
from PySide6.QtWidgets import (
    QColorDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from models.classes_model import (
    CLASS_NAME_ROLE,
    COLOR_ROLE,
    VISIBLE_ROLE,
    ClassColumns,
    ClassSortProxyModel,
    ClassTableModel,
)

from views.icons import material_icon

from ._shared import _ClickableFrame, _COLOR_SELECTED_BG

_ICON_BTN_SIZE = 16
_ICON_BTN_W = 28
_SWATCH_W = 24
_HEADER_FONT_PX = 10
_CELL_FONT_PX = 11
_COLUMN_TEXT_PADDING = 2
_HEADER_STYLE = f"font-size: {_HEADER_FONT_PX}px; font-weight: bold; color: black;"


def _header_label_width(text: str) -> int:
    """Width needed to show *text* in the header font, plus a little breathing room."""
    font = QFont()
    font.setPixelSize(_HEADER_FONT_PX)
    font.setBold(True)
    return QFontMetrics(font).horizontalAdvance(text) + _COLUMN_TEXT_PADDING


def _cell_text_width(text: str) -> int:
    """Width needed for a numeric cell value without clipping."""
    font = QFont()
    font.setPixelSize(_CELL_FONT_PX)
    return QFontMetrics(font).horizontalAdvance(text) + _COLUMN_TEXT_PADDING


class _ClassRow(_ClickableFrame):
    """One annotation class as a plain widget row (no table/delegates involved).

    Signals:
        activated (str): This class's name, on click anywhere outside the
            swatch / eye / delete controls.
        color_clicked (str): Class name whose swatch was clicked.
        visibility_toggled (str): Class name whose eye button was clicked.
        delete_requested (str): Class name whose trash button was clicked.
    """

    activated = Signal(str)
    color_clicked = Signal(str)
    visibility_toggled = Signal(str)
    delete_requested = Signal(str)

    def __init__(
        self,
        proxy_row: int,
        proxy_model,
        total_count_w: int,
        parent: QWidget = None,
    ) -> None:
        super().__init__(parent)
        self._name = str(
            proxy_model.index(proxy_row, ClassColumns.CLASS).data(CLASS_NAME_ROLE) or ""
        )
        self.setCursor(Qt.PointingHandCursor)
        self.clicked.connect(lambda: self.activated.emit(self._name))

        h = QHBoxLayout(self)
        h.setContentsMargins(4, 3, 4, 3)
        h.setSpacing(6)

        rgb = proxy_model.index(proxy_row, ClassColumns.COLOR).data(COLOR_ROLE)
        # A QToolButton (not _ClickableFrame) so its press is always accepted --
        # an unaccepted mouse press on a plain QFrame propagates to the parent
        # row and would also fire the row's own `activated`/select behavior.
        self._swatch = QToolButton()
        self._swatch.setFixedSize(_SWATCH_W, _SWATCH_W)
        self._swatch.setCursor(Qt.PointingHandCursor)
        self._swatch.setAutoRaise(True)
        self._swatch.setToolTip(
            proxy_model.index(proxy_row, ClassColumns.COLOR).data(Qt.ToolTipRole) or ""
        )
        self._set_swatch_color(rgb)
        self._swatch.clicked.connect(lambda: self.color_clicked.emit(self._name))
        h.addWidget(self._swatch)

        name_lbl = QLabel(self._name)
        name_lbl.setStyleSheet("color: black; font-size: 12px;")
        h.addWidget(name_lbl, 1)

        total_count = proxy_model.index(proxy_row, ClassColumns.TOTAL).data(
            Qt.DisplayRole
        )
        self._total_lbl = QLabel(str(total_count or "0"))
        self._total_lbl.setFixedWidth(total_count_w)
        self._total_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._total_lbl.setStyleSheet("color: black; font-size: 11px;")
        h.addWidget(self._total_lbl)

        visible = bool(
            proxy_model.index(proxy_row, ClassColumns.VISIBILITY).data(VISIBLE_ROLE)
        )
        self._eye_btn = QToolButton()
        self._eye_btn.setFixedSize(_ICON_BTN_W, _ICON_BTN_W)
        self._eye_btn.setAutoRaise(True)
        self._eye_btn.setToolTip("Show or hide this class's annotations")
        self._set_eye_icon(visible)
        self._eye_btn.clicked.connect(lambda: self.visibility_toggled.emit(self._name))
        h.addWidget(self._eye_btn)

        self._delete_btn = QToolButton()
        self._delete_btn.setFixedSize(_ICON_BTN_W, _ICON_BTN_W)
        self._delete_btn.setAutoRaise(True)
        self._delete_btn.setToolTip(f'Delete "{self._name}"')
        self._delete_btn.setIcon(material_icon("delete", size=_ICON_BTN_SIZE, color="black"))
        self._delete_btn.clicked.connect(lambda: self.delete_requested.emit(self._name))
        h.addWidget(self._delete_btn)

    def _set_swatch_color(self, rgb) -> None:
        if rgb:
            self._swatch.setStyleSheet(
                f"QToolButton {{ background-color: rgb{tuple(rgb)}; border: 1px solid "
                f"rgba(120,120,120,150); border-radius: 4px; }}"
            )

    def _set_eye_icon(self, visible: bool) -> None:
        name = "visibility" if visible else "visibility_off"
        self._eye_btn.setIcon(material_icon(name, size=_ICON_BTN_SIZE, color="black"))

    def set_selected(self, selected: bool) -> None:
        self.setStyleSheet(
            f"background-color: {_COLOR_SELECTED_BG};" if selected else ""
        )

    def set_visibility_control_visible(self, visible: bool) -> None:
        self._eye_btn.setVisible(visible)


class ClassesSection(QWidget):
    """Plain-widget list of annotation classes with add/delete and count summaries.

    Sorted alphabetically by class name (fixed, no user-facing sort control).
    Each class is a `_ClassRow`, rebuilt whenever the underlying model resets
    or its data changes (add/delete/recolor/visibility/mode-switch all do).

    Signals:
        class_selected (str): Emitted when the user clicks a class row.
        annotation_mode_changed (str): Emitted when the user switches annotation mode.
    """

    class_selected = Signal(str)
    annotation_mode_changed = Signal(str)  # "pixel" | "image_level"

    def __init__(self, dataset_model, parent: QWidget = None) -> None:
        super().__init__(parent)
        self.dataset_model = dataset_model
        self._selected_name: str = ""
        self._current_row: int = -1
        self._is_image_level: bool = False
        self._rows: dict[str, _ClassRow] = {}
        self._total_col_w: int = 0

        self._table_model = ClassTableModel(dataset_model, self)
        self._proxy = ClassSortProxyModel(self)
        self._proxy.setSourceModel(self._table_model)
        self._proxy.sort(ClassColumns.CLASS, Qt.AscendingOrder)

        self._init_ui()
        self._table_model.modelReset.connect(self._on_class_model_reset)
        self._table_model.dataChanged.connect(self._on_class_model_reset)
        dataset_model.dataChanged.connect(self._on_dataset_data_changed)
        dataset_model.annotation_mode_changed.connect(self._on_annotation_mode_changed)

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # Mode selector: two connected buttons acting as a segmented control
        mode_row = QWidget()
        mode_h = QHBoxLayout(mode_row)
        mode_h.setContentsMargins(16, 0, 16, 4)
        mode_h.setSpacing(0)

        _seg_base = (
            "QPushButton {"
            "  padding: 3px 10px;"
            "  font-weight: bold;"
            "  border: 1px solid palette(mid);"
            "  background: palette(button);"
            "  color: palette(button-text);"
            "}"
            "QPushButton:checked {"
            "  background: #ff9800;"
            "  color: white;"
            "  border-color: #c96800;"
            "}"
        )
        self._pixel_btn = QPushButton("Pixel Level")
        self._pixel_btn.setObjectName("classesModeToggle")
        self._pixel_btn.setCheckable(True)
        self._pixel_btn.setChecked(True)
        self._pixel_btn.setToolTip(
            "Pixel Level mode: a rejected image requires at least one polygon "
            "annotation to count as reviewed."
        )
        self._pixel_btn.setStyleSheet(
            _seg_base
            + "QPushButton { border-top-left-radius: 4px; border-bottom-left-radius: 4px;"
            "  border-right: none; }"
        )
        self._pixel_btn.clicked.connect(lambda: self._on_mode_btn_clicked("pixel"))

        self._image_btn = QPushButton("Image Level")
        self._image_btn.setCheckable(True)
        self._image_btn.setChecked(False)
        self._image_btn.setToolTip(
            "Image Level mode: a rejected image requires at least one class tag "
            "to count as reviewed. Polygon drawing tools are disabled."
        )
        self._image_btn.setStyleSheet(
            _seg_base
            + "QPushButton { border-top-right-radius: 4px; border-bottom-right-radius: 4px; }"
        )
        self._image_btn.clicked.connect(
            lambda: self._on_mode_btn_clicked("image_level")
        )

        mode_h.addWidget(self._pixel_btn, stretch=1)
        mode_h.addWidget(self._image_btn, stretch=1)
        layout.addWidget(mode_row)

        self._header_row = self._build_header_row()
        layout.addWidget(self._header_row)

        self._rows_container = QWidget()
        self._rows_layout = QVBoxLayout(self._rows_container)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(2)
        layout.addWidget(self._rows_container)

        self._rebuild_rows()

        add_row = QWidget()
        add_h = QHBoxLayout(add_row)
        add_h.setContentsMargins(4, 4, 4, 4)
        add_h.setSpacing(4)

        self._name_input = QLineEdit()
        self._name_input.setPlaceholderText("Enter new class name")
        self._name_input.returnPressed.connect(self._add_class)
        add_h.addWidget(self._name_input, stretch=1)

        btn_add = QPushButton(material_icon("add"), "Add Class")
        btn_add.clicked.connect(self._add_class)
        add_h.addWidget(btn_add)

        layout.addWidget(add_row)

    def _build_header_row(self) -> QWidget:
        """Column labels above the rows -- blank over the swatch/eye/delete slots."""
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(4, 2, 4, 2)
        h.setSpacing(6)

        swatch_spacer = QLabel()
        swatch_spacer.setFixedWidth(_SWATCH_W)
        h.addWidget(swatch_spacer)

        class_lbl = QLabel(
            self._table_model.headerData(ClassColumns.CLASS, Qt.Horizontal)
        )
        class_lbl.setStyleSheet(_HEADER_STYLE)
        h.addWidget(class_lbl, 1)

        total_header_text = self._table_model.headerData(
            ClassColumns.TOTAL, Qt.Horizontal
        )
        self._total_col_w = _header_label_width(total_header_text)
        self._total_header_lbl = QLabel(total_header_text)
        self._total_header_lbl.setToolTip(
            self._table_model.headerData(ClassColumns.TOTAL, Qt.Horizontal, Qt.ToolTipRole)
        )
        self._total_header_lbl.setFixedWidth(self._total_col_w)
        self._total_header_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._total_header_lbl.setStyleSheet(_HEADER_STYLE)
        h.addWidget(self._total_header_lbl)

        self._eye_header_spacer = QLabel()
        self._eye_header_spacer.setFixedWidth(_ICON_BTN_W)
        h.addWidget(self._eye_header_spacer)

        delete_spacer = QLabel()
        delete_spacer.setFixedWidth(_ICON_BTN_W)
        h.addWidget(delete_spacer)

        return row

    def set_current_row(self, row: int) -> None:
        self._current_row = row
        self._table_model.set_current_row(row)

    def _on_dataset_data_changed(self, top_left, bottom_right, roles=None) -> None:
        if self._current_row < 0:
            return
        if top_left.row() <= self._current_row <= bottom_right.row():
            self._table_model.set_current_row(self._current_row)

    def _on_mode_btn_clicked(self, mode: str) -> None:
        self.set_annotation_mode(mode)
        self.annotation_mode_changed.emit(mode)

    def set_annotation_mode(self, mode: str) -> None:
        """Sync the mode buttons without re-emitting the signal."""
        is_image = mode == "image_level"
        self._pixel_btn.blockSignals(True)
        self._image_btn.blockSignals(True)
        self._pixel_btn.setChecked(not is_image)
        self._image_btn.setChecked(is_image)
        self._pixel_btn.blockSignals(False)
        self._image_btn.blockSignals(False)

        self._is_image_level = is_image
        self._eye_header_spacer.setVisible(not is_image)
        for row_widget in self._rows.values():
            row_widget.set_visibility_control_visible(not is_image)

    def _on_annotation_mode_changed(self, mode: str) -> None:
        self.set_annotation_mode(mode)
        self._table_model.set_current_row(self._current_row)

    def _refresh_count_column_widths(self) -> None:
        """Fit the Tot column to its header or widest visible value."""
        self._total_col_w = self._widest_column_text(
            ClassColumns.TOTAL, self._total_header_lbl.text()
        )
        self._total_header_lbl.setFixedWidth(self._total_col_w)

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
            row_widget = _ClassRow(proxy_row, self._proxy, self._total_col_w)
            row_widget.set_visibility_control_visible(not self._is_image_level)
            row_widget.activated.connect(self._on_row_activated)
            row_widget.color_clicked.connect(self._change_color)
            row_widget.visibility_toggled.connect(self._toggle_visibility)
            row_widget.delete_requested.connect(self._delete_class)
            self._rows_layout.addWidget(row_widget)
            self._rows[row_widget._name] = row_widget

    def _on_row_activated(self, name: str) -> None:
        self._select_class(name, emit=True)

    def _select_class(self, name: str, emit: bool) -> None:
        if name not in self.dataset_model.get_class_names():
            return
        self._selected_name = name
        self._sync_selection()
        if emit:
            self.class_selected.emit(name)

    def _sync_selection(self) -> None:
        for name, row_widget in self._rows.items():
            row_widget.set_selected(name == self._selected_name)

    def _on_class_model_reset(self, *args) -> None:
        if self._selected_name not in self.dataset_model.get_class_names():
            self._selected_name = ""
        self._refresh_count_column_widths()
        self._rebuild_rows()
        self._sync_selection()

    def _change_color(self, name: str) -> None:
        color = QColorDialog.getColor(
            QColor(*self.dataset_model.get_class_color(name)),
            self,
            "Change Class Color",
        )
        if not color.isValid():
            return
        self.dataset_model.set_class_color(
            name,
            (color.red(), color.green(), color.blue()),
        )
        self._table_model.refresh_classes()
        self._select_class(name, emit=False)

    def _delete_class(self, name: str) -> None:
        pixel_count = self.dataset_model.get_class_annotation_count(name)
        tag_count = sum(
            1
            for row in range(self.dataset_model.rowCount())
            if name in self.dataset_model.get_image_classes(row)
        )
        if pixel_count > 0 or tag_count > 0:
            parts = []
            if pixel_count > 0:
                parts.append(f"{pixel_count} annotation(s)")
            if tag_count > 0:
                parts.append(f"{tag_count} image-level tag(s)")
            choice = QMessageBox.question(
                self,
                "Delete Annotation Class",
                f'Delete class "{name}" and its {" and ".join(parts)}?',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if choice != QMessageBox.Yes:
                return

        self.dataset_model.delete_class(name)
        if self._selected_name == name:
            self._selected_name = ""
        self._table_model.refresh_classes()
        self._sync_selection()

    def _toggle_visibility(self, name: str) -> None:
        self.dataset_model.toggle_class_visibility(name)

    def _pick_next_unique_color(self) -> tuple:
        from core.utils.constants import DEFAULT_CLASS_COLORS

        used = set(map(tuple, self.dataset_model.get_used_class_colors()))
        for cand in DEFAULT_CLASS_COLORS:
            if cand not in used:
                return cand
        return DEFAULT_CLASS_COLORS[0]

    def _add_class(self) -> None:
        name = self._name_input.text().strip().lower()
        if not name or name in self.dataset_model.get_class_names():
            return
        self.dataset_model.add_class(name, self._pick_next_unique_color())
        self._table_model.refresh_classes()
        self._name_input.clear()
        self._select_class(name, emit=True)
