from PySide6.QtCore import QItemSelectionModel, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QColorDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QStyle,
    QStyleOptionButton,
    QStyleOptionViewItem,
    QStyledItemDelegate,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from models.classes_model import (
    CLASS_NAME_ROLE,
    COLOR_ROLE,
    IMAGE_LEVEL_MODE_ROLE,
    IMAGE_TAG_KIND_ROLE,
    VISIBLE_ROLE,
    ClassColumns,
    ClassSortProxyModel,
    ClassTableModel,
)


_TAG_COL_W = 28
_COLOR_TAG_ACTIVE = QColor("#ff9800")  # amber — current mode's tags
_COLOR_TAG_INACTIVE = QColor("#4a90d9")  # blue  — other mode's tags
_COLOR_COL_W = 44
_SWATCH_W = 24
_COUNT_W = 56
_VISIBILITY_W = 40
_DELETE_W = 40
_NAME_MIN_W = 70


class _ColorSwatchDelegate(QStyledItemDelegate):
    """Paint class colors as compact swatches."""

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index) -> None:
        rgb = index.data(COLOR_ROLE)
        if not rgb:
            super().paint(painter, option, index)
            return

        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        opt.text = ""
        style = opt.widget.style() if opt.widget is not None else None
        if style is not None:
            style.drawControl(QStyle.CE_ItemViewItem, opt, painter)

        rect = option.rect
        x = rect.x() + (rect.width() - _SWATCH_W) // 2
        y = rect.y() + (rect.height() - _SWATCH_W) // 2
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(QPen(QColor(120, 120, 120), 1))
        painter.setBrush(QColor(*rgb))
        painter.drawRoundedRect(x, y, _SWATCH_W, _SWATCH_W, 3, 3)
        painter.restore()


class _IconButtonDelegate(QStyledItemDelegate):
    """Paint action cells with a native button frame and compact icon."""

    def __init__(self, action: str, parent=None) -> None:
        super().__init__(parent)
        self._action = action

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index) -> None:
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        opt.text = ""

        style = opt.widget.style() if opt.widget is not None else QApplication.style()
        style.drawControl(QStyle.CE_ItemViewItem, opt, painter)

        button = QStyleOptionButton()
        if option.widget is not None:
            button.initFrom(option.widget)
        button.rect = option.rect.adjusted(5, 3, -5, -3)
        button.state = QStyle.State_Enabled | QStyle.State_Raised
        if option.state & QStyle.State_MouseOver:
            button.state |= QStyle.State_MouseOver
        style.drawControl(QStyle.CE_PushButton, button, painter)
        self._paint_icon(painter, button.rect, opt.palette, index)

    def _paint_icon(self, painter: QPainter, rect, palette, index) -> None:
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        color = palette.buttonText().color()
        painter.setPen(QPen(color, 1.4))
        painter.setBrush(Qt.NoBrush)

        if self._action == "visibility":
            self._paint_eye_icon(painter, QRectF(rect).adjusted(8, 8, -8, -8), index)
        else:
            self._paint_trash_icon(painter, QRectF(rect).adjusted(9, 7, -9, -7))
        painter.restore()

    def _paint_eye_icon(self, painter: QPainter, rect: QRectF, index) -> None:
        center = rect.center()
        eye = QPainterPath()
        eye.moveTo(rect.left(), center.y())
        eye.cubicTo(
            rect.left() + rect.width() * 0.25,
            rect.top(),
            rect.right() - rect.width() * 0.25,
            rect.top(),
            rect.right(),
            center.y(),
        )
        eye.cubicTo(
            rect.right() - rect.width() * 0.25,
            rect.bottom(),
            rect.left() + rect.width() * 0.25,
            rect.bottom(),
            rect.left(),
            center.y(),
        )
        painter.drawPath(eye)
        painter.setBrush(painter.pen().color())
        radius = max(2.0, min(rect.width(), rect.height()) * 0.18)
        painter.drawEllipse(center, radius, radius)
        painter.setBrush(Qt.NoBrush)

        if not bool(index.data(VISIBLE_ROLE)):
            painter.drawLine(rect.topRight(), rect.bottomLeft())

    def _paint_trash_icon(self, painter: QPainter, rect: QRectF) -> None:
        w = rect.width()
        h = rect.height()
        lid_y = rect.top() + h * 0.22
        body = QRectF(
            rect.left() + w * 0.18,
            lid_y + h * 0.16,
            w * 0.64,
            h * 0.62,
        )
        painter.drawLine(rect.left() + w * 0.12, lid_y, rect.right() - w * 0.12, lid_y)
        painter.drawLine(
            rect.left() + w * 0.38,
            rect.top() + h * 0.08,
            rect.right() - w * 0.38,
            rect.top() + h * 0.08,
        )
        painter.drawRect(body)
        painter.drawLine(
            body.left() + body.width() * 0.35,
            body.top() + body.height() * 0.2,
            body.left() + body.width() * 0.35,
            body.bottom() - body.height() * 0.15,
        )
        painter.drawLine(
            body.right() - body.width() * 0.35,
            body.top() + body.height() * 0.2,
            body.right() - body.width() * 0.35,
            body.bottom() - body.height() * 0.15,
        )


class _ImageTagDelegate(QStyledItemDelegate):
    """Paint image-level class tag toggle cells in the class table.

    When interactive (image-level mode + rejected image): draws a colored
    rounded square with a checkmark if tagged, or an empty square if not.
    When not interactive: draws a faint grey square with a dimmed checkmark
    if tagged (read-only reference), or nothing if not tagged.
    """

    _TAG_W = 14

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index) -> None:
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        opt.text = ""
        style = opt.widget.style() if opt.widget is not None else None
        if style is not None:
            style.drawControl(QStyle.CE_ItemViewItem, opt, painter)

        kind = index.data(IMAGE_TAG_KIND_ROLE)  # "active" | "inactive" | "none"
        interactive = bool(index.flags() & Qt.ItemIsSelectable)
        in_mode = bool(index.data(IMAGE_LEVEL_MODE_ROLE))

        if kind == "none" and not interactive and not in_mode:
            return  # pixel mode, untagged — draw nothing beyond the row background

        rect = option.rect
        cx = rect.x() + rect.width() // 2
        cy = rect.y() + rect.height() // 2
        half = self._TAG_W // 2
        sq = QRectF(cx - half, cy - half, self._TAG_W, self._TAG_W)

        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)

        if kind == "active":
            painter.setBrush(_COLOR_TAG_ACTIVE)
            painter.setPen(QPen(_COLOR_TAG_ACTIVE.darker(130), 1))
            painter.drawRoundedRect(sq, 3, 3)
            painter.setPen(QPen(QColor(255, 255, 255), 1.5))
            self._draw_check(painter, sq)
        elif kind == "inactive":
            painter.setBrush(_COLOR_TAG_INACTIVE)
            painter.setPen(QPen(_COLOR_TAG_INACTIVE.darker(130), 1))
            painter.drawRoundedRect(sq, 3, 3)
            painter.setPen(QPen(QColor(255, 255, 255), 1.5))
            self._draw_check(painter, sq)
        elif interactive:
            # No tag yet but interactive — empty outlined square (click to tag)
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor(160, 160, 160), 1))
            painter.drawRoundedRect(sq, 3, 3)
        else:
            # In image-level mode, not interactive, not tagged — faint square
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor(190, 190, 190, 100), 1))
            painter.drawRoundedRect(sq, 3, 3)

        painter.restore()

    def _draw_check(self, painter: QPainter, sq: QRectF) -> None:
        x, t, w, h = sq.left(), sq.top(), sq.width(), sq.height()
        painter.drawLine(
            int(x + 2), int(t + h * 0.55), int(x + w * 0.4), int(t + h - 2)
        )
        painter.drawLine(int(x + w * 0.4), int(t + h - 2), int(x + w - 1), int(t + 2))


class ClassesSection(QWidget):
    """Sortable annotation class table with add/delete and count summaries.

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
        self._tag_interactive: bool = False

        self._table_model = ClassTableModel(dataset_model, self)
        self._proxy = ClassSortProxyModel(self)
        self._proxy.setSourceModel(self._table_model)

        self._init_ui()
        self._table_model.modelReset.connect(self._on_class_model_reset)
        self._proxy.layoutChanged.connect(self._sync_selection)
        self._proxy.modelReset.connect(self._sync_selection)
        dataset_model.dataChanged.connect(self._on_dataset_data_changed)
        dataset_model.annotation_mode_changed.connect(self._on_annotation_mode_changed)

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

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

        self._table = QTableView()
        self._table.setModel(self._proxy)
        self._table.setItemDelegateForColumn(
            ClassColumns.IMAGE_TAG, _ImageTagDelegate(self._table)
        )
        self._table.setItemDelegateForColumn(
            ClassColumns.COLOR, _ColorSwatchDelegate(self._table)
        )
        self._table.setItemDelegateForColumn(
            ClassColumns.VISIBILITY, _IconButtonDelegate("visibility", self._table)
        )
        self._table.setItemDelegateForColumn(
            ClassColumns.DELETE, _IconButtonDelegate("delete", self._table)
        )
        self._table.setFrameShape(QFrame.NoFrame)
        self._table.setAlternatingRowColors(True)
        self._table.setShowGrid(False)
        self._table.setWordWrap(False)
        self._table.setSortingEnabled(True)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setFocusPolicy(Qt.NoFocus)
        self._table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._table.clicked.connect(self._on_table_index_activated)
        self._table.activated.connect(self._on_table_index_activated)
        self._table.setStyleSheet(
            """
            QTableView {
                selection-background-color: palette(highlight);
                selection-color: palette(highlighted-text);
            }
            QTableView::item {
                padding: 1px 4px;
            }
            QTableView::item:focus {
                outline: none;
            }
            QHeaderView::section {
                font-size: 12px;
                font-weight: bold;
                padding: 2px 4px;
            }
            """
        )

        vertical_header = self._table.verticalHeader()
        vertical_header.setVisible(False)
        vertical_header.setDefaultSectionSize(28)
        vertical_header.setMinimumSectionSize(24)

        header = self._table.horizontalHeader()
        header.setHighlightSections(False)
        header.setSectionsClickable(True)
        header.setSortIndicatorShown(True)
        header.setMinimumSectionSize(24)
        header.setSectionResizeMode(ClassColumns.IMAGE_TAG, QHeaderView.Fixed)
        header.setSectionResizeMode(ClassColumns.COLOR, QHeaderView.Fixed)
        header.setSectionResizeMode(ClassColumns.CLASS, QHeaderView.Stretch)
        header.setSectionResizeMode(ClassColumns.IMAGE, QHeaderView.Fixed)
        header.setSectionResizeMode(ClassColumns.TOTAL, QHeaderView.Fixed)
        header.setSectionResizeMode(ClassColumns.VISIBILITY, QHeaderView.Fixed)
        header.setSectionResizeMode(ClassColumns.DELETE, QHeaderView.Fixed)
        self._table.setColumnWidth(ClassColumns.IMAGE_TAG, _TAG_COL_W)
        self._table.setColumnWidth(ClassColumns.COLOR, _COLOR_COL_W)
        self._table.setColumnWidth(ClassColumns.CLASS, _NAME_MIN_W)
        self._table.setColumnWidth(ClassColumns.IMAGE, _COUNT_W)
        self._table.setColumnWidth(ClassColumns.TOTAL, _COUNT_W)
        self._table.setColumnWidth(ClassColumns.VISIBILITY, _VISIBILITY_W)
        self._table.setColumnWidth(ClassColumns.DELETE, _DELETE_W)
        self._table.sortByColumn(ClassColumns.CLASS, Qt.AscendingOrder)
        self._sync_table_height()

        layout.addWidget(self._table)

        add_row = QWidget()
        add_h = QHBoxLayout(add_row)
        add_h.setContentsMargins(4, 4, 4, 4)
        add_h.setSpacing(4)

        self._name_input = QLineEdit()
        self._name_input.setPlaceholderText("Enter new class name")
        self._name_input.returnPressed.connect(self._add_class)
        add_h.addWidget(self._name_input, stretch=1)

        btn_add = QPushButton("Add Class")
        btn_add.clicked.connect(self._add_class)
        add_h.addWidget(btn_add)

        layout.addWidget(add_row)

    def _sync_table_height(self) -> None:
        """Grow the table viewport to fit every class row without scrolling."""
        if not hasattr(self, "_table"):
            return
        header_h = self._table.horizontalHeader().sizeHint().height()
        rows_h = 0
        for row in range(self._proxy.rowCount()):
            rows_h += self._table.verticalHeader().sectionSize(row)
        frame_h = self._table.frameWidth() * 2
        self._table.setFixedHeight(header_h + rows_h + frame_h + 2)
        self._table.updateGeometry()

    def set_current_row(self, row: int) -> None:
        self._current_row = row
        self._update_tag_interactive()
        self._table_model.set_current_row(row)

    def _update_tag_interactive(self) -> None:
        mode = self.dataset_model.get_annotation_mode()
        decision = self.dataset_model.get_review_decision(self._current_row)
        self._tag_interactive = mode == "image_level" and decision == "reject"

    def _on_dataset_data_changed(self, top_left, bottom_right, roles=None) -> None:
        if self._current_row < 0:
            return
        if top_left.row() <= self._current_row <= bottom_right.row():
            self._update_tag_interactive()
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
        self._table.setColumnHidden(ClassColumns.IMAGE, is_image)
        self._table.setColumnHidden(ClassColumns.VISIBILITY, is_image)

    def _on_annotation_mode_changed(self, mode: str) -> None:
        self.set_annotation_mode(mode)
        self._update_tag_interactive()
        self._table_model.set_current_row(self._current_row)

    def _on_table_index_activated(self, proxy_index) -> None:
        if not proxy_index.isValid():
            return
        name = self._class_name_from_proxy(proxy_index)
        if not name:
            return

        column = proxy_index.column()
        if column == ClassColumns.IMAGE_TAG:
            if self._tag_interactive:
                self._toggle_image_tag(name)
            return
        if column == ClassColumns.COLOR:
            self._change_color(name)
            return
        if column == ClassColumns.VISIBILITY:
            self._toggle_visibility(name)
            return
        if column == ClassColumns.DELETE:
            self._delete_class(name)
            return
        self._select_class(name, emit=True)

    def _toggle_image_tag(self, name: str) -> None:
        if self._current_row < 0:
            return
        current = self.dataset_model.get_image_classes(self._current_row)
        if name in current:
            pixel_count = self.dataset_model.get_pixel_annotation_count_for_class(
                self._current_row, name
            )
            if pixel_count > 0:
                QMessageBox.warning(
                    self,
                    "Cannot Remove Tag",
                    f"'{name}' has {pixel_count} pixel-level annotation(s) on this image.\n"
                    "To remove this class, switch to pixel-level mode and delete those "
                    "annotations first.",
                )
                return
            current.remove(name)
        else:
            current.append(name)
        self.dataset_model.set_image_classes(self._current_row, current)
        self._table_model.set_current_row(self._current_row)

    def _class_name_from_proxy(self, proxy_index) -> str:
        value = proxy_index.data(CLASS_NAME_ROLE)
        return str(value) if value else ""

    def _source_row_for_class(self, name: str) -> int:
        for row in range(self._table_model.rowCount()):
            if self._table_model.class_name(row) == name:
                return row
        return -1

    def _proxy_row_for_class(self, name: str) -> int:
        source_row = self._source_row_for_class(name)
        if source_row < 0:
            return -1
        source_index = self._table_model.index(source_row, ClassColumns.CLASS)
        proxy_index = self._proxy.mapFromSource(source_index)
        return proxy_index.row() if proxy_index.isValid() else -1

    def _select_class(self, name: str, emit: bool) -> None:
        if name not in self.dataset_model.get_class_names():
            return
        self._selected_name = name
        self._sync_selection()
        if emit:
            self.class_selected.emit(name)

    def _sync_selection(self, *args) -> None:
        if not hasattr(self, "_table"):
            return
        selection_model = self._table.selectionModel()
        if selection_model is None:
            return
        if not self._selected_name:
            selection_model.clearSelection()
            return

        proxy_row = self._proxy_row_for_class(self._selected_name)
        if proxy_row < 0:
            selection_model.clearSelection()
            return

        proxy_index = self._proxy.index(proxy_row, ClassColumns.CLASS)
        selection_model.setCurrentIndex(
            proxy_index,
            QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows,
        )
        self._table.scrollTo(proxy_index, QAbstractItemView.EnsureVisible)

    def _on_class_model_reset(self) -> None:
        if self._selected_name not in self.dataset_model.get_class_names():
            self._selected_name = ""
        self._sync_table_height()
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
