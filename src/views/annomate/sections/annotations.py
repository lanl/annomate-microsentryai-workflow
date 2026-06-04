from PySide6.QtCore import QItemSelectionModel, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QFrame,
    QHeaderView,
    QLabel,
    QSizePolicy,
    QStyle,
    QStyleOptionButton,
    QStyleOptionComboBox,
    QStyleOptionViewItem,
    QStyledItemDelegate,
    QTableView,
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


_COLOR_COL_W = 44
_DOT_W = 16
_COUNT_W = 52
_AREA_W = 86
_VISIBILITY_W = 40
_DELETE_W = 40
_NAME_MIN_W = 70


class _NoWheelComboBox(QComboBox):
    """Combo box that does not change value from accidental mouse-wheel scroll."""

    def wheelEvent(self, event) -> None:
        if self.view().isVisible():
            super().wheelEvent(event)
        else:
            event.ignore()


class _AnnotationClassDelegate(QStyledItemDelegate):
    """Combo-box editor for changing an annotation's class."""

    def __init__(self, table_model: AnnotationTableModel, parent=None) -> None:
        super().__init__(parent)
        self._table_model = table_model

    def createEditor(self, parent, option, index):
        editor = _NoWheelComboBox(parent)
        editor.addItems(self._table_model.class_names())
        editor.activated.connect(lambda *_: self._commit_and_close(editor))
        QTimer.singleShot(0, editor.showPopup)
        return editor

    def setEditorData(self, editor, index) -> None:
        text = index.data(Qt.EditRole) or ""
        pos = editor.findText(text)
        editor.setCurrentIndex(max(0, pos))

    def setModelData(self, editor, model, index) -> None:
        model.setData(index, editor.currentText(), Qt.EditRole)

    def _commit_and_close(self, editor) -> None:
        editor.hidePopup()
        self.commitData.emit(editor)
        self.closeEditor.emit(editor)

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index) -> None:
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        style = opt.widget.style() if opt.widget is not None else QApplication.style()
        style.drawControl(QStyle.CE_ItemViewItem, opt, painter)

        combo = QStyleOptionComboBox()
        if option.widget is not None:
            combo.initFrom(option.widget)
        combo.rect = option.rect.adjusted(3, 3, -3, -3)
        combo.currentText = str(index.data(Qt.DisplayRole) or "")
        combo.state = QStyle.State_Enabled
        if option.state & QStyle.State_MouseOver:
            combo.state |= QStyle.State_MouseOver
        if option.state & QStyle.State_Selected:
            combo.state |= QStyle.State_Selected
        style.drawComplexControl(QStyle.CC_ComboBox, combo, painter)
        style.drawControl(QStyle.CE_ComboBoxLabel, combo, painter)


class _ColorDotDelegate(QStyledItemDelegate):
    """Paint annotation colors as compact dots."""

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
        x = rect.x() + (rect.width() - _DOT_W) // 2
        y = rect.y() + (rect.height() - _DOT_W) // 2
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(QPen(QColor(120, 120, 120), 1))
        painter.setBrush(QColor(*rgb))
        painter.drawEllipse(x, y, _DOT_W, _DOT_W)
        painter.restore()


class _IconButtonDelegate(QStyledItemDelegate):
    """Paint action cells with a button frame and compact icon."""

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

        if self._action == "visibility":
            self._paint_eye_icon(
                painter, QRectF(button.rect).adjusted(8, 8, -8, -8), opt, index
            )
        else:
            self._paint_trash_icon(
                painter, QRectF(button.rect).adjusted(9, 7, -9, -7), opt
            )

    def _paint_eye_icon(self, painter: QPainter, rect: QRectF, opt, index) -> None:
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
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(QPen(opt.palette.buttonText().color(), 1.4))
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(eye)
        painter.setBrush(painter.pen().color())
        radius = max(2.0, min(rect.width(), rect.height()) * 0.18)
        painter.drawEllipse(center, radius, radius)
        painter.setBrush(Qt.NoBrush)
        if not bool(index.data(VISIBLE_ROLE)):
            painter.drawLine(rect.topRight(), rect.bottomLeft())
        painter.restore()

    def _paint_trash_icon(self, painter: QPainter, rect: QRectF, opt) -> None:
        w = rect.width()
        h = rect.height()
        lid_y = rect.top() + h * 0.22
        body = QRectF(
            rect.left() + w * 0.18,
            lid_y + h * 0.16,
            w * 0.64,
            h * 0.62,
        )
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(QPen(opt.palette.buttonText().color(), 1.4))
        painter.setBrush(Qt.NoBrush)
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
        painter.restore()


class AnnotationsSection(QWidget):
    """Sortable table of annotations for the currently displayed image.

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

        self._table_model = AnnotationTableModel(dataset_model, calibration_model, self)
        self._proxy = AnnotationSortProxyModel(self)
        self._proxy.setSourceModel(self._table_model)

        self._init_ui()
        self._table_model.modelReset.connect(self._on_model_reset)
        self._proxy.layoutChanged.connect(self._sync_selection)
        self._proxy.modelReset.connect(self._sync_selection)

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._table = QTableView()
        self._table.setModel(self._proxy)
        self._table.setItemDelegateForColumn(
            AnnotationColumns.COLOR, _ColorDotDelegate(self._table)
        )
        self._table.setItemDelegateForColumn(
            AnnotationColumns.CLASS,
            _AnnotationClassDelegate(self._table_model, self._table),
        )
        self._table.setItemDelegateForColumn(
            AnnotationColumns.VISIBILITY, _IconButtonDelegate("visibility", self._table)
        )
        self._table.setItemDelegateForColumn(
            AnnotationColumns.DELETE, _IconButtonDelegate("delete", self._table)
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
        header.setSectionResizeMode(AnnotationColumns.COLOR, QHeaderView.Fixed)
        header.setSectionResizeMode(AnnotationColumns.CLASS, QHeaderView.Stretch)
        header.setSectionResizeMode(AnnotationColumns.VERTICES, QHeaderView.Fixed)
        header.setSectionResizeMode(AnnotationColumns.AREA, QHeaderView.Fixed)
        header.setSectionResizeMode(AnnotationColumns.VISIBILITY, QHeaderView.Fixed)
        header.setSectionResizeMode(AnnotationColumns.DELETE, QHeaderView.Fixed)
        self._table.setColumnWidth(AnnotationColumns.COLOR, _COLOR_COL_W)
        self._table.setColumnWidth(AnnotationColumns.CLASS, _NAME_MIN_W)
        self._table.setColumnWidth(AnnotationColumns.VERTICES, _COUNT_W)
        self._table.setColumnWidth(AnnotationColumns.AREA, _AREA_W)
        self._table.setColumnWidth(AnnotationColumns.VISIBILITY, _VISIBILITY_W)
        self._table.setColumnWidth(AnnotationColumns.DELETE, _DELETE_W)
        self._table.sortByColumn(AnnotationColumns.CLASS, Qt.AscendingOrder)
        self._sync_table_height()

        layout.addWidget(self._table)

        self._empty_lbl = QLabel("No annotations")
        self._empty_lbl.setStyleSheet("color: gray; font-size: 11px;")
        self._empty_lbl.setContentsMargins(6, 4, 6, 4)
        layout.addWidget(self._empty_lbl)
        self._sync_empty_label()

    def set_current_row(self, row: int) -> None:
        self._current_row = row
        self._selected_idx = -1
        self._table_model.set_current_row(row)

    def select_annotation(self, idx: int) -> None:
        """Silently highlight *idx* without emitting annotation_selected."""
        self._selected_idx = idx
        self._sync_selection()

    def _on_table_index_activated(self, proxy_index) -> None:
        if not proxy_index.isValid():
            return
        idx = self._annotation_index_from_proxy(proxy_index)
        if idx < 0:
            return

        column = proxy_index.column()
        if column == AnnotationColumns.VISIBILITY:
            self._toggle_visibility(idx)
            return
        if column == AnnotationColumns.DELETE:
            self._delete_annotation(idx)
            return

        self.select_annotation(idx)
        self.annotation_selected.emit(idx)
        if column == AnnotationColumns.CLASS:
            self._table.edit(proxy_index)

    def _annotation_index_from_proxy(self, proxy_index) -> int:
        value = proxy_index.data(ANNOTATION_INDEX_ROLE)
        return int(value) if value is not None else -1

    def _proxy_row_for_annotation(self, idx: int) -> int:
        if idx < 0 or idx >= self._table_model.rowCount():
            return -1
        source_index = self._table_model.index(idx, AnnotationColumns.CLASS)
        proxy_index = self._proxy.mapFromSource(source_index)
        return proxy_index.row() if proxy_index.isValid() else -1

    def _sync_selection(self, *args) -> None:
        if not hasattr(self, "_table"):
            return
        selection_model = self._table.selectionModel()
        if selection_model is None:
            return
        if self._selected_idx < 0:
            selection_model.clearSelection()
            return

        proxy_row = self._proxy_row_for_annotation(self._selected_idx)
        if proxy_row < 0:
            selection_model.clearSelection()
            return

        proxy_index = self._proxy.index(proxy_row, AnnotationColumns.CLASS)
        selection_model.setCurrentIndex(
            proxy_index,
            QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows,
        )
        self._table.scrollTo(proxy_index, QAbstractItemView.EnsureVisible)

    def _delete_annotation(self, idx: int) -> None:
        if self._selected_idx == idx:
            self._selected_idx = -1
        elif self._selected_idx > idx:
            self._selected_idx -= 1
        self.dataset_model.delete_annotation(self._current_row, idx)

    def _toggle_visibility(self, idx: int) -> None:
        self.dataset_model.toggle_annotation_visibility(self._current_row, idx)

    def _on_model_reset(self) -> None:
        if self._selected_idx >= self._table_model.rowCount():
            self._selected_idx = -1
        self._sync_table_height()
        self._sync_empty_label()
        self._sync_selection()

    def _sync_empty_label(self) -> None:
        self._empty_lbl.setVisible(self._table_model.rowCount() == 0)

    def _sync_table_height(self) -> None:
        """Grow the table viewport to fit every annotation row without scrolling."""
        if not hasattr(self, "_table"):
            return
        header_h = self._table.horizontalHeader().sizeHint().height()
        rows_h = 0
        for row in range(self._proxy.rowCount()):
            rows_h += self._table.verticalHeader().sectionSize(row)
        frame_h = self._table.frameWidth() * 2
        self._table.setFixedHeight(header_h + rows_h + frame_h + 2)
        self._table.updateGeometry()
