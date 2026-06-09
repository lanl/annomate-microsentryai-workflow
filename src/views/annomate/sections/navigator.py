from PySide6.QtCore import QItemSelectionModel, Qt, Signal
from PySide6.QtGui import QAction, QColor, QPainter
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMenu,
    QSizePolicy,
    QStyledItemDelegate,
    QStyle,
    QStyleOptionViewItem,
    QTableView,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from models.navigator_model import (
    NavigatorColumns,
    NavigatorSortProxyModel,
    NavigatorTableModel,
    SOURCE_ROW_ROLE,
    STATUS_COLOR_ROLE,
)

from ._shared import _COLOR_IN_REVIEW, _COLOR_REVIEWED, _dot


_STATUS_COL_W = 34
_DOT_W = 10
_ANNOTS_W = 74
_DECISION_W = 82
_IMG_MIN_W = 90
_SCORE_W = 62
_CLASS_W = 76
_OPTIONAL_COLUMNS = (
    (NavigatorColumns.ANNOTS, "Annots"),
    (NavigatorColumns.DECISION, "Decision"),
    (NavigatorColumns.SCORE, "Score"),
    (NavigatorColumns.CLASS, "Class"),
)
_INFERENCE_COLUMNS = {NavigatorColumns.SCORE, NavigatorColumns.CLASS}


class _StatusDotDelegate(QStyledItemDelegate):
    """Paint the status column as a compact colored dot."""

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index) -> None:
        color = index.data(STATUS_COLOR_ROLE)
        if not color:
            super().paint(painter, option, index)
            return

        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        opt.text = ""
        style = opt.widget.style() if opt.widget is not None else QApplication.style()
        style.drawControl(QStyle.CE_ItemViewItem, opt, painter)

        rect = option.rect
        x = rect.x() + (rect.width() - _DOT_W) // 2
        y = rect.y() + (rect.height() - _DOT_W) // 2
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(color))
        painter.drawEllipse(x, y, _DOT_W, _DOT_W)
        painter.restore()


class DataNavigatorSection(QWidget):
    """Sortable dataset table for selecting images."""

    image_selected = Signal(int)
    prev_requested = Signal()
    next_requested = Signal()

    def __init__(
        self, dataset_model, inference_model=None, parent: QWidget = None
    ) -> None:
        super().__init__(parent)
        self.dataset_model = dataset_model
        self.inference_model = inference_model
        self._selected_row: int = -1
        self._microsentry_mode: bool = False
        self._column_actions: dict[int, QAction] = {}

        self._table_model = NavigatorTableModel(dataset_model, inference_model, self)
        self._proxy = NavigatorSortProxyModel(self)
        self._proxy.setSourceModel(self._table_model)

        self._init_ui()
        self.dataset_model.modelReset.connect(self._on_model_reset)
        self._proxy.layoutChanged.connect(self._on_proxy_order_changed)
        self._proxy.rowsMoved.connect(self._on_proxy_order_changed)
        self._proxy.modelReset.connect(self._on_proxy_order_changed)

        self._on_model_reset()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        nav_row = QWidget()
        nav_h = QHBoxLayout(nav_row)
        nav_h.setContentsMargins(0, 0, 0, 0)
        nav_h.setSpacing(4)

        self._btn_prev = QToolButton()
        self._btn_prev.setText("‹ Prev")
        self._btn_prev.setToolTip("Previous image")
        self._btn_prev.clicked.connect(self.prev_requested)
        nav_h.addWidget(self._btn_prev)

        self._btn_next = QToolButton()
        self._btn_next.setText("Next ›")
        self._btn_next.setToolTip("Next image")
        self._btn_next.clicked.connect(self.next_requested)
        nav_h.addWidget(self._btn_next)

        self._lbl_counter = QLabel("No images loaded")
        nav_h.addWidget(self._lbl_counter)
        nav_h.addStretch()
        nav_h.addWidget(_dot(_COLOR_IN_REVIEW))
        nav_h.addWidget(QLabel("In Review"))
        nav_h.addSpacing(4)
        nav_h.addWidget(_dot(_COLOR_REVIEWED))
        nav_h.addWidget(QLabel("Reviewed"))
        nav_h.addSpacing(4)

        self._btn_columns = QToolButton()
        self._btn_columns.setText("Columns")
        self._btn_columns.setToolTip("Choose navigator columns")
        self._btn_columns.setPopupMode(QToolButton.InstantPopup)
        self._btn_columns.setMenu(self._build_column_menu())
        nav_h.addWidget(self._btn_columns)

        layout.addWidget(nav_row)

        self._table = QTableView()
        self._table.setModel(self._proxy)
        self._table.setItemDelegateForColumn(
            NavigatorColumns.STATUS, _StatusDotDelegate(self._table)
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
        self._table.setMinimumHeight(80)
        self._table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
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
        vertical_header.setDefaultSectionSize(24)
        vertical_header.setMinimumSectionSize(20)

        header = self._table.horizontalHeader()
        header.setHighlightSections(False)
        header.setSectionsClickable(True)
        header.setSortIndicatorShown(True)
        header.setMinimumSectionSize(24)
        header.setSectionResizeMode(NavigatorColumns.STATUS, QHeaderView.Fixed)
        header.setSectionResizeMode(NavigatorColumns.IMG_ID, QHeaderView.Stretch)
        header.setSectionResizeMode(NavigatorColumns.ANNOTS, QHeaderView.Fixed)
        header.setSectionResizeMode(NavigatorColumns.DECISION, QHeaderView.Fixed)
        header.setSectionResizeMode(NavigatorColumns.SCORE, QHeaderView.Fixed)
        header.setSectionResizeMode(NavigatorColumns.CLASS, QHeaderView.Fixed)
        self._table.setColumnWidth(NavigatorColumns.STATUS, _STATUS_COL_W)
        self._table.setColumnWidth(NavigatorColumns.IMG_ID, _IMG_MIN_W)
        self._table.setColumnWidth(NavigatorColumns.ANNOTS, _ANNOTS_W)
        self._table.setColumnWidth(NavigatorColumns.DECISION, _DECISION_W)
        self._table.setColumnWidth(NavigatorColumns.SCORE, _SCORE_W)
        self._table.setColumnWidth(NavigatorColumns.CLASS, _CLASS_W)
        self._proxy.sort(-1)

        layout.addWidget(self._table)

    def _build_column_menu(self) -> QMenu:
        menu = QMenu(self)
        for column, label in _OPTIONAL_COLUMNS:
            action = QAction(label, self)
            action.setCheckable(True)
            action.setChecked(True)
            action.toggled.connect(
                lambda checked, col=column: self._on_column_toggled(col, checked)
            )
            menu.addAction(action)
            self._column_actions[column] = action
        return menu

    def _on_model_reset(self) -> None:
        has_images = self.dataset_model.rowCount() > 0
        self._btn_prev.setVisible(has_images)
        self._btn_next.setVisible(has_images)
        self._table.setVisible(has_images)
        self._selected_row = -1
        if has_images:
            total = self.dataset_model.rowCount()
            self._lbl_counter.setText(
                f"{total} image{'s' if total != 1 else ''} loaded"
            )
        else:
            self._lbl_counter.setText("No images loaded")
        self._sync_visible_columns()

    def _on_proxy_order_changed(self, *args) -> None:
        if self._selected_row >= 0:
            self.select_row(self._selected_row)

    def _on_table_index_activated(self, proxy_index) -> None:
        if not proxy_index.isValid():
            return
        source_row = self._source_row_from_proxy(proxy_index)
        if source_row < 0:
            return
        self._selected_row = source_row
        self._select_proxy_row(proxy_index.row(), scroll=False)
        self._update_counter(source_row)
        self.image_selected.emit(source_row)

    def _source_row_from_proxy(self, proxy_index) -> int:
        source_index = self._proxy.mapToSource(proxy_index)
        value = source_index.data(SOURCE_ROW_ROLE)
        return int(value) if value is not None else -1

    def _proxy_row_from_source(self, source_row: int) -> int:
        if not (0 <= source_row < self._table_model.rowCount()):
            return -1
        source_index = self._table_model.index(source_row, NavigatorColumns.IMG_ID)
        proxy_index = self._proxy.mapFromSource(source_index)
        return proxy_index.row() if proxy_index.isValid() else -1

    def _on_column_toggled(self, column: int, checked: bool) -> None:
        if (
            not checked
            and self._table.horizontalHeader().sortIndicatorSection() == column
        ):
            self._table.sortByColumn(NavigatorColumns.IMG_ID, Qt.AscendingOrder)
        self._sync_visible_columns()

    def _column_enabled(self, column: int) -> bool:
        action = self._column_actions.get(column)
        return True if action is None else action.isChecked()

    def _sync_visible_columns(self) -> None:
        self._table.setColumnHidden(NavigatorColumns.STATUS, False)
        self._table.setColumnHidden(NavigatorColumns.IMG_ID, False)
        self._table.setColumnHidden(
            NavigatorColumns.ANNOTS, not self._column_enabled(NavigatorColumns.ANNOTS)
        )
        self._table.setColumnHidden(
            NavigatorColumns.DECISION,
            not self._column_enabled(NavigatorColumns.DECISION),
        )
        self._table.setColumnHidden(
            NavigatorColumns.SCORE,
            not (
                self._microsentry_mode and self._column_enabled(NavigatorColumns.SCORE)
            ),
        )
        self._table.setColumnHidden(
            NavigatorColumns.CLASS,
            not (
                self._microsentry_mode and self._column_enabled(NavigatorColumns.CLASS)
            ),
        )

    def _update_counter(self, source_row: int) -> None:
        total = self.dataset_model.rowCount()
        if total <= 0:
            self._lbl_counter.setText("No images loaded")
            return
        proxy_row = self._proxy_row_from_source(source_row)
        position = proxy_row if proxy_row >= 0 else source_row
        self._lbl_counter.setText(f"{position + 1} / {total}")

    def _select_proxy_row(self, proxy_row: int, scroll: bool) -> None:
        proxy_index = self._proxy.index(proxy_row, NavigatorColumns.IMG_ID)
        if not proxy_index.isValid():
            return

        selection_model = self._table.selectionModel()
        selection_model.setCurrentIndex(
            proxy_index,
            QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows,
        )
        if scroll:
            self._table.scrollTo(proxy_index, QAbstractItemView.PositionAtCenter)

    # ── Public interface ──────────────────────────────────────────────────────

    def select_row(self, row: int) -> None:
        """Silently highlight *row* without emitting image_selected."""
        self._selected_row = row
        proxy_row = self._proxy_row_from_source(row)
        if proxy_row < 0:
            return

        self._select_proxy_row(proxy_row, scroll=True)
        self._update_counter(row)

    def set_counter(self, current: int, total: int) -> None:
        if total > 0:
            self._update_counter(current)

    def set_row_inference(self, row: int, score: float, label: str) -> None:
        """Update the Score and Class cells for a single source dataset row."""
        self._table_model.notify_inference_changed(row)
        if row == self._selected_row:
            self._update_counter(row)

    def set_microsentry_mode(self, enabled: bool) -> None:
        """Show or hide the Score and Class columns across the navigator."""
        self._microsentry_mode = enabled
        self._sync_visible_columns()
        self._table_model.refresh_inference()
        if self._selected_row >= 0:
            self.select_row(self._selected_row)

    def enable_inference_columns(self) -> None:
        """Check and show Score and Class columns if not already visible."""
        for col in _INFERENCE_COLUMNS:
            action = self._column_actions.get(col)
            if action and not action.isChecked():
                action.setChecked(True)

    def adjacent_source_row(self, current_source_row: int, step: int) -> int:
        """Return the source row adjacent in the current visible sort order."""
        proxy_row = self._proxy_row_from_source(current_source_row)
        if proxy_row < 0:
            return -1

        next_proxy_row = proxy_row + step
        if not (0 <= next_proxy_row < self._proxy.rowCount()):
            return -1

        proxy_index = self._proxy.index(next_proxy_row, NavigatorColumns.IMG_ID)
        return self._source_row_from_proxy(proxy_index)
