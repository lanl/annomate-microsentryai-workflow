from PySide6.QtCore import QCoreApplication, Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMenu,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from models.navigator_model import (
    NavigatorColumns,
    NavigatorSortProxyModel,
    NavigatorTableModel,
    SOURCE_ROW_ROLE,
)

from views.icons import material_icon

from .annotations import AnnotationsSection
from .metadata import MetadataSection
from ._filter_panel import _FilterPanel
from ._navigator_card import _NavigatorCard
from ._shared import (
    _ClickableFrame,
    _COLOR_REVIEWED,
    _COLOR_SELECTED_BG,
    _dot,
    _incomplete_badge,
    _ring_undecided,
)

_CHIP_ACTIVE_STYLE = f"background-color: {_COLOR_SELECTED_BG}; border-radius: 4px;"


class DataNavigatorSection(QWidget):
    """Scrollable, sortable list of expandable dataset cards for selecting images.

    Each card expands to show that image's annotations, inspector, and note
    inline. Since only one card is expanded at a time, the Annotations and
    Metadata sections are shared singleton widgets that get reparented into
    whichever card is currently expanded (see `_attach_shared_sections`/
    `_release_shared_sections`).
    """

    image_selected = Signal(int)
    prev_requested = Signal()
    next_requested = Signal()
    annotation_selected = Signal(int)
    state_counts_changed = Signal(int, int, int)

    def __init__(
        self,
        dataset_model,
        inference_model=None,
        calibration_model=None,
        parent: QWidget = None,
    ) -> None:
        super().__init__(parent)
        self.dataset_model = dataset_model
        self.inference_model = inference_model
        self._selected_row: int = -1
        self._microsentry_mode: bool = False
        self._annotation_mode: str = "pixel"
        self._cards: dict[int, _NavigatorCard] = {}
        self._filter_chips: dict[str, _ClickableFrame] = {}
        self._sort_column: int = NavigatorColumns.IMG_ID
        self._sort_order: Qt.SortOrder = Qt.AscendingOrder

        self._table_model = NavigatorTableModel(dataset_model, inference_model, self)
        self._proxy = NavigatorSortProxyModel(self)
        self._proxy.setSourceModel(self._table_model)
        self._proxy.sort(self._sort_column, self._sort_order)

        self.annotations = AnnotationsSection(dataset_model, calibration_model)
        self.annotations.annotation_selected.connect(self.annotation_selected)
        self.metadata = MetadataSection(dataset_model)

        self._init_ui()
        self.dataset_model.modelReset.connect(self._on_model_reset)
        self.dataset_model.dataChanged.connect(self._refresh_counts)
        self._table_model.dataChanged.connect(self._on_table_data_changed)
        self._proxy.layoutChanged.connect(self._on_proxy_order_changed)
        self._proxy.rowsMoved.connect(self._on_proxy_order_changed)
        self._proxy.modelReset.connect(self._on_proxy_order_changed)

        self._on_model_reset()

    def _init_ui(self) -> None:
        self.setStyleSheet(
            "QToolTip { padding: 2px 4px; border-radius: 4px; border: 1px solid palette(shadow); }"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        nav_row = QWidget()
        nav_h = QHBoxLayout(nav_row)
        nav_h.setContentsMargins(8, 0, 0, 0)
        nav_h.setSpacing(4)

        self._btn_prev = QToolButton()
        self._btn_prev.setText("Prev (A)")
        self._btn_prev.setStyleSheet("color: black;")
        self._btn_prev.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._btn_prev.setToolTip("Previous image")
        self._btn_prev.clicked.connect(self.prev_requested)
        nav_h.addWidget(self._btn_prev)

        self._btn_next = QToolButton()
        self._btn_next.setLayoutDirection(Qt.RightToLeft)
        self._btn_next.setText("Next (D)")
        self._btn_next.setStyleSheet("color: black;")
        self._btn_next.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._btn_next.setToolTip("Next image")
        self._btn_next.clicked.connect(self.next_requested)
        nav_h.addWidget(self._btn_next)

        self._lbl_counter = QLabel("No images loaded")
        self._lbl_counter.setStyleSheet("color: black;")
        nav_h.addWidget(self._lbl_counter)
        nav_h.addStretch()

        layout.addWidget(nav_row)

        filter_row = QWidget()
        filter_h = QHBoxLayout(filter_row)
        filter_h.setContentsMargins(8, 0, 0, 0)
        filter_h.setSpacing(6)

        _TIP_UNDECIDED = "Undecided: no Accept or Reject decision has been set."
        _TIP_REVIEWED = (
            "Reviewed: accepted, or rejected with a polygon annotation or class tag."
        )
        _TIP_INCOMPLETE = "Incomplete: action needed. Reject missing evidence, accepted image has annotations, or work present with no decision."

        self._lbl_count_undecided = QLabel("0")
        self._lbl_count_undecided.setStyleSheet("color: black;")
        self._add_filter_chip(
            filter_h, "undecided", _ring_undecided(), self._lbl_count_undecided,
            _TIP_UNDECIDED,
        )
        self._lbl_count_reviewed = QLabel("0")
        self._lbl_count_reviewed.setStyleSheet("color: black;")
        self._add_filter_chip(
            filter_h, "reviewed", _dot(_COLOR_REVIEWED), self._lbl_count_reviewed,
            _TIP_REVIEWED,
        )
        self._lbl_count_incomplete = QLabel("0")
        self._lbl_count_incomplete.setStyleSheet("color: black;")
        self._add_filter_chip(
            filter_h, "incomplete", _incomplete_badge(), self._lbl_count_incomplete,
            _TIP_INCOMPLETE,
        )
        filter_h.addStretch()

        self._btn_filter = QToolButton()
        self._btn_filter.setIcon(material_icon("filter_alt", size=16, color="black"))
        self._btn_filter.setText("Filter")
        self._btn_filter.setStyleSheet("color: black;")
        self._btn_filter.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._btn_filter.setToolTip("Filter and sort images")
        self._btn_filter.setPopupMode(QToolButton.InstantPopup)

        self._filter_panel = _FilterPanel()
        self._filter_panel.decision_toggled.connect(self._on_panel_decision_toggled)
        self._filter_panel.status_toggled.connect(self._on_panel_status_toggled)
        self._filter_panel.class_toggled.connect(self._on_panel_class_toggled)
        self._filter_panel.sort_field_clicked.connect(self._on_sort_field_chosen)
        self._filter_panel.clear_filters_clicked.connect(self._on_clear_filters_clicked)

        filter_menu = QMenu(self)
        filter_action = QWidgetAction(self)
        filter_action.setDefaultWidget(self._filter_panel)
        filter_menu.addAction(filter_action)
        self._btn_filter.setMenu(filter_menu)
        filter_h.addWidget(self._btn_filter)

        layout.addWidget(filter_row)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scroll.setFrameShape(QScrollArea.NoFrame)
        self._scroll.setMinimumHeight(80)
        self._scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self._cards_container = QWidget()
        self._cards_layout = QVBoxLayout(self._cards_container)
        self._cards_layout.setContentsMargins(0, 0, 0, 0)
        self._cards_layout.setSpacing(0)
        self._cards_layout.addStretch()
        self._scroll.setWidget(self._cards_container)

        layout.addWidget(self._scroll)

        # Permanent, invisible parent for the shared Annotations/Metadata
        # sections when no card is expanded. Keeps them real descendants of
        # this widget (and thus the window) at all times, independent of
        # whether any card currently hosts them.
        self._shared_slot = QWidget(self)
        self._shared_slot.setVisible(False)
        self._shared_slot_layout = QVBoxLayout(self._shared_slot)
        self._shared_slot_layout.setContentsMargins(0, 0, 0, 0)
        self._shared_slot_layout.addWidget(self.annotations)
        self._shared_slot_layout.addWidget(self.metadata)

    def _add_filter_chip(self, layout, mode: str, icon, count_label: QLabel, tooltip: str):
        chip = _ClickableFrame()
        chip.setCursor(Qt.PointingHandCursor)
        chip.setToolTip(tooltip)
        chip_h = QHBoxLayout(chip)
        chip_h.setContentsMargins(3, 2, 5, 2)
        chip_h.setSpacing(3)
        icon.setToolTip(tooltip)
        chip_h.addWidget(icon)
        count_label.setToolTip(tooltip)
        chip_h.addWidget(count_label)
        chip.clicked.connect(lambda: self._on_chip_clicked(mode))
        layout.addWidget(chip)
        self._filter_chips[mode] = chip

    def _on_sort_field_chosen(self, column: int) -> None:
        if column == self._sort_column:
            self._sort_order = (
                Qt.DescendingOrder
                if self._sort_order == Qt.AscendingOrder
                else Qt.AscendingOrder
            )
        else:
            self._sort_column = column
            self._sort_order = Qt.AscendingOrder
        self._proxy.sort(self._sort_column, self._sort_order)
        self._filter_panel.set_sort_state(self._sort_column, self._sort_order)

    def _on_chip_clicked(self, mode: str) -> None:
        active = mode in self._proxy.status_filter()
        self._proxy.set_status_filter_active(mode, not active)
        self._apply_filters()

    def _on_panel_decision_toggled(self, decision: str, checked: bool) -> None:
        self._proxy.set_decision_filter_active(decision, checked)
        self._apply_filters()

    def _on_panel_status_toggled(self, status: str, checked: bool) -> None:
        self._proxy.set_status_filter_active(status, checked)
        self._apply_filters()

    def _on_panel_class_toggled(self, class_name: str, checked: bool) -> None:
        self._proxy.set_class_filter_active(class_name, checked)
        self._apply_filters()

    def _on_clear_filters_clicked(self) -> None:
        self._proxy.clear_filters()
        self._apply_filters()

    def _apply_filters(self) -> None:
        for chip_mode, chip in self._filter_chips.items():
            chip.setStyleSheet(
                _CHIP_ACTIVE_STYLE if chip_mode in self._proxy.status_filter() else ""
            )
        self._filter_panel.set_decision_filter(self._proxy.decision_filter())
        self._filter_panel.set_status_filter(self._proxy.status_filter())
        self._filter_panel.set_class_filter(self._proxy.class_filter())
        n = self._proxy.active_filter_count()
        self._btn_filter.setText("Filter" if n == 0 else f"Filter ({n})")
        # invalidateFilter()/invalidateRowsFilter() don't reliably emit
        # layoutChanged in this Qt build, so _on_proxy_order_changed never
        # fires from a filter change alone -- rebuild explicitly instead of
        # depending on that signal.
        self._rebuild_cards()

    def _on_model_reset(self) -> None:
        self._release_shared_sections()
        has_images = self.dataset_model.rowCount() > 0
        self._btn_prev.setVisible(has_images)
        self._btn_next.setVisible(has_images)
        self._scroll.setVisible(has_images)
        self._selected_row = -1
        self.annotations.set_current_row(-1)
        self.metadata.set_current_row(-1)
        if has_images:
            total = self.dataset_model.rowCount()
            self._lbl_counter.setText(
                f"{total} image{'s' if total != 1 else ''} loaded"
            )
        else:
            self._lbl_counter.setText("No images loaded")
        self._proxy.clear_filters()
        self._apply_filters()  # also rebuilds the card list
        self._refresh_counts()

    def _refresh_counts(self, *args) -> None:
        facet_counts = self._table_model.get_filter_facet_counts()
        status_counts = facet_counts["status"]
        self._lbl_count_reviewed.setText(str(status_counts["reviewed"]))
        self._lbl_count_incomplete.setText(str(status_counts["incomplete"]))
        self._lbl_count_undecided.setText(str(status_counts["undecided"]))
        self.state_counts_changed.emit(
            status_counts["undecided"], status_counts["reviewed"], status_counts["incomplete"]
        )
        self._filter_panel.set_decision_counts(facet_counts["decision"])
        self._filter_panel.set_status_counts(status_counts)
        self._filter_panel.set_class_options(facet_counts["class_options"])
        self._filter_panel.set_class_filter(self._proxy.class_filter())

    def get_image_state_label(self, source_row: int) -> str:
        return self._table_model.get_image_state_label(source_row)

    def _on_proxy_order_changed(self, *args) -> None:
        self._rebuild_cards()
        if self._selected_row >= 0:
            self.select_row(self._selected_row)

    def _on_table_data_changed(self, top_left, bottom_right, roles=None) -> None:
        for row in range(top_left.row(), bottom_right.row() + 1):
            card = self._cards.get(row)
            if card is not None:
                card.refresh()

    def _rebuild_cards(self) -> None:
        valid_source_rows = set()
        for proxy_row in range(self._proxy.rowCount()):
            proxy_index = self._proxy.index(proxy_row, NavigatorColumns.IMG_ID)
            source_row = self._source_row_from_proxy(proxy_index)
            if source_row < 0:
                continue
            valid_source_rows.add(source_row)
            card = self._cards.get(source_row)
            if card is None:
                card = _NavigatorCard(
                    source_row,
                    self._table_model,
                    self._microsentry_mode,
                )
                card.clicked.connect(self._on_card_clicked)
                self._cards[source_row] = card
            else:
                card.refresh()
            self._cards_layout.insertWidget(proxy_row, card)

        for row in list(self._cards.keys()):
            if row not in valid_source_rows:
                if row == self._selected_row:
                    self._release_shared_sections()
                card = self._cards.pop(row)
                self._cards_layout.removeWidget(card)
                card.deleteLater()

    def _on_card_clicked(self, source_row: int) -> None:
        card = self._cards.get(source_row)
        if (
            source_row == self._selected_row
            and card is not None
            and card.is_expanded()
        ):
            self._release_shared_sections()
            card.set_expanded(False)
            return

        self._select_source_row(source_row, scroll=False)
        self._update_counter(source_row)
        self.image_selected.emit(source_row)

    def _select_source_row(self, row: int, scroll: bool) -> None:
        if self._selected_row in self._cards and self._selected_row != row:
            self._cards[self._selected_row].set_expanded(False)
            self._release_shared_sections()
        self._selected_row = row
        card = self._cards.get(row)
        if card is not None:
            card.set_expanded(True)
            self._attach_shared_sections(card)
            if scroll:
                self._scroll_card_to_top(card)

    def _scroll_card_to_top(self, card: _NavigatorCard) -> None:
        """Scroll the list so *card* sits at the very top of the visible area.

        Used for Prev/Next (A/D) navigation so the currently viewed image's
        card stays anchored at the top instead of merely being scrolled into
        view somewhere in the middle or bottom. Expanding the card just now
        dirtied the layout, so force it to settle before reading card.y() --
        otherwise it (and the scrollbar's range) would still reflect stale,
        pre-expansion geometry.
        """
        self._cards_layout.activate()
        QCoreApplication.sendPostedEvents()
        QCoreApplication.processEvents()
        self._scroll.verticalScrollBar().setValue(card.y())

    def _release_shared_sections(self) -> None:
        """Move the shared Annotations/Metadata widgets back to the holding slot.

        Must run before a card that might currently host them is deleted —
        otherwise Qt would cascade-delete these singleton widgets along with
        the card.
        """
        self.metadata.commit_pending_edits()
        self._shared_slot_layout.addWidget(self.annotations)
        self._shared_slot_layout.addWidget(self.metadata)

    def _attach_shared_sections(self, card: _NavigatorCard) -> None:
        """Reparent the shared Annotations/Metadata widgets into *card*'s body."""
        body_layout = card.body_container().layout()
        body_layout.addWidget(self.annotations)
        body_layout.addWidget(self.metadata)
        self.annotations.set_current_row(card.source_row())
        self.metadata.set_current_row(card.source_row())

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

    def _update_counter(self, source_row: int) -> None:
        total = self.dataset_model.rowCount()
        if total <= 0:
            self._lbl_counter.setText("No images loaded")
            return
        proxy_row = self._proxy_row_from_source(source_row)
        position = proxy_row if proxy_row >= 0 else source_row
        self._lbl_counter.setText(f"{position + 1} / {total}")

    # ── Public interface ──────────────────────────────────────────────────────

    def select_row(self, row: int) -> None:
        """Silently highlight *row* without emitting image_selected."""
        self._select_source_row(row, scroll=True)
        self._update_counter(row)

    def set_counter(self, current: int, total: int) -> None:
        if total > 0:
            self._update_counter(current)

    def set_row_inference(self, row: int, score: float) -> None:
        """Update the Score cell for a single source dataset row."""
        self._table_model.notify_inference_changed(row)
        if row == self._selected_row:
            self._update_counter(row)

    def set_microsentry_mode(self, enabled: bool) -> None:
        """Show or hide the Score field across the navigator cards."""
        self._microsentry_mode = enabled
        for card in self._cards.values():
            card.set_microsentry_mode(enabled)
        self._table_model.refresh_inference()

    def enable_inference_columns(self) -> None:
        """Reveal the Score field; called once inference data is available."""
        self.set_microsentry_mode(True)

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

    def select_annotation(self, idx: int) -> None:
        """Silently highlight annotation *idx* in the expanded card's annotation list."""
        self.annotations.select_annotation(idx)

    def set_annotation_mode(self, mode: str) -> None:
        """Show or hide the Annotations sub-block based on pixel vs image-level mode."""
        self._annotation_mode = mode
        self.annotations.setVisible(mode == "pixel")
