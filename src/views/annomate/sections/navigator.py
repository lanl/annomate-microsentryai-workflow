from PySide6.QtCore import QEvent, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListView,
    QMenu,
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
from ._navigator_delegate import _NavigatorRowDelegate
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

    Collapsed rows are virtualized: a QListView paints only the rows in the
    viewport via `_NavigatorRowDelegate`, instead of constructing a real
    QWidget per dataset row up front (the old approach, which froze the UI
    while loading hundreds of images). The one row that IS expanded is a
    real `_NavigatorCard`, manually parented as a child of the list's
    viewport and positioned with `setGeometry(self._list.visualRect(index))`
    -- deliberately NOT via `QAbstractItemView.setIndexWidget()`, which Qt's
    own docs mark as meant for static content only; there's no reliable way
    to get the view to size a row to fit a dynamically-sized index widget,
    and the view takes ownership of (and can silently delete) whatever's
    assigned to it. Normal Qt parent/child widget lifetime applies instead:
    we show/hide/delete the card ourselves, and reposition it on every
    scroll, resize, sort, filter, or content change that could move or
    resize its row (see `_reposition_expanded_card`).
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
        self._expanded_card: _NavigatorCard | None = None
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

        self._list = QListView()
        self._list.setFrameShape(QFrame.NoFrame)
        # Image navigation hotkeys are handled by AnnoMateWindow. QListView's
        # default StrongFocus consumes letter keys for keyboard search, so A/D
        # never reach the window after the navigator is clicked. Rows do not
        # use Qt selection or keyboard editing, so the view should not focus.
        self._list.setFocusPolicy(Qt.NoFocus)
        self._list.setSelectionMode(QAbstractItemView.NoSelection)
        self._list.setSpacing(0)
        self._list.setUniformItemSizes(False)  # the expanded row's height varies
        self._list.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self._list.viewport().setCursor(Qt.PointingHandCursor)
        self._list.setMinimumHeight(80)
        self._list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._list.setModel(self._proxy)
        # Must match the column used everywhere else in this file
        # (_proxy_row_from_source, _select_source_row, etc.) -- QListView
        # only lays out/reports visualRect() for its configured
        # modelColumn(); an index built with any other column silently
        # returns an empty rect, which is what was causing the expanded
        # card to sit at (0, 0, 0, 0) even though sizeHint()/doItemsLayout()
        # were computing the right row height all along.
        self._list.setModelColumn(NavigatorColumns.IMG_ID)

        self._delegate = _NavigatorRowDelegate(self._table_model, self._list, self)
        self._list.setItemDelegate(self._delegate)
        self._list.clicked.connect(self._on_list_clicked)
        # The expanded row's card is a manually-positioned child of the
        # viewport (see class docstring) -- anything that can move or
        # resize its row needs to reposition it explicitly.
        self._list.verticalScrollBar().valueChanged.connect(
            self._reposition_expanded_card
        )
        self._list.verticalScrollBar().rangeChanged.connect(
            self._reposition_expanded_card
        )
        self._list.viewport().installEventFilter(self)

        layout.addWidget(self._list)

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
        # layoutChanged in this Qt build -- force a relayout explicitly
        # rather than depending on QListView picking it up on its own.
        self._list.doItemsLayout()
        self._prune_selection_if_filtered_out()
        self._reposition_expanded_card()

    def _on_model_reset(self) -> None:
        self._collapse_expanded_widget()
        has_images = self.dataset_model.rowCount() > 0
        self._btn_prev.setVisible(has_images)
        self._btn_next.setVisible(has_images)
        self._list.setVisible(has_images)
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
        self._apply_filters()
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
        self._list.doItemsLayout()
        self._reposition_expanded_card()

    def _on_table_data_changed(self, top_left, bottom_right, roles=None) -> None:
        if self._selected_row < 0 or self._expanded_card is None:
            return
        if top_left.row() <= self._selected_row <= bottom_right.row():
            self._expanded_card.refresh()
            self._list.doItemsLayout()
            self._reposition_expanded_card()

    def _prune_selection_if_filtered_out(self) -> None:
        if self._selected_row < 0:
            return
        if self._proxy_row_from_source(self._selected_row) < 0:
            self._collapse_expanded_widget()

    def eventFilter(self, obj, event) -> bool:
        if obj is self._list.viewport() and event.type() == QEvent.Resize:
            self._reposition_expanded_card()
        elif (
            obj is self._expanded_card
            and event.type() == QEvent.LayoutRequest
        ):
            # The expanded card's body content (annotation count, expandable
            # notes, ...) can change size after the fact. Qt sends
            # LayoutRequest to a widget whenever its own layout's sizeHint
            # is invalidated -- catching that here is what tells QListView
            # to re-measure the row and the card to grow/shrink to match,
            # instead of clipping or leaving a gap.
            self._list.doItemsLayout()
            self._reposition_expanded_card()
        return super().eventFilter(obj, event)

    def _on_list_clicked(self, proxy_index) -> None:
        source_row = self._source_row_from_proxy(proxy_index)
        if source_row < 0:
            return
        self._on_card_clicked(source_row)

    def _on_card_clicked(self, source_row: int) -> None:
        if (
            source_row == self._selected_row
            and self._expanded_card is not None
            and self._expanded_card.is_expanded()
        ):
            self._collapse_expanded_widget()
            return

        self._select_source_row(source_row, scroll=False)
        self._update_counter(source_row)
        self.image_selected.emit(source_row)

    def _select_source_row(self, row: int, scroll: bool) -> None:
        proxy_row = self._proxy_row_from_source(row)
        if proxy_row < 0:
            self._selected_row = row
            return
        index = self._proxy.index(proxy_row, NavigatorColumns.IMG_ID)

        if row == self._selected_row and self._expanded_card is not None:
            # Already expanded on this exact row -- e.g. a duplicate/racy
            # click signal. Building a second card here would silently
            # steal the shared Annotations/Metadata widgets away from the
            # one already on screen (via _attach_shared_sections()),
            # leaving that one's body blank while a second, orphaned card
            # never gets shown -- exactly the "expands again into a bugged
            # state" symptom. Just make sure the existing card is
            # positioned correctly and stop.
            if scroll:
                self._scroll_row_to_top(index)
            else:
                self._reposition_expanded_card()
            return

        if self._selected_row >= 0 and self._selected_row != row:
            self._collapse_expanded_widget()
        self._selected_row = row

        # A real widget, but deliberately NOT via QListView.setIndexWidget()
        # -- see class docstring. Manually parented to the viewport instead,
        # so normal Qt widget ownership applies: we show/hide/delete it
        # ourselves, no risk of Qt's view-internal bookkeeping deleting it
        # (or us) out from under the other.
        card = _NavigatorCard(
            row, self._table_model, self._microsentry_mode, parent=self._list.viewport()
        )
        card.setAutoFillBackground(True)  # opaque -- otherwise the (unpainted) row shows through
        card.set_expanded(True)
        card.clicked.connect(self._on_card_clicked)
        self._expanded_card = card
        self._delegate.set_expanded_row(row)
        self._delegate.set_expanded_card(card)
        card.installEventFilter(self)
        self._attach_shared_sections()

        # show() BEFORE doItemsLayout(): Qt layouts skip hidden widgets when
        # computing sizeHint(), and a widget only counts as "visible" once
        # its whole ancestor chain is shown. Before this card is shown, its
        # body (and everything inside it -- annotations, metadata) is
        # effectively invisible to Qt's layout system no matter how many
        # times set_expanded(True)/addWidget() ran, so card.sizeHint() would
        # silently report just the header's height. Show first so the
        # delegate's sizeHint() (called from doItemsLayout()) sees the real,
        # fully-visible content.
        card.show()
        self._list.doItemsLayout()
        card.setGeometry(self._list.visualRect(index))
        card.raise_()

        if scroll:
            self._scroll_row_to_top(index)
        else:
            self._reposition_expanded_card()
        # doItemsLayout() doesn't always finish settling QListView's
        # internal row-position cache synchronously -- visualRect() can
        # still report stale/empty geometry for a beat afterward, even
        # though it's correct one event-loop turn later. Reposition once
        # more once that's had a chance to happen.
        QTimer.singleShot(0, self._reposition_expanded_card)

    def _collapse_expanded_widget(self) -> None:
        if self._expanded_card is None:
            return
        card = self._expanded_card
        self._release_shared_sections()  # detach the singletons BEFORE deleting the card
        card.removeEventFilter(self)
        card.set_expanded(False)  # deleteLater() is deferred -- don't leave it visually stale meanwhile
        self._delegate.set_expanded_row(-1)
        self._delegate.set_expanded_card(None)
        self._expanded_card = None
        self._selected_row = -1
        card.setParent(None)
        card.deleteLater()
        self._list.doItemsLayout()  # the row shrinks back to collapsed height
        self._list.viewport().update()  # force a repaint of the area the card vacated

    def _reposition_expanded_card(self, *args) -> None:
        """Move the expanded row's card to wherever its row currently sits.

        Needed because the card is a plain child widget with manually set
        geometry, not something QListView positions on its own -- call this
        after anything that could shift row positions: scrolling, resizing,
        sorting, filtering, or the card's own content changing size.
        """
        if self._expanded_card is None or self._selected_row < 0:
            return
        proxy_row = self._proxy_row_from_source(self._selected_row)
        if proxy_row < 0:
            return
        index = self._proxy.index(proxy_row, NavigatorColumns.IMG_ID)
        self._expanded_card.setGeometry(self._list.visualRect(index))

    def _scroll_row_to_top(self, index) -> None:
        """Scroll the list so *index* sits at the very top of the visible area.

        Used for Prev/Next (A/D) navigation so the currently viewed image's
        row stays anchored at the top instead of merely being scrolled into
        view somewhere in the middle or bottom.
        """
        self._list.scrollTo(index, QAbstractItemView.PositionAtTop)
        self._reposition_expanded_card()

    def _release_shared_sections(self) -> None:
        """Move the shared Annotations/Metadata widgets back to the holding slot.

        Must run before the expanded card is detached from a row -- keeps
        these singleton widgets from ever being reparented into limbo.
        """
        self.metadata.commit_pending_edits()
        self._shared_slot_layout.addWidget(self.annotations)
        self._shared_slot_layout.addWidget(self.metadata)

    def _attach_shared_sections(self) -> None:
        """Reparent the shared Annotations/Metadata widgets into the expanded card's body."""
        card = self._expanded_card
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
        if self._expanded_card is not None:
            self._expanded_card.set_microsentry_mode(enabled)
        self._delegate.set_microsentry_mode(enabled)
        self._table_model.refresh_inference()
        self._list.viewport().update()

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
