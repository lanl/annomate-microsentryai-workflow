from collections import OrderedDict

from PySide6.QtCore import QEvent, QModelIndex, QPoint, QSize, Qt
from PySide6.QtGui import QPainter, QPixmap
from PySide6.QtWidgets import QStyle, QStyledItemDelegate, QToolTip

from models.navigator_model import SOURCE_ROW_ROLE

from ._navigator_card import _NavigatorCard


class _NavigatorRowDelegate(QStyledItemDelegate):
    """Paints collapsed navigator rows without building a widget per row.

    QListView only calls paint()/sizeHint() for rows actually inside the
    viewport -- that's what makes the list virtualized. This delegate leans
    on that by painting with a single reused ("flyweight") hidden
    _NavigatorCard: on each call its content is retargeted to the row being
    drawn via set_source_row(), then its header is rendered into the target
    rect with QWidget.render(). Reusing the real card widget (rather than
    hand-rolling the pill tray/badges/status-dot in raw QPainter) guarantees
    the collapsed look stays pixel-identical to the expanded card's header,
    since it's the same widget code and stylesheets either way.

    The row that's actually expanded is skipped here entirely -- it's a
    real _NavigatorCard, manually parented into the list's viewport and
    positioned by DataNavigatorSection (see _select_source_row /
    _reposition_expanded_card), so this delegate just reports its live
    sizeHint instead of painting over it.
    """

    def __init__(self, table_model, list_view, parent=None) -> None:
        super().__init__(parent)
        self._list_view = list_view
        self._expanded_source_row = -1
        self._expanded_card: _NavigatorCard | None = None

        self._flyweight = _NavigatorCard(0, table_model, parent=list_view)
        self._flyweight.setVisible(False)
        self._collapsed_height = self._flyweight.sizeHint().height()
        self._pixmap_cache: OrderedDict[tuple, QPixmap] = OrderedDict()
        self._cache_limit = 512
        table_model.dataChanged.connect(self._on_model_data_changed)
        table_model.modelReset.connect(self.clear_cache)

    def set_expanded_card(self, card: _NavigatorCard) -> None:
        self._expanded_card = card

    def set_expanded_row(self, source_row: int) -> None:
        self._expanded_source_row = source_row

    def set_microsentry_mode(self, enabled: bool) -> None:
        self._flyweight.set_microsentry_mode(enabled)
        self._collapsed_height = self._flyweight.sizeHint().height()
        self.clear_cache()

    def clear_cache(self) -> None:
        self._pixmap_cache.clear()

    def _on_model_data_changed(self, top_left, bottom_right, roles=None) -> None:
        first = top_left.row()
        last = bottom_right.row()
        for key in list(self._pixmap_cache):
            if first <= key[0] <= last:
                del self._pixmap_cache[key]

    def paint(self, painter, option, index: QModelIndex) -> None:
        source_row = index.data(SOURCE_ROW_ROLE)
        if source_row is None or source_row == self._expanded_source_row:
            return  # the expanded row is real widget, placed via setIndexWidget

        hovered = bool(option.state & QStyle.State_MouseOver)
        device_pixel_ratio = option.widget.devicePixelRatioF() if option.widget else 1.0
        row_pixmap = self._row_pixmap(
            source_row, option.rect.width(), hovered, device_pixel_ratio
        )
        painter.drawPixmap(option.rect.topLeft(), row_pixmap)

    def _row_pixmap(
        self, source_row: int, width: int, hovered: bool, device_pixel_ratio: float
    ) -> QPixmap:
        """Return a cached, pixel-identical rendering of one collapsed row."""
        key = (source_row, width, hovered, device_pixel_ratio)
        cached = self._pixmap_cache.get(key)
        if cached is not None:
            self._pixmap_cache.move_to_end(key)
            return cached

        pixmap = QPixmap(
            max(1, round(width * device_pixel_ratio)),
            max(1, round(self._collapsed_height * device_pixel_ratio)),
        )
        pixmap.setDevicePixelRatio(device_pixel_ratio)
        pixmap.fill(Qt.transparent)

        self._flyweight.set_source_row(source_row)
        self._flyweight.set_hovered(hovered)
        self._flyweight.prepare_collapsed_render(width, self._collapsed_height)
        pixmap_painter = QPainter(pixmap)
        self._flyweight.render(pixmap_painter, QPoint(0, 0))
        pixmap_painter.end()

        self._pixmap_cache[key] = pixmap
        if len(self._pixmap_cache) > self._cache_limit:
            self._pixmap_cache.popitem(last=False)
        return pixmap

    def helpEvent(self, event, view, option, index: QModelIndex) -> bool:
        """Expose the real card's child tooltips for delegate-painted rows."""
        if event.type() != QEvent.ToolTip or not index.isValid():
            return super().helpEvent(event, view, option, index)

        source_row = index.data(SOURCE_ROW_ROLE)
        if source_row is None or source_row == self._expanded_source_row:
            return False  # the expanded row's real widget handles its own tips

        self._flyweight.set_source_row(source_row)
        self._flyweight.prepare_collapsed_render(
            option.rect.width(), self._collapsed_height
        )
        local_pos = event.pos() - option.rect.topLeft()
        tooltip = self._tooltip_at(local_pos)
        if tooltip:
            QToolTip.showText(event.globalPos(), tooltip, view)
            return True

        QToolTip.hideText()
        return True

    def _tooltip_at(self, local_pos: QPoint) -> str:
        """Return the nearest tooltip in the flyweight's child hierarchy."""
        child = self._flyweight.childAt(local_pos)
        while child is not None:
            tooltip = child.toolTip()
            if tooltip:
                return tooltip
            if child is self._flyweight:
                break
            child = child.parentWidget()
        return ""

    def sizeHint(self, option, index: QModelIndex) -> QSize:
        source_row = index.data(SOURCE_ROW_ROLE)
        width = self._list_view.viewport().width() or option.rect.width()
        if source_row == self._expanded_source_row and self._expanded_card is not None:
            # Read the card's own sizeHint directly -- do NOT resize() it
            # here first. Resizing using card.height() as the seed is a
            # trap: the first time this runs (before the card has ever been
            # shown), height() is some small stale default, that gets
            # force-applied, and every later call reseeds from that same
            # now-permanently-wrong value -- the row never grows past it.
            # sizeHint() is geometry-independent (computed from the layout
            # tree, invalidated whenever a child is added/removed), so it
            # reflects the real preferred height regardless of the card's
            # current on-screen size.
            return QSize(width, self._expanded_card.sizeHint().height())
        return QSize(width, self._collapsed_height)
