from PySide6.QtCore import QModelIndex, QPoint, QSize
from PySide6.QtWidgets import QStyledItemDelegate

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

    def set_expanded_card(self, card: _NavigatorCard) -> None:
        self._expanded_card = card

    def set_expanded_row(self, source_row: int) -> None:
        self._expanded_source_row = source_row

    def set_microsentry_mode(self, enabled: bool) -> None:
        self._flyweight.set_microsentry_mode(enabled)
        self._collapsed_height = self._flyweight.sizeHint().height()

    def paint(self, painter, option, index: QModelIndex) -> None:
        source_row = index.data(SOURCE_ROW_ROLE)
        if source_row is None or source_row == self._expanded_source_row:
            return  # the expanded row is real widget, placed via setIndexWidget

        painter.save()
        self._flyweight.set_source_row(source_row)
        self._flyweight.prepare_collapsed_render(
            option.rect.width(), self._collapsed_height
        )
        painter.translate(option.rect.topLeft())
        self._flyweight.render(painter, QPoint(0, 0))
        painter.restore()

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
