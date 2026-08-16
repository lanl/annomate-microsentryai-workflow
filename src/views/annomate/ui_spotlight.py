"""spotlight_widget() — briefly draws a pulsing highlight border around a live
widget, used by the Help search dialog's "Show Me" button (see help_dialog.py)
to point at whatever button/panel a HelpEntry.object_name refers to.

Handles the wrinkles real widgets in this app have before a target can
actually be shown: a target collapsed inside a _CollapsibleSection is
expanded first, a target inside a collapsed RightPanel tab has that tab
opened, a target inside the collapsed LeftPanel has it expanded, and a
target inside a QScrollArea is scrolled into view.
"""

from PySide6.QtCore import QPoint, QRect, QTimer, Qt, Property, QPropertyAnimation
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QScrollArea, QWidget

from views.annomate.left_panel import LeftPanel
from views.annomate.right_panel import RightPanel
from views.annomate.sections._collapsible import _CollapsibleSection

_PULSE_COLOR = QColor(255, 153, 0)
_PULSE_DURATION_MS = 900
_PULSE_COUNT = 3
_MARGIN = 4
_REPOSITION_INTERVAL_MS = 30


class _SpotlightOverlay(QWidget):
    """Transparent, click-through overlay that tracks *target*'s on-screen
    rect within *window* and paints a pulsing border around it."""

    def __init__(self, target: QWidget, window: QWidget) -> None:
        super().__init__(window)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_NoSystemBackground)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self._target = target
        self._window = window
        self._alpha = 60

        self._reposition_timer = QTimer(self)
        self._reposition_timer.timeout.connect(self._reposition)
        self._reposition_timer.start(_REPOSITION_INTERVAL_MS)

        self._anim = QPropertyAnimation(self, b"alpha", self)
        self._anim.setDuration(_PULSE_DURATION_MS)
        self._anim.setKeyValueAt(0.0, 60)
        self._anim.setKeyValueAt(0.5, 235)
        self._anim.setKeyValueAt(1.0, 60)
        self._anim.setLoopCount(_PULSE_COUNT)

        self._reposition()
        self.show()
        self.raise_()
        self._anim.start()

        QTimer.singleShot(_PULSE_DURATION_MS * _PULSE_COUNT + 150, self._finish)

    def _get_alpha(self) -> int:
        return self._alpha

    def _set_alpha(self, value: int) -> None:
        self._alpha = value
        self.update()

    alpha = Property(int, _get_alpha, _set_alpha)

    def _reposition(self) -> None:
        if self._target is None:
            self._finish()
            return
        try:
            visible = self._target.isVisible()
        except RuntimeError:
            visible = False
        if not visible:
            self._finish()
            return
        top_left = self._target.mapTo(self._window, QPoint(0, 0))
        rect = QRect(top_left, self._target.size())
        rect = rect.adjusted(-_MARGIN, -_MARGIN, _MARGIN, _MARGIN)
        self.setGeometry(rect)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        color = QColor(_PULSE_COLOR)
        color.setAlpha(self._alpha)
        painter.setPen(QPen(color, 3))
        painter.drawRoundedRect(self.rect().adjusted(2, 2, -2, -2), 8, 8)

    def _finish(self) -> None:
        self._reposition_timer.stop()
        self._anim.stop()
        self.close()
        self.deleteLater()


def spotlight_widget(widget: QWidget) -> bool:
    """Expand any collapsed ancestor section, scroll *widget* into view, and
    flash a pulsing highlight border around it.

    Returns False (and does nothing else) if *widget* is None or not
    currently visible/enabled-for-display — e.g. a review bar that only
    appears once an image is loaded.
    """
    if widget is None:
        return False

    node = widget.parentWidget()
    while node is not None:
        if isinstance(node, _CollapsibleSection):
            node.set_expanded(True)
        elif isinstance(node, RightPanel):
            node.show_tab_for(widget)
        elif isinstance(node, LeftPanel):
            node.set_collapsed(False)
        node = node.parentWidget()

    if not widget.isVisible():
        return False

    def _reveal() -> None:
        if not widget.isVisible():
            return
        node = widget.parentWidget()
        while node is not None:
            if isinstance(node, QScrollArea):
                node.ensureWidgetVisible(widget, 40, 40)
            node = node.parentWidget()
        window = widget.window()
        window.raise_()
        window.activateWindow()
        _SpotlightOverlay(widget, window)

    QTimer.singleShot(0, _reveal)
    return True
