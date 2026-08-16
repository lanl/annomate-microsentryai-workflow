"""TourOverlay — dimmed spotlight scrim that highlights one widget at a time.

Painting technique mirrors ImageLabel._paint_center_crop (see image_label.py):
a QPainterPath combining the full rect with the highlight rect, filled with
Qt.OddEvenFill so only the area *outside* the highlight is dimmed.
"""

from typing import Optional

from PySide6.QtCore import QPoint, QRect, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget

from .callout import _TourCallout
from .steps import TourStep


class TourOverlay(QWidget):
    """Full-window child widget parented to AnnoMateWindow.

    Signals:
        next_requested (): Forwarded from the callout's Next/Done button.
        back_requested (): Forwarded from the callout's Back button.
        skip_requested (): Forwarded from the callout's Skip button, or Esc.
    """

    next_requested = Signal()
    back_requested = Signal()
    skip_requested = Signal()

    _MARGIN = 14
    _CORNER_PAD = 6
    _DIM_COLOR = QColor(0, 0, 0, 140)
    _HIGHLIGHT_COLOR = QColor(255, 255, 255, 220)

    def __init__(self, main_window: QWidget) -> None:
        super().__init__(main_window)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setFocusPolicy(Qt.StrongFocus)

        self._main_window = main_window
        self._step: Optional[TourStep] = None
        self._highlight_rect: Optional[QRect] = None

        self._callout = _TourCallout(self)
        self._callout.next_requested.connect(self.next_requested)
        self._callout.back_requested.connect(self.back_requested)
        self._callout.skip_requested.connect(self.skip_requested)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def show_step(
        self, step: TourStep, index: int, total: int, can_go_back: bool
    ) -> None:
        """Display *step*, updating the callout and spotlight position."""
        self._step = step
        is_last = index == total - 1
        self._callout.set_step(
            step.title, step.body, index, total, can_go_back, is_last
        )
        self.reposition(self._main_window)
        self.show()
        self.raise_()
        self.setFocus()

    def reposition(self, main_window: QWidget) -> None:
        """Match the overlay to *main_window*'s current size and re-place the highlight/callout."""
        self._main_window = main_window
        self.setGeometry(main_window.rect())
        self._update_highlight()
        self._place_callout()
        self.update()

    # ------------------------------------------------------------------ #
    # Internal layout
    # ------------------------------------------------------------------ #

    def _update_highlight(self) -> None:
        target = (
            self._step.target(self._main_window)
            if self._step is not None and self._step.target is not None
            else None
        )
        if target is not None and target.isVisible():
            top_left = target.mapTo(self._main_window, QPoint(0, 0))
            self._highlight_rect = QRect(top_left, target.size())
        else:
            self._highlight_rect = None

    def _place_callout(self) -> None:
        size = self._callout.size()
        bounds = self.rect()
        if self._highlight_rect is None:
            self._callout.move(
                (bounds.width() - size.width()) // 2,
                (bounds.height() - size.height()) // 2,
            )
            return

        r = self._highlight_rect
        margin = self._MARGIN
        candidates = [
            QPoint(r.right() + margin, r.center().y() - size.height() // 2),
            QPoint(
                r.left() - margin - size.width(), r.center().y() - size.height() // 2
            ),
            QPoint(r.center().x() - size.width() // 2, r.bottom() + margin),
            QPoint(
                r.center().x() - size.width() // 2, r.top() - margin - size.height()
            ),
        ]
        for pt in candidates:
            if bounds.contains(QRect(pt, size)):
                self._callout.move(pt)
                return
        self._callout.move(
            (bounds.width() - size.width()) // 2,
            (bounds.height() - size.height()) // 2,
        )

    # ------------------------------------------------------------------ #
    # Painting / input
    # ------------------------------------------------------------------ #

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        scrim = QPainterPath()
        scrim.setFillRule(Qt.OddEvenFill)
        scrim.addRect(QRectF(self.rect()))
        if self._highlight_rect is not None:
            pad = self._CORNER_PAD
            scrim.addRoundedRect(
                QRectF(self._highlight_rect).adjusted(-pad, -pad, pad, pad), 6, 6
            )
        painter.fillPath(scrim, self._DIM_COLOR)

        if self._highlight_rect is not None:
            pad = self._CORNER_PAD
            painter.setPen(QPen(self._HIGHLIGHT_COLOR, 2))
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(
                QRectF(self._highlight_rect).adjusted(-pad, -pad, pad, pad), 6, 6
            )

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Escape:
            self.skip_requested.emit()
            return
        super().keyPressEvent(event)
