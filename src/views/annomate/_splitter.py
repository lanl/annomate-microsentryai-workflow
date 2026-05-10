"""Reusable styled splitter handle for the AnnoMate main window.

Handle colors (plain RGB tuples — converted to QColor only at paint time):
  HANDLE_DEFAULT  subtle grey  — resting state
  HANDLE_HOVER    orange       — mouse over handle
  HANDLE_DRAG     dark grey    — actively dragging
"""

from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QColor, QPalette
from PySide6.QtWidgets import QSplitter, QSplitterHandle

# ── Reusable handle color palette ────────────────────────────────────────────
HANDLE_DEFAULT = (176, 176, 176)  # subtle grey
HANDLE_HOVER = (255, 140, 0)  # orange
HANDLE_DRAG = (64, 64, 64)  # dark grey
# ─────────────────────────────────────────────────────────────────────────────

_MARGIN_DEFAULT = 2  # px inset on the long axis — shortens the visible bar
_CORNER_RADIUS = 4  # px rounded corner radius
_DOT_COUNT = 3  # number of grip dots
_DOT_SIZE = 3  # px diameter of each dot
_DOT_SPACING = 5  # px between dot centres
_DOT_ALPHA = 160  # opacity of grip dots (0–255)


class _StyledHandle(QSplitterHandle):
    """Splitter handle with rounded pill shape, grip dots, and hover/drag colours."""

    def __init__(self, orientation, parent, margin: int = _MARGIN_DEFAULT) -> None:
        super().__init__(orientation, parent)
        self.setMouseTracking(True)
        self._hovered = False
        self._dragging = False
        self._margin = margin

    # ── State transitions ─────────────────────────────────────────────────────

    def enterEvent(self, event) -> None:
        self._hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        self._hovered = False
        self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event) -> None:
        self._dragging = True
        self.update()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        self._dragging = False
        self._hovered = True  # mouse is still over the handle at release
        self.update()
        super().mouseReleaseEvent(event)

    # ── Rendering ─────────────────────────────────────────────────────────────

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Clear to the surrounding panel background
        painter.fillRect(self.rect(), self.palette().color(QPalette.Window))

        # Bar colour
        if self._dragging:
            r, g, b = HANDLE_DRAG
        elif self._hovered:
            r, g, b = HANDLE_HOVER
        else:
            r, g, b = HANDLE_DEFAULT

        # Shortened, rounded pill — inset on the long axis
        rect = self.rect()
        if self.orientation() == Qt.Horizontal:
            # Vertical bar (canvas ↔ panel splitter): inset top + bottom
            bar = rect.adjusted(1, self._margin, -1, -self._margin)
        else:
            # Horizontal bar (navigator ↕ sections splitter): inset left + right
            bar = rect.adjusted(self._margin, 1, -self._margin, -1)

        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(r, g, b))
        painter.drawRoundedRect(bar, _CORNER_RADIUS, _CORNER_RADIUS)

        # Grip dots — white, centred on the bar
        painter.setBrush(QColor(255, 255, 255, _DOT_ALPHA))
        cx = bar.center().x()
        cy = bar.center().y()
        half_span = (_DOT_COUNT - 1) * _DOT_SPACING // 2

        for i in range(_DOT_COUNT):
            offset = -half_span + i * _DOT_SPACING
            if self.orientation() == Qt.Horizontal:
                # Dots stacked vertically along the vertical bar
                dx = cx - _DOT_SIZE // 2
                dy = cy + offset - _DOT_SIZE // 2
            else:
                # Dots spaced horizontally along the horizontal bar
                dx = cx + offset - _DOT_SIZE // 2
                dy = cy - _DOT_SIZE // 2
            painter.drawEllipse(dx, dy, _DOT_SIZE, _DOT_SIZE)


class StyledSplitter(QSplitter):
    """QSplitter whose handle shows a rounded pill that highlights on hover/drag."""

    def __init__(self, orientation, margin: int = _MARGIN_DEFAULT, parent=None) -> None:
        super().__init__(orientation, parent)
        self._handle_margin = margin

    def createHandle(self) -> _StyledHandle:
        return _StyledHandle(self.orientation(), self, self._handle_margin)
