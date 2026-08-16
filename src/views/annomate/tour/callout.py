"""_TourCallout — the small text card shown next to each highlighted feature."""

from PySide6.QtGui import QFont, QTextLayout, QTextOption
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import QPointF, Signal


class _TourCallout(QFrame):
    """Title + body + step counter + Back/Next/Skip buttons.

    Signals:
        next_requested (): User clicked Next/Done.
        back_requested (): User clicked Back.
        skip_requested (): User clicked Skip.
    """

    next_requested = Signal()
    back_requested = Signal()
    skip_requested = Signal()

    _WIDTH = 300
    _HEIGHT_PAD = 6

    def __init__(self, parent: QWidget = None) -> None:
        super().__init__(parent)
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        self.setAutoFillBackground(True)
        self.setFixedWidth(self._WIDTH)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(6)

        self._title_lbl = QLabel()
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(11)
        self._title_lbl.setFont(title_font)
        self._title_lbl.setWordWrap(True)
        layout.addWidget(self._title_lbl)

        self._body_lbl = QLabel()
        self._body_lbl.setWordWrap(True)
        layout.addWidget(self._body_lbl)

        self._counter_lbl = QLabel()
        self._counter_lbl.setStyleSheet("color: palette(mid); font-size: 10px;")
        layout.addWidget(self._counter_lbl)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)

        self._skip_btn = QPushButton("Skip")
        self._skip_btn.setToolTip("Close the tour (Esc)")
        self._skip_btn.clicked.connect(self.skip_requested)
        btn_row.addWidget(self._skip_btn)

        btn_row.addStretch()

        self._back_btn = QPushButton("Back")
        self._back_btn.clicked.connect(self.back_requested)
        btn_row.addWidget(self._back_btn)

        self._next_btn = QPushButton("Next")
        self._next_btn.clicked.connect(self.next_requested)
        btn_row.addWidget(self._next_btn)

        layout.addLayout(btn_row)

    def set_step(
        self,
        title: str,
        body: str,
        index: int,
        total: int,
        can_go_back: bool,
        is_last: bool,
    ) -> None:
        """Update the card's contents and button state for a step."""
        self._title_lbl.setText(title)
        self._body_lbl.setText(body)
        self._counter_lbl.setText(f"{index + 1} of {total}")
        self._back_btn.setVisible(can_go_back)
        self._next_btn.setText("Done" if is_last else "Next")
        self._sync_wrapped_label_heights()
        self.adjustSize()

    def _sync_wrapped_label_heights(self) -> None:
        """Explicitly size the title/body labels to fit their wrapped text.

        QLabel + QVBoxLayout's automatic heightForWidth pass doesn't reliably
        account for word-wrapped height on the first adjustSize() call
        (worse under HiDPI scaling), which clips or overlaps the last line.
        QLabel.heightForWidth() itself is unreliable here too — it returns
        stale values once a fixed height has previously been applied to the
        label. Measuring with QTextLayout instead — the same engine QLabel
        uses internally to paint wrapped text — avoids both: it's a pure
        text-measurement query with no dependency on the label's current
        size, and (unlike QFontMetrics.boundingRect, which Qt's own docs
        note is only an approximation for word-wrapped text and was seen
        here to undercount a wrapped line under the app's real font/style
        stack, clipping the last line) it reproduces the actual line breaks
        QLabel will paint. One extra line of headroom is still reserved on
        top of that as a safety margin.
        """
        margins = self.layout().contentsMargins()
        content_width = self._WIDTH - margins.left() - margins.right()
        for label in (self._title_lbl, self._body_lbl):
            height = self._wrapped_text_height(label.text(), label.font(), content_width)
            line_height = label.fontMetrics().lineSpacing()
            label.setFixedHeight(round(height) + line_height + self._HEIGHT_PAD)

    @staticmethod
    def _wrapped_text_height(text: str, font: QFont, width: int) -> float:
        layout = QTextLayout(text, font)
        option = QTextOption()
        option.setWrapMode(QTextOption.WordWrap)
        layout.setTextOption(option)
        layout.beginLayout()
        height = 0.0
        while True:
            line = layout.createLine()
            if not line.isValid():
                break
            line.setLineWidth(width)
            line.setPosition(QPointF(0, height))
            height += line.height()
        layout.endLayout()
        return height
