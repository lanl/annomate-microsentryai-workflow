from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from models.navigator_model import IMAGE_STATE_ROLE, NavigatorColumns

from ._shared import _ClickableFrame, _COLOR_INCOMPLETE, _COLOR_REVIEWED, _COLOR_UNDECIDED

_DOT_W = 10
_INCOMPLETE_STATES = ("reject_incomplete", "accept_conflict", "undecided_work")
_REVIEWED_STATES = ("accept_clean", "reject_reviewed")


def _apply_status_icon(label: QLabel, state: str) -> None:
    """Style *label* in place to match the status dot/ring/badge for *state*."""
    label.setFixedSize(_DOT_W, _DOT_W)
    label.setAlignment(Qt.AlignCenter)
    if state in _INCOMPLETE_STATES:
        label.setText("!")
        label.setStyleSheet(
            f"QLabel {{ color: {_COLOR_INCOMPLETE}; font-size: {_DOT_W}px; "
            "font-weight: bold; background: transparent; border: none; }"
        )
    elif state in _REVIEWED_STATES:
        label.setText("")
        label.setStyleSheet(
            f"QLabel {{ background-color: {_COLOR_REVIEWED}; border-radius: "
            f"{_DOT_W // 2}px; }}"
        )
    else:
        label.setText("")
        label.setStyleSheet(
            f"QLabel {{ border: 1.5px solid {_COLOR_UNDECIDED}; border-radius: "
            f"{_DOT_W // 2}px; background: transparent; }}"
        )


class _NavigatorCard(QWidget):
    """One row of the dataset navigator: a clickable header, expandable in place.

    Signals:
        clicked (int): Emitted with this card's source row when its header is clicked.
    """

    clicked = Signal(int)

    def __init__(
        self,
        source_row: int,
        table_model,
        microsentry_mode: bool = False,
        parent: QWidget = None,
    ) -> None:
        super().__init__(parent)
        self._source_row = source_row
        self._table_model = table_model
        self._microsentry_mode = microsentry_mode
        self._expanded = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._header = _ClickableFrame()
        self._header.setFrameShape(QFrame.StyledPanel)
        self._header.setCursor(Qt.PointingHandCursor)
        self._header.clicked.connect(
            lambda: self.clicked.emit(self._source_row)
        )
        h = QHBoxLayout(self._header)
        h.setContentsMargins(2, 4, 6, 4)
        h.setSpacing(6)

        self._status_lbl = QLabel()
        h.addWidget(self._status_lbl)

        text_col = QVBoxLayout()
        text_col.setSpacing(1)

        self._filename_lbl = QLabel()
        self._filename_lbl.setStyleSheet("font-weight: bold;")
        text_col.addWidget(self._filename_lbl)

        second_row = QHBoxLayout()
        second_row.setSpacing(8)
        self._decision_lbl = QLabel()
        second_row.addWidget(self._decision_lbl)
        self._annots_lbl = QLabel()
        self._annots_lbl.setStyleSheet("color: palette(mid);")
        second_row.addWidget(self._annots_lbl)
        second_row.addStretch()
        text_col.addLayout(second_row)

        h.addLayout(text_col, 1)

        self._score_lbl = QLabel()
        h.addWidget(self._score_lbl)

        self._arrow_lbl = QLabel("▸")
        h.addWidget(self._arrow_lbl)

        outer.addWidget(self._header)

        self._body = QWidget()
        self._body.setVisible(False)
        self._body.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(8, 4, 8, 8)
        outer.addWidget(self._body)

        self.refresh()

    def source_row(self) -> int:
        return self._source_row

    def is_expanded(self) -> bool:
        return self._expanded

    def set_expanded(self, expanded: bool) -> None:
        self._expanded = expanded
        self._body.setVisible(expanded)
        self._arrow_lbl.setText("▾" if expanded else "▸")
        self._header.setStyleSheet(
            "background-color: palette(highlight);" if expanded else ""
        )

    def body_container(self) -> QWidget:
        return self._body

    def set_microsentry_mode(self, enabled: bool) -> None:
        self._microsentry_mode = enabled
        self.refresh()

    def refresh(self) -> None:
        row = self._source_row
        model = self._table_model

        filename = model.data(model.index(row, NavigatorColumns.IMG_ID))
        self._filename_lbl.setText(filename or "")

        decision = model.data(model.index(row, NavigatorColumns.DECISION))
        self._decision_lbl.setText(decision or "")
        self._decision_lbl.setVisible(bool(decision))
        decision_color = model.data(
            model.index(row, NavigatorColumns.DECISION), Qt.ForegroundRole
        )
        if decision_color is not None:
            self._decision_lbl.setStyleSheet(
                f"color: {decision_color.color().name()}; font-weight: bold;"
            )

        annots = model.data(model.index(row, NavigatorColumns.ANNOTS))
        self._annots_lbl.setText(f"{annots} annotation(s)" if annots else "")

        score = model.data(model.index(row, NavigatorColumns.SCORE))
        self._score_lbl.setVisible(self._microsentry_mode)
        self._score_lbl.setText(score or "")

        state = model.data(model.index(row, NavigatorColumns.STATUS), IMAGE_STATE_ROLE)
        _apply_status_icon(self._status_lbl, state)

        tooltip = model.data(model.index(row, NavigatorColumns.STATUS), Qt.ToolTipRole)
        self._header.setToolTip(tooltip or "")
