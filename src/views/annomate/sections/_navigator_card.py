from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from models.navigator_model import (
    HAS_INSPECTOR_ROLE,
    HAS_NOTE_ROLE,
    IMAGE_STATE_ROLE,
    NavigatorColumns,
)

from views.icons import material_icon

from ._shared import _ClickableFrame, _COLOR_INCOMPLETE, _COLOR_REVIEWED, _COLOR_UNDECIDED

_CHEVRON_SIZE = 16
_ICON_EXPANDED = "expand_more"  # chevron down -- body visible
_ICON_COLLAPSED = "chevron_right"  # chevron right -- body hidden

_BADGE_ICON_SIZE = 14
_BADGE_ICON_COLOR = "#888888"
_ANNOTATIONS_ICON = "pentagon"
_INSPECTOR_ICON = "person"
_NOTE_ICON = "comment"

_STATUS_DOT_W = 10
_INCOMPLETE_STATES = ("reject_incomplete", "accept_conflict", "undecided_work")
_REVIEWED_STATES = ("accept_clean", "reject_reviewed")


def _badge_icon_label(name: str, tooltip: str) -> QLabel:
    lbl = QLabel()
    lbl.setPixmap(
        material_icon(name, size=_BADGE_ICON_SIZE, color=_BADGE_ICON_COLOR).pixmap(
            _BADGE_ICON_SIZE, _BADGE_ICON_SIZE
        )
    )
    lbl.setToolTip(tooltip)
    return lbl


def _apply_status_icon(label: QLabel, state: str) -> None:
    """Style *label* in place to match the status dot/ring/badge for *state*."""
    label.setFixedSize(_STATUS_DOT_W, _STATUS_DOT_W)
    label.setAlignment(Qt.AlignCenter)
    if state in _INCOMPLETE_STATES:
        label.setText("!")
        label.setStyleSheet(
            f"QLabel {{ color: {_COLOR_INCOMPLETE}; font-size: {_STATUS_DOT_W}px; "
            "font-weight: bold; background: transparent; border: none; }"
        )
    elif state in _REVIEWED_STATES:
        label.setText("")
        label.setStyleSheet(
            f"QLabel {{ background-color: {_COLOR_REVIEWED}; border-radius: "
            f"{_STATUS_DOT_W // 2}px; }}"
        )
    else:
        label.setText("")
        label.setStyleSheet(
            f"QLabel {{ border: 1.5px solid {_COLOR_UNDECIDED}; border-radius: "
            f"{_STATUS_DOT_W // 2}px; background: transparent; }}"
        )


class _NavigatorCard(QWidget):
    """One row of the dataset navigator: a clickable header, expandable in place.

    The collapsed header is a two-line at-a-glance summary:
        Row 1: filename (left)               annotation/inspector/note badges (right)
        Row 2: decision -- Accept/Reject/Undecided (left)     MicroSentry score (right)

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
        h.setContentsMargins(6, 4, 6, 4)
        h.setSpacing(6)

        self._status_lbl = QLabel()
        h.addWidget(self._status_lbl)

        text_col = QVBoxLayout()
        text_col.setSpacing(1)

        # Row 1: filename (left) -- annotation/inspector/note badges (right)
        row1 = QHBoxLayout()
        row1.setSpacing(4)

        self._filename_lbl = QLabel()
        self._filename_lbl.setStyleSheet("font-weight: bold;")
        row1.addWidget(self._filename_lbl)
        row1.addStretch()

        self._annot_count_lbl = QLabel()
        self._annot_count_lbl.setStyleSheet("color: palette(mid); font-size: 11px;")
        row1.addWidget(self._annot_count_lbl)
        self._annot_icon_lbl = _badge_icon_label(_ANNOTATIONS_ICON, "Has annotations")
        row1.addWidget(self._annot_icon_lbl)

        row1.addSpacing(4)
        self._inspector_icon_lbl = _badge_icon_label(
            _INSPECTOR_ICON, "Has an assigned inspector"
        )
        row1.addWidget(self._inspector_icon_lbl)

        self._note_icon_lbl = _badge_icon_label(_NOTE_ICON, "Has a note")
        row1.addWidget(self._note_icon_lbl)

        text_col.addLayout(row1)

        # Row 2: decision (left) -- MicroSentry score (right)
        row2 = QHBoxLayout()
        row2.setSpacing(8)

        self._decision_lbl = QLabel()
        row2.addWidget(self._decision_lbl)
        row2.addStretch()

        self._score_lbl = QLabel()
        self._score_lbl.setStyleSheet("color: palette(mid);")
        row2.addWidget(self._score_lbl)

        text_col.addLayout(row2)

        h.addLayout(text_col, 1)

        self._arrow_lbl = QLabel()
        self._arrow_lbl.setPixmap(
            material_icon(_ICON_COLLAPSED, size=_CHEVRON_SIZE).pixmap(
                _CHEVRON_SIZE, _CHEVRON_SIZE
            )
        )
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
        self._arrow_lbl.setPixmap(
            material_icon(
                _ICON_EXPANDED if expanded else _ICON_COLLAPSED, size=_CHEVRON_SIZE
            ).pixmap(_CHEVRON_SIZE, _CHEVRON_SIZE)
        )
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

        annots = model.data(model.index(row, NavigatorColumns.ANNOTS))
        has_annots = bool(annots)
        self._annot_count_lbl.setText(str(annots) if has_annots else "")
        self._annot_count_lbl.setVisible(has_annots)
        self._annot_icon_lbl.setVisible(has_annots)

        has_inspector = bool(
            model.data(model.index(row, NavigatorColumns.STATUS), HAS_INSPECTOR_ROLE)
        )
        self._inspector_icon_lbl.setVisible(has_inspector)

        has_note = bool(
            model.data(model.index(row, NavigatorColumns.STATUS), HAS_NOTE_ROLE)
        )
        self._note_icon_lbl.setVisible(has_note)

        decision = model.data(model.index(row, NavigatorColumns.DECISION))
        if decision:
            decision_color = model.data(
                model.index(row, NavigatorColumns.DECISION), Qt.ForegroundRole
            )
            color = decision_color.color().name() if decision_color is not None else None
            self._decision_lbl.setText(decision)
            self._decision_lbl.setStyleSheet(
                f"color: {color}; font-weight: bold;" if color else "font-weight: bold;"
            )
        else:
            self._decision_lbl.setText("Undecided")
            self._decision_lbl.setStyleSheet("color: palette(mid);")

        score = model.data(model.index(row, NavigatorColumns.SCORE))
        self._score_lbl.setVisible(self._microsentry_mode)
        self._score_lbl.setText(score or "")

        state = model.data(model.index(row, NavigatorColumns.STATUS), IMAGE_STATE_ROLE)
        _apply_status_icon(self._status_lbl, state)

        tooltip = model.data(model.index(row, NavigatorColumns.STATUS), Qt.ToolTipRole)
        self._header.setToolTip(tooltip or "")
