from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QFont, QFontMetrics
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

_PILL_FONT_PX = 10
_PILL_PADDING_X = 4
_PILL_BORDER_W = 1
_PILL_SPACING = 4


def _vline() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.VLine)
    line.setFrameShadow(QFrame.Sunken)
    line.setFixedWidth(2)
    return line


def _pill_text_width(name: str) -> int:
    font = QFont()
    font.setPixelSize(_PILL_FONT_PX)
    return (
        QFontMetrics(font).horizontalAdvance(name)
        + 2 * _PILL_PADDING_X
        + 2 * _PILL_BORDER_W
    )


def _make_pill(name: str, rgb) -> QLabel:
    lbl = QLabel(name)
    lbl.setAlignment(Qt.AlignCenter)
    lbl.setStyleSheet(
        f"QLabel {{ border: {_PILL_BORDER_W}px solid rgb{tuple(rgb)}; "
        f"border-radius: 7px; padding: 0px {_PILL_PADDING_X}px; color: black; "
        f"font-size: {_PILL_FONT_PX}px; background: transparent; }}"
    )
    return lbl


class _ClassPillTray(QWidget):
    """Row of colored class-name pills, showing only as many as fit its width.

    Extras beyond what fits are simply dropped -- this lives in the compact
    "at a glance" card row, not a place for wrapping or eliding. Pills are
    packed against the left edge (next to the divider after the decision
    label); any leftover space trails to the right.
    """

    def __init__(self, parent: QWidget = None) -> None:
        super().__init__(parent)
        self._entries: list = []
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(_PILL_SPACING)

    def sizeHint(self) -> QSize:
        # Pinned regardless of content: this widget is purely reactive to the
        # width its parent layout hands it. If its own sizeHint grew/shrank
        # with the pills it renders, that would feed back into the parent's
        # space negotiation, resize this widget again, retrigger _refit(),
        # and potentially loop -- exactly what happened before this was fixed.
        return QSize(0, super().sizeHint().height())

    def minimumSizeHint(self) -> QSize:
        return QSize(0, 0)

    def set_classes(self, entries: list) -> None:
        self._entries = entries
        self._refit()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._refit()

    def _refit(self) -> None:
        while self._layout.count():
            item = self._layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        for name, rgb in self._fitting_entries():
            self._layout.addWidget(_make_pill(name, rgb))
        self._layout.addStretch(1)

    def _fitting_entries(self) -> list:
        available = self.width()
        if available <= 0 or not self._entries:
            return []
        visible = []
        used = 0
        for name, rgb in self._entries:
            pill_w = _pill_text_width(name)
            extra = pill_w if not visible else _PILL_SPACING + pill_w
            if used + extra > available:
                break
            used += extra
            visible.append((name, rgb))
        return visible


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

        # Row 2: decision (left) -- class pills (middle) -- MicroSentry score (right)
        row2 = QHBoxLayout()
        row2.setSpacing(6)

        self._decision_lbl = QLabel()
        row2.addWidget(self._decision_lbl)

        self._pill_divider_left = _vline()
        row2.addWidget(self._pill_divider_left)

        self._pill_tray = _ClassPillTray()
        row2.addWidget(self._pill_tray, 1)

        self._pill_divider_right = _vline()
        row2.addWidget(self._pill_divider_right)

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

        class_entries = model.class_entries(row)
        has_classes = bool(class_entries)
        self._pill_divider_left.setVisible(has_classes)
        self._pill_divider_right.setVisible(has_classes and self._microsentry_mode)
        self._pill_tray.set_classes(class_entries)

        state = model.data(model.index(row, NavigatorColumns.STATUS), IMAGE_STATE_ROLE)
        _apply_status_icon(self._status_lbl, state)

        tooltip = model.data(model.index(row, NavigatorColumns.STATUS), Qt.ToolTipRole)
        self._header.setToolTip(tooltip or "")
