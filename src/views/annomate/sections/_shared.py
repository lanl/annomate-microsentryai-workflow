from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QSlider, QVBoxLayout

_COLOR_REVIEWED = "#4caf50"
_COLOR_UNDECIDED = "#888888"
_COLOR_INCOMPLETE = "#ff9800"
_COLOR_SELECTED_BG = "#d6d6d6"  # light grey, standing in for palette(highlight)'s accent blue

# Applied to the QApplication itself (see main.py) so every tooltip in the
# app looks the same, instead of each widget/section falling back to
# whatever the native platform style happens to render for QToolTip.
TOOLTIP_STYLESHEET = (
    f"QToolTip {{ background-color: {_COLOR_SELECTED_BG}; color: black; "
    "padding: 2px 4px; border: 1px solid palette(shadow); }"
)

# Same orange used by the Pixel Level / Image Level mode selector in
# Annotation Classes, reused so every on/off toggle in the right panel
# reads the same way: a plain button that turns orange when engaged.
TOGGLE_BUTTON_STYLESHEET = (
    "QPushButton {"
    "  padding: 3px 10px;"
    "  border: 1px solid palette(mid);"
    "  border-radius: 4px;"
    "  background: palette(button);"
    "  color: palette(button-text);"
    "}"
    "QPushButton:checked {"
    "  background: #ff9800;"
    "  color: white;"
    "  border-color: #c96800;"
    "}"
)


def _toggle_button(text: str, tooltip: str | None = None) -> QPushButton:
    """A checkable QPushButton styled as an on/off toggle (orange when checked)."""
    btn = QPushButton(text)
    btn.setCheckable(True)
    btn.setStyleSheet(TOGGLE_BUTTON_STYLESHEET)
    if tooltip:
        btn.setToolTip(tooltip)
    return btn


def _add_slider_row(
    layout: QVBoxLayout,
    label: str,
    minimum: int,
    maximum: int,
    default: int,
    on_change,
    suffix: str = "",
    value_width: int = 30,
    tooltip: str | None = None,
) -> tuple[QSlider, QLabel]:
    """Append a "label - slider - value" row to *layout*; returns (slider, value_label)."""
    row = QHBoxLayout()
    row.setSpacing(8)
    name_lbl = QLabel(label)
    name_lbl.setFixedWidth(70)
    row.addWidget(name_lbl)
    slider = QSlider(Qt.Horizontal)
    slider.setRange(minimum, maximum)
    slider.setValue(default)
    slider.valueChanged.connect(on_change)
    if tooltip:
        slider.setToolTip(tooltip)
    row.addWidget(slider)
    value_lbl = QLabel(f"{default}{suffix}")
    value_lbl.setFixedWidth(value_width)
    row.addWidget(value_lbl)
    layout.addLayout(row)
    return slider, value_lbl


class _ClickableFrame(QFrame):
    """A QFrame that emits clicked() on a left-button press."""

    clicked = Signal()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


_STATUS_DOT_W = 10
_INCOMPLETE_ICON_FONT_PX = 18
_INCOMPLETE_STATES = ("reject_incomplete", "accept_conflict", "undecided_work")
_REVIEWED_STATES = ("accept_clean", "reject_reviewed")


def _apply_status_icon(label: QLabel, state: str) -> None:
    """Style *label* in place to match the status dot/ring/badge for *state*.

    Shared by the per-card status icon and the navigator header's summary
    chips so both render the exact same dot/ring/"!" for a given state --
    passing any representative state from a bucket (e.g. "accept_clean" for
    "reviewed") renders that bucket's style.
    """
    label.setFixedSize(_STATUS_DOT_W, _STATUS_DOT_W)
    label.setAlignment(Qt.AlignCenter)
    if state in _INCOMPLETE_STATES:
        label.setText("!")
        label.setStyleSheet(
            f"QLabel {{ color: {_COLOR_INCOMPLETE}; font-size: {_INCOMPLETE_ICON_FONT_PX}px; "
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


def _status_icon(state: str) -> QLabel:
    """A standalone QLabel pre-styled via _apply_status_icon -- for chip-style uses."""
    lbl = QLabel()
    _apply_status_icon(lbl, state)
    return lbl
