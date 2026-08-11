from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFrame, QLabel, QPushButton

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


def _toggle_button(text: str) -> QPushButton:
    """A checkable QPushButton styled as an on/off toggle (orange when checked)."""
    btn = QPushButton(text)
    btn.setCheckable(True)
    btn.setStyleSheet(TOGGLE_BUTTON_STYLESHEET)
    return btn


class _ClickableFrame(QFrame):
    """A QFrame that emits clicked() on a left-button press."""

    clicked = Signal()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


def _dot(color: str) -> QLabel:
    lbl = QLabel()
    lbl.setFixedSize(10, 10)
    lbl.setStyleSheet(f"QLabel {{ background-color: {color}; border-radius: 5px; }}")
    return lbl


def _ring_undecided() -> QLabel:
    lbl = QLabel()
    lbl.setFixedSize(10, 10)
    lbl.setStyleSheet(
        f"QLabel {{ border: 2px solid {_COLOR_UNDECIDED}; border-radius: 5px; }}"
    )
    return lbl


def _incomplete_badge() -> QLabel:
    lbl = QLabel("!")
    lbl.setFixedSize(10, 10)
    lbl.setAlignment(Qt.AlignCenter)
    lbl.setStyleSheet(
        f"QLabel {{ color: {_COLOR_INCOMPLETE}; font-size: 10px; font-weight: bold; }}"
    )
    return lbl
