from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel

_COLOR_REVIEWED = "#4caf50"
_COLOR_IN_REVIEW = "#ff9800"
_COLOR_OMITTED = "#ff9800"


def _dot(color: str) -> QLabel:
    lbl = QLabel()
    lbl.setFixedSize(10, 10)
    lbl.setStyleSheet(f"background-color: {color}; border-radius: 5px;")
    return lbl


def _omit_badge() -> QLabel:
    lbl = QLabel("!")
    lbl.setFixedSize(10, 10)
    lbl.setAlignment(Qt.AlignCenter)
    lbl.setStyleSheet(f"color: {_COLOR_OMITTED}; font-size: 10px; font-weight: bold;")
    return lbl
