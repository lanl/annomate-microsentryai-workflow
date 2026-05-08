from PySide6.QtWidgets import QLabel

_COLOR_REVIEWED  = "#4caf50"
_COLOR_IN_REVIEW = "#ff9800"


def _dot(color: str) -> QLabel:
    lbl = QLabel()
    lbl.setFixedSize(10, 10)
    lbl.setStyleSheet(f"background-color: {color}; border-radius: 5px;")
    return lbl
