import re

from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QDialogButtonBox,
    QMessageBox,
)

_VALUE_UNIT_RE = re.compile(r"^\s*([\d.]+)\s*([a-zA-Z]+)\s*$")


class CalibrationDialog(QDialog):
    """Dialog for entering the real-world distance between two calibration points.

    Constructor takes the pixel distance already computed by the caller so it
    can be shown as a read-only reference.  On accept, call get_result() to
    retrieve (real_distance, unit).
    """

    def __init__(self, pixel_dist: float, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Set Calibration Distance")
        self.setModal(True)
        self.setMinimumWidth(320)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        layout.addWidget(
            QLabel(f"Pixel distance between points: <b>{pixel_dist:.2f} px</b>")
        )
        layout.addWidget(QLabel("Enter the real-world distance this represents:"))

        row = QHBoxLayout()
        self._edit = QLineEdit()
        self._edit.setPlaceholderText("e.g. 5mm, 100um, 0.5in")
        row.addWidget(self._edit)
        layout.addLayout(row)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_accept(self) -> None:
        m = _VALUE_UNIT_RE.match(self._edit.text())
        if not m or float(m.group(1)) <= 0:
            QMessageBox.warning(
                self,
                "Invalid Input",
                "Enter a positive value with a unit, e.g. 5mm or 100um.",
            )
            return
        self.accept()

    def get_result(self) -> tuple:
        """Return (real_distance: float, unit: str)."""
        m = _VALUE_UNIT_RE.match(self._edit.text())
        return float(m.group(1)), m.group(2)
