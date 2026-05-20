from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QComboBox,
    QDialogButtonBox,
    QMessageBox,
)


class CalibrationDialog(QDialog):
    """Dialog for entering the real-world distance between two calibration points.

    Constructor takes the pixel distance already computed by the caller so it
    can be shown as a read-only reference.  On accept, call get_result() to
    retrieve (real_distance, unit).
    """

    _UNITS = ["mm", "um", "nm", "cm", "m", "in", "ft"]

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
        self._edit = QLineEdit("1.0")
        self._edit.setPlaceholderText("e.g. 5.0")
        row.addWidget(self._edit)

        self._combo = QComboBox()
        self._combo.addItems(self._UNITS)
        self._combo.setCurrentText("mm")
        row.addWidget(self._combo)
        layout.addLayout(row)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_accept(self) -> None:
        try:
            val = float(self._edit.text())
            if val <= 0:
                raise ValueError
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Enter a positive number.")
            return
        self.accept()

    def get_result(self) -> tuple:
        """Return (real_distance: float, unit: str)."""
        return float(self._edit.text()), self._combo.currentText()
