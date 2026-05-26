"""
CanvasToolbar — Collapsible top bar for the picture canvas.
Provides context-sensitive controls for Polygon shapes, SAM variations, and edit settings.
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QComboBox,
    QSlider,
    QWidget,
    QPushButton,
)

_SAM_VARIANT_MAP = {
    "SAM2.1 Tiny": "sam2_t.pt",
    "SAM2.1 Small": "sam2_s.pt",
    "SAM2.1 Base+": "sam2_b.pt",
    "SAM2.1 Large": "sam2_l.pt",
}


class CanvasToolbar(QFrame):
    thickness_changed = Signal(float)
    shape_changed = Signal(str)
    sam_variant_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        self.setAutoFillBackground(True)

        self.setStyleSheet("""
            QFrame {
                background: palette(window);
                border-bottom: 1px solid palette(mid);
            }
        """)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Top bar (Toggle handle)
        self._top_bar = QWidget()
        top_layout = QHBoxLayout(self._top_bar)
        top_layout.setContentsMargins(4, 2, 4, 2)

        self._btn_toggle = QPushButton("Settings (None)")
        self._btn_toggle.setFlat(True)
        self._btn_toggle.setCheckable(True)
        self._btn_toggle.setChecked(True)
        self._btn_toggle.clicked.connect(self._on_toggle)
        self._btn_toggle.setStyleSheet("font-weight: bold; text-align: left;")
        self._btn_toggle.setEnabled(False)
        top_layout.addWidget(self._btn_toggle)
        top_layout.addStretch()

        main_layout.addWidget(self._top_bar)

        # Bottom container (Collapsible content)
        self._content = QWidget()
        content_layout = QHBoxLayout(self._content)
        content_layout.setContentsMargins(8, 4, 8, 4)
        content_layout.setSpacing(12)

        # Mode / Shape
        self._lbl_mode = QLabel("Shape Mode:")
        self._combo_shape = QComboBox()
        self._combo_shape.currentTextChanged.connect(self._on_shape_changed)
        content_layout.addWidget(self._lbl_mode)
        content_layout.addWidget(self._combo_shape)

        # Thickness
        self._lbl_thickness = QLabel("Brush Thickness:")
        self._slider_thickness = QSlider(Qt.Horizontal)
        self._slider_thickness.setRange(1, 40)
        self._slider_thickness.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._slider_thickness.setTickInterval(4)
        self._slider_thickness.setFixedWidth(150)
        self._slider_thickness.valueChanged.connect(self._on_thickness_changed)
        self._lbl_thickness_val = QLabel("2.00 px")
        self._lbl_thickness_val.setFixedWidth(55)
        content_layout.addWidget(self._lbl_thickness)
        content_layout.addWidget(self._slider_thickness)
        content_layout.addWidget(self._lbl_thickness_val)

        # SAM specific
        self._lbl_sam = QLabel("SAM Variant:")
        self._combo_sam = QComboBox()
        self._combo_sam.addItems(list(_SAM_VARIANT_MAP.keys()))
        self._combo_sam.currentTextChanged.connect(self._on_sam_changed)
        self._lbl_sam_status = QLabel("Ready")
        content_layout.addWidget(self._lbl_sam)
        content_layout.addWidget(self._combo_sam)
        content_layout.addWidget(self._lbl_sam_status)

        content_layout.addStretch()
        main_layout.addWidget(self._content)

        self._context = ""
        self.set_context("")

    def _on_toggle(self, checked: bool):
        self._content.setVisible(checked)
        self._btn_toggle.setText("▾ Settings" if checked else "▸ Settings")

    def _on_shape_changed(self, text: str):
        mapping = {
            "Point-by-point": "point",
            "Free Brush": "brush",
            "Rectangle": "rectangle",
            "Circle": "circle",
        }
        self.shape_changed.emit(mapping.get(text, "point"))

    def _on_thickness_changed(self, val: int):
        thickness = val * 0.25
        self._lbl_thickness_val.setText(f"{thickness:.2f} px")
        self.thickness_changed.emit(thickness)

    def _on_sam_changed(self, text: str):
        self.sam_variant_changed.emit(_SAM_VARIANT_MAP.get(text, "sam2_t.pt"))

    def current_sam_variant(self) -> str:
        return _SAM_VARIANT_MAP.get(self._combo_sam.currentText(), "sam2_t.pt")

    def set_sam_status(self, status: str, color: str):
        self._lbl_sam_status.setText(status)
        self._lbl_sam_status.setStyleSheet(
            f"color: {color}; font-style: {'italic' if color == 'grey' else 'normal'};"
        )

    def set_context(self, context: str, thickness: float = 2.0):
        """Update toolbar state based on selected canvas context."""
        self._context = context
        if context == "":
            self._btn_toggle.setEnabled(False)
            self._content.setVisible(False)
            self._btn_toggle.setText("Settings (None)")
            return

        self._btn_toggle.setEnabled(True)
        self._btn_toggle.setText(
            "▾ Settings" if self._btn_toggle.isChecked() else "▸ Settings"
        )
        self._content.setVisible(self._btn_toggle.isChecked())

        self._combo_shape.blockSignals(True)
        self._combo_shape.clear()

        show_shape = context in ("polygon", "sam_bbox")
        show_thickness = context in ("polygon", "edit", "sam_bbox")
        show_sam = context == "sam_bbox"

        self._lbl_mode.setVisible(show_shape)
        self._combo_shape.setVisible(show_shape)

        self._lbl_thickness.setVisible(show_thickness)
        self._slider_thickness.setVisible(show_thickness)
        self._lbl_thickness_val.setVisible(show_thickness)

        self._lbl_sam.setVisible(show_sam)
        self._combo_sam.setVisible(show_sam)
        self._lbl_sam_status.setVisible(show_sam)

        if context in ("polygon", "sam_bbox"):
            self._combo_shape.addItems(
                ["Point-by-point", "Free Brush", "Rectangle", "Circle"]
            )

        self._combo_shape.blockSignals(False)

        if show_thickness:
            self._slider_thickness.blockSignals(True)
            self._slider_thickness.setValue(int(thickness * 4))
            self._lbl_thickness_val.setText(f"{thickness:.2f} px")
            self._slider_thickness.blockSignals(False)

        if show_shape:
            self._on_shape_changed(self._combo_shape.currentText())
