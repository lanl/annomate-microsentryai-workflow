"""
MicrosentryOverlay — floating AI view controls panel.

Positioned at the top-right of the canvas, visible only when Microsentry is
enabled. Provides toggles for heatmap/segmentation and sliders for threshold
and transparency.
"""

from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QToolButton, QWidget, QPushButton,
)


def _slider_row(parent_layout: QVBoxLayout, label: str, lo: int, hi: int, default: int) -> tuple:
    """Add a label+slider pair to parent_layout. Returns (value_label, slider)."""
    row_w = QWidget()
    row_h = QHBoxLayout(row_w)
    row_h.setContentsMargins(0, 0, 0, 0)
    row_h.setSpacing(4)
    lbl = QLabel(label)
    lbl.setStyleSheet("font-size: 10px;")
    row_h.addWidget(lbl)
    row_h.addStretch()
    val_lbl = QLabel(str(default))
    val_lbl.setStyleSheet("font-size: 10px;")
    val_lbl.setFixedWidth(26)
    val_lbl.setAlignment(Qt.AlignRight)
    row_h.addWidget(val_lbl)
    parent_layout.addWidget(row_w)

    slider = QSlider(Qt.Horizontal)
    slider.setRange(lo, hi)
    slider.setValue(default)
    slider.setFixedWidth(130)
    parent_layout.addWidget(slider)
    return val_lbl, slider


class MicrosentryOverlay(QFrame):
    """Floating top-right panel with AI view toggles and sliders.

    Parented to the canvas widget so it floats on top. Call reposition()
    whenever the canvas resizes.

    Signals:
        settings_changed (): Emitted (debounced 150 ms) when any control changes.
    """

    _MARGIN = 10

    settings_changed = Signal()
    accept_polygons_requested = Signal()

    def __init__(self, canvas: QWidget, parent: QWidget = None) -> None:
        super().__init__(parent or canvas)
        self._canvas = canvas
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(150)
        self._debounce.timeout.connect(self.settings_changed)

        self.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        self.setAutoFillBackground(True)

        font = QFont()
        font.setPointSize(9)
        self.setFont(font)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)

        title = QLabel("AI View")
        title.setStyleSheet("font-weight: bold; font-size: 11px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Toggle buttons
        btn_row = QWidget()
        btn_h = QHBoxLayout(btn_row)
        btn_h.setContentsMargins(0, 0, 0, 0)
        btn_h.setSpacing(4)

        self._btn_heatmap = QToolButton()
        self._btn_heatmap.setText("Heatmap")
        self._btn_heatmap.setCheckable(True)
        self._btn_heatmap.setChecked(False)
        self._btn_heatmap.setToolTip("Overlay heatmap on image")
        self._btn_heatmap.toggled.connect(self._debounce.start)
        btn_h.addWidget(self._btn_heatmap)

        self._btn_seg = QToolButton()
        self._btn_seg.setText("Segments")
        self._btn_seg.setCheckable(True)
        self._btn_seg.setChecked(False)
        self._btn_seg.setToolTip("Show AI segmentation polygons")
        self._btn_seg.toggled.connect(self._on_seg_toggled)
        btn_h.addWidget(self._btn_seg)

        layout.addWidget(btn_row)

        self._btn_accept = QPushButton("Accept AI Polygons")
        self._btn_accept.setToolTip("Add AI segmentation polygons as annotations on the active class")
        self._btn_accept.setEnabled(False)
        self._btn_accept.clicked.connect(self.accept_polygons_requested)
        layout.addWidget(self._btn_accept)

        # Threshold slider (percentile 0–100, default 95)
        self._thresh_val, self._thresh = _slider_row(layout, "Threshold", 0, 100, 95)
        self._thresh.valueChanged.connect(
            lambda v: (self._thresh_val.setText(str(v)), self._debounce.start())
        )

        # Transparency slider (alpha 0–100 → 0.0–1.0, default 45)
        self._alpha_val, self._alpha = _slider_row(layout, "Transparency", 0, 100, 45)
        self._alpha.valueChanged.connect(
            lambda v: (self._alpha_val.setText(str(v)), self._debounce.start())
        )

        self.adjustSize()
        self.setVisible(False)

    def _on_seg_toggled(self, checked: bool) -> None:
        self._btn_accept.setEnabled(checked)
        self._debounce.start()

    # ------------------------------------------------------------------ #
    # Positioning
    # ------------------------------------------------------------------ #

    def reposition(self, canvas_size) -> None:
        x = canvas_size.width() - self.width() - self._MARGIN
        y = self._MARGIN
        self.move(x, y)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def set_enabled_visible(self, visible: bool) -> None:
        self.setVisible(visible)
        if visible:
            self.raise_()
            self.reposition(self._canvas.size())

    def get_settings(self) -> dict:
        return {
            "heatmap_enabled": self._btn_heatmap.isChecked(),
            "seg_enabled": self._btn_seg.isChecked(),
            "seg_pct": self._thresh.value(),
            "alpha": self._alpha.value() / 100.0,
        }
