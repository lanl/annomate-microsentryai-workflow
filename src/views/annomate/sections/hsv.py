"""
HSVSection — "HSV" section body for the right activity bar's Image
Adjustments tab.

A read-only, view-local color preview: hue/saturation/value sliders that
recompute the canvas's displayed pixmap directly, the same pattern
CenterCropSection uses for its overlay settings (self._canvas.set_center_crop
/ center_crop_settings). Settings persist across image switches for as long
as Enable stays checked -- ImageLabel re-applies them to every newly loaded
image itself, so this section never needs to push anything on an image
change. It never touches image data, annotations, or the heatmap overlay.
"""

from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import QPushButton, QVBoxLayout, QWidget

from ._shared import _add_slider_row, _toggle_button

_DEBOUNCE_MS = 50


class HSVSection(QWidget):
    """Hue/Saturation/Value adjustment controls, driven directly against the canvas.

    Signals:
        hsv_enabled_toggled (bool): "Enable HSV" button toggled.
    """

    hsv_enabled_toggled = Signal(bool)

    def __init__(self, canvas, parent: QWidget = None) -> None:
        super().__init__(parent)
        self._canvas = canvas
        self._has_image = False
        self._refreshing = False

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(_DEBOUNCE_MS)
        self._debounce.timeout.connect(self._apply_sliders)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self._enable_chk = _toggle_button(
            "Enable HSV",
            tooltip="Turn on the hue/saturation/value adjustment for the canvas view",
        )
        self._enable_chk.toggled.connect(self._on_enable_toggled)
        layout.addWidget(self._enable_chk)

        self._hue_slider, self._hue_lbl = _add_slider_row(
            layout,
            "Hue",
            -179,
            179,
            0,
            self._on_slider_changed,
            value_width=40,
            tooltip="Rotate the hue of displayed pixels",
        )
        self._sat_slider, self._sat_lbl = _add_slider_row(
            layout,
            "Saturation",
            0,
            200,
            100,
            self._on_slider_changed,
            suffix="%",
            value_width=40,
            tooltip="Scale the color saturation of displayed pixels",
        )
        self._val_slider, self._val_lbl = _add_slider_row(
            layout,
            "Value",
            0,
            200,
            100,
            self._on_slider_changed,
            suffix="%",
            value_width=40,
            tooltip="Scale the brightness (value) of displayed pixels",
        )

        self._btn_reset = QPushButton("Reset")
        self._btn_reset.setToolTip("Reset hue/saturation/value to their defaults")
        self._btn_reset.clicked.connect(self._on_reset_clicked)
        layout.addWidget(self._btn_reset)

        if hasattr(canvas, "hsvChanged"):
            canvas.hsvChanged.connect(lambda _: self._refresh_controls())
        if hasattr(canvas, "image_loaded"):
            canvas.image_loaded.connect(lambda *_: self.set_has_image(True))
        already_loaded = hasattr(canvas, "is_image_loaded") and canvas.is_image_loaded()
        self.set_has_image(already_loaded)
        self._refresh_controls()

    # ------------------------------------------------------------------ #
    # External state
    # ------------------------------------------------------------------ #

    def set_has_image(self, has_image: bool) -> None:
        self._has_image = bool(has_image)
        self._refresh_availability()

    # ------------------------------------------------------------------ #
    # Controls
    # ------------------------------------------------------------------ #

    def _on_enable_toggled(self, checked: bool) -> None:
        if self._refreshing:
            return
        self._canvas.set_hsv_adjustment(enabled=checked)
        self.hsv_enabled_toggled.emit(checked)

    def _on_slider_changed(self, _value: int) -> None:
        self._hue_lbl.setText(f"{self._hue_slider.value()}")
        self._sat_lbl.setText(f"{self._sat_slider.value()}%")
        self._val_lbl.setText(f"{self._val_slider.value()}%")
        if self._refreshing:
            return
        self._debounce.start()

    def _apply_sliders(self) -> None:
        self._canvas.set_hsv_adjustment(
            hue=self._hue_slider.value(),
            saturation=self._sat_slider.value(),
            value=self._val_slider.value(),
        )

    def _on_reset_clicked(self) -> None:
        self._refreshing = True
        self._hue_slider.setValue(0)
        self._sat_slider.setValue(100)
        self._val_slider.setValue(100)
        self._hue_lbl.setText("0")
        self._sat_lbl.setText("100%")
        self._val_lbl.setText("100%")
        self._refreshing = False
        self._canvas.set_hsv_adjustment(hue=0, saturation=100, value=100)

    # ------------------------------------------------------------------ #
    # Refresh
    # ------------------------------------------------------------------ #

    def _refresh_controls(self) -> None:
        if self._canvas is None or not hasattr(self._canvas, "hsv_settings"):
            self._refresh_availability()
            return
        settings = self._canvas.hsv_settings()
        self._refreshing = True
        self._enable_chk.setChecked(bool(settings.get("enabled")))
        self._hue_slider.setValue(int(settings.get("hue", 0)))
        self._sat_slider.setValue(int(settings.get("saturation", 100)))
        self._val_slider.setValue(int(settings.get("value", 100)))
        self._hue_lbl.setText(f"{self._hue_slider.value()}")
        self._sat_lbl.setText(f"{self._sat_slider.value()}%")
        self._val_lbl.setText(f"{self._val_slider.value()}%")
        self._refreshing = False
        self._refresh_availability()

    def _refresh_availability(self) -> None:
        self._enable_chk.setEnabled(self._has_image)
        self._hue_slider.setEnabled(self._has_image)
        self._sat_slider.setEnabled(self._has_image)
        self._val_slider.setEnabled(self._has_image)
        self._btn_reset.setEnabled(self._has_image)
