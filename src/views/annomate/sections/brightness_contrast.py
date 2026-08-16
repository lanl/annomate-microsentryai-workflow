"""
BrightnessContrastSection — "Brightness/Contrast" section body for the
right activity bar's Image Adjustments tab.

A read-only, view-local min/max linear contrast stretch -- the same
window/level Fiji applies to RGB pixel values directly -- driven against
the canvas the same way CenterCropSection and HSVSection are
(self._canvas.set_contrast_adjustment / contrast_settings). Settings
persist across image switches for as long as Enable stays checked;
ImageLabel re-applies them to every newly loaded image itself. Never
touches image data, annotations, or the heatmap overlay.
"""

from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import QPushButton, QVBoxLayout, QWidget

from ._shared import _add_slider_row, _toggle_button

_DEBOUNCE_MS = 50


class BrightnessContrastSection(QWidget):
    """Min/Max contrast-stretch controls, driven directly against the canvas.

    Signals:
        contrast_enabled_toggled (bool): "Enable" button toggled.
    """

    contrast_enabled_toggled = Signal(bool)

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
            "Enable Brightness/Contrast",
            tooltip="Turn on the min/max contrast stretch for the canvas view",
        )
        self._enable_chk.toggled.connect(self._on_enable_toggled)
        layout.addWidget(self._enable_chk)

        self._min_slider, self._min_lbl = _add_slider_row(
            layout,
            "Min",
            0,
            254,
            0,
            self._on_slider_changed,
            tooltip="Pixel values at or below this level display as black",
        )
        self._max_slider, self._max_lbl = _add_slider_row(
            layout,
            "Max",
            1,
            255,
            255,
            self._on_slider_changed,
            tooltip="Pixel values at or above this level display as white",
        )

        self._btn_reset = QPushButton("Reset")
        self._btn_reset.setToolTip("Reset min/max back to 0/255")
        self._btn_reset.clicked.connect(self._on_reset_clicked)
        layout.addWidget(self._btn_reset)

        if hasattr(canvas, "contrastChanged"):
            canvas.contrastChanged.connect(lambda _: self._refresh_controls())
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
        self._canvas.set_contrast_adjustment(enabled=checked)
        self.contrast_enabled_toggled.emit(checked)

    def _on_slider_changed(self, _value: int) -> None:
        if self._refreshing:
            return
        # Cross-clamp: Min can never reach/pass Max and vice versa, since
        # two independent sliders stand in for one dual-handle range slider.
        self._refreshing = True
        if self._min_slider.value() >= self._max_slider.value():
            if self.sender() is self._min_slider:
                self._max_slider.setValue(self._min_slider.value() + 1)
            else:
                self._min_slider.setValue(self._max_slider.value() - 1)
        self._refreshing = False
        self._min_lbl.setText(f"{self._min_slider.value()}")
        self._max_lbl.setText(f"{self._max_slider.value()}")
        self._debounce.start()

    def _apply_sliders(self) -> None:
        self._canvas.set_contrast_adjustment(
            min_value=self._min_slider.value(),
            max_value=self._max_slider.value(),
        )

    def _on_reset_clicked(self) -> None:
        self._refreshing = True
        self._min_slider.setValue(0)
        self._max_slider.setValue(255)
        self._min_lbl.setText("0")
        self._max_lbl.setText("255")
        self._refreshing = False
        self._canvas.set_contrast_adjustment(min_value=0, max_value=255)

    # ------------------------------------------------------------------ #
    # Refresh
    # ------------------------------------------------------------------ #

    def _refresh_controls(self) -> None:
        if self._canvas is None or not hasattr(self._canvas, "contrast_settings"):
            self._refresh_availability()
            return
        settings = self._canvas.contrast_settings()
        self._refreshing = True
        self._enable_chk.setChecked(bool(settings.get("enabled")))
        self._min_slider.setValue(int(settings.get("min", 0)))
        self._max_slider.setValue(int(settings.get("max", 255)))
        self._min_lbl.setText(f"{self._min_slider.value()}")
        self._max_lbl.setText(f"{self._max_slider.value()}")
        self._refreshing = False
        self._refresh_availability()

    def _refresh_availability(self) -> None:
        self._enable_chk.setEnabled(self._has_image)
        self._min_slider.setEnabled(self._has_image)
        self._max_slider.setEnabled(self._has_image)
        self._btn_reset.setEnabled(self._has_image)
