"""
CenterCropSection — "Center Crop" section body for the right activity
bar's View Overlays tab.

Owns everything related to the center-crop overlay: enable/shape/size/
opacity/border/center-dot controls, plus the template calibration
workflow (Calibrate/Accept/Import/Clear) used to line the crop up on a
matching template image. All of it used to live in the viewport's
floating Center Crop popup menu.
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QColorDialog,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ._shared import _toggle_button


class CenterCropSection(QWidget):
    """Center-crop overlay controls + template calibration workflow.

    Signals:
        crop_overlay_toggled (bool): "Enable Center Crop" button toggled.
        center_calibration_started (): "Calibrate Center" clicked.
        center_calibration_accepted (): "Accept" clicked.
        center_template_cleared (): "Clear" clicked.
        center_template_import_requested (str): "Import" picked a PNG path.
    """

    crop_overlay_toggled = Signal(bool)
    center_calibration_started = Signal()
    center_calibration_accepted = Signal()
    center_template_cleared = Signal()
    center_template_import_requested = Signal(str)

    def __init__(
        self,
        canvas,
        center_template_model=None,
        parent: QWidget = None,
    ) -> None:
        super().__init__(parent)
        self._canvas = canvas
        self._model = None
        self._has_image = False
        self._image_w = 0
        self._image_h = 0
        self._refreshing = False
        self._calibrating = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Enable
        self._crop_chk = _toggle_button(
            "Enable Center Crop",
            tooltip="Crop the canvas view to the region around the template match",
        )
        self._crop_chk.toggled.connect(self._on_crop_toggled)
        layout.addWidget(self._crop_chk)

        # Shape
        shape_row = QHBoxLayout()
        shape_row.setSpacing(8)
        shape_lbl = QLabel("Shape")
        shape_lbl.setFixedWidth(58)
        shape_lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        shape_row.addWidget(shape_lbl)
        self._crop_shape_combo = QComboBox()
        self._crop_shape_combo.addItems(["Rectangle", "Circle"])
        self._crop_shape_combo.setMaximumWidth(90)
        self._crop_shape_combo.currentTextChanged.connect(self._on_crop_shape_changed)
        shape_row.addWidget(self._crop_shape_combo)
        shape_row.addStretch()
        layout.addLayout(shape_row)

        # Width / Diameter
        width_row = QHBoxLayout()
        width_row.setSpacing(8)
        self._crop_primary_lbl = QLabel("Width")
        self._crop_primary_lbl.setFixedWidth(58)
        self._crop_primary_lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        width_row.addWidget(self._crop_primary_lbl)
        self._crop_width_spin = QSpinBox()
        self._crop_width_spin.setRange(1, 999999)
        self._crop_width_spin.setSuffix(" px")
        self._crop_width_spin.setMaximumWidth(90)
        self._crop_width_spin.valueChanged.connect(self._on_crop_primary_changed)
        width_row.addWidget(self._crop_width_spin)
        width_row.addStretch()
        layout.addLayout(width_row)

        # Height / Radius
        height_row = QHBoxLayout()
        height_row.setSpacing(8)
        self._crop_secondary_lbl = QLabel("Height")
        self._crop_secondary_lbl.setFixedWidth(58)
        self._crop_secondary_lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        height_row.addWidget(self._crop_secondary_lbl)
        self._crop_height_spin = QSpinBox()
        self._crop_height_spin.setRange(1, 999999)
        self._crop_height_spin.setSuffix(" px")
        self._crop_height_spin.setMaximumWidth(90)
        self._crop_height_spin.valueChanged.connect(self._on_crop_secondary_changed)
        height_row.addWidget(self._crop_height_spin)
        height_row.addStretch()
        layout.addLayout(height_row)

        # Outside opacity
        opacity_row = QHBoxLayout()
        opacity_row.setSpacing(8)
        opacity_row.addWidget(QLabel("Outside opacity"))
        self._crop_opacity_slider = QSlider(Qt.Horizontal)
        self._crop_opacity_slider.setRange(0, 100)
        self._crop_opacity_slider.valueChanged.connect(self._on_crop_opacity_changed)
        opacity_row.addWidget(self._crop_opacity_slider)
        self._crop_opacity_lbl = QLabel("37%")
        self._crop_opacity_lbl.setFixedWidth(34)
        opacity_row.addWidget(self._crop_opacity_lbl)
        layout.addLayout(opacity_row)

        # Border color
        border_row = QHBoxLayout()
        border_row.setSpacing(8)
        border_lbl = QLabel("Border")
        border_lbl.setFixedWidth(58)
        border_lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        border_row.addWidget(border_lbl)
        self._crop_color_btn = QPushButton()
        self._crop_color_btn.setFixedSize(32, 20)
        self._crop_color_btn.setToolTip("Click to set a custom border color")
        self._crop_color_btn.clicked.connect(self._on_crop_color_clicked)
        border_row.addWidget(self._crop_color_btn)
        self._crop_color_auto_btn = QPushButton("Auto")
        self._crop_color_auto_btn.setFixedHeight(20)
        self._crop_color_auto_btn.setToolTip("Reset to auto-contrast color")
        self._crop_color_auto_btn.clicked.connect(self._on_crop_color_auto)
        border_row.addWidget(self._crop_color_auto_btn)
        border_row.addStretch()
        layout.addLayout(border_row)
        self._update_crop_color_swatch(None)

        # Center dot
        self._crop_center_dot_chk = _toggle_button(
            "Enable Center Dot", tooltip="Show center dot"
        )
        self._crop_center_dot_chk.toggled.connect(self._on_crop_center_dot_toggled)
        layout.addWidget(self._crop_center_dot_chk)

        # Hint
        self._crop_hint_lbl = QLabel("Centered on image")
        self._crop_hint_lbl.setStyleSheet("color: grey; font-style: italic;")
        layout.addWidget(self._crop_hint_lbl)

        # Template divider
        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setFrameShadow(QFrame.Sunken)
        layout.addWidget(divider)

        template_header = QLabel("Template")
        template_header.setStyleSheet("font-weight: bold;")
        layout.addWidget(template_header)

        self._template_status_lbl = QLabel("Template: none")
        self._template_status_lbl.setWordWrap(True)
        self._template_status_lbl.setStyleSheet("color: grey; font-style: italic;")
        layout.addWidget(self._template_status_lbl)

        self._btn_calibrate = QPushButton("Calibrate Center")
        self._btn_calibrate.setToolTip("Move the crop and dot together")
        self._btn_calibrate.clicked.connect(self.center_calibration_started)
        layout.addWidget(self._btn_calibrate)

        self._btn_accept = QPushButton("Accept")
        self._btn_accept.setToolTip("Save this center as the matching template")
        self._btn_accept.clicked.connect(self.center_calibration_accepted)
        layout.addWidget(self._btn_accept)

        import_clear_row = QHBoxLayout()
        import_clear_row.setSpacing(6)
        self._btn_import = QPushButton("Import")
        self._btn_import.setToolTip(
            "Import a PNG file as the center template (looks for a companion "
            "template.annoproj in the same folder to restore all settings)"
        )
        self._btn_import.clicked.connect(self._on_import_clicked)
        import_clear_row.addWidget(self._btn_import)
        self._btn_clear = QPushButton("Clear")
        self._btn_clear.setToolTip(
            "Clear the saved center template, or cancel an in-progress calibration"
        )
        self._btn_clear.clicked.connect(self.center_template_cleared)
        import_clear_row.addWidget(self._btn_clear)
        layout.addLayout(import_clear_row)

        # Reset at the bottom
        self._btn_reset_crop = QPushButton("Reset Defaults")
        self._btn_reset_crop.setToolTip("Reset crop shape, size, and opacity to defaults")
        self._btn_reset_crop.clicked.connect(self._on_reset_crop_clicked)
        layout.addWidget(self._btn_reset_crop)

        if hasattr(canvas, "centerCropChanged"):
            canvas.centerCropChanged.connect(lambda _: self._refresh_crop_controls())
        if hasattr(canvas, "image_loaded"):
            canvas.image_loaded.connect(self.set_image_dimensions)
        already_loaded = hasattr(canvas, "is_image_loaded") and canvas.is_image_loaded()
        self.set_has_image(already_loaded)

        if center_template_model is not None:
            self.set_center_template_model(center_template_model)
        self._refresh_crop_controls()

    # ------------------------------------------------------------------ #
    # External state
    # ------------------------------------------------------------------ #

    def set_center_template_model(self, model) -> None:
        self._model = model
        model.template_changed.connect(self._refresh_crop_controls)
        model.match_changed.connect(self._refresh_crop_controls)
        self._refresh_crop_controls()

    def set_has_image(self, has_image: bool) -> None:
        self._has_image = bool(has_image)
        self._refresh_availability()

    def set_image_dimensions(self, width: int, height: int) -> None:
        self._image_w = max(0, int(width))
        self._image_h = max(0, int(height))
        self._has_image = self._image_w > 0 and self._image_h > 0
        self._refresh_crop_controls()

    def set_calibrating(self, active: bool) -> None:
        # Deliberately does not go through _refresh_crop_controls(), which
        # re-syncs _calibrating from the canvas's own crop settings -- that
        # would immediately clobber the value being set here.
        self._calibrating = bool(active)
        self._refresh_template_status()
        self._refresh_availability()

    # ------------------------------------------------------------------ #
    # Crop controls
    # ------------------------------------------------------------------ #

    def _on_crop_toggled(self, checked: bool) -> None:
        if self._refreshing:
            return
        if not checked:
            self._calibrating = False
        self._canvas.set_center_crop(
            enabled=checked,
            calibrating=False if not checked else None,
        )
        self._refresh_crop_controls()
        self.crop_overlay_toggled.emit(checked)

    def _on_crop_shape_changed(self, display_name: str) -> None:
        if self._refreshing:
            return
        shape = display_name.lower()
        if shape == "circle":
            diameter = min(
                self._crop_width_spin.value(),
                self._crop_height_spin.value(),
            )
            self._canvas.set_center_crop(shape=shape, width=diameter, height=diameter)
        else:
            self._canvas.set_center_crop(shape=shape)
        self._refresh_crop_controls()

    def _on_crop_primary_changed(self, value: int) -> None:
        if self._refreshing:
            return
        if self._crop_shape_combo.currentText() == "Circle":
            diameter = max(1, value)
            self._canvas.set_center_crop(width=diameter, height=diameter)
        else:
            self._canvas.set_center_crop(
                width=value,
                height=self._crop_height_spin.value(),
            )
        self._refresh_crop_controls()

    def _on_crop_secondary_changed(self, value: int) -> None:
        if self._refreshing:
            return
        if self._crop_shape_combo.currentText() == "Circle":
            diameter = max(1, value * 2)
            self._canvas.set_center_crop(width=diameter, height=diameter)
        else:
            self._canvas.set_center_crop(
                width=self._crop_width_spin.value(),
                height=value,
            )
        self._refresh_crop_controls()

    def _on_crop_opacity_changed(self, value: int) -> None:
        self._crop_opacity_lbl.setText(f"{value}%")
        if self._refreshing:
            return
        self._canvas.set_center_crop(opacity=value / 100.0)

    def _on_crop_center_dot_toggled(self, checked: bool) -> None:
        if self._refreshing:
            return
        self._canvas.set_center_crop(center_dot=checked)

    def _on_crop_color_clicked(self) -> None:
        current = self._canvas.center_crop_settings().get("border_color")
        initial = QColor(*current) if current else QColor(255, 255, 255)
        color = QColorDialog.getColor(initial, self, "Border Color")
        if color.isValid():
            rgb = (color.red(), color.green(), color.blue())
            self._canvas.set_center_crop(border_color=rgb)
            self._update_crop_color_swatch(rgb)

    def _on_crop_color_auto(self) -> None:
        self._canvas.set_center_crop(border_color=None)
        self._update_crop_color_swatch(None)

    def _update_crop_color_swatch(self, rgb) -> None:
        if rgb is None:
            self._crop_color_btn.setStyleSheet(
                "background: qlineargradient(x1:0,y1:0,x2:1,y2:1,"
                "stop:0 #ffffff,stop:0.49 #ffffff,stop:0.5 #000000,stop:1 #000000);"
                "border: 1px solid #888;"
            )
            self._crop_color_btn.setToolTip("Auto-contrast (click to override)")
        else:
            r, g, b = rgb
            self._crop_color_btn.setStyleSheet(
                f"background-color: rgb({r},{g},{b}); border: 1px solid #888;"
            )
            self._crop_color_btn.setToolTip(f"Border color: rgb({r},{g},{b}), click to change")

    def _on_reset_crop_clicked(self) -> None:
        if self._refreshing:
            return
        self._calibrating = False
        self._canvas.set_center_crop(
            shape="circle",
            width=1210,
            height=1210,
            opacity=0.37,
            center_dot=False,
            calibrating=False,
        )
        self._refresh_crop_controls()

    # ------------------------------------------------------------------ #
    # Template controls
    # ------------------------------------------------------------------ #

    def _on_import_clicked(self) -> None:
        if self._refreshing:
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Center Template", "", "PNG Image (*.png)"
        )
        if path:
            self.center_template_import_requested.emit(path)

    # ------------------------------------------------------------------ #
    # Refresh
    # ------------------------------------------------------------------ #

    def _refresh_crop_controls(self) -> None:
        if self._canvas is None:
            self._refresh_availability()
            return
        settings = self._canvas.center_crop_settings()
        max_w = max(1, self._image_w)
        max_h = max(1, self._image_h)
        width = settings.get("width") or max_w // 2
        height = settings.get("height") or max_h // 2
        shape = settings.get("shape") or "rectangle"
        opacity_pct = int(round((settings.get("opacity") or 0.0) * 100))
        center_dot = bool(settings.get("center_dot"))

        self._refreshing = True
        self._crop_chk.setChecked(bool(settings.get("enabled")))
        if shape == "circle":
            max_diameter = max(1210, min(max_w, max_h))
            diameter = max(1, min(int(min(width, height)), max_diameter))
            self._crop_primary_lbl.setText("Diameter")
            self._crop_secondary_lbl.setText("Radius")
            self._crop_width_spin.setToolTip("Circle diameter")
            self._crop_height_spin.setToolTip("Circle radius")
            self._crop_width_spin.setRange(1, max_diameter)
            self._crop_height_spin.setRange(1, max(1, max_diameter // 2))
            self._crop_width_spin.setValue(diameter)
            self._crop_height_spin.setValue(max(1, diameter // 2))
        else:
            self._crop_primary_lbl.setText("W")
            self._crop_secondary_lbl.setText("H")
            self._crop_width_spin.setToolTip("Rectangle width")
            self._crop_height_spin.setToolTip("Rectangle height")
            self._crop_width_spin.setRange(1, max(1210, max_w))
            self._crop_height_spin.setRange(1, max(1210, max_h))
            self._crop_width_spin.setValue(
                max(1, min(int(width), self._crop_width_spin.maximum()))
            )
            self._crop_height_spin.setValue(
                max(1, min(int(height), self._crop_height_spin.maximum()))
            )
        shape_label = {"rectangle": "Rectangle", "circle": "Circle"}.get(shape, "Rectangle")
        self._crop_shape_combo.setCurrentText(shape_label)
        self._crop_opacity_slider.setValue(opacity_pct)
        self._crop_opacity_lbl.setText(f"{opacity_pct}%")
        self._crop_center_dot_chk.setChecked(center_dot)
        self._update_crop_color_swatch(settings.get("border_color"))
        if settings.get("calibrating") != self._calibrating:
            self._calibrating = bool(settings.get("calibrating"))
        self._crop_hint_lbl.setText(
            (
                f"Center: {settings.get('center_x'):.0f}, {settings.get('center_y'):.0f}"
                if settings.get("center_x") is not None
                else f"Image: {self._image_w} × {self._image_h} px"
            )
            if self._has_image
            else "Load an image to preview a crop"
        )
        self._refreshing = False
        self._refresh_template_status()
        self._refresh_availability()

    def _refresh_template_status(self) -> None:
        if self._calibrating:
            self._template_status_lbl.setText("Template: move crop, then Accept")
            self._template_status_lbl.setStyleSheet("color: black; font-style: normal;")
        elif self._model is None or not self._model.has_template():
            self._template_status_lbl.setText("Template: none")
            self._template_status_lbl.setStyleSheet("color: grey; font-style: italic;")
        else:
            score = self._model.last_score()
            if score is None:
                self._template_status_lbl.setText("Template: saved")
            else:
                self._template_status_lbl.setText(f"Template match: {score:.3f}")
            self._template_status_lbl.setStyleSheet("color: black; font-style: normal;")

    def _refresh_availability(self) -> None:
        self._crop_chk.setEnabled(self._has_image)
        self._crop_shape_combo.setEnabled(self._has_image)
        self._crop_width_spin.setEnabled(self._has_image)
        self._crop_height_spin.setEnabled(self._has_image)
        self._crop_opacity_slider.setEnabled(self._has_image)
        self._crop_center_dot_chk.setEnabled(self._has_image)
        self._btn_reset_crop.setEnabled(self._has_image)
        self._btn_calibrate.setEnabled(self._has_image)
        self._btn_accept.setEnabled(self._has_image and self._calibrating)
        self._btn_import.setEnabled(self._has_image)
        # Also enabled mid-calibration so an unwanted calibrate can be
        # cancelled — clearing exits calibration and hides the crop overlay.
        self._btn_clear.setEnabled(
            self._calibrating or (self._model is not None and self._model.has_template())
        )
