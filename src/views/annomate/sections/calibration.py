from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox,
    QSlider, QPushButton, QRadioButton, QLineEdit, QFrame,
    QColorDialog, QButtonGroup,
)
from PySide6.QtGui import QColor


class CalibrationSection(QWidget):
    """Grid & Calibration settings panel.

    Talks directly to CalibrationModel via set_calibration_model().
    Emits no forwarded signals — parent widgets do not need to relay anything.
    """

    calibration_reset_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._model = None
        self._refreshing = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(6)

        # ---- Status label ----
        self._status_lbl = QLabel("Not calibrated")
        self._status_lbl.setWordWrap(True)
        self._status_lbl.setStyleSheet("color: grey; font-style: italic;")
        layout.addWidget(self._status_lbl)

        # ---- Grid toggle + opacity ----
        grid_row = QHBoxLayout()
        self._grid_chk = QCheckBox("Show Grid")
        self._grid_chk.setChecked(False)
        self._grid_chk.toggled.connect(self._on_grid_toggled)
        grid_row.addWidget(self._grid_chk)
        grid_row.addStretch()
        grid_row.addWidget(QLabel("Opacity:"))
        self._opacity_slider = QSlider(Qt.Horizontal)
        self._opacity_slider.setRange(0, 100)
        self._opacity_slider.setValue(50)
        self._opacity_slider.setFixedWidth(80)
        self._opacity_slider.valueChanged.connect(self._on_opacity_changed)
        grid_row.addWidget(self._opacity_slider)
        self._opacity_lbl = QLabel("50%")
        self._opacity_lbl.setFixedWidth(32)
        grid_row.addWidget(self._opacity_lbl)
        layout.addLayout(grid_row)

        # ---- Grid color ----
        color_row = QHBoxLayout()
        color_row.addWidget(QLabel("Grid Color:"))
        self._color_btn = QPushButton()
        self._color_btn.setFixedSize(40, 22)
        self._color_btn.setToolTip("Click to change grid color")
        self._color_btn.clicked.connect(self._on_color_clicked)
        color_row.addWidget(self._color_btn)
        color_row.addStretch()
        layout.addLayout(color_row)
        self._update_color_swatch((58, 90, 122))

        # ---- Spacing ----
        spacing_row = QHBoxLayout()
        self._radio_auto = QRadioButton("Auto spacing")
        self._radio_fixed = QRadioButton("Fixed:")
        self._radio_auto.setChecked(True)
        spacing_group = QButtonGroup(self)
        spacing_group.addButton(self._radio_auto)
        spacing_group.addButton(self._radio_fixed)
        self._radio_auto.toggled.connect(self._on_spacing_mode_changed)
        spacing_row.addWidget(self._radio_auto)
        spacing_row.addWidget(self._radio_fixed)
        self._spacing_edit = QLineEdit()
        self._spacing_edit.setPlaceholderText("e.g. 1.0")
        self._spacing_edit.setFixedWidth(60)
        self._spacing_edit.setEnabled(False)
        self._spacing_edit.editingFinished.connect(self._on_spacing_edited)
        spacing_row.addWidget(self._spacing_edit)
        self._unit_lbl = QLabel("mm")
        spacing_row.addWidget(self._unit_lbl)
        layout.addLayout(spacing_row)

        # ---- Divider ----
        div = QFrame()
        div.setFrameShape(QFrame.HLine)
        div.setFrameShadow(QFrame.Sunken)
        layout.addWidget(div)

        # ---- Measurement display ----
        self._meas_lbl = QLabel("Distance: —")
        self._meas_lbl.setStyleSheet("font-weight: bold;")
        layout.addWidget(self._meas_lbl)

        # ---- Reset button ----
        reset_btn = QPushButton("Reset Calibration")
        reset_btn.clicked.connect(self._on_reset_clicked)
        layout.addWidget(reset_btn)

    # ------------------------------------------------------------------ #
    # Model binding
    # ------------------------------------------------------------------ #

    def set_calibration_model(self, model) -> None:
        self._model = model
        model.calibration_changed.connect(self._refresh_status)
        model.calibration_changed.connect(self._refresh_controls)
        model.grid_changed.connect(self._refresh_controls)
        model.measurement_updated.connect(self._refresh_measurement)
        self._refresh_status()
        self._refresh_controls()
        self._refresh_measurement()

    # ------------------------------------------------------------------ #
    # Slots
    # ------------------------------------------------------------------ #

    def _on_grid_toggled(self, checked: bool) -> None:
        if self._model is not None and not self._refreshing:
            self._model.set_grid_visible(checked)

    def _on_opacity_changed(self, value: int) -> None:
        self._opacity_lbl.setText(f"{value}%")
        if self._model is not None and not self._refreshing:
            self._model.set_grid_opacity(value / 100.0)

    def _on_color_clicked(self) -> None:
        if self._model is None:
            return
        r, g, b = self._model.grid_color()
        initial = QColor(r, g, b)
        color = QColorDialog.getColor(initial, self, "Grid Color")
        if color.isValid():
            rgb = (color.red(), color.green(), color.blue())
            self._model.set_grid_color(rgb)
            self._update_color_swatch(rgb)

    def _on_spacing_mode_changed(self, auto_checked: bool) -> None:
        self._spacing_edit.setEnabled(not auto_checked)
        if self._model is not None and not self._refreshing:
            if auto_checked:
                self._model.set_grid_spacing_auto()
            else:
                self._try_apply_spacing()

    def _on_spacing_edited(self) -> None:
        if self._model is not None and not self._refreshing and self._radio_fixed.isChecked():
            self._try_apply_spacing()

    def _try_apply_spacing(self) -> None:
        try:
            val = float(self._spacing_edit.text())
            if val > 0:
                self._model.set_grid_spacing(val)
        except ValueError:
            pass

    def _on_reset_clicked(self) -> None:
        if self._model is not None:
            self._model.clear_calibration()

    # ------------------------------------------------------------------ #
    # Refresh helpers
    # ------------------------------------------------------------------ #

    def _refresh_status(self) -> None:
        if self._model is None or not self._model.is_calibrated():
            self._status_lbl.setText("Not calibrated")
            self._status_lbl.setStyleSheet("color: grey; font-style: italic;")
        else:
            scale = self._model.scale()
            unit = self._model.unit()
            step = self._model.grid_spacing_world()
            self._status_lbl.setText(
                f"1 px = {scale:.4g} {unit}\n"
                f"Grid step: {step:g} {unit}"
            )
            self._status_lbl.setStyleSheet("color: black; font-style: normal;")

    def _refresh_controls(self) -> None:
        if self._model is None:
            return
        self._refreshing = True
        self._grid_chk.setChecked(self._model.grid_visible())
        opacity_pct = int(self._model.grid_opacity() * 100)
        self._opacity_slider.setValue(opacity_pct)
        self._opacity_lbl.setText(f"{opacity_pct}%")
        self._update_color_swatch(self._model.grid_color())
        self._radio_auto.setChecked(self._model.grid_spacing_auto())
        self._radio_fixed.setChecked(not self._model.grid_spacing_auto())
        self._spacing_edit.setEnabled(not self._model.grid_spacing_auto())
        self._spacing_edit.setText(f"{self._model.grid_spacing_world():g}")
        self._unit_lbl.setText(self._model.unit())
        self._refreshing = False

    def _refresh_measurement(self) -> None:
        if self._model is None:
            self._meas_lbl.setText("Distance: —")
            return
        dist = self._model.measured_distance()
        if dist is None:
            p1, p2 = self._model.meas_points()
            if p1 is not None and p2 is None:
                self._meas_lbl.setText("Distance: click point B…")
            else:
                self._meas_lbl.setText("Distance: —")
        else:
            self._meas_lbl.setText(f"Distance: {dist:.4g} {self._model.unit()}")

    def _update_color_swatch(self, rgb: tuple) -> None:
        r, g, b = rgb
        self._color_btn.setStyleSheet(
            f"background-color: rgb({r},{g},{b}); border: 1px solid #888;"
        )
