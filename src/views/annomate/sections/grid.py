"""
GridSection — "Grid" section body for the right activity bar's View
Overlays tab.

Owns calibration (the pixels-to-real-units scale that everything here
depends on) and the grid overlay display controls (visibility, opacity,
spacing, color) that consume it. Calibration used to live in the
viewport's floating toolbar as a separate concern shared with the Measure
tool and Anomaly Constraints; it moved here because grid calibration is
its primary use, and the other two consumers just read the resulting
CalibrationModel scale wherever they already live (Measure's readout is
drawn live on-canvas; Anomaly Constraints reads the unit passively).
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QButtonGroup,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ._shared import _toggle_button

_DEFAULT_GRID_OPACITY = 0.5
_DEFAULT_GRID_COLOR = (58, 90, 122)


def _hline() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setFrameShadow(QFrame.Sunken)
    return line


class GridSection(QWidget):
    """Calibration + grid overlay controls, driven directly by a CalibrationModel.

    Signals:
        calibrate_tool_toggled (bool): The "Click two points…" button was
            checked/unchecked -- the parent arms/disarms the canvas's
            CALIBRATE tool in response.
    """

    calibrate_tool_toggled = Signal(bool)

    def __init__(self, calibration_model=None, parent: QWidget = None) -> None:
        super().__init__(parent)
        self._model = None
        self._refreshing = False
        self._has_image = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        calib_header = QLabel("Calibration")
        calib_header.setStyleSheet("font-weight: bold;")
        layout.addWidget(calib_header)

        self._calib_status_lbl = QLabel("Current Calibration: None")
        self._calib_status_lbl.setWordWrap(True)
        layout.addWidget(self._calib_status_lbl)

        # Ratio input: [ 1 ] px : [ 0.05 ] [ mm▾ ] [ Apply ]
        ratio_row = QHBoxLayout()
        ratio_row.setSpacing(4)
        self._ratio_px_spin = QSpinBox()
        self._ratio_px_spin.setRange(1, 999999)
        self._ratio_px_spin.setValue(1)
        self._ratio_px_spin.setToolTip("Number of pixels on the left side of the ratio")
        self._ratio_px_spin.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        ratio_row.addWidget(self._ratio_px_spin)
        ratio_row.addWidget(QLabel("px :"))
        self._ratio_val_num_spin = QDoubleSpinBox()
        self._ratio_val_num_spin.setRange(0.0, 999999.0)
        self._ratio_val_num_spin.setDecimals(2)
        self._ratio_val_num_spin.setSingleStep(0.1)
        self._ratio_val_num_spin.setToolTip(
            "Real-world value for the right side of the ratio"
        )
        self._ratio_val_num_spin.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        ratio_row.addWidget(self._ratio_val_num_spin)
        self._ratio_unit_combo = QComboBox()
        for _u in ("mm", "um", "nm", "pm", "fm", "cm", "dm", "m", "km"):
            self._ratio_unit_combo.addItem(_u)
        self._ratio_unit_combo.setFixedWidth(52)
        ratio_row.addWidget(self._ratio_unit_combo)
        self._btn_apply_ratio = QPushButton("Apply")
        self._btn_apply_ratio.setFixedWidth(50)
        self._btn_apply_ratio.setToolTip("Apply this pixel-to-real-world ratio as the calibration")
        self._btn_apply_ratio.clicked.connect(self._on_apply_ratio_clicked)
        ratio_row.addWidget(self._btn_apply_ratio)
        layout.addLayout(ratio_row)

        # Import / Export calibration file buttons
        ratio_file_row = QHBoxLayout()
        ratio_file_row.setSpacing(4)
        self._btn_import_ratio = QPushButton("Import")
        self._btn_import_ratio.setToolTip("Load a ratio from a plain-text .txt file")
        self._btn_import_ratio.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self._btn_import_ratio.clicked.connect(self._on_import_ratio_clicked)
        ratio_file_row.addWidget(self._btn_import_ratio)
        self._btn_export_ratio = QPushButton("Export")
        self._btn_export_ratio.setToolTip(
            "Save the current ratio to a plain-text .txt file"
        )
        self._btn_export_ratio.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self._btn_export_ratio.setEnabled(False)  # enabled once a calibration is set
        self._btn_export_ratio.clicked.connect(self._on_export_ratio_clicked)
        ratio_file_row.addWidget(self._btn_export_ratio)
        layout.addLayout(ratio_file_row)

        self._btn_calibrate_points = QPushButton("Click two points…")
        self._btn_calibrate_points.setCheckable(True)
        self._btn_calibrate_points.setToolTip(
            "Click two known points on the image, then enter the real distance"
        )
        self._btn_calibrate_points.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self._btn_calibrate_points.clicked.connect(self._on_calibrate_clicked)
        layout.addWidget(self._btn_calibrate_points)

        layout.addWidget(_hline())

        self._grid_chk = _toggle_button(
            "Enable Grid", tooltip="Overlay a calibrated reference grid on the canvas"
        )
        self._grid_chk.toggled.connect(self._on_grid_toggled)
        layout.addWidget(self._grid_chk)

        opacity_row = QHBoxLayout()
        opacity_row.setSpacing(6)
        opacity_row.addWidget(QLabel("Opacity"))
        self._opacity_slider = QSlider(Qt.Horizontal)
        self._opacity_slider.setRange(0, 100)
        self._opacity_slider.setValue(50)
        self._opacity_slider.valueChanged.connect(self._on_opacity_changed)
        opacity_row.addWidget(self._opacity_slider)
        self._opacity_lbl = QLabel("50%")
        self._opacity_lbl.setFixedWidth(34)
        opacity_row.addWidget(self._opacity_lbl)
        layout.addLayout(opacity_row)

        spacing_mode_row = QHBoxLayout()
        spacing_mode_row.setSpacing(6)
        spacing_mode_row.addWidget(QLabel("Spacing"))
        spacing_mode_row.addStretch()
        self._radio_auto = QRadioButton("Auto")
        self._radio_auto.setToolTip("Space grid lines automatically based on zoom level")
        self._radio_fixed = QRadioButton("Fixed")
        self._radio_fixed.setToolTip("Space grid lines at a fixed real-world distance")
        self._radio_auto.setChecked(True)
        spacing_group = QButtonGroup(self)
        spacing_group.addButton(self._radio_auto)
        spacing_group.addButton(self._radio_fixed)
        self._radio_auto.toggled.connect(self._on_spacing_mode_changed)
        spacing_mode_row.addWidget(self._radio_auto)
        spacing_mode_row.addWidget(self._radio_fixed)
        layout.addLayout(spacing_mode_row)

        spacing_val_row = QHBoxLayout()
        spacing_val_row.setSpacing(6)
        self._spacing_edit = QLineEdit()
        self._spacing_edit.setPlaceholderText("1.0")
        self._spacing_edit.setEnabled(False)
        self._spacing_edit.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self._spacing_edit.editingFinished.connect(self._on_spacing_edited)
        spacing_val_row.addWidget(self._spacing_edit)
        self._unit_lbl = QLabel("px")
        spacing_val_row.addWidget(self._unit_lbl)
        layout.addLayout(spacing_val_row)

        color_row = QHBoxLayout()
        color_row.setSpacing(6)
        color_row.addWidget(QLabel("Color"))
        self._color_btn = QPushButton()
        self._color_btn.setFixedSize(32, 20)
        self._color_btn.setToolTip("Change grid color")
        self._color_btn.clicked.connect(self._on_color_clicked)
        color_row.addWidget(self._color_btn)
        color_row.addStretch()
        layout.addLayout(color_row)

        self._btn_reset_defaults = QPushButton("Reset to Defaults")
        self._btn_reset_defaults.setToolTip(
            "Remove calibration and restore grid display settings to their defaults"
        )
        self._btn_reset_defaults.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self._btn_reset_defaults.clicked.connect(self._on_reset_defaults_clicked)
        layout.addWidget(self._btn_reset_defaults)

        self._update_color_swatch(_DEFAULT_GRID_COLOR)

        if calibration_model is not None:
            self.set_calibration_model(calibration_model)
        else:
            self._refresh_availability()

    def set_calibration_model(self, model) -> None:
        self._model = model
        model.calibration_changed.connect(self._refresh_controls)
        model.grid_changed.connect(self._refresh_controls)
        self._refresh_controls()

    def set_has_image(self, has_image: bool) -> None:
        self._has_image = bool(has_image)
        self._refresh_availability()

    def set_active_tool(self, tool_key: str) -> None:
        """Sync the calibrate button's checked state without emitting toggled."""
        self._refreshing = True
        self._btn_calibrate_points.setChecked(tool_key == "calibrate")
        self._refreshing = False

    def toggle_calibrate(self) -> None:
        """Toggle the calibrate tool on/off (no-op without a loaded image)."""
        if self._btn_calibrate_points.isEnabled():
            checked = not self._btn_calibrate_points.isChecked()
            self._btn_calibrate_points.setChecked(checked)
            self._on_calibrate_clicked(checked)

    # ------------------------------------------------------------------ #
    # Calibration
    # ------------------------------------------------------------------ #

    def _on_calibrate_clicked(self, checked: bool) -> None:
        if self._refreshing:
            return
        self.calibrate_tool_toggled.emit(checked)

    def _on_apply_ratio_clicked(self) -> None:
        if self._model is None:
            return
        px_count = self._ratio_px_spin.value()
        world_val = self._ratio_val_num_spin.value()
        unit = self._ratio_unit_combo.currentText()
        if world_val <= 0:
            QMessageBox.warning(
                self, "Invalid Ratio", "World value must be greater than zero."
            )
            return
        self._model.apply_scale_direct(px_count, world_val, unit)

    def _on_import_ratio_clicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Calibration Ratio",
            "",
            "Calibration Ratio (*.txt)",
        )
        if not path:
            return
        from core.persistence.calibration_io import read_calibration_ratio

        try:
            data = read_calibration_ratio(path)
            self._model.apply_scale_direct(
                data["px_count"], data["world_val"], data["unit"]
            )
        except Exception as exc:
            QMessageBox.critical(self, "Import Error", str(exc))

    def _on_export_ratio_clicked(self) -> None:
        if self._model is None or not self._model.has_scale():
            return  # button is disabled in this state; guard kept as a safety net
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Calibration Ratio",
            "calibration.txt",
            "Calibration Ratio (*.txt)",
        )
        if not path:
            return
        from core.persistence.calibration_io import write_calibration_ratio

        try:
            write_calibration_ratio(
                path,
                self._model.px_count(),
                self._model.world_val(),
                self._model.unit(),
            )
            QMessageBox.information(
                self, "Export Calibration Ratio", f"Saved to:\n{path}"
            )
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))

    def _on_reset_defaults_clicked(self) -> None:
        if self._model is None:
            return
        self._model.clear_calibration()
        self._model.set_grid_opacity(_DEFAULT_GRID_OPACITY)
        self._model.set_grid_color(_DEFAULT_GRID_COLOR)

    def _refresh_calib_status(self) -> None:
        has_scale = self._model is not None and self._model.has_scale()
        self._btn_export_ratio.setEnabled(has_scale)
        if not has_scale:
            self._calib_status_lbl.setText("Current Calibration: None")
            return
        from core.persistence.calibration_io import format_ratio_string

        px_count = self._model.px_count()
        world_val = self._model.world_val()
        unit = self._model.unit()
        ratio_str = format_ratio_string(px_count, world_val, unit)
        self._calib_status_lbl.setText(f"Current Calibration: {ratio_str}")
        if not self._ratio_px_spin.hasFocus():
            self._ratio_px_spin.setValue(int(round(px_count)))
        if not self._ratio_val_num_spin.hasFocus():
            self._ratio_val_num_spin.setValue(world_val)
            idx = self._ratio_unit_combo.findText(unit)
            if idx >= 0:
                self._ratio_unit_combo.setCurrentIndex(idx)
            else:
                self._ratio_unit_combo.addItem(unit)
                self._ratio_unit_combo.setCurrentText(unit)

    # ------------------------------------------------------------------ #
    # Grid display
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
        color = QColorDialog.getColor(QColor(r, g, b), self, "Grid Color")
        if color.isValid():
            rgb = (color.red(), color.green(), color.blue())
            self._model.set_grid_color(rgb)
            self._update_color_swatch(rgb)

    def _on_spacing_mode_changed(self, auto_checked: bool) -> None:
        self._spacing_edit.setEnabled(not auto_checked)
        if self._model is None or self._refreshing:
            return
        if auto_checked:
            self._model.set_grid_spacing_auto()
        else:
            self._try_apply_spacing()

    def _on_spacing_edited(self) -> None:
        if (
            self._model is not None
            and not self._refreshing
            and self._radio_fixed.isChecked()
        ):
            self._try_apply_spacing()

    def _try_apply_spacing(self) -> None:
        try:
            value = float(self._spacing_edit.text())
        except ValueError:
            return
        if value > 0:
            self._model.set_grid_spacing(value)

    def _refresh_controls(self) -> None:
        if self._model is None:
            self._refresh_availability()
            return
        self._refreshing = True
        self._refresh_calib_status()
        grid_visible = self._model.grid_visible() and self._model.has_scale()
        self._grid_chk.setChecked(grid_visible)
        opacity_pct = int(self._model.grid_opacity() * 100)
        self._opacity_slider.setValue(opacity_pct)
        self._opacity_lbl.setText(f"{opacity_pct}%")
        self._update_color_swatch(self._model.grid_color())
        auto_spacing = self._model.grid_spacing_auto()
        self._radio_auto.setChecked(auto_spacing)
        self._radio_fixed.setChecked(not auto_spacing)
        self._spacing_edit.setEnabled(not auto_spacing)
        self._spacing_edit.setText(f"{self._model.grid_spacing_world():g}")
        self._unit_lbl.setText(self._model.unit())
        self._refreshing = False
        self._refresh_availability()

    def _refresh_availability(self) -> None:
        scale_available = self._model is not None and self._model.has_scale()
        self._btn_calibrate_points.setEnabled(self._has_image)
        self._btn_export_ratio.setEnabled(scale_available)
        self._btn_reset_defaults.setEnabled(scale_available)
        self._grid_chk.setEnabled(scale_available)
        self._opacity_slider.setEnabled(scale_available)
        self._color_btn.setEnabled(scale_available)
        self._radio_auto.setEnabled(scale_available)
        self._radio_fixed.setEnabled(scale_available)
        self._spacing_edit.setEnabled(scale_available and self._radio_fixed.isChecked())

    def _update_color_swatch(self, rgb: tuple) -> None:
        r, g, b = rgb
        self._color_btn.setStyleSheet(
            f"background-color: rgb({r},{g},{b}); border: 1px solid #888;"
        )
