"""
GridSection — "Grid" section body for the right activity bar's View
Overlays tab.

Owns the grid overlay display controls (visibility, opacity, spacing,
color). These used to live bundled into the viewport's Calibration/Grid
popup menu; only the grid-display portion moved here -- calibration and
measurement stayed in the viewport bar since they aren't grid-specific.
"""

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QColorDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)


class GridSection(QWidget):
    """Grid overlay controls, driven directly by a CalibrationModel."""

    def __init__(self, calibration_model=None, parent: QWidget = None) -> None:
        super().__init__(parent)
        self._model = None
        self._refreshing = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self._grid_chk = QCheckBox("Show Grid")
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
        self._radio_fixed = QRadioButton("Fixed")
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

        self._update_color_swatch((58, 90, 122))

        if calibration_model is not None:
            self.set_calibration_model(calibration_model)
        else:
            self._refresh_availability()

    def set_calibration_model(self, model) -> None:
        self._model = model
        model.calibration_changed.connect(self._refresh_controls)
        model.grid_changed.connect(self._refresh_controls)
        self._refresh_controls()

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
