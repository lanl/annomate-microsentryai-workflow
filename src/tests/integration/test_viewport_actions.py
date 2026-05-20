import numpy as np
import pytest
from PySide6.QtCore import Qt

from core.states.calibration_state import CalibrationState
from models.calibration_model import CalibrationModel
from views.annomate.image_label import ImageLabel
from views.annomate.viewport_actions import ViewportActionsBar


@pytest.fixture
def calibrated_model():
    model = CalibrationModel(CalibrationState())
    model.set_calib_points((0.0, 0.0), (100.0, 0.0))
    assert model.apply_calibration(10.0, "mm")
    return model


@pytest.fixture
def canvas(qtbot):
    widget = ImageLabel()
    widget.resize(320, 240)
    widget.set_image(np.zeros((100, 100, 3), dtype=np.uint8))
    qtbot.addWidget(widget)
    widget.show()
    return widget


def test_zoom_buttons_drive_canvas(canvas, calibrated_model, qtbot):
    bar = ViewportActionsBar(canvas, calibrated_model, canvas)
    qtbot.addWidget(bar)

    qtbot.mouseClick(bar._btn_zoom_in, Qt.LeftButton)
    assert canvas._zoom > 1.0

    qtbot.mouseClick(bar._btn_reset, Qt.LeftButton)
    assert canvas._zoom == 1.0

    qtbot.mouseClick(bar._btn_zoom_out, Qt.LeftButton)
    assert canvas._zoom < 1.0


def test_calibrate_and_measure_emit_tool_requests(canvas, calibrated_model, qtbot):
    bar = ViewportActionsBar(canvas, calibrated_model, canvas)
    bar.set_image_loaded(True)
    qtbot.addWidget(bar)

    requested = []
    bar.tool_selected.connect(requested.append)

    qtbot.mouseClick(bar._btn_calibrate, Qt.LeftButton)
    assert requested[-1] == "calibrate"
    assert bar._btn_calibrate.isChecked()
    assert not bar._btn_measure.isChecked()

    qtbot.mouseClick(bar._btn_measure, Qt.LeftButton)
    assert requested[-1] == "measure"
    assert bar._btn_measure.isChecked()
    assert not bar._btn_calibrate.isChecked()

    qtbot.mouseClick(bar._btn_measure, Qt.LeftButton)
    assert requested[-1] == ""
    assert not bar._btn_measure.isChecked()


def test_measure_and_grid_settings_disabled_until_calibrated(canvas, qtbot):
    model = CalibrationModel(CalibrationState())
    bar = ViewportActionsBar(canvas, model, canvas)
    bar.set_image_loaded(True)
    qtbot.addWidget(bar)

    assert bar._btn_calibrate.isEnabled()
    assert not bar._btn_measure.isEnabled()
    assert not bar._grid_chk.isEnabled()

    model.set_calib_points((0.0, 0.0), (100.0, 0.0))
    assert model.apply_calibration(5.0, "mm")

    assert bar._btn_measure.isEnabled()
    assert bar._grid_chk.isEnabled()
    assert model.grid_visible() is True
    assert bar._grid_chk.isChecked()


def test_grid_toggle_in_settings_updates_model(canvas, calibrated_model, qtbot):
    bar = ViewportActionsBar(canvas, calibrated_model, canvas)
    qtbot.addWidget(bar)

    assert calibrated_model.grid_visible() is True
    bar._grid_chk.click()

    assert calibrated_model.grid_visible() is False
    assert not bar._grid_chk.isChecked()


def test_settings_controls_update_calibration_model(canvas, calibrated_model, qtbot):
    bar = ViewportActionsBar(canvas, calibrated_model, canvas)
    qtbot.addWidget(bar)

    bar._opacity_slider.setValue(75)
    assert calibrated_model.grid_opacity() == 0.75

    bar._spacing_edit.setText("2.5")
    bar._radio_fixed.setChecked(True)
    assert calibrated_model.grid_spacing_auto() is False
    assert calibrated_model.grid_spacing_world() == 2.5

    calibrated_model.set_meas_p1((0.0, 0.0))
    calibrated_model.set_meas_p2((100.0, 0.0))
    assert "10" in bar._meas_lbl.text()

    qtbot.mouseClick(bar._btn_clear_measurement, Qt.LeftButton)
    assert calibrated_model.meas_points() == (None, None)

    qtbot.mouseClick(bar._btn_reset_calibration, Qt.LeftButton)
    assert calibrated_model.is_calibrated() is False
    assert not bar._btn_measure.isEnabled()
    assert not bar._grid_chk.isEnabled()


def test_center_crop_controls_update_canvas(canvas, calibrated_model, qtbot):
    bar = ViewportActionsBar(canvas, calibrated_model, canvas)
    bar.set_image_dimensions(100, 100)
    qtbot.addWidget(bar)

    assert not canvas.center_crop_settings()["enabled"]

    bar._crop_chk.click()
    bar._crop_width_spin.setValue(40)
    bar._crop_height_spin.setValue(30)
    bar._crop_shape_combo.setCurrentText("Circle")
    bar._crop_height_spin.setValue(15)
    bar._crop_opacity_slider.setValue(80)
    bar._crop_center_dot_chk.setChecked(True)

    settings = canvas.center_crop_settings()
    assert settings["enabled"] is True
    assert settings["width"] == 30
    assert settings["height"] == 30
    assert settings["shape"] == "circle"
    assert settings["opacity"] == 0.8
    assert settings["center_dot"] is True
    assert bar._crop_width_spin.value() == 30
    assert bar._crop_height_spin.value() == 15


def test_center_crop_defaults_to_605px_radius(qtbot):
    widget = ImageLabel()
    widget.set_image(np.zeros((1080, 1080, 3), dtype=np.uint8))
    qtbot.addWidget(widget)

    settings = widget.center_crop_settings()
    assert settings["shape"] == "circle"
    assert settings["width"] == 1210
    assert settings["height"] == 1210


def test_center_crop_reset_restores_defaults(canvas, calibrated_model, qtbot):
    bar = ViewportActionsBar(canvas, calibrated_model, canvas)
    bar.set_image_dimensions(100, 100)
    qtbot.addWidget(bar)

    bar._crop_shape_combo.setCurrentText("Rectangle")
    bar._crop_width_spin.setValue(40)
    bar._crop_height_spin.setValue(30)
    bar._crop_opacity_slider.setValue(80)

    qtbot.mouseClick(bar._btn_reset_crop, Qt.LeftButton)

    settings = canvas.center_crop_settings()
    assert settings["shape"] == "circle"
    assert settings["width"] == 1210
    assert settings["height"] == 1210
    assert settings["opacity"] == 0.37
    assert settings["center_dot"] is False
    assert bar._crop_height_spin.value() == 605
