import pytest
from PySide6.QtCore import Qt

from core.states.calibration_state import CalibrationState
from models.calibration_model import CalibrationModel
from views.annomate.sections.grid import GridSection


@pytest.fixture
def calibrated_model():
    model = CalibrationModel(CalibrationState())
    model.set_calib_points((0.0, 0.0), (100.0, 0.0))
    assert model.apply_calibration(10.0, "mm")
    return model


def test_grid_enabled_and_checked_once_calibrated(qtbot):
    """Verify grid controls are enabled but unchecked in the default pixel mode.

    The default 1px:1px scale still counts as "has scale", so grid controls
    are enabled immediately; grid_visible defaults to False until an actual
    calibration is applied, at which point it flips on.
    """
    model = CalibrationModel(CalibrationState())
    section = GridSection(model)
    qtbot.addWidget(section)

    assert section._grid_chk.isEnabled()
    assert model.grid_visible() is False
    assert not section._grid_chk.isChecked()

    model.set_calib_points((0.0, 0.0), (100.0, 0.0))
    assert model.apply_calibration(5.0, "mm")

    assert section._grid_chk.isEnabled()
    assert model.grid_visible() is True
    assert section._grid_chk.isChecked()


def test_grid_toggle_updates_model(calibrated_model, qtbot):
    """Verify clicking the grid visibility checkbox updates the calibration model's grid_visible flag.

    The grid starts visible. Clicking the checkbox should hide the grid in
    the model and uncheck the control.
    """
    section = GridSection(calibrated_model)
    qtbot.addWidget(section)

    assert calibrated_model.grid_visible() is True
    section._grid_chk.click()

    assert calibrated_model.grid_visible() is False
    assert not section._grid_chk.isChecked()


def test_opacity_and_spacing_controls_update_model(calibrated_model, qtbot):
    """Verify the opacity slider and fixed-spacing input update the calibration model.

    Success means grid_opacity() reflects the slider value, and switching to
    Fixed spacing with a typed value updates grid_spacing_world().
    """
    section = GridSection(calibrated_model)
    qtbot.addWidget(section)

    section._opacity_slider.setValue(75)
    assert calibrated_model.grid_opacity() == 0.75

    section._spacing_edit.setText("2.5")
    section._radio_fixed.setChecked(True)
    assert calibrated_model.grid_spacing_auto() is False
    assert calibrated_model.grid_spacing_world() == 2.5


def test_calibrate_button_emits_tool_toggled(qtbot):
    """Verify clicking the calibrate button emits calibrate_tool_toggled and toggles as expected.

    Clicking calibrate emits True and checks the button; clicking it again
    emits False and unchecks it.
    """
    model = CalibrationModel(CalibrationState())
    section = GridSection(model)
    section.set_has_image(True)
    qtbot.addWidget(section)

    requested = []
    section.calibrate_tool_toggled.connect(requested.append)

    qtbot.mouseClick(section._btn_calibrate_points, Qt.LeftButton)
    assert requested[-1] is True
    assert section._btn_calibrate_points.isChecked()

    qtbot.mouseClick(section._btn_calibrate_points, Qt.LeftButton)
    assert requested[-1] is False
    assert not section._btn_calibrate_points.isChecked()


def test_calibrate_points_enabled_in_default_pixel_mode(qtbot):
    """Verify calibrate-points is enabled and status shows '1px:1px' before any calibration.

    In the default uncalibrated state, the calibrate-points button should be
    enabled (once an image is loaded) and the status label should display
    '1px:1px'.
    """
    model = CalibrationModel(CalibrationState())
    section = GridSection(model)
    section.set_has_image(True)
    qtbot.addWidget(section)

    assert section._btn_calibrate_points.isEnabled()
    assert "1px:1px" in section._calib_status_lbl.text()


def test_reset_defaults_button_resets_calibration_and_grid_display(
    calibrated_model, qtbot
):
    """Verify Reset to Defaults clears calibration AND restores grid display settings.

    Unlike the old "Reset to pixels" button, this resets the whole section --
    calibration, plus opacity/color, which aren't touched by clear_calibration()
    on their own.
    """
    section = GridSection(calibrated_model)
    section.set_has_image(True)
    qtbot.addWidget(section)

    calibrated_model.set_grid_opacity(0.9)
    calibrated_model.set_grid_color((10, 20, 30))

    qtbot.mouseClick(section._btn_reset_defaults, Qt.LeftButton)

    assert calibrated_model.is_calibrated() is False
    assert calibrated_model.has_scale() is True
    assert calibrated_model.unit() == "px"
    assert calibrated_model.grid_opacity() == pytest.approx(0.5)
    assert calibrated_model.grid_color() == (58, 90, 122)
