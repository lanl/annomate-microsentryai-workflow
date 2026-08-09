import pytest

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
