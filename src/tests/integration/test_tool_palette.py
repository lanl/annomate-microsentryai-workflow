import pytest
from PySide6.QtCore import Qt

from core.states.calibration_state import CalibrationState
from models.calibration_model import CalibrationModel
from views.annomate.tool_palette import ToolPalette


@pytest.fixture
def calibrated_model():
    model = CalibrationModel(CalibrationState())
    model.set_calib_points((0.0, 0.0), (100.0, 0.0))
    assert model.apply_calibration(10.0, "mm")
    return model


def test_polygon_sam_measure_are_mutually_exclusive(qtbot, calibrated_model):
    """Verify Polygon, SAM, and Measure share one exclusive button group.

    Selecting one tool emits its name and unchecks the others -- no manual
    deselect_all() call is needed since QButtonGroup enforces exclusivity.
    """
    palette = ToolPalette(calibration_model=calibrated_model)
    qtbot.addWidget(palette)

    requested = []
    palette.tool_selected.connect(requested.append)

    poly_btn = next(b for b, name in palette._btn_tool.items() if name == "polygon")
    sam_btn = next(b for b, name in palette._btn_tool.items() if name == "sam_bbox")

    qtbot.mouseClick(poly_btn, Qt.LeftButton)
    assert requested[-1] == "polygon"
    assert poly_btn.isChecked()

    qtbot.mouseClick(palette._btn_measure, Qt.LeftButton)
    assert requested[-1] == "measure"
    assert palette._btn_measure.isChecked()
    assert not poly_btn.isChecked()

    qtbot.mouseClick(sam_btn, Qt.LeftButton)
    assert requested[-1] == "sam_bbox"
    assert sam_btn.isChecked()
    assert not palette._btn_measure.isChecked()


def test_measure_disabled_without_calibration_model(qtbot):
    """Verify the measure tool stays disabled when no calibration model is attached.

    Without a model reference there's no way to ever query a scale, so the
    button must be permanently unusable rather than silently no-op on click.
    """
    palette = ToolPalette()
    qtbot.addWidget(palette)

    assert not palette._btn_measure.isEnabled()

    requested = []
    palette.tool_selected.connect(requested.append)
    palette.toggle_measure()
    assert requested == []  # toggle is a no-op while disabled


def test_measure_enabled_once_any_calibration_model_attached(qtbot):
    """Verify the measure tool is enabled as soon as a calibration model exists.

    A fresh CalibrationModel already reports has_scale()=True via its
    default 1px:1px ratio (matching ViewportActionsBar's Grid/Measure
    controls, which are likewise enabled before the user ever calibrates)
    -- real calibration narrows the *unit*, it isn't a prerequisite for
    the button being usable at all.
    """
    model = CalibrationModel(CalibrationState())
    assert model.has_scale() is True

    palette = ToolPalette(calibration_model=model)
    qtbot.addWidget(palette)

    assert palette._btn_measure.isEnabled()


def test_toggle_hotkey_path_still_deselects_the_other_tool(qtbot, calibrated_model):
    """Verify switching tools via the toggle_*() hotkey path fully unchecks the old one.

    Regression guard: toggle_measure()/toggle_polygon()/toggle_sam() go
    through _toggle_tool(), a different code path than a direct button
    click (_on_btn_clicked). Both must leave exactly one tool checked.
    """
    palette = ToolPalette(calibration_model=calibrated_model)
    qtbot.addWidget(palette)

    palette.toggle_measure()
    assert palette._btn_measure.isChecked()

    palette.toggle_polygon()
    poly_btn = next(b for b, name in palette._btn_tool.items() if name == "polygon")
    assert poly_btn.isChecked()
    assert not palette._btn_measure.isChecked()


def test_toggle_measure_hotkey_matches_polygon_and_sam(qtbot, calibrated_model):
    """Verify toggle_measure() follows the same on/off pattern as toggle_polygon/toggle_sam."""
    palette = ToolPalette(calibration_model=calibrated_model)
    qtbot.addWidget(palette)

    requested = []
    palette.tool_selected.connect(requested.append)

    palette.toggle_measure()
    assert requested[-1] == "measure"
    assert palette._btn_measure.isChecked()

    palette.toggle_measure()
    assert requested[-1] == ""
    assert not palette._btn_measure.isChecked()
