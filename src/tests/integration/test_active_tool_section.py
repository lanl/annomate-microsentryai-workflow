import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QPushButton

from core.states.calibration_state import CalibrationState
from models.calibration_model import CalibrationModel
from views.annomate.sections.active_tool import ActiveToolSection


@pytest.fixture
def calibrated_model():
    model = CalibrationModel(CalibrationState())
    model.set_calib_points((0.0, 0.0), (100.0, 0.0))
    assert model.apply_calibration(10.0, "mm")
    return model


def test_measure_tool_options_page_registered(qtbot):
    """Verify Measure has a Tool Options page, like SAM does."""
    section = ActiveToolSection()
    qtbot.addWidget(section)

    assert "measure" in section._tool_pages
    section.set_active_tool("measure")
    assert section._tool_section.isVisibleTo(section)
    assert section._tool_stack.currentWidget() is section._tool_stack.widget(
        section._tool_pages["measure"]
    )


def test_clear_measurement_button_clears_calibration_model(calibrated_model, qtbot):
    """Verify Measure's Tool Options page can clear an in-progress measurement.

    The live distance readout is drawn on the canvas, not here -- this page
    only holds the clear action.
    """
    section = ActiveToolSection(calibrated_model)
    qtbot.addWidget(section)
    section.set_active_tool("measure")

    calibrated_model.set_meas_p1((0.0, 0.0))
    calibrated_model.set_meas_p2((100.0, 0.0))
    assert calibrated_model.meas_points() != (None, None)

    page = section._tool_stack.widget(section._tool_pages["measure"])
    btn_clear = page.findChild(QPushButton)
    qtbot.mouseClick(btn_clear, Qt.LeftButton)

    assert calibrated_model.meas_points() == (None, None)
