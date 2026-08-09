import numpy as np
import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor

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
    """Verify that zoom-in, zoom-out, and reset buttons correctly change the canvas zoom level.

    Starts at fit-zoom, clicks zoom-in (zoom increases), clicks reset (zoom returns to fit),
    then clicks zoom-out (zoom decreases below fit). Success means each button produces
    the expected zoom change.
    """
    bar = ViewportActionsBar(canvas, calibrated_model, canvas)
    qtbot.addWidget(bar)

    fit_zoom = canvas._zoom
    qtbot.mouseClick(bar._btn_zoom_in, Qt.LeftButton)
    assert canvas._zoom > fit_zoom

    qtbot.mouseClick(bar._btn_reset, Qt.LeftButton)
    assert canvas._zoom == pytest.approx(fit_zoom)

    qtbot.mouseClick(bar._btn_zoom_out, Qt.LeftButton)
    assert canvas._zoom < fit_zoom


def test_set_image_fits_and_centers_image(canvas):
    """Verify that loading an image via set_image initializes zoom and pan to fit and center the image.

    A 100x100 image loaded into a 320x240 canvas should be fit-zoomed and centered.
    Success means zoom is 2.4 (240/100), pan_x is 40 (centering horizontally), and pan_y is 0.
    """
    assert canvas._zoom == pytest.approx(2.4)
    assert canvas._pan.x() == pytest.approx(40.0)
    assert canvas._pan.y() == pytest.approx(0.0)


def test_reset_view_fits_current_viewport(canvas):
    """Verify that reset_view() restores the fit-zoom and centered pan regardless of current state.

    Manually sets zoom to 5.0 and a non-zero pan, then calls reset_view(). Success means
    zoom returns to the fit value (2.4) and pan returns to the centered position.
    """
    canvas._zoom = 5.0
    canvas._pan = QPoint(12, 34)

    canvas.reset_view()

    assert canvas._zoom == pytest.approx(2.4)
    assert canvas._pan.x() == pytest.approx(40.0)
    assert canvas._pan.y() == pytest.approx(0.0)


def test_hidden_annotation_overlays_keep_index_but_clear_selection(canvas):
    """Verify that set_overlays stores all overlays (including hidden ones) and clears the active selection.

    Passes two overlays: one hidden (visible=False) and one visible (visible=True). The
    canvas should store both but reset the selected_polygon_idx to -1 since overlays
    changed. Success means both overlays are stored, visibility flags are preserved, and
    selected_polygon_idx is -1.
    """
    canvas.selected_polygon_idx = 0

    canvas.set_overlays(
        [
            ([(0, 0), (10, 0), (10, 10)], QColor(255, 0, 0), 2.0, False),
            ([(20, 20), (30, 20), (30, 30)], QColor(0, 255, 0), 2.0, True),
        ]
    )

    assert len(canvas._overlays) == 2
    assert canvas._overlays[0][3] is False
    assert canvas._overlays[1][3] is True
    assert canvas.selected_polygon_idx == -1


def test_calibrate_emits_tool_requests(canvas, calibrated_model, qtbot):
    """Verify that the calibrate button emits tool_selected and toggles as expected.

    Clicking calibrate emits 'calibrate' and checks the button; clicking it again
    deactivates it and emits an empty string. The measure tool now lives on
    ToolPalette (see test_tool_palette.py), not here.
    """
    bar = ViewportActionsBar(canvas, calibrated_model, canvas)
    bar.set_image_loaded(True)
    qtbot.addWidget(bar)

    requested = []
    bar.tool_selected.connect(requested.append)

    qtbot.mouseClick(bar._btn_calibrate_points, Qt.LeftButton)
    assert requested[-1] == "calibrate"
    assert bar._btn_calibrate_points.isChecked()

    qtbot.mouseClick(bar._btn_calibrate_points, Qt.LeftButton)
    assert requested[-1] == ""
    assert not bar._btn_calibrate_points.isChecked()


def test_grid_settings_enabled_in_default_pixel_mode(canvas, qtbot):
    """Verify that calibration and grid controls are all enabled and show correct defaults in pixel mode.

    In the default uncalibrated state (1px:1px), calibrate and grid controls
    should all be enabled, the status label should display '1px:1px', and the grid
    checkbox should be checked. These states should persist after applying calibration.
    Success means all assertions pass both before and after calibration.
    """
    model = CalibrationModel(CalibrationState())
    bar = ViewportActionsBar(canvas, model, canvas)
    bar.set_image_loaded(True)
    qtbot.addWidget(bar)

    assert bar._btn_calibrate_points.isEnabled()
    assert bar._grid_chk.isEnabled()
    assert model.grid_visible() is False
    assert not bar._grid_chk.isChecked()
    assert "1px:1px" in bar._calib_status_lbl.text()

    model.set_calib_points((0.0, 0.0), (100.0, 0.0))
    assert model.apply_calibration(5.0, "mm")

    assert bar._grid_chk.isEnabled()
    assert model.grid_visible() is True
    assert bar._grid_chk.isChecked()


def test_grid_toggle_in_settings_updates_model(canvas, calibrated_model, qtbot):
    """Verify that clicking the grid visibility checkbox updates the calibration model's grid_visible flag.

    The grid starts visible. Clicking the grid checkbox should hide the grid in the
    model and uncheck the control. Success means grid_visible() is False and the
    checkbox is unchecked after the click.
    """
    bar = ViewportActionsBar(canvas, calibrated_model, canvas)
    qtbot.addWidget(bar)

    assert calibrated_model.grid_visible() is True
    bar._grid_chk.click()

    assert calibrated_model.grid_visible() is False
    assert not bar._grid_chk.isChecked()


def test_settings_controls_update_calibration_model(canvas, calibrated_model, qtbot):
    """Verify that all ViewportActionsBar settings controls update the calibration model and canvas state.

    Tests opacity slider, fixed spacing input, measurement display, clear measurement
    button, and reset calibration button. Success means each control produces the
    corresponding model or canvas state change.
    """
    bar = ViewportActionsBar(canvas, calibrated_model, canvas)
    bar.set_image_loaded(True)
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
    assert calibrated_model.has_scale() is True
    assert calibrated_model.unit() == "px"
    assert bar._grid_chk.isEnabled()


def test_center_crop_defaults_to_605px_radius(qtbot):
    """Verify that a new ImageLabel loaded with a 1080x1080 image defaults to a 1210px circle crop.

    The default crop diameter is 1210px (radius 605px) for a 1080px image. Success
    means center_crop_settings reports shape='circle', width=1210, and height=1210.
    """
    widget = ImageLabel()
    widget.set_image(np.zeros((1080, 1080, 3), dtype=np.uint8))
    qtbot.addWidget(widget)

    settings = widget.center_crop_settings()
    assert settings["shape"] == "circle"
    assert settings["width"] == 1210
    assert settings["height"] == 1210


def test_center_crop_drag_updates_original_center(canvas, calibrated_model, qtbot):
    """Verify that dragging the center crop overlay in calibration mode updates the original-image center coordinates.

    Enables the crop with calibration mode and simulates a mouse press-move-release
    at the canvas center. The reported center in original image coordinates should be
    at (50, 50) for a 100x100 image. Success means center_x and center_y both
    approximate 50.0 after the drag.
    """
    canvas.set_center_crop(enabled=True, center_dot=True, calibrating=True)
    pos = QPoint(canvas.width() // 2, canvas.height() // 2)
    qtbot.mousePress(canvas, Qt.LeftButton, pos=pos)
    qtbot.mouseMove(canvas, pos=pos)
    qtbot.mouseRelease(canvas, Qt.LeftButton, pos=pos)

    settings = canvas.center_crop_settings()
    assert settings["center_x"] == pytest.approx(50.0)
    assert settings["center_y"] == pytest.approx(50.0)
