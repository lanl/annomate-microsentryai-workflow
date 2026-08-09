import numpy as np
import pytest
from PySide6.QtCore import Qt

from core.states.center_template_state import CenterTemplateState
from models.center_template_model import CenterTemplateModel
from views.annomate.image_label import ImageLabel
from views.annomate.sections.center_crop import CenterCropSection


@pytest.fixture
def canvas(qtbot):
    widget = ImageLabel()
    widget.resize(320, 240)
    widget.set_image(np.zeros((100, 100, 3), dtype=np.uint8))
    qtbot.addWidget(widget)
    widget.show()
    return widget


@pytest.fixture
def template_model():
    return CenterTemplateModel(CenterTemplateState())


def test_crop_controls_update_canvas(canvas, qtbot):
    """Verify that the crop controls in the section drive canvas crop settings.

    Enables the crop overlay, changes width, height, shape to circle, and opacity.
    For circular crops, width and height are kept equal (the smaller dimension). Success
    means all canvas crop settings reflect the control values, with width and height
    constrained to the circle constraint.
    """
    section = CenterCropSection(canvas)
    section.set_image_dimensions(100, 100)
    qtbot.addWidget(section)

    assert not canvas.center_crop_settings()["enabled"]

    section._crop_chk.click()
    section._crop_width_spin.setValue(40)
    section._crop_height_spin.setValue(30)
    section._crop_shape_combo.setCurrentText("Circle")
    section._crop_height_spin.setValue(15)
    section._crop_opacity_slider.setValue(80)
    section._crop_center_dot_chk.setChecked(True)

    settings = canvas.center_crop_settings()
    assert settings["enabled"] is True
    assert settings["width"] == 30
    assert settings["height"] == 30
    assert settings["shape"] == "circle"
    assert settings["opacity"] == 0.8
    assert settings["center_dot"] is True
    assert section._crop_width_spin.value() == 30
    assert section._crop_height_spin.value() == 15


def test_crop_reset_restores_defaults(canvas, qtbot):
    """Verify that clicking the crop reset button restores all crop settings to factory defaults.

    Changes shape, dimensions, and opacity away from defaults, then clicks reset.
    Success means all settings return to their defaults: circle shape, 1210x1210 size,
    opacity 0.37, center_dot False, and the height spin shows 605 (the radius input).
    """
    section = CenterCropSection(canvas)
    section.set_image_dimensions(100, 100)
    qtbot.addWidget(section)

    section._crop_shape_combo.setCurrentText("Rectangle")
    section._crop_width_spin.setValue(40)
    section._crop_height_spin.setValue(30)
    section._crop_opacity_slider.setValue(80)

    qtbot.mouseClick(section._btn_reset_crop, Qt.LeftButton)

    settings = canvas.center_crop_settings()
    assert settings["shape"] == "circle"
    assert settings["width"] == 1210
    assert settings["height"] == 1210
    assert settings["opacity"] == 0.37
    assert settings["center_dot"] is False
    assert section._crop_height_spin.value() == 605


def test_calibrate_accept_clear_emit_signals(canvas, template_model, qtbot):
    """Verify the Calibrate/Accept/Clear buttons emit their respective signals.

    Success means each button's click produces exactly one signal, in the
    order a real calibration workflow would trigger them.
    """
    section = CenterCropSection(canvas, template_model)
    qtbot.addWidget(section)
    section.set_has_image(True)

    started = []
    accepted = []
    cleared = []
    section.center_calibration_started.connect(lambda: started.append(True))
    section.center_calibration_accepted.connect(lambda: accepted.append(True))
    section.center_template_cleared.connect(lambda: cleared.append(True))

    qtbot.mouseClick(section._btn_calibrate, Qt.LeftButton)
    section.set_calibrating(True)
    qtbot.mouseClick(section._btn_accept, Qt.LeftButton)
    qtbot.mouseClick(section._btn_clear, Qt.LeftButton)

    assert started == [True]
    assert accepted == [True]
    assert cleared == [True]


def test_import_button_emits_path_from_file_dialog(canvas, template_model, qtbot, monkeypatch):
    """Verify the Import button emits the path chosen in the file dialog.

    The dialog itself isn't testable headlessly, so getOpenFileName is
    monkeypatched to return a fixed path -- success means that exact path
    is forwarded on the import-requested signal.
    """
    section = CenterCropSection(canvas, template_model)
    qtbot.addWidget(section)
    section.set_has_image(True)

    monkeypatch.setattr(
        "views.annomate.sections.center_crop.QFileDialog.getOpenFileName",
        lambda *a, **k: ("/tmp/template.png", ""),
    )
    requested = []
    section.center_template_import_requested.connect(requested.append)

    qtbot.mouseClick(section._btn_import, Qt.LeftButton)

    assert requested == ["/tmp/template.png"]


def test_template_status_reflects_calibrating_then_saved_template(canvas, template_model, qtbot):
    """Verify the status label tracks calibrating state, then template save state.

    Starts at "Template: none", switches to the mid-calibration hint once
    set_calibrating(True) is called, then shows the saved/match text once
    calibration ends and a template has been saved to the model.
    """
    section = CenterCropSection(canvas, template_model)
    qtbot.addWidget(section)

    assert section._template_status_lbl.text() == "Template: none"

    section.set_calibrating(True)
    assert "Accept" in section._template_status_lbl.text()

    section.set_calibrating(False)
    template_model.set_template("t.png", "/tmp/t.png", 0, 0, "circle", 100, 100, 50, 50)
    assert section._template_status_lbl.text() == "Template: saved"

    template_model.set_match(51, 51, 0.874)
    assert section._template_status_lbl.text() == "Template match: 0.874"


def test_template_buttons_gated_by_has_image_and_calibrating(canvas, template_model, qtbot):
    """Verify Calibrate/Import/Accept/Clear enablement follows has_image/calibrating/has_template.

    Calibrate and Import require an image; Accept additionally requires an
    in-progress calibration; Clear is enabled either mid-calibration or once
    a template has actually been saved.
    """
    section = CenterCropSection(canvas, template_model)
    qtbot.addWidget(section)
    section.set_has_image(False)

    assert section._btn_calibrate.isEnabled() is False
    assert section._btn_import.isEnabled() is False
    assert section._btn_accept.isEnabled() is False
    assert section._btn_clear.isEnabled() is False

    section.set_has_image(True)
    assert section._btn_calibrate.isEnabled() is True
    assert section._btn_import.isEnabled() is True
    assert section._btn_accept.isEnabled() is False  # not calibrating yet

    section.set_calibrating(True)
    assert section._btn_accept.isEnabled() is True
    assert section._btn_clear.isEnabled() is True  # cancel-mid-calibration path

    section.set_calibrating(False)
    assert section._btn_clear.isEnabled() is False

    template_model.set_template("t.png", "/tmp/t.png", 0, 0, "circle", 100, 100, 50, 50)
    assert section._btn_clear.isEnabled() is True


def test_disabling_crop_cancels_in_progress_calibration(canvas, template_model, qtbot):
    """Verify unchecking Enable exits an in-progress calibration.

    Matches the original popup-menu behavior: disabling the crop overlay
    mid-calibration cancels the calibration rather than leaving it dangling.
    """
    section = CenterCropSection(canvas, template_model)
    qtbot.addWidget(section)
    section.set_has_image(True)
    section._crop_chk.setChecked(True)
    section.set_calibrating(True)

    section._crop_chk.setChecked(False)

    assert section._calibrating is False
    assert canvas.center_crop_settings()["calibrating"] is False
