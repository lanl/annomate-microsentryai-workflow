"""Integration tests for the Anomaly Constraints panel in ViewportActionsBar.

Tests verify widget construction, public refresh methods, and that panel
interactions update the AnomalyConstraintModel.
"""

import numpy as np
import pytest

from core.states.anomaly_constraint_state import AnomalyConstraintState
from models.anomaly_constraint_model import AnomalyConstraintModel
from views.annomate.image_label import ImageLabel
from views.annomate.viewport_actions import ViewportActionsBar


@pytest.fixture
def anomaly_model():
    return AnomalyConstraintModel(AnomalyConstraintState())


@pytest.fixture
def canvas(qtbot):
    widget = ImageLabel()
    widget.resize(320, 240)
    widget.set_image(np.zeros((100, 100, 3), dtype=np.uint8))
    qtbot.addWidget(widget)
    widget.show()
    return widget


@pytest.fixture
def bar(canvas, anomaly_model, qtbot):
    b = ViewportActionsBar(canvas, None, canvas, anomaly_constraint_model=anomaly_model)
    qtbot.addWidget(b)
    b.show()
    return b


class TestAnomalyPanelWidgets:
    def test_anomaly_button_exists(self, bar):
        """The anomaly constraints button is present on the bar."""
        assert hasattr(bar, "_btn_anomaly")
        assert bar._btn_anomaly is not None

    def test_panel_widgets_built(self, bar):
        """All expected anomaly panel widgets are built during construction."""
        assert hasattr(bar, "_anomaly_enable_chk")
        assert hasattr(bar, "_anomaly_area_chk")
        assert hasattr(bar, "_anomaly_area_spin")
        assert hasattr(bar, "_anomaly_area_unit_lbl")
        assert hasattr(bar, "_anomaly_area_count_lbl")
        assert hasattr(bar, "_anomaly_dist_chk")
        assert hasattr(bar, "_anomaly_dist_spin")
        assert hasattr(bar, "_anomaly_dist_unit_lbl")
        assert hasattr(bar, "_anomaly_dist_count_lbl")
        assert hasattr(bar, "_anomaly_centroid_radio")
        assert hasattr(bar, "_anomaly_edge_radio")

    def test_default_units_are_pixels(self, bar):
        """Without calibration, unit labels default to 'px' and 'px²'."""
        assert bar._anomaly_area_unit_lbl.text() == "px²"
        assert bar._anomaly_dist_unit_lbl.text() == "px"

    def test_color_swatch_buttons_exist(self, bar):
        """Color picker swatch buttons are present for area and distance."""
        assert hasattr(bar, "_anomaly_area_color_btn")
        assert hasattr(bar, "_anomaly_dist_color_btn")

    def test_color_swatch_reflects_model_color(self, bar, anomaly_model, qtbot):
        """Changing the model's area_color updates the swatch button stylesheet."""
        anomaly_model.set_area_color((10, 20, 30))
        style = bar._anomaly_area_color_btn.styleSheet()
        assert "10" in style and "20" in style and "30" in style


class TestRefreshAnomalyViolations:
    def test_no_violations_clears_labels(self, bar, anomaly_model):
        """refresh_anomaly_violations(0, 0) clears both inline count labels."""
        bar.refresh_anomaly_violations(0, 0)
        assert bar._anomaly_area_count_lbl.text() == ""
        assert bar._anomaly_dist_count_lbl.text() == ""

    def test_area_violations_shown_with_threshold(self, bar, anomaly_model):
        """refresh_anomaly_violations(2, 0) shows count and threshold in area label."""
        anomaly_model.set_area_threshold(100.0)
        bar.refresh_anomaly_violations(2, 0)
        text = bar._anomaly_area_count_lbl.text()
        assert "2" in text
        assert "100" in text
        assert bar._anomaly_dist_count_lbl.text() == ""

    def test_distance_violations_shown_with_threshold(self, bar, anomaly_model):
        """refresh_anomaly_violations(0, 3) shows count and threshold in distance label."""
        anomaly_model.set_distance_threshold(30.0)
        bar.refresh_anomaly_violations(0, 3)
        text = bar._anomaly_dist_count_lbl.text()
        assert "3" in text
        assert "30" in text
        assert bar._anomaly_area_count_lbl.text() == ""

    def test_singular_defect_label(self, bar, anomaly_model):
        """A count of 1 uses 'defect' (singular) in the label."""
        anomaly_model.set_area_threshold(50.0)
        bar.refresh_anomaly_violations(1, 0)
        assert "defect" in bar._anomaly_area_count_lbl.text()
        assert "defects" not in bar._anomaly_area_count_lbl.text()


class TestUpdateAnomalyUnits:
    def test_unit_labels_updated(self, bar):
        """update_anomaly_units('mm') sets area label to 'mm²' and dist label to 'mm'."""
        bar.update_anomaly_units("mm")
        assert bar._anomaly_area_unit_lbl.text() == "mm²"
        assert bar._anomaly_dist_unit_lbl.text() == "mm"

    def test_unit_labels_cm(self, bar):
        """update_anomaly_units('cm') reflects correctly."""
        bar.update_anomaly_units("cm")
        assert bar._anomaly_area_unit_lbl.text() == "cm²"
        assert bar._anomaly_dist_unit_lbl.text() == "cm"


class TestPanelModelSync:
    def test_enable_checkbox_updates_model(self, bar, anomaly_model, qtbot):
        """Toggling the Enable checkbox updates the anomaly model.

        Programmatically check the Enable checkbox and verify the model's
        enabled() flag becomes True.
        """
        bar._anomaly_enable_chk.setChecked(True)
        assert anomaly_model.enabled() is True

    def test_area_threshold_spin_updates_model(self, bar, anomaly_model, qtbot):
        """Changing the area spinbox value updates the model's area_threshold."""
        bar._anomaly_area_spin.setValue(123.0)
        assert anomaly_model.area_threshold() == pytest.approx(123.0)

    def test_distance_threshold_spin_updates_model(self, bar, anomaly_model, qtbot):
        """Changing the distance spinbox value updates the model's distance_threshold."""
        bar._anomaly_dist_spin.setValue(45.0)
        assert anomaly_model.distance_threshold() == pytest.approx(45.0)

    def test_edge_radio_updates_model(self, bar, anomaly_model, qtbot):
        """Selecting the Edge radio button updates the model's distance_method to 'edge'."""
        bar._anomaly_edge_radio.setChecked(True)
        assert anomaly_model.distance_method() == "edge"

    def test_centroid_radio_updates_model(self, bar, anomaly_model, qtbot):
        """Selecting the Centroid radio button updates the model's distance_method to 'centroid'."""
        bar._anomaly_edge_radio.setChecked(True)
        bar._anomaly_centroid_radio.setChecked(True)
        assert anomaly_model.distance_method() == "centroid"
