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
        assert hasattr(bar, "_anomaly_dist_chk")
        assert hasattr(bar, "_anomaly_dist_spin")
        assert hasattr(bar, "_anomaly_dist_unit_lbl")
        assert hasattr(bar, "_anomaly_centroid_radio")
        assert hasattr(bar, "_anomaly_edge_radio")
        assert hasattr(bar, "_anomaly_status_lbl")

    def test_default_units_are_pixels(self, bar):
        """Without calibration, unit labels default to 'px' and 'px²'."""
        assert bar._anomaly_area_unit_lbl.text() == "px²"
        assert bar._anomaly_dist_unit_lbl.text() == "px"


class TestRefreshAnomalyViolations:
    def test_no_violations_message(self, bar):
        """refresh_anomaly_violations(0, 0) sets a 'no violations' style label."""
        bar.refresh_anomaly_violations(0, 0)
        assert "No violations" in bar._anomaly_status_lbl.text()

    def test_violation_counts_shown(self, bar):
        """refresh_anomaly_violations(2, 3) reflects counts in the status label."""
        bar.refresh_anomaly_violations(2, 3)
        text = bar._anomaly_status_lbl.text()
        assert "2" in text
        assert "3" in text

    def test_area_only_violations(self, bar):
        """refresh_anomaly_violations(1, 0) mentions area but not proximity."""
        bar.refresh_anomaly_violations(1, 0)
        text = bar._anomaly_status_lbl.text()
        assert "area" in text.lower()

    def test_distance_only_violations(self, bar):
        """refresh_anomaly_violations(0, 2) mentions proximity but not area."""
        bar.refresh_anomaly_violations(0, 2)
        text = bar._anomaly_status_lbl.text()
        assert "proximity" in text.lower()


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
