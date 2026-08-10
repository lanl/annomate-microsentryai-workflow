"""Integration tests for AnomalyConstraintsSection.

Tests verify widget construction, public refresh methods, and that panel
interactions update the AnomalyConstraintModel.
"""

import pytest

from core.states.anomaly_constraint_state import AnomalyConstraintState
from models.anomaly_constraint_model import AnomalyConstraintModel
from views.annomate.sections.anomaly import AnomalyConstraintsSection


@pytest.fixture
def anomaly_model():
    return AnomalyConstraintModel(AnomalyConstraintState())


@pytest.fixture
def section(anomaly_model, qtbot):
    s = AnomalyConstraintsSection(anomaly_model)
    qtbot.addWidget(s)
    s.show()
    return s


class TestAnomalyPanelWidgets:
    def test_panel_widgets_built(self, section):
        """All expected anomaly panel widgets are built during construction."""
        assert hasattr(section, "_anomaly_enable_chk")
        assert hasattr(section, "_anomaly_area_chk")
        assert hasattr(section, "_anomaly_area_spin")
        assert hasattr(section, "_anomaly_area_unit_lbl")
        assert hasattr(section, "_anomaly_area_count_lbl")
        assert hasattr(section, "_anomaly_dist_chk")
        assert hasattr(section, "_anomaly_dist_spin")
        assert hasattr(section, "_anomaly_dist_unit_lbl")
        assert hasattr(section, "_anomaly_dist_count_lbl")
        assert hasattr(section, "_anomaly_centroid_radio")
        assert hasattr(section, "_anomaly_edge_radio")

    def test_default_units_are_pixels(self, section):
        """Without calibration, unit labels default to 'px' and 'px²'."""
        assert section._anomaly_area_unit_lbl.text() == "px²"
        assert section._anomaly_dist_unit_lbl.text() == "px"

    def test_color_swatch_buttons_exist(self, section):
        """Color picker swatch buttons are present for area and distance."""
        assert hasattr(section, "_anomaly_area_color_btn")
        assert hasattr(section, "_anomaly_dist_color_btn")

    def test_color_swatch_reflects_model_color(self, section, anomaly_model, qtbot):
        """Changing the model's area_color updates the swatch button stylesheet."""
        anomaly_model.set_area_color((10, 20, 30))
        style = section._anomaly_area_color_btn.styleSheet()
        assert "10" in style and "20" in style and "30" in style


class TestRefreshViolations:
    def test_no_violations_clears_labels(self, section, anomaly_model):
        """refresh_violations(0, 0) clears both inline count labels."""
        section.refresh_violations(0, 0)
        assert section._anomaly_area_count_lbl.text() == ""
        assert section._anomaly_dist_count_lbl.text() == ""

    def test_area_violations_shown_with_threshold(self, section, anomaly_model):
        """refresh_violations(2, 0) shows count and threshold in area label."""
        anomaly_model.set_area_threshold(100.0)
        section.refresh_violations(2, 0)
        text = section._anomaly_area_count_lbl.text()
        assert "2" in text
        assert "100" in text
        assert section._anomaly_dist_count_lbl.text() == ""

    def test_distance_violations_shown_with_threshold(self, section, anomaly_model):
        """refresh_violations(0, 3) shows count and threshold in distance label."""
        anomaly_model.set_distance_threshold(30.0)
        section.refresh_violations(0, 3)
        text = section._anomaly_dist_count_lbl.text()
        assert "3" in text
        assert "30" in text
        assert section._anomaly_area_count_lbl.text() == ""

    def test_singular_defect_label(self, section, anomaly_model):
        """A count of 1 uses 'defect' (singular) in the label."""
        anomaly_model.set_area_threshold(50.0)
        section.refresh_violations(1, 0)
        assert "defect" in section._anomaly_area_count_lbl.text()
        assert "defects" not in section._anomaly_area_count_lbl.text()


class TestUpdateUnits:
    def test_unit_labels_updated(self, section):
        """update_units('mm') sets area label to 'mm²' and dist label to 'mm'."""
        section.update_units("mm")
        assert section._anomaly_area_unit_lbl.text() == "mm²"
        assert section._anomaly_dist_unit_lbl.text() == "mm"

    def test_unit_labels_cm(self, section):
        """update_units('cm') reflects correctly."""
        section.update_units("cm")
        assert section._anomaly_area_unit_lbl.text() == "cm²"
        assert section._anomaly_dist_unit_lbl.text() == "cm"


class TestPanelModelSync:
    def test_enable_checkbox_updates_model(self, section, anomaly_model, qtbot):
        """Toggling the Enable checkbox updates the anomaly model.

        Programmatically check the Enable checkbox and verify the model's
        enabled() flag becomes True.
        """
        section._anomaly_enable_chk.setChecked(True)
        assert anomaly_model.enabled() is True

    def test_area_threshold_spin_updates_model(self, section, anomaly_model, qtbot):
        """Changing the area spinbox value updates the model's area_threshold."""
        section._anomaly_area_spin.setValue(123.0)
        assert anomaly_model.area_threshold() == pytest.approx(123.0)

    def test_distance_threshold_spin_updates_model(self, section, anomaly_model, qtbot):
        """Changing the distance spinbox value updates the model's distance_threshold."""
        section._anomaly_dist_spin.setValue(45.0)
        assert anomaly_model.distance_threshold() == pytest.approx(45.0)

    def test_edge_radio_updates_model(self, section, anomaly_model, qtbot):
        """Selecting the Edge radio button updates the model's distance_method to 'edge'."""
        section._anomaly_edge_radio.setChecked(True)
        assert anomaly_model.distance_method() == "edge"

    def test_centroid_radio_updates_model(self, section, anomaly_model, qtbot):
        """Selecting the Centroid radio button updates the model's distance_method to 'centroid'."""
        section._anomaly_edge_radio.setChecked(True)
        section._anomaly_centroid_radio.setChecked(True)
        assert anomaly_model.distance_method() == "centroid"
