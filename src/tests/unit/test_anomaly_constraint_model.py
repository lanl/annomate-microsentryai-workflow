"""Unit tests for AnomalyConstraintModel.

Requires a QApplication (pytest-qt qtbot fixture).
"""

import pytest

from models.anomaly_constraint_model import AnomalyConstraintModel


@pytest.fixture
def model():
    return AnomalyConstraintModel()


class TestAnomalyConstraintModelSetters:
    def test_set_enabled_emits_signal(self, model, qtbot):
        """set_enabled(True) changes state and emits constraints_changed."""
        with qtbot.waitSignal(model.constraints_changed, timeout=500):
            model.set_enabled(True)
        assert model.enabled() is True

    def test_set_enabled_no_signal_when_unchanged(self, model, qtbot):
        """set_enabled with same value does not emit constraints_changed."""
        model.set_enabled(False)
        with qtbot.assertNotEmitted(model.constraints_changed):
            model.set_enabled(False)

    def test_set_area_check_enabled(self, model, qtbot):
        """set_area_check_enabled flips the flag and emits."""
        with qtbot.waitSignal(model.constraints_changed, timeout=500):
            model.set_area_check_enabled(False)
        assert model.area_check_enabled() is False

    def test_set_area_threshold(self, model, qtbot):
        """set_area_threshold updates the value and emits."""
        with qtbot.waitSignal(model.constraints_changed, timeout=500):
            model.set_area_threshold(150.0)
        assert model.area_threshold() == pytest.approx(150.0)

    def test_set_distance_check_enabled(self, model, qtbot):
        """set_distance_check_enabled flips the flag and emits."""
        with qtbot.waitSignal(model.constraints_changed, timeout=500):
            model.set_distance_check_enabled(False)
        assert model.distance_check_enabled() is False

    def test_set_distance_threshold(self, model, qtbot):
        """set_distance_threshold updates the value and emits."""
        with qtbot.waitSignal(model.constraints_changed, timeout=500):
            model.set_distance_threshold(25.0)
        assert model.distance_threshold() == pytest.approx(25.0)

    def test_set_distance_method(self, model, qtbot):
        """set_distance_method changes the method and emits."""
        with qtbot.waitSignal(model.constraints_changed, timeout=500):
            model.set_distance_method("edge")
        assert model.distance_method() == "edge"

    def test_set_distance_method_no_signal_when_unchanged(self, model, qtbot):
        """set_distance_method with same value does not emit constraints_changed."""
        model.set_distance_method("centroid")
        with qtbot.assertNotEmitted(model.constraints_changed):
            model.set_distance_method("centroid")

    def test_set_area_color(self, model, qtbot):
        """set_area_color changes the color tuple and emits constraints_changed."""
        with qtbot.waitSignal(model.constraints_changed, timeout=500):
            model.set_area_color((100, 200, 50))
        assert model.area_color() == (100, 200, 50)

    def test_set_area_color_no_signal_when_unchanged(self, model, qtbot):
        """set_area_color with the same tuple does not emit constraints_changed."""
        model.set_area_color((10, 20, 30))
        with qtbot.assertNotEmitted(model.constraints_changed):
            model.set_area_color((10, 20, 30))

    def test_set_distance_color(self, model, qtbot):
        """set_distance_color changes the color tuple and emits constraints_changed."""
        with qtbot.waitSignal(model.constraints_changed, timeout=500):
            model.set_distance_color((0, 128, 255))
        assert model.distance_color() == (0, 128, 255)


class TestAnomalyConstraintModelPersistence:
    def test_to_dict_round_trip(self, model, qtbot):
        """to_dict / from_dict restores all values and emits constraints_changed."""
        model.set_enabled(True)
        model.set_area_threshold(500.0)
        model.set_area_color((10, 20, 30))
        model.set_distance_threshold(15.0)
        model.set_distance_method("edge")
        model.set_distance_color((40, 50, 60))

        d = model.to_dict()
        fresh = AnomalyConstraintModel()
        with qtbot.waitSignal(fresh.constraints_changed, timeout=500):
            fresh.from_dict(d)

        assert fresh.enabled() is True
        assert fresh.area_threshold() == pytest.approx(500.0)
        assert fresh.area_color() == (10, 20, 30)
        assert fresh.distance_threshold() == pytest.approx(15.0)
        assert fresh.distance_method() == "edge"
        assert fresh.distance_color() == (40, 50, 60)

    def test_from_dict_with_defaults(self, model, qtbot):
        """from_dict with an empty dict resets to defaults."""
        model.set_enabled(True)
        model.from_dict({})
        assert model.enabled() is False
        assert model.area_threshold() == 0.0
        assert model.distance_method() == "centroid"
