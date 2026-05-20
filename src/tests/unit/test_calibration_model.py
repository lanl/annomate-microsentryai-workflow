"""Unit tests for CalibrationState and CalibrationModel.

These run without a QApplication (CalibrationModel is QObject-based, but
signal connections are not exercised here — only method logic is tested).
"""

import math
import pytest

from core.states.calibration_state import CalibrationState
from models.calibration_model import CalibrationModel


@pytest.fixture()
def model():
    state = CalibrationState()
    return CalibrationModel(state)


class TestApplyCalibration:
    def test_sets_scale(self, model):
        model.set_calib_points((0.0, 0.0), (100.0, 0.0))
        ok = model.apply_calibration(5.0, "mm")
        assert ok
        assert model.scale() == pytest.approx(0.05)
        assert model.unit() == "mm"
        assert model.is_calibrated()
        assert model.grid_visible() is True

    def test_diagonal_points(self, model):
        model.set_calib_points((0.0, 0.0), (3.0, 4.0))  # pixel_dist = 5
        ok = model.apply_calibration(10.0, "cm")
        assert ok
        assert model.scale() == pytest.approx(2.0)

    def test_zero_distance_returns_false(self, model):
        model.set_calib_points((50.0, 50.0), (50.0, 50.0))
        ok = model.apply_calibration(5.0, "mm")
        assert not ok
        assert not model.is_calibrated()

    def test_no_points_returns_false(self, model):
        ok = model.apply_calibration(5.0, "mm")
        assert not ok

    def test_clear_calibration(self, model):
        model.set_calib_points((0.0, 0.0), (100.0, 0.0))
        model.apply_calibration(5.0, "mm")
        model.clear_calibration()
        assert not model.is_calibrated()
        assert model.scale() is None
        assert model.calib_points() == (None, None)


class TestNiceSpacing:
    def _spacing(self, scale):
        return CalibrationModel._nice_spacing(scale)

    def test_returns_positive(self):
        assert self._spacing(0.05) > 0

    def test_nice_numbers_only(self):
        # Nice spacing must be 1, 2, 5, or 10 × 10^n
        for scale in [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]:
            s = self._spacing(scale)
            # Normalize: divide by the correct power of 10
            exp = math.floor(math.log10(s))
            mantissa = round(s / (10**exp), 6)
            assert mantissa in (1.0, 2.0, 5.0, 10.0), (
                f"Bad spacing {s} for scale {scale}"
            )

    def test_targets_roughly_80px(self):
        # Spacing / scale should be close to 80 original pixels (within 1.25×)
        for scale in [0.01, 0.1, 1.0]:
            s = self._spacing(scale)
            px = s / scale
            assert 16 <= px <= 800, (
                f"Grid spacing {s} at scale {scale} → {px}px (out of range)"
            )


class TestMeasurement:
    def test_measured_distance(self, model):
        model.set_calib_points((0.0, 0.0), (100.0, 0.0))
        model.apply_calibration(5.0, "mm")
        model.set_meas_p1((0.0, 0.0))
        model.set_meas_p2((200.0, 0.0))
        assert model.measured_distance() == pytest.approx(10.0)

    def test_no_measurement_when_uncalibrated(self, model):
        model.set_meas_p1((0.0, 0.0))
        model.set_meas_p2((100.0, 0.0))
        assert model.measured_distance() is None

    def test_clear_measurement(self, model):
        model.set_meas_p1((0.0, 0.0))
        model.set_meas_p2((100.0, 0.0))
        model.clear_measurement()
        assert model.meas_points() == (None, None)


class TestSerialization:
    def test_roundtrip(self, model):
        model.set_calib_points((10.0, 20.0), (110.0, 20.0))
        model.apply_calibration(5.0, "cm")
        model.set_grid_visible(True)
        model.set_grid_color((255, 0, 128))
        model.set_grid_opacity(0.75)
        model.set_grid_spacing(2.0)

        data = model.to_dict()

        state2 = CalibrationState()
        model2 = CalibrationModel(state2)
        model2.from_dict(data)

        assert model2.scale() == pytest.approx(model.scale())
        assert model2.unit() == "cm"
        assert model2.calib_points() == ((10.0, 20.0), (110.0, 20.0))
        assert model2.grid_visible() is True
        assert model2.grid_color() == (255, 0, 128)
        assert model2.grid_opacity() == pytest.approx(0.75)
        assert model2.grid_spacing_world() == pytest.approx(2.0)
        assert model2.grid_spacing_auto() is False

    def test_empty_dict_defaults(self, model):
        model.from_dict({})
        assert not model.is_calibrated()
        assert model.unit() == "mm"
        assert model.grid_visible() is False
        assert model.grid_opacity() == pytest.approx(0.5)

    def test_null_points_roundtrip(self, model):
        data = model.to_dict()
        assert data["calib_p1"] is None
        assert data["calib_p2"] is None
        model.from_dict(data)
        assert model.calib_points() == (None, None)
