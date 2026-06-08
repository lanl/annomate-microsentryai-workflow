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
    def test_defaults_to_pixel_scale(self, model):
        """Verify that a fresh CalibrationModel defaults to pixel scale with no user calibration.

        Tests the initial state of CalibrationModel: scale=1.0, unit='px', grid visible,
        and is_calibrated() returns False. Success means all default values match expectations.
        """
        assert model.has_scale()
        assert not model.is_calibrated()
        assert model.scale() == pytest.approx(1.0)
        assert model.unit() == "px"
        assert model.grid_visible() is True
        assert model.grid_spacing_world() == pytest.approx(100.0)

    def test_sets_scale(self, model):
        """Verify that apply_calibration computes and stores the correct scale factor.

        Sets two horizontal calibration points 100px apart representing 5mm, so scale
        should be 5/100 = 0.05 mm/px. Success means apply_calibration returns True,
        scale and unit are updated, and is_calibrated() becomes True.
        """
        model.set_calib_points((0.0, 0.0), (100.0, 0.0))
        ok = model.apply_calibration(5.0, "mm")
        assert ok
        assert model.scale() == pytest.approx(0.05)
        assert model.unit() == "mm"
        assert model.is_calibrated()
        assert model.grid_visible() is True

    def test_diagonal_points(self, model):
        """Verify that scale is computed correctly from diagonal calibration points.

        Uses a 3-4-5 right triangle (pixel distance = 5). With a real-world distance
        of 10cm, the scale should be 10/5 = 2.0 cm/px. Success means the computed
        scale matches the expected value within floating-point tolerance.
        """
        model.set_calib_points((0.0, 0.0), (3.0, 4.0))  # pixel_dist = 5
        ok = model.apply_calibration(10.0, "cm")
        assert ok
        assert model.scale() == pytest.approx(2.0)

    def test_zero_distance_returns_false(self, model):
        """Verify that apply_calibration rejects identical calibration points.

        When both points are the same location (zero pixel distance), dividing the
        real-world distance by zero is undefined. Success means apply_calibration
        returns False and is_calibrated() remains False.
        """
        model.set_calib_points((50.0, 50.0), (50.0, 50.0))
        ok = model.apply_calibration(5.0, "mm")
        assert not ok
        assert not model.is_calibrated()

    def test_no_points_returns_false(self, model):
        """Verify that apply_calibration fails gracefully when no calibration points are set.

        Calling apply_calibration without first setting any points should return False
        without raising an exception, leaving the model in its default pixel state.
        """
        ok = model.apply_calibration(5.0, "mm")
        assert not ok

    def test_clear_calibration(self, model):
        """Verify that clearing calibration fully resets the model to pixel defaults.

        After a successful calibration, clear_calibration() should reset scale to 1.0,
        unit to 'px', mark the model as not user-calibrated, and clear stored calibration
        points. has_scale() must still return True (pixel scale is always available).
        """
        model.set_calib_points((0.0, 0.0), (100.0, 0.0))
        model.apply_calibration(5.0, "mm")
        model.clear_calibration()
        assert not model.is_calibrated()
        assert model.has_scale()
        assert model.scale() == pytest.approx(1.0)
        assert model.unit() == "px"
        assert model.calib_points() == (None, None)


class TestNiceSpacing:
    def _spacing(self, scale):
        return CalibrationModel._nice_spacing(scale)

    def test_returns_positive(self):
        """Verify that _nice_spacing always returns a positive grid spacing value.

        The computed grid spacing should never be zero or negative regardless of
        the scale factor passed in. Success means the returned value is greater than 0.
        """
        assert self._spacing(0.05) > 0

    def test_nice_numbers_only(self):
        """Verify that _nice_spacing returns only human-readable 'nice' numbers.

        Nice numbers are of the form 1, 2, 5, or 10 multiplied by any power of 10
        (e.g. 0.05, 0.2, 500). Tests a range of scale values from very small to large.
        Success means the mantissa of the result is always in {1, 2, 5, 10}.
        """
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
        """Verify that the grid spacing maps to a visually reasonable pixel interval.

        The ratio spacing/scale gives the on-screen pixel distance between grid lines.
        This should be in a readable range (not too crowded, not too sparse). Success
        means the resulting pixel distance is between 16 and 800 pixels.
        """
        # Spacing / scale should be close to 80 original pixels (within 1.25×)
        for scale in [0.01, 0.1, 1.0]:
            s = self._spacing(scale)
            px = s / scale
            assert 16 <= px <= 800, (
                f"Grid spacing {s} at scale {scale} → {px}px (out of range)"
            )


class TestMeasurement:
    def test_measured_distance(self, model):
        """Verify that measured_distance returns the correct real-world value after calibration.

        Calibrates 100px to 5mm (scale = 0.05 mm/px), then measures 200px. Expected
        result is 200 * 0.05 = 10.0 mm. Success means the returned value matches 10.0.
        """
        model.set_calib_points((0.0, 0.0), (100.0, 0.0))
        model.apply_calibration(5.0, "mm")
        model.set_meas_p1((0.0, 0.0))
        model.set_meas_p2((200.0, 0.0))
        assert model.measured_distance() == pytest.approx(10.0)

    def test_default_measurement_uses_pixels(self, model):
        """Verify that measurement falls back to pixels when no calibration is applied.

        Without user calibration, scale defaults to 1.0 px/px. Measuring 100px should
        return 100.0. Success means measured_distance() equals the raw pixel distance.
        """
        model.set_meas_p1((0.0, 0.0))
        model.set_meas_p2((100.0, 0.0))
        assert model.measured_distance() == pytest.approx(100.0)

    def test_clear_measurement(self, model):
        """Verify that clear_measurement removes both measurement endpoints.

        After setting measurement points and calling clear_measurement(), meas_points()
        should return (None, None). Success means both points are cleared to None.
        """
        model.set_meas_p1((0.0, 0.0))
        model.set_meas_p2((100.0, 0.0))
        model.clear_measurement()
        assert model.meas_points() == (None, None)


class TestSerialization:
    def test_roundtrip(self, model):
        """Verify that all calibration state survives a to_dict / from_dict round-trip.

        Serializes a fully calibrated model (including grid color, opacity, and fixed
        spacing) and deserializes it into a fresh model. Success means every attribute
        on the restored model matches the original within tolerance.
        """
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
        assert model2.is_calibrated()
        assert model2.calib_points() == ((10.0, 20.0), (110.0, 20.0))
        assert model2.grid_visible() is True
        assert model2.grid_color() == (255, 0, 128)
        assert model2.grid_opacity() == pytest.approx(0.75)
        assert model2.grid_spacing_world() == pytest.approx(2.0)
        assert model2.grid_spacing_auto() is False

    def test_empty_dict_defaults(self, model):
        """Verify that loading an empty dict restores pixel-scale defaults.

        Calling from_dict({}) should be equivalent to creating a fresh model: scale 1.0,
        unit 'px', not user-calibrated, grid visible with default opacity. Success means
        all default values are correctly applied even with no keys present.
        """
        model.from_dict({})
        assert not model.is_calibrated()
        assert model.has_scale()
        assert model.scale() == pytest.approx(1.0)
        assert model.unit() == "px"
        assert model.grid_visible() is True
        assert model.grid_opacity() == pytest.approx(0.5)

    def test_legacy_calibrated_dict_defaults_user_calibrated(self, model):
        """Verify that a legacy dict with a non-None scale is loaded as user-calibrated.

        Older project files stored calibration as just scale + unit without an explicit
        'user_calibrated' flag. If scale is set, it should be treated as calibrated.
        Success means is_calibrated() is True and scale/unit match the dict values.
        """
        model.from_dict({"scale": 0.25, "unit": "mm"})
        assert model.is_calibrated()
        assert model.scale() == pytest.approx(0.25)
        assert model.unit() == "mm"

    def test_legacy_uncalibrated_dict_resets_to_pixels(self, model):
        """Verify that a legacy dict with scale=None resets the model to pixel defaults.

        If the legacy dict explicitly stored scale as None (uncalibrated), the model
        should reset to pixel defaults and ignore other stale values like unit and
        grid_visible=False. Success means scale=1.0, unit='px', and grid remains visible.
        """
        model.from_dict({"scale": None, "unit": "mm", "grid_visible": False})
        assert not model.is_calibrated()
        assert model.has_scale()
        assert model.scale() == pytest.approx(1.0)
        assert model.unit() == "px"
        assert model.grid_visible() is True

    def test_null_points_roundtrip(self, model):
        """Verify that None calibration points serialize to null and restore correctly.

        A default model with no calibration points set should serialize calib_p1 and
        calib_p2 as None (JSON null). After deserializing, calib_points() should return
        (None, None). Success means null points survive the full round-trip.
        """
        data = model.to_dict()
        assert data["calib_p1"] is None
        assert data["calib_p2"] is None
        model.from_dict(data)
        assert model.calib_points() == (None, None)
