import math

from PySide6.QtCore import QObject, Signal

from core.states.calibration_state import CalibrationState


class CalibrationModel(QObject):
    """Qt adapter wrapping CalibrationState.

    Signals:
        calibration_changed: Scale or unit changed; canvas must repaint.
        grid_changed: Grid color, opacity, spacing, or visibility changed.
        measurement_updated: meas_p1 or meas_p2 changed.
    """

    calibration_changed = Signal()
    grid_changed = Signal()
    measurement_updated = Signal()

    def __init__(self, state: CalibrationState, parent=None) -> None:
        super().__init__(parent)
        self._state = state

    # ------------------------------------------------------------------ #
    # Calibration commands
    # ------------------------------------------------------------------ #

    def set_calib_points(self, p1: tuple, p2: tuple) -> None:
        self._state.calib_p1 = p1
        self._state.calib_p2 = p2

    def apply_calibration(self, real_distance: float, unit: str) -> bool:
        """Compute scale from stored calib points and real_distance.

        Returns False if points are too close (pixel_dist < 1e-6).
        Auto-computes a nice grid spacing when grid_spacing_auto is True.
        """
        p1, p2 = self._state.calib_p1, self._state.calib_p2
        if p1 is None or p2 is None:
            return False
        pixel_dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        if pixel_dist < 1e-6:
            return False

        self._state.scale = real_distance / pixel_dist
        self._state.unit = unit
        self._state.real_distance = real_distance
        self._state.grid_visible = True

        if self._state.grid_spacing_auto:
            self._state.grid_spacing_world = self._nice_spacing(self._state.scale)

        self.calibration_changed.emit()
        self.grid_changed.emit()
        return True

    def clear_calibration(self) -> None:
        self._state.clear_calibration()
        self.calibration_changed.emit()

    # ------------------------------------------------------------------ #
    # Grid commands
    # ------------------------------------------------------------------ #

    def set_grid_visible(self, visible: bool) -> None:
        self._state.grid_visible = visible
        self.grid_changed.emit()

    def set_grid_color(self, rgb: tuple) -> None:
        self._state.grid_color = rgb
        self.grid_changed.emit()

    def set_grid_opacity(self, alpha: float) -> None:
        self._state.grid_opacity = max(0.0, min(1.0, alpha))
        self.grid_changed.emit()

    def set_grid_spacing(self, world: float) -> None:
        if world > 0:
            self._state.grid_spacing_world = world
            self._state.grid_spacing_auto = False
            self.grid_changed.emit()

    def set_grid_spacing_auto(self) -> None:
        self._state.grid_spacing_auto = True
        if self._state.scale is not None:
            self._state.grid_spacing_world = self._nice_spacing(self._state.scale)
        self.grid_changed.emit()

    # ------------------------------------------------------------------ #
    # Measure commands
    # ------------------------------------------------------------------ #

    def set_meas_p1(self, p: tuple) -> None:
        self._state.meas_p1 = p
        self._state.meas_p2 = None
        self.measurement_updated.emit()

    def set_meas_p2(self, p: tuple) -> None:
        self._state.meas_p2 = p
        self.measurement_updated.emit()

    def clear_measurement(self) -> None:
        self._state.clear_measurement()
        self.measurement_updated.emit()

    # ------------------------------------------------------------------ #
    # Query API
    # ------------------------------------------------------------------ #

    def is_calibrated(self) -> bool:
        return self._state.is_calibrated()

    def scale(self) -> float | None:
        return self._state.scale

    def unit(self) -> str:
        return self._state.unit

    def grid_visible(self) -> bool:
        return self._state.grid_visible

    def grid_color(self) -> tuple:
        return self._state.grid_color

    def grid_opacity(self) -> float:
        return self._state.grid_opacity

    def grid_spacing_world(self) -> float:
        return self._state.grid_spacing_world

    def grid_spacing_auto(self) -> bool:
        return self._state.grid_spacing_auto

    def calib_points(self) -> tuple:
        return (self._state.calib_p1, self._state.calib_p2)

    def meas_points(self) -> tuple:
        return (self._state.meas_p1, self._state.meas_p2)

    def measured_distance(self) -> float | None:
        p1, p2 = self._state.meas_p1, self._state.meas_p2
        if p1 is None or p2 is None or self._state.scale is None:
            return None
        pixel_dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        return pixel_dist * self._state.scale

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict:
        s = self._state
        return {
            "scale": s.scale,
            "unit": s.unit,
            "calib_p1": list(s.calib_p1) if s.calib_p1 else None,
            "calib_p2": list(s.calib_p2) if s.calib_p2 else None,
            "real_distance": s.real_distance,
            "grid_visible": s.grid_visible,
            "grid_color": list(s.grid_color),
            "grid_opacity": s.grid_opacity,
            "grid_spacing_world": s.grid_spacing_world,
            "grid_spacing_auto": s.grid_spacing_auto,
        }

    def from_dict(self, data: dict) -> None:
        s = self._state
        s.scale = data.get("scale", None)
        s.unit = data.get("unit", "mm")
        p1 = data.get("calib_p1")
        s.calib_p1 = tuple(p1) if p1 else None
        p2 = data.get("calib_p2")
        s.calib_p2 = tuple(p2) if p2 else None
        s.real_distance = data.get("real_distance", 1.0)
        s.grid_visible = data.get("grid_visible", False)
        color = data.get("grid_color", [58, 90, 122])
        s.grid_color = tuple(color)
        s.grid_opacity = data.get("grid_opacity", 0.5)
        s.grid_spacing_world = data.get("grid_spacing_world", 1.0)
        s.grid_spacing_auto = data.get("grid_spacing_auto", True)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _nice_spacing(scale: float) -> float:
        """Return a human-friendly grid spacing in world units.

        Targets ~80 original-image pixels between lines. Same algorithm as
        size_demo.py lines 520-533.
        """
        target_px = 80
        ideal_world = target_px * scale
        exponent = math.floor(math.log10(ideal_world))
        base = ideal_world / (10**exponent)
        if base < 1.5:
            nice = 1
        elif base < 3.5:
            nice = 2
        elif base < 7.5:
            nice = 5
        else:
            nice = 10
        return nice * (10**exponent)
