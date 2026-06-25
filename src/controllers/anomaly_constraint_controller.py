from PySide6.QtCore import QObject, Signal

from core.logic.anomaly_constraints import (
    check_area_violations,
    check_distance_violations,
)
from models.anomaly_constraint_model import AnomalyConstraintModel


class AnomalyConstraintController(QObject):
    """Headless service that runs area and proximity checks against a set of annotations.

    Signals:
        violations_updated: Emitted after every ``run_checks`` call (including when
            the feature is disabled, in which case both sets are empty).
            Args: (area_violations: set[int], distance_pairs: set[frozenset])
    """

    violations_updated = Signal(set, set, object)

    def __init__(self, model: AnomalyConstraintModel, parent=None) -> None:
        super().__init__(parent)
        self._model = model
        self._distance_cache: dict[tuple[int, int], float] = {}
        self._last_scale: float | None = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def run_checks(self, annotations: list[dict], scale: float | None) -> None:
        """Evaluate area and distance thresholds for *annotations*.

        Args:
            annotations: All annotation dicts for the currently displayed image.
            scale: World-units-per-pixel from CalibrationModel, or None if uncalibrated.
        """
        self._last_scale = scale

        if not self._model.enabled():
            self.violations_updated.emit(set(), set(), {})
            return

        area_violations: set[int] = set()
        distance_pairs: set[frozenset] = set()

        if self._model.area_check_enabled() and self._model.area_threshold() > 0.0:
            area_violations = check_area_violations(
                annotations,
                self._model.area_threshold(),
                scale,
            )

        if (
            self._model.distance_check_enabled()
            and self._model.distance_threshold() > 0.0
        ):
            distance_pairs = check_distance_violations(
                annotations,
                self._model.distance_threshold(),
                scale,
                self._model.distance_method(),
                self._distance_cache,
            )

        dist_values: dict[frozenset, float] = {}
        for pair in distance_pairs:
            i, j = sorted(pair)
            dist_px = self._distance_cache.get((i, j), 0.0)
            dist_values[pair] = dist_px * scale if scale is not None else dist_px

        self.violations_updated.emit(area_violations, distance_pairs, dist_values)

    def invalidate_cache(self) -> None:
        """Clear the pairwise distance cache.

        Must be called when the annotation list changes structurally (deletion,
        point edit, image navigation) or when distance_method changes.
        """
        self._distance_cache.clear()
