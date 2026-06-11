from PySide6.QtCore import QObject, Signal

from core.logic.anomaly_constraints import check_area_violations, check_distance_violations
from models.anomaly_constraint_model import AnomalyConstraintModel


class AnomalyConstraintController(QObject):
    """Headless service that runs area and proximity checks against a set of annotations.

    Signals:
        violations_updated: Emitted after every ``run_checks`` call (including when
            the feature is disabled, in which case both sets are empty).
            Args: (area_violations: set[int], distance_pairs: set[frozenset])
    """

    violations_updated = Signal(set, set)

    def __init__(self, model: AnomalyConstraintModel, parent=None) -> None:
        super().__init__(parent)
        self._model = model
        self._distance_cache: dict[tuple[int, int], float] = {}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def run_checks(self, annotations: list[dict], scale: float | None) -> None:
        """Evaluate area and distance thresholds for *annotations*.

        Args:
            annotations: All annotation dicts for the currently displayed image.
            scale: World-units-per-pixel from CalibrationModel, or None if uncalibrated.
        """
        if not self._model.enabled():
            self.violations_updated.emit(set(), set())
            return

        area_violations: set[int] = set()
        distance_pairs: set[frozenset] = set()

        if self._model.area_check_enabled() and self._model.area_threshold() > 0.0:
            area_violations = check_area_violations(
                annotations,
                self._model.area_threshold(),
                scale,
            )

        if self._model.distance_check_enabled() and self._model.distance_threshold() > 0.0:
            distance_pairs = check_distance_violations(
                annotations,
                self._model.distance_threshold(),
                scale,
                self._model.distance_method(),
                self._distance_cache,
            )

        self.violations_updated.emit(area_violations, distance_pairs)

    def invalidate_cache(self) -> None:
        """Clear the pairwise distance cache.

        Must be called when the annotation list changes structurally (deletion,
        point edit, image navigation) or when distance_method changes.
        """
        self._distance_cache.clear()
