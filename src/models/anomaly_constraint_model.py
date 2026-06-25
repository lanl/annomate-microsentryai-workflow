from PySide6.QtCore import QObject, Signal

from core.states.anomaly_constraint_state import AnomalyConstraintState


class AnomalyConstraintModel(QObject):
    """Qt adapter wrapping AnomalyConstraintState.

    Signals:
        constraints_changed: Any setting changed; controller should re-run checks.
    """

    constraints_changed = Signal()

    def __init__(
        self, state: AnomalyConstraintState | None = None, parent=None
    ) -> None:
        super().__init__(parent)
        self._state = state if state is not None else AnomalyConstraintState()

    # ------------------------------------------------------------------ #
    # Query API
    # ------------------------------------------------------------------ #

    def enabled(self) -> bool:
        return self._state.enabled

    def area_check_enabled(self) -> bool:
        return self._state.area_check_enabled

    def area_threshold(self) -> float:
        return self._state.area_threshold

    def distance_check_enabled(self) -> bool:
        return self._state.distance_check_enabled

    def distance_threshold(self) -> float:
        return self._state.distance_threshold

    def distance_method(self) -> str:
        return self._state.distance_method

    def area_color(self) -> tuple:
        return self._state.area_color

    def distance_color(self) -> tuple:
        return self._state.distance_color

    # ------------------------------------------------------------------ #
    # Command API
    # ------------------------------------------------------------------ #

    def set_enabled(self, v: bool) -> None:
        if self._state.enabled == v:
            return
        self._state.enabled = v
        self.constraints_changed.emit()

    def set_area_check_enabled(self, v: bool) -> None:
        if self._state.area_check_enabled == v:
            return
        self._state.area_check_enabled = v
        self.constraints_changed.emit()

    def set_area_threshold(self, v: float) -> None:
        if self._state.area_threshold == v:
            return
        self._state.area_threshold = v
        self.constraints_changed.emit()

    def set_distance_check_enabled(self, v: bool) -> None:
        if self._state.distance_check_enabled == v:
            return
        self._state.distance_check_enabled = v
        self.constraints_changed.emit()

    def set_distance_threshold(self, v: float) -> None:
        if self._state.distance_threshold == v:
            return
        self._state.distance_threshold = v
        self.constraints_changed.emit()

    def set_distance_method(self, v: str) -> None:
        if self._state.distance_method == v:
            return
        self._state.distance_method = v
        self.constraints_changed.emit()

    def set_area_color(self, v: tuple) -> None:
        if self._state.area_color == v:
            return
        self._state.area_color = v
        self.constraints_changed.emit()

    def set_distance_color(self, v: tuple) -> None:
        if self._state.distance_color == v:
            return
        self._state.distance_color = v
        self.constraints_changed.emit()

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict:
        return self._state.to_dict()

    def from_dict(self, data: dict) -> None:
        self._state = AnomalyConstraintState.from_dict(data)
        self.constraints_changed.emit()
