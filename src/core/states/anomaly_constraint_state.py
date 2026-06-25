from __future__ import annotations

from dataclasses import dataclass


_DEFAULT_AREA_COLOR = (255, 165, 0)
_DEFAULT_DIST_COLOR = (220, 50, 50)


@dataclass
class AnomalyConstraintState:
    """Persistent settings for the anomaly area and proximity constraint checks."""

    enabled: bool = False

    area_check_enabled: bool = True
    # Threshold in world-unit² (px² when uncalibrated). 0.0 means check is unconfigured.
    area_threshold: float = 0.0
    # RGB tuple for the area-violation outline overlay.
    area_color: tuple = _DEFAULT_AREA_COLOR

    distance_check_enabled: bool = True
    # Threshold in world-units (px when uncalibrated). 0.0 means check is unconfigured.
    distance_threshold: float = 0.0
    # "centroid" = arithmetic-mean centroid distance; "edge" = minimum vertex-to-vertex distance
    distance_method: str = "centroid"
    # RGB tuple for the proximity-violation connecting line.
    distance_color: tuple = _DEFAULT_DIST_COLOR

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "area_check_enabled": self.area_check_enabled,
            "area_threshold": self.area_threshold,
            "area_color": list(self.area_color),
            "distance_check_enabled": self.distance_check_enabled,
            "distance_threshold": self.distance_threshold,
            "distance_method": self.distance_method,
            "distance_color": list(self.distance_color),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AnomalyConstraintState":
        return cls(
            enabled=bool(data.get("enabled", False)),
            area_check_enabled=bool(data.get("area_check_enabled", True)),
            area_threshold=float(data.get("area_threshold", 0.0)),
            area_color=tuple(data.get("area_color", list(_DEFAULT_AREA_COLOR))),
            distance_check_enabled=bool(data.get("distance_check_enabled", True)),
            distance_threshold=float(data.get("distance_threshold", 0.0)),
            distance_method=str(data.get("distance_method", "centroid")),
            distance_color=tuple(data.get("distance_color", list(_DEFAULT_DIST_COLOR))),
        )
