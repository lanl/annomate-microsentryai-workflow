from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AnomalyConstraintState:
    """Persistent settings for the anomaly area and proximity constraint checks."""

    enabled: bool = False

    area_check_enabled: bool = True
    # Threshold in world-unit² (px² when uncalibrated). 0.0 means check is unconfigured.
    area_threshold: float = 0.0

    distance_check_enabled: bool = True
    # Threshold in world-units (px when uncalibrated). 0.0 means check is unconfigured.
    distance_threshold: float = 0.0
    # "centroid" = arithmetic-mean centroid distance; "edge" = minimum vertex-to-vertex distance
    distance_method: str = "centroid"

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "area_check_enabled": self.area_check_enabled,
            "area_threshold": self.area_threshold,
            "distance_check_enabled": self.distance_check_enabled,
            "distance_threshold": self.distance_threshold,
            "distance_method": self.distance_method,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AnomalyConstraintState":
        return cls(
            enabled=bool(data.get("enabled", False)),
            area_check_enabled=bool(data.get("area_check_enabled", True)),
            area_threshold=float(data.get("area_threshold", 0.0)),
            distance_check_enabled=bool(data.get("distance_check_enabled", True)),
            distance_threshold=float(data.get("distance_threshold", 0.0)),
            distance_method=str(data.get("distance_method", "centroid")),
        )
