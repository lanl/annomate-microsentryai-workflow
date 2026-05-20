class CalibrationState:
    """Pure Python calibration state. Zero Qt dependencies."""

    def __init__(self) -> None:
        # Calibration reference (original image coords)
        self.scale: float | None = None  # world units per original image pixel
        self.unit: str = "mm"
        self.calib_p1: tuple | None = None  # (x, y) in original pixels
        self.calib_p2: tuple | None = None  # (x, y) in original pixels
        self.real_distance: float = 1.0  # known real-world distance

        # Grid appearance
        self.grid_visible: bool = False
        self.grid_color: tuple = (58, 90, 122)  # (r, g, b)
        self.grid_opacity: float = 0.5  # 0.0–1.0
        self.grid_spacing_world: float = 1.0  # world units per grid step
        self.grid_spacing_auto: bool = True  # recompute on calibration

        # Measurement points — session-only, never persisted
        self.meas_p1: tuple | None = None
        self.meas_p2: tuple | None = None

    def is_calibrated(self) -> bool:
        return self.scale is not None

    def clear_calibration(self) -> None:
        self.scale = None
        self.calib_p1 = None
        self.calib_p2 = None
        self.real_distance = 1.0
        self.meas_p1 = None
        self.meas_p2 = None

    def clear_measurement(self) -> None:
        self.meas_p1 = None
        self.meas_p2 = None
