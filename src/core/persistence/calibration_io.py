"""Pure Python read/write for plain-text calibration ratio files.

Format:
    {px_count}px:{world_value}{unit}

Examples:
    500px:1cm       500 pixels = 1 cm
    1px:0.05mm      1 pixel = 0.05 mm
    50px:1mm        50 pixels = 1 mm

The left side must always be in pixels. The right side defines the world unit.
Scale stored internally as world_units_per_pixel.
"""

import re

_RATIO_RE = re.compile(
    r"^\s*([\d.]+)\s*px\s*:\s*([\d.]+)\s*([a-zA-Z]+)\s*$",
    re.IGNORECASE,
)


def parse_ratio_string(s: str) -> tuple[float, float, str]:
    """Parse a ratio string into (px_count, world_val, unit).

    Accepts '500px:1cm', '1px:0.05mm', '50px:1mm', etc.
    """
    m = _RATIO_RE.match(s.strip())
    if not m:
        raise ValueError(
            f"Invalid ratio format: {s!r}\nExpected e.g. '500px:1cm' or '50px:1mm'"
        )
    px_count = float(m.group(1))
    world_val = float(m.group(2))
    unit = m.group(3)
    if px_count <= 0:
        raise ValueError("Pixel count must be greater than zero")
    if world_val <= 0:
        raise ValueError("World value must be greater than zero")
    return px_count, world_val, unit


def format_ratio_string(px_count: float, world_val: float, unit: str) -> str:
    """Format as '{px_count:g}px:{world_val:g}{unit}'."""
    return f"{px_count:g}px:{world_val:g}{unit}"


def write_calibration_ratio(
    path: str, px_count: float, world_val: float, unit: str
) -> None:
    """Write calibration ratio to a plain-text .txt file."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(format_ratio_string(px_count, world_val, unit) + "\n")


def read_calibration_ratio(path: str) -> dict:
    """Read a calibration ratio .txt file.

    Returns dict with keys: scale_world_per_px, unit, px_count, world_val.
    """
    with open(path, encoding="utf-8") as f:
        content = f.read().strip()
    px_count, world_val, unit = parse_ratio_string(content)
    return {
        "scale_world_per_px": world_val / px_count,
        "unit": unit,
        "px_count": px_count,
        "world_val": world_val,
    }
