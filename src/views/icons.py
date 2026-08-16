"""Loads vendored Material Symbols SVGs as tinted QIcons.

SVG sources live in src/resources/icons/ (Google Material Symbols,
Apache-2.0 — see src/resources/icons/NOTICE). Each is a single-color
outlined glyph; recoloring happens here at render time via QPainter
compositing, so one asset per icon serves every tint a caller needs.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

_ICON_DIR = Path(__file__).resolve().parent.parent / "resources" / "icons"

# Render at 4x the requested logical size so icons stay crisp on scaled
# (HiDPI) displays instead of being upscaled from a too-small raster.
_OVERSAMPLE = 4


@lru_cache(maxsize=None)
def _pixmap(name: str, size: int, color: str) -> QPixmap:
    renderer = QSvgRenderer(str(_ICON_DIR / f"{name}.svg"))
    px_size = size * _OVERSAMPLE
    pixmap = QPixmap(QSize(px_size, px_size))
    pixmap.fill(Qt.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    renderer.render(painter)
    painter.setCompositionMode(QPainter.CompositionMode_SourceIn)
    painter.fillRect(pixmap.rect(), QColor(color))
    painter.end()

    pixmap.setDevicePixelRatio(_OVERSAMPLE)
    return pixmap


def material_icon(name: str, size: int = 20, color: str = "#3c3c3c") -> QIcon:
    """Return a QIcon for a vendored Material Symbol, tinted to `color`.

    Args:
        name: Icon filename stem under src/resources/icons/ (e.g. "polyline").
        size: Rendered square size in px.
        color: Any QColor-parsable string (hex, name, etc).
    """
    return QIcon(_pixmap(name, size, color))
