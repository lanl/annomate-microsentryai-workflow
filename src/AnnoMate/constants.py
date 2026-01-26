"""
Global Constants for AnnoMate.

This module defines application-wide constants, including metadata (name, version)
and default visual styling options (such as the default color palette for classes).
"""

from typing import List
from PyQt5.QtGui import QColor

APP_NAME = "AnnoMate and MicroSentryAI"
APP_VERSION = "1.0"

# Default palette for new classes.
# Colors are cycled when the user adds more classes than defined here.
DEFAULT_CLASS_COLORS: List[QColor] = [
    QColor(255, 0, 0),      # Red
    QColor(0, 200, 0),      # Green
    QColor(0, 120, 255),    # Blue
    QColor(255, 165, 0),    # Orange
    QColor(255, 0, 255),    # Magenta
    QColor(0, 200, 200),    # Cyan
    QColor(255, 230, 0),    # Yellow
]