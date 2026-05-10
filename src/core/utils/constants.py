APP_NAME = "AnnoMate and MicroSentryAI"
APP_VERSION = "1.0"

# Pure Python RGB tuples — no Qt dependency in the domain layer.
# Cycled when the user adds more classes than colours defined here.
DEFAULT_CLASS_COLORS: list[tuple] = [
    (255, 0, 0),  # Red
    (0, 200, 0),  # Green
    (0, 120, 255),  # Blue
    (255, 165, 0),  # Orange
    (255, 0, 255),  # Magenta
    (0, 200, 200),  # Cyan
    (255, 230, 0),  # Yellow
]

DEFAULT_CLASSES: dict[str, tuple] = {}
