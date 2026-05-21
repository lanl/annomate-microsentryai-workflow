APP_NAME = "AnnoMate and MicroSentryAI"
APP_VERSION = "1.0"

# Pure Python RGB tuples — no Qt dependency in the domain layer.
# Cycled when the user adds more classes than colours defined here.
DEFAULT_CLASS_COLORS: list[tuple] = [
    (232, 102, 116),  # Soft red
    (239, 151, 87),  # Soft orange
    (228, 204, 88),  # Soft yellow
    (123, 194, 112),  # Soft green
    (93, 190, 190),  # Soft cyan
    (100, 152, 220),  # Soft blue
    (145, 119, 213),  # Soft violet
    (204, 112, 183),  # Soft magenta
    (214, 124, 102),  # Coral
    (213, 174, 82),  # Amber
    (169, 199, 91),  # Olive
    (102, 184, 133),  # Jade
    (92, 171, 211),  # Azure
    (122, 132, 216),  # Indigo
    (176, 112, 205),  # Purple
    (218, 118, 151),  # Rose
    (190, 138, 94),  # Tan
    (139, 181, 105),  # Moss
    (93, 169, 153),  # Teal
    (126, 164, 203),  # Steel blue
]

DEFAULT_CLASSES: dict[str, tuple] = {}
