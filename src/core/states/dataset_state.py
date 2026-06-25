from datetime import datetime, timezone

from core.utils.constants import DEFAULT_CLASSES


class DatasetState:
    """State container for dataset annotations, metadata, and class registry.

    Manages per-image annotations, inspector assignments, and free-text
    notes alongside a global class registry that persists across folder
    loads. Contains zero Qt dependencies.

    Attributes:
        image_dir (str): Absolute path to the currently loaded image folder.
        image_files (list[str]): Ordered list of image filenames in the folder.
        annotations (dict[str, list[dict]]): Per-image annotation records
            keyed by filename. Each record contains ``category_name`` (str)
            and ``polygon`` (list of (x, y) coordinate pairs).
        inspectors (dict[str, str]): Maps image filename to the name of the
            inspector who reviewed it.
        notes (dict[str, str]): Maps image filename to a free-text note.
        class_names (list[str]): Ordered class name registry.
        class_colors (dict[str, tuple]): Maps class name to an RGB color tuple.
        class_visibility (dict[str, bool]): Maps class name to viewport visibility.
    """

    def __init__(self) -> None:
        """Initialize DatasetState from default class definitions."""
        # File management
        self.image_dir = ""
        self.image_files = []

        # Annotations & Metadata
        self.annotations = {}  # { "img.jpg": [ { "category_name": str, "polygon": [...] } ] }
        self.inspectors = {}  # { "img.jpg": "John Doe" }
        self.notes = {}  # { "img.jpg": "Needs review" }
        self.review_decisions = {}  # { "img.jpg": "accept" | "reject" | "omitted" }
        self.decision_timestamps = {}  # { "img.jpg": ISO-8601 UTC string }
        self.omit_reasons = {}  # { "img.jpg": "no_decision" | "no_annotation" }
        self.image_sizes = {}  # { "img.jpg": (width, height) } — cached to avoid PIL reads on save

        # Class registry — initialized from defaults, NOT cleared on folder load
        self.class_names = list(DEFAULT_CLASSES.keys())
        self.class_colors = dict(DEFAULT_CLASSES)  # { name: (r, g, b) }
        self.class_visibility = {name: True for name in self.class_names}

    def clear(self) -> None:
        """Reset per-folder data. Class registry is intentionally preserved."""
        self.image_dir = ""
        self.image_files = []
        self.annotations.clear()
        self.inspectors.clear()
        self.notes.clear()
        self.review_decisions.clear()
        self.decision_timestamps.clear()
        self.omit_reasons.clear()
        self.image_sizes.clear()

    def reset_classes(self) -> None:
        """Reset the class registry back to defaults."""
        self.class_names = list(DEFAULT_CLASSES.keys())
        self.class_colors = dict(DEFAULT_CLASSES)
        self.class_visibility = {name: True for name in self.class_names}

    def is_reviewed(self, img_name: str) -> bool:
        """Return whether an image has been fully reviewed.

        Accept decisions are reviewed unconditionally. Reject decisions require
        at least one annotation to be considered reviewed.

        Args:
            img_name (str): Image filename to check.

        Returns:
            bool: ``True`` if the image is reviewed; ``False`` otherwise.
        """
        decision = self.review_decisions.get(img_name)
        if decision == "accept":
            return True
        if decision == "reject":
            return bool(self.annotations.get(img_name))
        return False

    # --- Annotation CRUD ---

    def add_annotation(
        self, image_name: str, category: str, polygon: list, thickness: float = 2.0
    ) -> None:
        """Append a new polygon annotation to an image.

        Args:
            image_name (str): Target image filename.
            category (str): Class name for the annotation.
            polygon (list): Sequence of (x, y) coordinate pairs defining
                the polygon boundary.
            thickness (float): Line thickness for the annotation (default: 2.0).
        """
        self.annotations.setdefault(image_name, []).append(
            {
                "category_name": category.lower(),
                "polygon": polygon,
                "thickness": thickness,
                "visible": True,
            }
        )

    def update_annotation_thickness(
        self, image_name: str, index: int, thickness: float
    ) -> None:
        """Update the thickness of a specific polygon."""
        annos = self.annotations.get(image_name, [])
        if 0 <= index < len(annos):
            annos[index]["thickness"] = thickness

    def update_annotation_class(
        self, image_name: str, index: int, new_class: str
    ) -> None:
        """Change the category_name of a specific annotation."""
        annos = self.annotations.get(image_name, [])
        if 0 <= index < len(annos):
            annos[index]["category_name"] = new_class.lower()

    def delete_annotation(self, image_name: str, index: int) -> None:
        """Remove the annotation at *index* for *image_name*.

        Args:
            image_name (str): Target image filename.
            index (int): Zero-based index of the annotation to remove.
                Out-of-range indices are silently ignored.
        """
        annos = self.annotations.get(image_name, [])
        if 0 <= index < len(annos):
            annos.pop(index)

    def update_annotation_points(
        self, image_name: str, index: int, points: list
    ) -> None:
        """Replace the polygon points of an existing annotation.

        Args:
            image_name (str): Target image filename.
            index (int): Zero-based index of the annotation to update.
                Out-of-range indices are silently ignored.
            points (list): New sequence of (x, y) coordinate pairs.
        """
        annos = self.annotations.get(image_name, [])
        if 0 <= index < len(annos):
            annos[index]["polygon"] = points

    def set_annotation_visible(
        self, image_name: str, index: int, visible: bool
    ) -> None:
        """Set viewport visibility for a specific annotation."""
        annos = self.annotations.get(image_name, [])
        if 0 <= index < len(annos):
            annos[index]["visible"] = bool(visible)

    def is_annotation_visible(self, image_name: str, index: int) -> bool:
        """Return whether a specific annotation should render in the viewport."""
        annos = self.annotations.get(image_name, [])
        if 0 <= index < len(annos):
            return annos[index].get("visible", True)
        return True

    # --- Class Registry ---

    def add_class(self, name: str, color: tuple) -> None:
        """Register a new class in the global class registry.

        Args:
            name (str): Class label to register. Duplicates are ignored.
            color (tuple): RGB color tuple to associate with the class.
        """
        name = name.lower()
        if name not in self.class_names:
            self.class_names.append(name)
            self.class_colors[name] = color
            self.class_visibility[name] = True

    def delete_class(self, name: str) -> None:
        """Remove a class and all annotations that reference it.

        Args:
            name (str): Class label to remove. Unregistered names are
                silently ignored.
        """
        if name in self.class_names:
            self.class_names.remove(name)
            self.class_colors.pop(name, None)
            self.class_visibility.pop(name, None)
            for img in self.annotations:
                self.annotations[img] = [
                    a for a in self.annotations[img] if a.get("category_name") != name
                ]

    def set_class_visible(self, name: str, visible: bool) -> None:
        """Set viewport visibility for an annotation class."""
        if name in self.class_names:
            self.class_visibility[name] = bool(visible)

    def is_class_visible(self, name: str) -> bool:
        """Return whether an annotation class should render in the viewport."""
        return self.class_visibility.get(name, True)

    # --- Per-image Metadata ---

    def set_inspector(self, image_name: str, value: str) -> None:
        """Assign an inspector name to an image.

        Args:
            image_name (str): Target image filename.
            value (str): Inspector's name or identifier.
        """
        self.inspectors[image_name] = value

    def set_note(self, image_name: str, value: str) -> None:
        """Attach a free-text note to an image.

        Args:
            image_name (str): Target image filename.
            value (str): Note content to store.
        """
        self.notes[image_name] = value

    def set_review_decision(
        self, image_name: str, decision, omit_reason: str | None = None
    ) -> None:
        """Set the image-level review decision.

        Args:
            image_name (str): Target image filename.
            decision (str | None): ``"accept"``, ``"reject"``, ``"omitted"``, or
                ``None`` to clear.
            omit_reason (str | None): Reason key stored when decision is
                ``"omitted"`` (``"no_decision"`` or ``"no_annotation"``).
        """
        if decision is None:
            self.review_decisions.pop(image_name, None)
            self.decision_timestamps.pop(image_name, None)
            self.omit_reasons.pop(image_name, None)
        else:
            self.review_decisions[image_name] = decision
            self.decision_timestamps[image_name] = datetime.now(
                timezone.utc
            ).isoformat()
            if decision == "omitted" and omit_reason:
                self.omit_reasons[image_name] = omit_reason
            elif decision in ("accept", "reject"):
                self.omit_reasons.pop(image_name, None)

    def get_review_decision(self, image_name: str):
        """Return the image-level review decision, or None if not set."""
        return self.review_decisions.get(image_name)

    def get_omit_reason(self, image_name: str) -> str | None:
        """Return the omit reason key for *image_name*, or None if not omitted."""
        return self.omit_reasons.get(image_name)
