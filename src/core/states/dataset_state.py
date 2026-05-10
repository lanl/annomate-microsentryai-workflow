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
        self.review_decisions = {}  # { "img.jpg": "accept" | "reject" }

        # Class registry — initialized from defaults, NOT cleared on folder load
        self.class_names = list(DEFAULT_CLASSES.keys())
        self.class_colors = dict(DEFAULT_CLASSES)  # { name: (r, g, b) }

    def clear(self) -> None:
        """Reset per-folder data. Class registry is intentionally preserved."""
        self.image_dir = ""
        self.image_files = []
        self.annotations.clear()
        self.inspectors.clear()
        self.notes.clear()
        self.review_decisions.clear()

    def reset_classes(self) -> None:
        """Reset the class registry back to defaults."""
        self.class_names = list(DEFAULT_CLASSES.keys())
        self.class_colors = dict(DEFAULT_CLASSES)

    def is_reviewed(self, img_name: str) -> bool:
        """Return whether an image has at least one annotation or metadata entry.

        Args:
            img_name (str): Image filename to check.

        Returns:
            bool: ``True`` if the image has any annotation, inspector
                assignment, or note; ``False`` otherwise.
        """
        has_anno = bool(self.annotations.get(img_name))
        has_meta = bool(self.inspectors.get(img_name) or self.notes.get(img_name))
        has_decision = img_name in self.review_decisions
        return has_anno or has_meta or has_decision

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
            {"category_name": category, "polygon": polygon, "thickness": thickness}
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
            annos[index]["category_name"] = new_class

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

    # --- Class Registry ---

    def add_class(self, name: str, color: tuple) -> None:
        """Register a new class in the global class registry.

        Args:
            name (str): Class label to register. Duplicates are ignored.
            color (tuple): RGB color tuple to associate with the class.
        """
        if name not in self.class_names:
            self.class_names.append(name)
            self.class_colors[name] = color

    def delete_class(self, name: str) -> None:
        """Remove a class and all annotations that reference it.

        Args:
            name (str): Class label to remove. Unregistered names are
                silently ignored.
        """
        if name in self.class_names:
            self.class_names.remove(name)
            self.class_colors.pop(name, None)
            for img in self.annotations:
                self.annotations[img] = [
                    a for a in self.annotations[img] if a.get("category_name") != name
                ]

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

    def set_review_decision(self, image_name: str, decision) -> None:
        """Set the image-level review decision.

        Args:
            image_name (str): Target image filename.
            decision (str | None): ``"accept"``, ``"reject"``, or ``None`` to clear.
        """
        if decision is None:
            self.review_decisions.pop(image_name, None)
        else:
            self.review_decisions[image_name] = decision

    def get_review_decision(self, image_name: str):
        """Return the image-level review decision, or None if not set."""
        return self.review_decisions.get(image_name)
