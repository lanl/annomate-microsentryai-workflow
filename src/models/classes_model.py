from PySide6.QtCore import QAbstractTableModel, QModelIndex, QSortFilterProxyModel, Qt


class ClassColumns:
    IMAGE_TAG = 0  # image-level defect class tag toggle
    COLOR = 1
    CLASS = 2
    IMAGE = 3
    TOTAL = 4
    VISIBILITY = 5
    DELETE = 6


CLASS_NAME_ROLE = Qt.UserRole + 1
SORT_ROLE = Qt.UserRole + 2
COLOR_ROLE = Qt.UserRole + 3
VISIBLE_ROLE = Qt.UserRole + 4
IMAGE_TAG_ROLE = (
    Qt.UserRole + 5
)  # bool — tagged in any mode (union of pixel and image-level)
IMAGE_LEVEL_MODE_ROLE = (
    Qt.UserRole + 6
)  # bool — whether image-level annotation mode is active
IMAGE_TAG_KIND_ROLE = Qt.UserRole + 7  # str — "active" | "inactive" | "none"

_HEADERS = ["", "", "Class", "Img", "Tot", "", ""]
_TOOLTIPS = {
    ClassColumns.IMAGE_TAG: "Image-level defect class tag (active in Image Level Mode for rejected images)",
    ClassColumns.COLOR: "Annotation class color",
    ClassColumns.CLASS: "Annotation class name",
    ClassColumns.IMAGE: "Class count for this image",
    ClassColumns.TOTAL: "Class count for the whole dataset",
    ClassColumns.VISIBILITY: "Show or hide this class's annotations",
    ClassColumns.DELETE: "Delete annotation class",
}


class ClassTableModel(QAbstractTableModel):
    """Read-only table model for the annotation class registry."""

    def __init__(self, dataset_model, parent=None) -> None:
        super().__init__(parent)
        self._dataset_model = dataset_model
        self._current_row = -1
        self._class_names = self._dataset_model.get_class_names()
        self._pixel_tags: set = set()
        self._image_level_tags: set = set()
        self._image_tags: set = set()  # union of both — used for sort
        self._tag_interactive: bool = False
        self._in_image_level_mode: bool = False

        self._dataset_model.modelReset.connect(self._on_model_reset)
        self._dataset_model.dataChanged.connect(self._on_source_data_changed)
        self._dataset_model.classVisibilityChanged.connect(
            self._on_class_visibility_changed
        )
        self._dataset_model.annotation_mode_changed.connect(
            self._on_annotation_mode_changed
        )

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.isValid():
            return 0
        return len(self._class_names)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.isValid():
            return 0
        return len(_HEADERS)

    def flags(self, index: QModelIndex) -> Qt.ItemFlag:
        if not index.isValid():
            return Qt.NoItemFlags
        if index.column() == ClassColumns.IMAGE_TAG:
            if self._tag_interactive:
                return Qt.ItemIsEnabled | Qt.ItemIsSelectable
            return Qt.ItemIsEnabled  # visible but not selectable when inactive
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable

    def headerData(
        self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole
    ) -> object:
        if orientation != Qt.Horizontal or not (0 <= section < len(_HEADERS)):
            return None
        if role == Qt.DisplayRole:
            return _HEADERS[section]
        if role == Qt.ToolTipRole:
            if section == ClassColumns.TOTAL and self._in_image_level_mode:
                return "Number of images tagged with this defect class"
            return _TOOLTIPS.get(section)
        return None

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> object:
        if not index.isValid() or not (0 <= index.row() < self.rowCount()):
            return None

        row = index.row()
        col = index.column()
        name = self._class_names[row]

        if role == CLASS_NAME_ROLE:
            return name
        if role == SORT_ROLE:
            return self.sort_value(row, col)
        if role == COLOR_ROLE and col == ClassColumns.COLOR:
            return self._dataset_model.get_class_color(name)
        if role == VISIBLE_ROLE:
            return self._dataset_model.is_class_visible(name)
        if role == IMAGE_TAG_ROLE:
            return name in self._image_tags
        if role == IMAGE_TAG_KIND_ROLE:
            return self._tag_kind(name)
        if role == IMAGE_LEVEL_MODE_ROLE and col == ClassColumns.IMAGE_TAG:
            return self._in_image_level_mode
        if role == Qt.ToolTipRole:
            return self._tooltip(name, col)
        if role == Qt.TextAlignmentRole:
            return self._alignment(col)
        if role == Qt.DisplayRole:
            return self._display(name, col)

        return None

    def set_current_row(self, row: int) -> None:
        self._current_row = row
        self._refresh_tag_state()
        self.refresh_counts()

    def _refresh_tag_state(self) -> None:
        """Update image tag state and interactivity for the current row."""
        if self._current_row >= 0:
            mode = self._dataset_model.get_annotation_mode()
            decision = self._dataset_model.get_review_decision(self._current_row)
            self._tag_interactive = mode == "image_level" and decision == "reject"
            self._in_image_level_mode = mode == "image_level"
            self._pixel_tags = {
                a["category_name"]
                for a in self._dataset_model.get_annotations(self._current_row)
            }
            self._image_level_tags = set(
                self._dataset_model.get_image_classes(self._current_row)
            )
            self._image_tags = self._pixel_tags | self._image_level_tags
        else:
            self._pixel_tags = set()
            self._image_level_tags = set()
            self._image_tags = set()
            self._tag_interactive = False
            self._in_image_level_mode = False
        if self.rowCount() > 0:
            self.dataChanged.emit(
                self.index(0, ClassColumns.IMAGE_TAG),
                self.index(self.rowCount() - 1, ClassColumns.IMAGE_TAG),
                [
                    Qt.DisplayRole,
                    IMAGE_TAG_ROLE,
                    IMAGE_TAG_KIND_ROLE,
                    IMAGE_LEVEL_MODE_ROLE,
                ],
            )

    def refresh_classes(self) -> None:
        self.beginResetModel()
        self._class_names = self._dataset_model.get_class_names()
        self.endResetModel()

    def refresh_counts(self) -> None:
        if self.rowCount() == 0:
            return
        self.dataChanged.emit(
            self.index(0, ClassColumns.IMAGE),
            self.index(self.rowCount() - 1, ClassColumns.TOTAL),
            [Qt.DisplayRole, SORT_ROLE, Qt.ToolTipRole],
        )

    def class_name(self, row: int) -> str:
        if not (0 <= row < self.rowCount()):
            return ""
        return self._class_names[row]

    def _tag_kind(self, name: str) -> str:
        """Return tag classification for name.

        "active"   — amber: has pixel annotation (pixel mode) OR purely image-level tag
                     with no pixel backing (image-level mode)
        "inactive" — blue: has pixel annotation (image-level mode) OR only image-level
                     tag with no pixel backing (pixel mode)
        "none"     — not tagged in either mode

        The amber/blue meaning is consistent: amber = active-mode-exclusive,
        blue = other-mode-backed. In both modes, blue always means pixel-backed.
        """
        has_pixel = name in self._pixel_tags
        has_image = name in self._image_level_tags
        if not has_pixel and not has_image:
            return "none"
        if self._in_image_level_mode:
            # Amber = exclusively image-level (no pixel backing)
            # Blue  = pixel-backed (with or without an image-level tag)
            return "active" if (has_image and not has_pixel) else "inactive"
        # Pixel mode:
        # Amber = has pixel annotation   Blue = image-level tag only
        return "active" if has_pixel else "inactive"

    def sort_value(self, row: int, col: int):
        name = self.class_name(row)
        if col == ClassColumns.IMAGE_TAG:
            return int(name in self._image_tags)
        if col == ClassColumns.COLOR:
            return tuple(self._dataset_model.get_class_color(name))
        if col == ClassColumns.CLASS:
            return name.casefold()
        if col == ClassColumns.IMAGE:
            return self._count_image_annotations(name)
        if col == ClassColumns.TOTAL:
            return self._count_total_annotations(name)
        return row

    def tie_break_value(self, row: int) -> str:
        return self.class_name(row).casefold()

    def _on_model_reset(self) -> None:
        self.refresh_classes()
        self._refresh_tag_state()

    def _on_source_data_changed(self, top_left, bottom_right, roles=None) -> None:
        class_names = self._dataset_model.get_class_names()
        if class_names != self._class_names:
            self.refresh_classes()
            return
        self._refresh_tag_state()
        if self.rowCount() == 0:
            return
        self.dataChanged.emit(
            self.index(0, 0),
            self.index(self.rowCount() - 1, self.columnCount() - 1),
            [
                Qt.DisplayRole,
                SORT_ROLE,
                COLOR_ROLE,
                VISIBLE_ROLE,
                IMAGE_TAG_ROLE,
                IMAGE_TAG_KIND_ROLE,
                Qt.ToolTipRole,
            ],
        )

    def _on_annotation_mode_changed(self, mode: str) -> None:
        self._refresh_tag_state()

    def _on_class_visibility_changed(self, name: str, visible: bool) -> None:
        row = self._class_names.index(name) if name in self._class_names else -1
        if row < 0:
            return
        self.dataChanged.emit(
            self.index(row, ClassColumns.VISIBILITY),
            self.index(row, ClassColumns.VISIBILITY),
            [Qt.DisplayRole, VISIBLE_ROLE, Qt.ToolTipRole],
        )

    def _display(self, name: str, col: int) -> str:
        if col == ClassColumns.IMAGE_TAG:
            return ""
        if col == ClassColumns.COLOR:
            return ""
        if col == ClassColumns.CLASS:
            return name
        if col == ClassColumns.IMAGE:
            return str(self._count_image_annotations(name))
        if col == ClassColumns.TOTAL:
            return str(self._count_total_annotations(name))
        if col == ClassColumns.VISIBILITY:
            return "Hide" if self._dataset_model.is_class_visible(name) else "Show"
        if col == ClassColumns.DELETE:
            return "Delete"
        return ""

    def _tooltip(self, name: str, col: int) -> str:
        if col == ClassColumns.IMAGE_TAG:
            kind = self._tag_kind(name)
            if kind == "none":
                if self._tag_interactive:
                    return f'Click to add "{name}" as an image-level tag'
                return "Switch to Image Level mode and reject this image to assign class tags"
            if self._in_image_level_mode:
                if kind == "active":
                    return f'"{name}" is exclusively tagged at image level. Click to remove.'
                # inactive = pixel-backed (may or may not also have an image-level tag)
                pixel_n = self._count_image_annotations(name)
                if name in self._image_level_tags:
                    return (
                        f'"{name}" has {pixel_n} pixel-level annotation(s) and an '
                        f"image-level tag. To untag, remove the pixel annotations in "
                        f"Pixel Level mode first."
                    )
                return (
                    f'"{name}" has {pixel_n} pixel-level annotation(s) on this image. '
                    f"Click to also add an image-level tag."
                )
            else:
                if kind == "active":
                    pixel_n = self._count_image_annotations(name)
                    return f'"{name}" has {pixel_n} pixel-level annotation(s) on this image'
                return f'"{name}" is tagged at image level — switch to Image Level mode to manage'
        if col == ClassColumns.COLOR:
            r, g, b = self._dataset_model.get_class_color(name)
            return f"{name}: rgb({r}, {g}, {b})"
        if col == ClassColumns.TOTAL and self._in_image_level_mode:
            n = self._count_total_annotations(name)
            return f'"{name}" tagged on {n} image(s) across the dataset'
        if col == ClassColumns.DELETE:
            return f'Delete "{name}"'
        if col == ClassColumns.VISIBILITY:
            action = "Hide" if self._dataset_model.is_class_visible(name) else "Show"
            return f'{action} annotations for "{name}"'
        return self._display(name, col) or (_TOOLTIPS.get(col) or "")

    def _alignment(self, col: int) -> Qt.AlignmentFlag:
        if col in (ClassColumns.IMAGE, ClassColumns.TOTAL):
            return Qt.AlignRight | Qt.AlignVCenter
        if col in (
            ClassColumns.IMAGE_TAG,
            ClassColumns.COLOR,
            ClassColumns.VISIBILITY,
            ClassColumns.DELETE,
        ):
            return Qt.AlignCenter
        return Qt.AlignLeft | Qt.AlignVCenter

    def _count_image_annotations(self, class_name: str) -> int:
        if self._current_row < 0:
            return 0
        return sum(
            1
            for annotation in self._dataset_model.get_annotations(self._current_row)
            if annotation["category_name"] == class_name
        )

    def _count_total_annotations(self, class_name: str) -> int:
        if self._in_image_level_mode:
            return sum(
                1
                for row in range(self._dataset_model.rowCount())
                if class_name in self._dataset_model.get_image_classes(row)
            )
        total = 0
        for row in range(self._dataset_model.rowCount()):
            for annotation in self._dataset_model.get_annotations(row):
                if annotation["category_name"] == class_name:
                    total += 1
        return total


class ClassSortProxyModel(QSortFilterProxyModel):
    """Type-aware proxy for annotation class sorting."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setDynamicSortFilter(True)
        self.setSortCaseSensitivity(Qt.CaseInsensitive)

    def lessThan(self, left: QModelIndex, right: QModelIndex) -> bool:
        model = self.sourceModel()
        if model is None:
            return super().lessThan(left, right)

        col = left.column()
        left_value = model.sort_value(left.row(), col)
        right_value = model.sort_value(right.row(), col)

        if left_value == right_value:
            return model.tie_break_value(left.row()) < model.tie_break_value(
                right.row()
            )
        return left_value < right_value
