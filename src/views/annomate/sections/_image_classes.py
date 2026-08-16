from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from views.icons import material_icon

from ._shared import _ClickableFrame

_DOT_W = 16
_CHECK_ICON_SIZE = 12


class _ImageClassRow(_ClickableFrame):
    """One class as a row: color dot, name, and a check icon if tagged.

    The check icon sits immediately after the name (not pushed to the row's
    far edge) so it reads as "this name is checked" rather than as a
    separate trailing column.

    Signals:
        toggled (str): This row's class name, emitted on click -- only
            connected to anything meaningful when the row is interactive.
    """

    toggled = Signal(str)

    def __init__(
        self, name: str, rgb, tagged: bool, interactive: bool, parent: QWidget = None
    ) -> None:
        super().__init__(parent)
        self._name = name
        if interactive:
            self.setCursor(Qt.PointingHandCursor)
            self.clicked.connect(lambda: self.toggled.emit(self._name))
            self.setToolTip(
                "Click to remove this tag from the image"
                if tagged
                else "Click to tag this image with this class"
            )

        h = QHBoxLayout(self)
        h.setContentsMargins(4, 3, 4, 3)
        h.setSpacing(6)

        dot = QLabel()
        dot.setFixedSize(_DOT_W, _DOT_W)
        dot.setStyleSheet(
            f"QLabel {{ background-color: rgb{tuple(rgb)}; border: 1px solid "
            f"rgba(120,120,120,150); border-radius: {_DOT_W // 2}px; }}"
        )
        h.addWidget(dot)

        name_lbl = QLabel(name)
        name_lbl.setStyleSheet("color: black; font-size: 11px;")
        h.addWidget(name_lbl)

        check_lbl = QLabel()
        check_lbl.setFixedSize(_CHECK_ICON_SIZE, _CHECK_ICON_SIZE)
        if tagged:
            check_lbl.setPixmap(
                material_icon("check", size=_CHECK_ICON_SIZE, color="black").pixmap(
                    _CHECK_ICON_SIZE, _CHECK_ICON_SIZE
                )
            )
        h.addWidget(check_lbl)

        h.addStretch(1)


class ImageClassesSection(QWidget):
    """Toggleable image-level class tags for the currently displayed image.

    Shown in the navigator card's expanded body in place of the polygon
    Annotations list when the dataset is in image-level mode. Every dataset
    class is listed as a row while the image is rejected -- clicking a row
    tags/untags it, mirroring the same "reject to unlock tagging" rule the
    Annotation Classes panel enforces, so both surfaces agree on when
    *new* tagging is allowed. Outside of a reject decision, only the tags
    already assigned are shown -- but those existing tags stay clickable so
    a tag applied while rejected can still be removed after the decision is
    changed back (e.g. to undecided), without having to re-reject first.
    """

    def __init__(self, dataset_model, parent: QWidget = None) -> None:
        super().__init__(parent)
        self.dataset_model = dataset_model
        self._current_row: int = -1
        self._rows: dict[str, _ImageClassRow] = {}

        self._init_ui()
        dataset_model.modelReset.connect(self._on_model_reset)
        dataset_model.dataChanged.connect(self._on_data_changed)

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self._hint_lbl = QLabel("Click a class to tag or untag this image.")
        self._hint_lbl.setWordWrap(True)
        self._hint_lbl.setStyleSheet("color: black; font-size: 11px;")
        self._hint_lbl.setContentsMargins(6, 4, 6, 0)
        layout.addWidget(self._hint_lbl)

        self._rows_container = QWidget()
        self._rows_layout = QVBoxLayout(self._rows_container)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(2)
        layout.addWidget(self._rows_container)

        self._empty_lbl = QLabel()
        self._empty_lbl.setWordWrap(True)
        self._empty_lbl.setStyleSheet("color: black; font-size: 11px;")
        self._empty_lbl.setContentsMargins(6, 4, 6, 4)
        layout.addWidget(self._empty_lbl)

    def set_current_row(self, row: int) -> None:
        self._current_row = row
        self._rebuild()

    def _is_interactive(self) -> bool:
        if self._current_row < 0:
            return False
        return self.dataset_model.get_review_decision(self._current_row) == "reject"

    def _rebuild(self) -> None:
        while self._rows_layout.count():
            item = self._rows_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._rows.clear()

        reject_unlocked = self._is_interactive()
        tagged = (
            set(self.dataset_model.get_image_classes(self._current_row))
            if self._current_row >= 0
            else set()
        )
        visible_names = (
            self.dataset_model.get_class_names() if reject_unlocked else sorted(tagged)
        )

        for name in visible_names:
            rgb = self.dataset_model.get_class_color(name)
            is_tagged = name in tagged
            # Adding a new tag still requires the reject unlock; removing an
            # already-tagged class never needs it, so a tag applied while
            # rejected stays removable after the decision changes.
            row_interactive = reject_unlocked or is_tagged
            row = _ImageClassRow(name, rgb, is_tagged, row_interactive)
            row.toggled.connect(self._on_row_toggled)
            self._rows_layout.addWidget(row)
            self._rows[name] = row

        has_rows = bool(visible_names)
        self._rows_container.setVisible(has_rows)
        self._hint_lbl.setVisible(reject_unlocked or bool(tagged))
        self._hint_lbl.setText(
            "Click a class to tag or untag this image."
            if reject_unlocked
            else "Click a tag to remove it."
        )
        self._empty_lbl.setVisible(not has_rows)
        if not has_rows:
            self._empty_lbl.setText(
                "No classes defined yet. Add one in the Annotation Classes panel."
                if reject_unlocked
                else "No class tags. Reject this image to assign class tags."
            )

    def _on_row_toggled(self, name: str) -> None:
        if self._current_row < 0:
            return
        current = self.dataset_model.get_image_classes(self._current_row)
        if name in current:
            pixel_count = self.dataset_model.get_pixel_annotation_count_for_class(
                self._current_row, name
            )
            if pixel_count > 0:
                QMessageBox.warning(
                    self,
                    "Cannot Remove Tag",
                    f"'{name}' has {pixel_count} pixel-level annotation(s) on this "
                    "image.\nRemove those annotations in Pixel Level mode first.",
                )
                return
            current = [n for n in current if n != name]
        else:
            current = current + [name]
        self.dataset_model.set_image_classes(self._current_row, current)

    def _on_model_reset(self) -> None:
        self._rebuild()

    def _on_data_changed(self, top_left, bottom_right, roles=None) -> None:
        if self._current_row < 0:
            return
        if top_left.row() <= self._current_row <= bottom_right.row():
            self._rebuild()
