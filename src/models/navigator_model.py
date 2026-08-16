import os

from PySide6.QtCore import QAbstractTableModel, QModelIndex, QSortFilterProxyModel, Qt
from PySide6.QtGui import QColor, QBrush, QFont


class NavigatorColumns:
    STATUS = 0
    IMG_ID = 1
    ANNOTS = 2
    DECISION = 3
    SCORE = 4


SOURCE_ROW_ROLE = Qt.UserRole + 1
SORT_ROLE = Qt.UserRole + 2
STATUS_COLOR_ROLE = Qt.UserRole + 3
FILTER_DECISION_ROLE = Qt.UserRole + 5  # raw decision string for proxy filtering
FILTER_COMPLETE_ROLE = (
    Qt.UserRole + 6
)  # bool: reject + sufficient work for current mode
IMAGE_STATE_ROLE = Qt.UserRole + 7  # str: one of the six _image_state() keys
HAS_INSPECTOR_ROLE = Qt.UserRole + 8  # bool: row has a non-empty inspector name
HAS_NOTE_ROLE = Qt.UserRole + 9  # bool: row has a non-empty note


DECISION_FILTER_OPTIONS = (("accept", "Accept"), ("reject", "Reject"))
STATUS_FILTER_OPTIONS = (
    ("undecided", "Undecided"),
    ("reviewed", "Reviewed"),
    ("incomplete", "Incomplete"),
    ("conflicting", "Conflicting"),
)
# Maps the six _image_state() keys down to the three status-filter buckets --
# matches NavigatorTableModel.get_filter_facet_counts()'s partition exactly,
# so the chip/checkbox counts always agree with what checking them filters to.
# "conflicting" isn't in this map -- it's checked separately against the raw
# "accept_conflict" state, since it's an intentional subset of "incomplete"
# (both true for accept_conflict rows), not a fourth disjoint bucket.
_STATUS_BUCKET = {
    "undecided": "undecided",
    "undecided_work": "incomplete",
    "accept_clean": "reviewed",
    "reject_reviewed": "reviewed",
    "reject_incomplete": "incomplete",
    "accept_conflict": "incomplete",
}

_HEADERS = ["", "Img ID", "Annots", "Decision", "Score"]
_TOOLTIPS = {
    NavigatorColumns.STATUS: "Review status",
    NavigatorColumns.IMG_ID: "Image identifier",
    NavigatorColumns.ANNOTS: "Annotation count",
    NavigatorColumns.DECISION: "Review decision",
    NavigatorColumns.SCORE: "MicroSentry anomaly score",
}
_DECISION_LABELS = {"accept": "Accept", "reject": "Reject"}


class NavigatorTableModel(QAbstractTableModel):
    """Read-only table model for the Dataset Navigator view."""

    def __init__(self, dataset_model, inference_model=None, parent=None) -> None:
        super().__init__(parent)
        self._dataset_model = dataset_model
        self._inference_model = inference_model

        self._dataset_model.modelReset.connect(self._on_source_reset)
        self._dataset_model.dataChanged.connect(self._on_source_data_changed)

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.isValid():
            return 0
        return self._dataset_model.rowCount()

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.isValid():
            return 0
        return len(_HEADERS)

    def flags(self, index: QModelIndex) -> Qt.ItemFlag:
        if not index.isValid():
            return Qt.NoItemFlags
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable

    def headerData(
        self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole
    ) -> object:
        if orientation != Qt.Horizontal or not (0 <= section < len(_HEADERS)):
            return None
        if role == Qt.DisplayRole:
            return _HEADERS[section]
        if role == Qt.ToolTipRole:
            return _TOOLTIPS.get(section)
        return None

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> object:
        if not index.isValid() or not (0 <= index.row() < self.rowCount()):
            return None

        row = index.row()
        col = index.column()

        if role == SOURCE_ROW_ROLE:
            return row
        if role == SORT_ROLE:
            return self.sort_value(row, col)
        if role == FILTER_DECISION_ROLE:
            return self._dataset_model.get_review_decision(row)
        if role == FILTER_COMPLETE_ROLE:
            return self._is_complete(row)
        if role == IMAGE_STATE_ROLE:
            return self._image_state(row)
        if role == HAS_INSPECTOR_ROLE:
            return bool(self._dataset_model.get_inspector(row))
        if role == HAS_NOTE_ROLE:
            return bool(self._dataset_model.get_note(row))
        if role == STATUS_COLOR_ROLE and col == NavigatorColumns.STATUS:
            return "#4caf50" if self._dataset_model.is_reviewed(row) else "#ff9800"
        if role == Qt.ToolTipRole:
            return self._tooltip(row, col)
        if role == Qt.TextAlignmentRole:
            return self._alignment(col)
        if role == Qt.FontRole:
            return self._font(row, col)
        if role == Qt.ForegroundRole:
            return self._foreground(row, col)
        if role == Qt.DisplayRole:
            return self._display(row, col)

        return None

    def source_row(self, index: QModelIndex) -> int:
        if not index.isValid():
            return -1
        return index.row()

    def sort_value(self, row: int, col: int):
        if col == NavigatorColumns.STATUS:
            return 1 if self._dataset_model.is_reviewed(row) else 0
        if col == NavigatorColumns.IMG_ID:
            return self._image_stem(row).casefold()
        if col == NavigatorColumns.ANNOTS:
            return self._work_count(row)
        if col == NavigatorColumns.SCORE:
            score = self._score(row)
            return None if score is None else float(score)
        return ""

    def tie_break_value(self, row: int) -> str:
        return self._image_stem(row).casefold()

    def notify_inference_changed(self, row: int) -> None:
        if not (0 <= row < self.rowCount()):
            return
        self.dataChanged.emit(
            self.index(row, NavigatorColumns.SCORE),
            self.index(row, NavigatorColumns.SCORE),
            [Qt.DisplayRole, Qt.ToolTipRole, SORT_ROLE, Qt.ForegroundRole, Qt.FontRole],
        )

    def refresh_inference(self) -> None:
        if self.rowCount() == 0:
            return
        self.dataChanged.emit(
            self.index(0, NavigatorColumns.SCORE),
            self.index(self.rowCount() - 1, NavigatorColumns.SCORE),
            [Qt.DisplayRole, Qt.ToolTipRole, SORT_ROLE, Qt.ForegroundRole, Qt.FontRole],
        )

    _STATE_LABELS = {
        "undecided": "No decision",
        "undecided_work": "Has work, no decision",
        "accept_clean": "Accepted",
        "accept_conflict": "Accepted with annotations",
        "reject_incomplete": "Reject incomplete",
        "reject_reviewed": "Reviewed",
    }

    def get_image_state_label(self, row: int) -> str:
        return self._STATE_LABELS.get(self._image_state(row), "")

    def get_annotation_mode(self) -> str:
        """Return the current annotation workflow mode (``"pixel"`` or ``"image_level"``)."""
        return self._dataset_model.get_annotation_mode()

    def class_entries(self, row: int) -> list:
        """Unique classes on *row* for the current mode, alphabetical, each with its color.

        Pixel mode: classes from polygon annotations. Image-level mode: the
        image's assigned class tags -- these are two different sources of
        truth, so which one backs the card's pill tray must follow the mode.
        """
        if not (0 <= row < self.rowCount()):
            return []
        if self._dataset_model.get_annotation_mode() == "image_level":
            names = sorted(set(self._dataset_model.get_image_classes(row)))
        else:
            names = sorted(
                {a["category_name"] for a in self._dataset_model.get_annotations(row)}
            )
        return [(name, self._dataset_model.get_class_color(name)) for name in names]

    def _work_count(self, row: int) -> int:
        """Mode-aware "how much work exists on this image" count.

        Pixel mode: number of polygon annotation instances. Image-level mode:
        number of class tags assigned (tags don't repeat, so this is also
        the unique class count).
        """
        if self._dataset_model.get_annotation_mode() == "image_level":
            return len(self._dataset_model.get_image_classes(row))
        return self._dataset_model.get_annotation_count(row)

    def get_filter_facet_counts(self) -> dict:
        """Image counts for populating the Filter menu's checkbox labels.

        Returns {"decision": {"accept": n, "reject": n},
                 "status": {"undecided": n, "reviewed": n, "incomplete": n, "conflicting": n},
                 "class_options": [(name, rgb, image_count), ...]} (class_options
        alphabetical). Counts are images, not annotation instances -- an image
        with 3 "crack" annotations counts once toward "crack"'s total.
        """
        decision_counts = {"accept": 0, "reject": 0}
        status_counts = {"undecided": 0, "reviewed": 0, "incomplete": 0, "conflicting": 0}
        class_counts: dict = {}
        class_colors: dict = {}
        for row in range(self.rowCount()):
            decision = self._dataset_model.get_review_decision(row)
            if decision in decision_counts:
                decision_counts[decision] += 1

            state = self._image_state(row)
            bucket = _STATUS_BUCKET.get(state)
            if bucket in status_counts:
                status_counts[bucket] += 1
            if state == "accept_conflict":
                status_counts["conflicting"] += 1

            for name, rgb in self.class_entries(row):
                class_counts[name] = class_counts.get(name, 0) + 1
                class_colors[name] = rgb

        class_options = [
            (name, class_colors[name], class_counts[name]) for name in sorted(class_counts)
        ]
        return {
            "decision": decision_counts,
            "status": status_counts,
            "class_options": class_options,
        }

    def _image_state(self, row: int) -> str:
        """Return a string key describing the review completeness of this image.

        "Work" is mode-aware: pixel mode counts polygon annotations; image-level
        mode counts image-level class tags.  This keeps the state consistent with
        what the user can actually produce in the current mode.

        States:
            undecided         -- no decision, no work in the current mode
            undecided_work    -- no decision, but has work in the current mode
            accept_clean      -- accepted, no work in the current mode
            accept_conflict   -- accepted, but has work in the current mode
            reject_incomplete -- rejected, no work in the current mode
            reject_reviewed   -- rejected, has work in the current mode
        """
        decision = self._dataset_model.get_review_decision(row)
        mode = self._dataset_model.get_annotation_mode()
        if mode == "image_level":
            has_work = bool(self._dataset_model.get_image_classes(row))
        else:
            has_work = self._dataset_model.get_annotation_count(row) > 0
        if decision is None:
            return "undecided_work" if has_work else "undecided"
        if decision == "accept":
            return "accept_conflict" if has_work else "accept_clean"
        # reject
        return "reject_reviewed" if has_work else "reject_incomplete"

    def _is_complete(self, row: int) -> bool:
        """Return True if the image needs no further work for the current mode."""
        decision = self._dataset_model.get_review_decision(row)
        if decision != "reject":
            return True
        mode = self._dataset_model.get_annotation_mode()
        if mode == "image_level":
            return bool(self._dataset_model.get_image_classes(row))
        return self._dataset_model.get_annotation_count(row) > 0

    def _on_source_reset(self) -> None:
        self.beginResetModel()
        self.endResetModel()

    def _on_source_data_changed(self, top_left, bottom_right, roles=None) -> None:
        if self.rowCount() == 0:
            return
        top = max(0, top_left.row())
        bottom = min(self.rowCount() - 1, bottom_right.row())
        if top > bottom:
            return
        self.dataChanged.emit(
            self.index(top, 0),
            self.index(bottom, self.columnCount() - 1),
            [
                Qt.DisplayRole,
                Qt.ToolTipRole,
                SORT_ROLE,
                STATUS_COLOR_ROLE,
                HAS_INSPECTOR_ROLE,
                HAS_NOTE_ROLE,
            ],
        )

    def _display(self, row: int, col: int) -> str:
        if col == NavigatorColumns.STATUS:
            return ""
        if col == NavigatorColumns.IMG_ID:
            return self._image_stem(row)
        if col == NavigatorColumns.ANNOTS:
            count = self._work_count(row)
            return str(count) if count > 0 else ""
        if col == NavigatorColumns.DECISION:
            return _DECISION_LABELS.get(
                self._dataset_model.get_review_decision(row), ""
            )
        if col == NavigatorColumns.SCORE:
            score = self._score(row)
            return "" if score is None else f"{score:.2f}"
        return ""

    def _tooltip(self, row: int, col: int) -> str:
        if col == NavigatorColumns.STATUS:
            return self._status_tooltip(row)
        value = self._display(row, col)
        return value or (_TOOLTIPS.get(col) or "")

    def _status_tooltip(self, row: int) -> str:
        state = self._image_state(row)
        ann_count = self._dataset_model.get_annotation_count(row)
        img_classes = self._dataset_model.get_image_classes(row)

        if state == "undecided":
            return (
                "No decision set. Mark this image Accept or Reject to complete review."
            )

        if state == "undecided_work":
            parts = []
            if ann_count:
                parts.append(f"{ann_count} polygon annotation(s)")
            if img_classes:
                parts.append(f"class tag(s): {', '.join(img_classes)}")
            work = " and ".join(parts)
            return f"No decision set. This image has {work} but no Accept or Reject has been assigned. Set a decision to complete."

        if state == "accept_clean":
            return "Accepted as defect-free."

        if state == "accept_conflict":
            parts = []
            if ann_count:
                parts.append(f"{ann_count} polygon annotation(s)")
            if img_classes:
                parts.append(f"class tag(s): {', '.join(img_classes)}")
            work = " and ".join(parts)
            return f"Accepted but has {work}. Remove the annotations or tags, or change the decision to Reject."

        if state == "reject_incomplete":
            return "Reject with no supporting evidence. Add a polygon annotation or a class tag to complete."

        # reject_reviewed
        parts = []
        if ann_count:
            classes = sorted(
                {a["category_name"] for a in self._dataset_model.get_annotations(row)}
            )
            parts.append(f"{ann_count} polygon annotation(s): {', '.join(classes)}")
        if img_classes:
            parts.append(f"class tag(s): {', '.join(img_classes)}")
        detail = " and ".join(parts)
        return f"Reviewed. Rejected with {detail}."

    def _alignment(self, col: int) -> Qt.AlignmentFlag:
        if col in (NavigatorColumns.ANNOTS, NavigatorColumns.SCORE):
            return Qt.AlignRight | Qt.AlignVCenter
        if col in (NavigatorColumns.STATUS, NavigatorColumns.DECISION):
            return Qt.AlignCenter
        return Qt.AlignLeft | Qt.AlignVCenter

    def _font(self, row: int, col: int) -> QFont | None:
        if col == NavigatorColumns.DECISION and self._dataset_model.get_review_decision(
            row
        ):
            font = QFont()
            font.setBold(True)
            return font
        return None

    def _foreground(self, row: int, col: int) -> QBrush | None:
        if col == NavigatorColumns.DECISION:
            decision = self._dataset_model.get_review_decision(row)
            if decision == "accept":
                return QBrush(QColor("#4caf50"))
            if decision == "reject":
                return QBrush(QColor("#f44336"))
        return None

    def _image_stem(self, row: int) -> str:
        name = self._dataset_model.get_image_filename(row)
        return os.path.splitext(name)[0]

    def _image_path(self, row: int) -> str:
        return self._dataset_model.get_image_path(row)

    def _score(self, row: int) -> float | None:
        if self._inference_model is None:
            return None
        return self._inference_model.get_score(self._image_path(row))


class NavigatorSortProxyModel(QSortFilterProxyModel):
    """Type-aware proxy for navigator column sorting and row filtering."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setDynamicSortFilter(True)
        self.setSortCaseSensitivity(Qt.CaseInsensitive)
        self._decision_filter: set = set()
        self._status_filter: set = set()
        self._class_filter: set = set()
        self._pinned_source_row: int = -1

    def set_pinned_source_row(self, source_row: int) -> None:
        """Exempt *source_row* from filtering so editing the open image can't
        make its own row vanish out from under it. Pass -1 to clear.
        """
        if source_row == self._pinned_source_row:
            return
        self._pinned_source_row = source_row
        self.invalidateFilter()

    def set_decision_filter_active(self, decision: str, active: bool) -> None:
        """decision is "accept" or "reject". Empty set imposes no restriction."""
        if active:
            self._decision_filter.add(decision)
        else:
            self._decision_filter.discard(decision)
        self.invalidateFilter()

    def set_status_filter_active(self, status: str, active: bool) -> None:
        """status is one of STATUS_FILTER_OPTIONS' keys. Empty set imposes no restriction."""
        if active:
            self._status_filter.add(status)
        else:
            self._status_filter.discard(status)
        self.invalidateFilter()

    def set_class_filter_active(self, class_name: str, active: bool) -> None:
        """Empty set imposes no restriction; a non-empty set matches images with ANY of them."""
        if active:
            self._class_filter.add(class_name)
        else:
            self._class_filter.discard(class_name)
        self.invalidateFilter()

    def clear_filters(self) -> None:
        self._decision_filter.clear()
        self._status_filter.clear()
        self._class_filter.clear()
        self.invalidateFilter()

    def decision_filter(self) -> frozenset:
        return frozenset(self._decision_filter)

    def status_filter(self) -> frozenset:
        return frozenset(self._status_filter)

    def class_filter(self) -> frozenset:
        return frozenset(self._class_filter)

    def active_filter_count(self) -> int:
        return (
            len(self._decision_filter)
            + len(self._status_filter)
            + len(self._class_filter)
        )

    def filterAcceptsRow(self, source_row: int, parent: QModelIndex) -> bool:
        if source_row == self._pinned_source_row:
            return True
        if not self._decision_filter and not self._status_filter and not self._class_filter:
            return True
        model = self.sourceModel()
        if model is None:
            return True
        idx = model.index(source_row, 0)

        if self._decision_filter:
            decision = model.data(idx, FILTER_DECISION_ROLE)
            if decision not in self._decision_filter:
                return False

        if self._status_filter:
            state = model.data(idx, IMAGE_STATE_ROLE)
            bucket_match = _STATUS_BUCKET.get(state) in self._status_filter
            conflict_match = (
                "conflicting" in self._status_filter and state == "accept_conflict"
            )
            if not (bucket_match or conflict_match):
                return False

        if self._class_filter:
            row_classes = {name for name, _rgb in model.class_entries(source_row)}
            if not (row_classes & self._class_filter):
                return False

        return True

    def lessThan(self, left: QModelIndex, right: QModelIndex) -> bool:
        model = self.sourceModel()
        if model is None:
            return super().lessThan(left, right)

        col = left.column()
        left_row = left.row()
        right_row = right.row()
        left_value = model.sort_value(left_row, col)
        right_value = model.sort_value(right_row, col)

        if col == NavigatorColumns.SCORE:
            left_missing = left_value is None
            right_missing = right_value is None
            if left_missing != right_missing:
                if self.sortOrder() == Qt.DescendingOrder:
                    return left_missing
                return right_missing

        if left_value == right_value:
            return model.tie_break_value(left_row) < model.tie_break_value(right_row)
        return left_value < right_value
