from PySide6.QtCore import Qt, Signal, QSettings
from PySide6.QtWidgets import QWidget, QFrame, QVBoxLayout, QScrollArea, QSizePolicy

from views.annomate._splitter import StyledSplitter

from views.annomate.sections import (
    _CollapsibleSection,
    DataNavigatorSection,
    ClassesSection,
    AnnotationsSection,
    MetadataSection,
    MicrosentrySection,
)


class RightPanel(QWidget):
    """Scrollable right panel with collapsible sections for the AnnoMate main window.

    Signals:
        image_selected (int): Forwarded from DataNavigatorSection.
        class_selected (str): Forwarded from ClassesSection.
        prev_requested (): Forwarded from DataNavigatorSection.
        next_requested (): Forwarded from DataNavigatorSection.
        load_model_requested (): Forwarded from MicrosentrySection.
        microsentry_settings_changed (): Forwarded from MicrosentrySection.
    """

    image_selected = Signal(int)
    class_selected = Signal(str)
    prev_requested = Signal()
    next_requested = Signal()
    annotation_selected = Signal(int)
    load_model_requested = Signal()
    load_previous_model_requested = Signal()
    microsentry_settings_changed = Signal()
    accept_polygons_requested = Signal()

    def __init__(
        self,
        dataset_model,
        inference_model=None,
        calibration_model=None,
        parent: QWidget = None,
    ) -> None:
        super().__init__(parent)
        # Left border separating the panel from the canvas
        self.setStyleSheet("RightPanel { border-left: 1px solid palette(mid); }")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Microsentry — always visible above the splitter, collapsed by default
        _ms_settings = QSettings("LANL", "AnnoMateMicroSentryAI")
        _ms_expanded = _ms_settings.value("ui/microsentry_expanded", False, type=bool)
        ms_sec = _CollapsibleSection("Microsentry", expanded=_ms_expanded)
        self._ms_section = MicrosentrySection()
        self._ms_section.load_model_requested.connect(self.load_model_requested)
        self._ms_section.load_previous_model_requested.connect(
            self.load_previous_model_requested
        )
        self._ms_section.settings_changed.connect(self.microsentry_settings_changed)
        self._ms_section.accept_polygons_requested.connect(
            self.accept_polygons_requested
        )
        ms_sec.body_layout().addWidget(self._ms_section)
        ms_sec.toggled.connect(
            lambda checked: QSettings("LANL", "AnnoMateMicroSentryAI").setValue(
                "ui/microsentry_expanded", checked
            )
        )
        outer.addWidget(ms_sec)
        self._ms_collapsible = ms_sec

        # Splitter: Dataset Navigator (top, resizable) | rest (bottom, scrollable)
        splitter = StyledSplitter(Qt.Vertical)
        splitter.setHandleWidth(8)
        splitter.setChildrenCollapsible(False)

        nav_sec = _CollapsibleSection("Dataset Navigator", expandable=True)
        self.navigator = DataNavigatorSection(dataset_model, inference_model)
        self.navigator.set_microsentry_mode(True)
        self.navigator.image_selected.connect(self.image_selected)
        self.navigator.prev_requested.connect(self.prev_requested)
        self.navigator.next_requested.connect(self.next_requested)
        nav_sec.body_layout().addWidget(self.navigator)

        nav_frame = QFrame()
        nav_frame.setFrameShape(QFrame.StyledPanel)
        nav_frame.setFrameShadow(QFrame.Sunken)
        nav_frame.setMinimumHeight(60)
        nav_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        nav_fl = QVBoxLayout(nav_frame)
        nav_fl.setContentsMargins(0, 0, 0, 0)
        nav_fl.setSpacing(0)
        nav_fl.addWidget(nav_sec)
        splitter.addWidget(nav_frame)
        self._nav_frame = nav_frame
        nav_sec.toggled.connect(self._on_navigator_toggled)

        bottom_scroll = QScrollArea()
        bottom_scroll.setWidgetResizable(True)
        bottom_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        bottom_scroll.setFrameShape(QFrame.StyledPanel)
        bottom_scroll.setFrameShadow(QFrame.Sunken)
        bottom_content = QWidget()
        cl = QVBoxLayout(bottom_content)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(0)

        classes_sec = _CollapsibleSection("Annotation Classes")
        self.classes = ClassesSection(dataset_model)
        self.classes.class_selected.connect(self.class_selected)
        classes_sec.body_layout().addWidget(self.classes)
        cl.addWidget(classes_sec)

        annos_sec = _CollapsibleSection("Current Image Annotations")
        self.annotations = AnnotationsSection(dataset_model, calibration_model)
        self.annotations.annotation_selected.connect(self.annotation_selected)
        annos_sec.body_layout().addWidget(self.annotations)
        cl.addWidget(annos_sec)

        meta_sec = _CollapsibleSection("Inspector/Notes")
        self.metadata = MetadataSection(dataset_model)
        meta_sec.body_layout().addWidget(self.metadata)
        cl.addWidget(meta_sec)

        cl.addStretch()
        bottom_scroll.setWidget(bottom_content)
        splitter.addWidget(bottom_scroll)

        self._splitter = splitter
        self._dataset_model = dataset_model
        splitter.setSizes([20, 560])
        outer.addWidget(splitter, stretch=1)

        dataset_model.modelReset.connect(self._on_model_reset)

    def _on_model_reset(self) -> None:
        if self._dataset_model.rowCount() > 0:
            self._splitter.setSizes([220, 400])
        else:
            self._splitter.setSizes([20, 560])

    def _on_navigator_toggled(self, expanded: bool) -> None:
        if expanded:
            self._nav_frame.setMinimumHeight(60)
            self._splitter.setSizes([220, 400])
        else:
            self._nav_frame.setMinimumHeight(0)
            self._splitter.setSizes([34, 10000])

    def select_row(self, row: int) -> None:
        """Silently highlight *row* in the navigator list."""
        self.navigator.select_row(row)

    def set_counter(self, current: int, total: int) -> None:
        """Update the image position counter in the navigator."""
        self.navigator.set_counter(current, total)

    def navigator_adjacent_source_row(self, current_source_row: int, step: int) -> int:
        """Return the navigator-adjacent source row in current visible order."""
        return self.navigator.adjacent_source_row(current_source_row, step)

    def set_current_row(self, row: int) -> None:
        """Update per-image counts, annotations list, and metadata for the new image."""
        self.classes.set_current_row(row)
        self.annotations.set_current_row(row)
        self.metadata.set_current_row(row)

    # ------------------------------------------------------------------ #
    # Microsentry pass-throughs
    # ------------------------------------------------------------------ #

    def set_model_loaded(self, name: str, path: str = "") -> None:
        self._ms_section.set_model_loaded(name, path)

    def set_no_model(self) -> None:
        self._ms_section.set_no_model()

    def get_microsentry_settings(self) -> dict:
        return self._ms_section.get_settings()

    def navigator_set_inference(self, row: int, score: float, label: str) -> None:
        self.navigator.set_row_inference(row, score, label)

    def navigator_set_microsentry_mode(self, enabled: bool) -> None:
        self.navigator.set_microsentry_mode(enabled)
