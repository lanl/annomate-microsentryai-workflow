from PySide6.QtCore import Qt, Signal, QSettings
from PySide6.QtWidgets import QWidget, QFrame, QVBoxLayout, QScrollArea

from views.annomate.sections import (
    _CollapsibleSection,
    ClassesSection,
    MicrosentrySection,
)


class RightPanel(QWidget):
    """Scrollable right panel with collapsible sections for the AnnoMate main window.

    Signals:
        class_selected (str): Forwarded from ClassesSection.
        load_model_requested (): Forwarded from MicrosentrySection.
        microsentry_settings_changed (): Forwarded from MicrosentrySection.
    """

    class_selected = Signal(str)
    load_model_requested = Signal()
    load_previous_model_requested = Signal()
    microsentry_settings_changed = Signal()
    accept_polygons_requested = Signal()
    annotation_mode_changed = Signal(str)  # "pixel" | "image_level"

    def __init__(
        self,
        dataset_model,
        inference_model=None,
        parent: QWidget = None,
    ) -> None:
        super().__init__(parent)
        # Left border separating the panel from the canvas
        self.setStyleSheet("RightPanel { border-left: 1px solid palette(mid); }")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Microsentry — always visible above the scroll area, collapsed by default
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
        self._classes_collapsible = classes_sec
        self.classes = ClassesSection(dataset_model)
        self.classes.class_selected.connect(self.class_selected)
        self.classes.annotation_mode_changed.connect(self.annotation_mode_changed)
        classes_sec.body_layout().setContentsMargins(8, 0, 8, 8)
        classes_sec.body_layout().addWidget(self.classes)
        cl.addWidget(classes_sec)

        cl.addStretch()
        bottom_scroll.setWidget(bottom_content)
        outer.addWidget(bottom_scroll, stretch=1)

    def set_current_row(self, row: int) -> None:
        """Update the annotation-classes section for the new image."""
        self.classes.set_current_row(row)

    # ------------------------------------------------------------------ #
    # Microsentry pass-throughs
    # ------------------------------------------------------------------ #

    def set_project_saved(self, has_project: bool) -> None:
        self._ms_section.set_project_saved(has_project)

    def set_model_loaded(self, name: str, path: str = "") -> None:
        self._ms_section.set_model_loaded(name, path)

    def set_scoremaps_loaded(self) -> None:
        self._ms_section.set_scoremaps_loaded()

    def set_no_model(self) -> None:
        self._ms_section.set_no_model()

    def get_microsentry_settings(self) -> dict:
        return self._ms_section.get_settings()

    # ------------------------------------------------------------------ #
    # Section header accessors (for tour/onboarding targeting)
    # ------------------------------------------------------------------ #

    def classes_header(self) -> QWidget:
        return self._classes_collapsible.header_widget()

    def microsentry_header(self) -> QWidget:
        return self._ms_collapsible.header_widget()

    # ------------------------------------------------------------------ #
    # Annotation mode
    # ------------------------------------------------------------------ #

    def set_annotation_mode(self, mode: str) -> None:
        """Sync the mode button state without re-emitting the signal."""
        self.classes.set_annotation_mode(mode)
