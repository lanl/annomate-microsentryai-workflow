"""
RightPanel — VS Code-style activity bar for the AnnoMate main window.

A vertical icon rail (always visible, tooltip per tab) sits at the outer
edge of the window; clicking a tab swaps the content pane to that tab's
page and expands the panel. Clicking the active tab again collapses the
panel back down to just the rail, so the icons stay reachable without the
panel taking up canvas space.

Four tabs total: Active Tool, Dataset Setup, Microsentry, and View
Overlays. View Overlays' three sections -- Center Crop, Grid, and Anomaly
Constraints -- have all migrated in from the viewport floating bar.

The panel always starts collapsed at construction (no project is loaded
yet at that point). Two explicit calls decide what happens once one is:
show_dataset_setup() forces Dataset Setup open for a brand-new project;
restore_last_state() reapplies whatever tab/expanded state was persisted
via QSettings the last time a project was open, for everything else
(opening an existing project or image folder).
"""

from PySide6.QtCore import QSettings, QSize, Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from views.annomate.sections import (
    ActiveToolSection,
    AnomalyConstraintsSection,
    CenterCropSection,
    ClassesSection,
    GridSection,
    MicrosentrySection,
    _CollapsibleSection,
)
from views.icons import material_icon

_RAIL_WIDTH = 48
_TAB_BTN_SIZE = 40
_TAB_ICON_SIZE = 20
_EXPANDED_MIN_WIDTH = 220

_SETTINGS_ORG = "LANL"
_SETTINGS_APP = "AnnoMateMicroSentryAI"
_SETTINGS_TAB_KEY = "ui/right_panel_active_tab"
_SETTINGS_COLLAPSED_KEY = "ui/right_panel_collapsed"


def _stack_sections(sections: list) -> QWidget:
    """Stack one or more independently-collapsible sections into a tab page.

    Each activity-bar tab groups features by idea (e.g. every AI capability
    lives under the same tab); this keeps each feature collapsible on its
    own so more can be added to a tab later without disturbing the rest.
    """
    page = QWidget()
    layout = QVBoxLayout(page)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)
    for section in sections:
        layout.addWidget(section)
    layout.addStretch()
    return page


def _scroll_wrap(widget: QWidget) -> QScrollArea:
    padded = QWidget()
    padded_layout = QVBoxLayout(padded)
    padded_layout.setContentsMargins(8, 8, 8, 8)
    padded_layout.addWidget(widget)

    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    scroll.setFrameShape(QFrame.NoFrame)
    scroll.setWidget(padded)
    return scroll


class _PlaceholderPage(QWidget):
    """Stand-in body for an activity-bar tab whose settings haven't moved in yet."""

    def __init__(self, title: str, icon_name: str, parent: QWidget = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)

        icon_lbl = QLabel()
        icon_lbl.setPixmap(material_icon(icon_name, size=32, color="grey").pixmap(32, 32))
        icon_lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(icon_lbl)

        text_lbl = QLabel(f"{title}\nComing soon")
        text_lbl.setAlignment(Qt.AlignCenter)
        text_lbl.setStyleSheet("color: grey;")
        layout.addWidget(text_lbl)




class _ActivityRail(QFrame):
    """Fixed-width vertical icon rail; one checkable tab button per page.

    Signals:
        tab_clicked (str): Emitted with the tab key, or "" when the
            already-active tab was clicked again (collapse request).
    """

    tab_clicked = Signal(str)

    def __init__(self, parent: QWidget = None) -> None:
        super().__init__(parent)
        self.setFixedWidth(_RAIL_WIDTH)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(4, 6, 4, 6)
        self._layout.setSpacing(2)
        self._layout.setAlignment(Qt.AlignTop)

        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._buttons: dict[str, QToolButton] = {}
        self._active_key = ""

    def add_tab(self, key: str, icon_name: str, tooltip: str) -> QToolButton:
        btn = QToolButton()
        btn.setIcon(material_icon(icon_name, size=_TAB_ICON_SIZE, color="black"))
        btn.setIconSize(QSize(_TAB_ICON_SIZE, _TAB_ICON_SIZE))
        btn.setToolTip(tooltip)
        btn.setCheckable(True)
        btn.setAutoRaise(True)
        btn.setFixedSize(_TAB_BTN_SIZE, _TAB_BTN_SIZE)
        btn.clicked.connect(lambda: self._on_clicked(key))
        self._group.addButton(btn)
        self._layout.addWidget(btn)
        self._buttons[key] = btn
        return btn

    def set_active(self, key: str) -> None:
        """Sync the checked button without emitting tab_clicked."""
        self._active_key = key
        self._group.setExclusive(False)
        for btn_key, btn in self._buttons.items():
            btn.setChecked(btn_key == key)
        self._group.setExclusive(True)

    def button(self, key: str) -> QToolButton:
        return self._buttons[key]

    def _on_clicked(self, key: str) -> None:
        if self._active_key == key:
            # Clicking the active tab again collapses the panel. QButtonGroup
            # won't let a click uncheck the already-checked button, so force it.
            self._group.setExclusive(False)
            self._buttons[key].setChecked(False)
            self._group.setExclusive(True)
            self._active_key = ""
        else:
            self._active_key = key
        self.tab_clicked.emit(self._active_key)


class RightPanel(QWidget):
    """Activity-bar-style right panel for the AnnoMate main window.

    Signals:
        class_selected (str): Forwarded from ClassesSection.
        load_model_requested (): Forwarded from MicrosentrySection.
        microsentry_settings_changed (): Forwarded from MicrosentrySection.
        collapsed_changed (bool): Emitted after the panel expands/collapses.
    """

    class_selected = Signal(str)
    load_model_requested = Signal()
    load_previous_model_requested = Signal()
    microsentry_settings_changed = Signal()
    accept_polygons_requested = Signal()
    annotation_mode_changed = Signal(str)  # "pixel" | "image_level"
    collapsed_changed = Signal(bool)
    thickness_changed = Signal(float)
    sam_variant_changed = Signal(str)
    crop_overlay_toggled = Signal(bool)
    center_calibration_started = Signal()
    center_calibration_accepted = Signal()
    center_template_cleared = Signal()
    center_template_import_requested = Signal(str)

    def __init__(
        self,
        dataset_model,
        inference_model=None,
        canvas=None,
        center_template_model=None,
        calibration_model=None,
        anomaly_constraint_model=None,
        parent: QWidget = None,
    ) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet("RightPanel { border-left: 1px solid palette(mid); }")

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._stack = QStackedWidget()
        outer.addWidget(self._stack, stretch=1)

        rail_divider = QFrame()
        rail_divider.setFrameShape(QFrame.VLine)
        rail_divider.setFrameShadow(QFrame.Sunken)
        outer.addWidget(rail_divider)

        self._rail = _ActivityRail()
        self._rail.tab_clicked.connect(self._on_tab_clicked)
        outer.addWidget(self._rail)

        self._page_index: dict[str, int] = {}
        self._collapsed = False

        # ---- Active Tool tab -- Common (stroke width) plus a per-tool
        # section that swaps its body to match whichever tool is selected;
        # new tools register their own settings widget the same way SAM did. ----
        self.active_tool = ActiveToolSection()
        self.active_tool.thickness_changed.connect(self.thickness_changed)
        self.active_tool.sam_variant_changed.connect(self.sam_variant_changed)
        self._add_tab("active_tool", "draw", "Active Tool", self.active_tool)

        # ---- Dataset Setup tab -- one collapsible section per feature, so
        # future dataset-setup features can join "Annotation Classes" here
        # without disturbing it. ----
        self.classes = ClassesSection(dataset_model)
        self.classes.class_selected.connect(self.class_selected)
        self.classes.annotation_mode_changed.connect(self.annotation_mode_changed)
        classes_section = _CollapsibleSection("Annotation Classes", expanded=True)
        classes_section.body_layout().setContentsMargins(0, 4, 0, 0)
        classes_section.body_layout().addWidget(self.classes)
        classes_page = _stack_sections([classes_section])
        self._add_tab("classes", "data_table", "Dataset Setup", classes_page)

        # ---- AI / Microsentry tab -- same idea: current AI capabilities
        # (Microsentry) and any future ones each get their own collapsible
        # section stacked in this one tab. ----
        self._ms_section = MicrosentrySection()
        self._ms_section.load_model_requested.connect(self.load_model_requested)
        self._ms_section.load_previous_model_requested.connect(
            self.load_previous_model_requested
        )
        self._ms_section.settings_changed.connect(self.microsentry_settings_changed)
        self._ms_section.accept_polygons_requested.connect(
            self.accept_polygons_requested
        )
        _ms_settings = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        _ms_expanded = _ms_settings.value("ui/microsentry_expanded", False, type=bool)
        ms_section = _CollapsibleSection("Microsentry", expanded=_ms_expanded)
        ms_section.body_layout().setContentsMargins(0, 4, 0, 0)
        ms_section.body_layout().addWidget(self._ms_section)
        ms_section.toggled.connect(
            lambda checked: QSettings(_SETTINGS_ORG, _SETTINGS_APP).setValue(
                "ui/microsentry_expanded", checked
            )
        )
        ms_page = _stack_sections([ms_section])
        self._add_tab("microsentry", "auto_awesome", "Microsentry AI", ms_page)

        # ---- View Overlays tab -- Center Crop, Grid, and Anomaly
        # Constraints have all migrated in from the viewport floating bar. ----
        self.center_crop = CenterCropSection(canvas, center_template_model)
        self.center_crop.crop_overlay_toggled.connect(self.crop_overlay_toggled)
        self.center_crop.center_calibration_started.connect(
            self.center_calibration_started
        )
        self.center_crop.center_calibration_accepted.connect(
            self.center_calibration_accepted
        )
        self.center_crop.center_template_cleared.connect(
            self.center_template_cleared
        )
        self.center_crop.center_template_import_requested.connect(
            self.center_template_import_requested
        )
        center_crop_section = _CollapsibleSection("Center Crop", expanded=False)
        center_crop_section.body_layout().setContentsMargins(0, 4, 0, 0)
        center_crop_section.body_layout().addWidget(self.center_crop)

        self.grid = GridSection(calibration_model)
        grid_section = _CollapsibleSection("Grid", expanded=False)
        grid_section.body_layout().setContentsMargins(0, 4, 0, 0)
        grid_section.body_layout().addWidget(self.grid)

        self.anomaly = AnomalyConstraintsSection(anomaly_constraint_model)
        anomaly_section = _CollapsibleSection("Anomaly Constraints", expanded=False)
        anomaly_section.body_layout().setContentsMargins(0, 4, 0, 0)
        anomaly_section.body_layout().addWidget(self.anomaly)

        overlay_sections = [center_crop_section, grid_section, anomaly_section]
        overlays_page = _stack_sections(overlay_sections)
        self._add_tab("overlays", "layers", "View Overlays", overlays_page)

        settings = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        last_tab = settings.value(_SETTINGS_TAB_KEY, "classes", type=str)
        if last_tab not in self._page_index:
            last_tab = "classes"
        # Pre-select the remembered tab's content without marking the rail
        # active, so the first click on that same tab correctly expands it
        # instead of reading as "click the already-active tab" (collapse).
        self._set_stack_page(last_tab)
        # Always collapsed until a project is opened -- no project is loaded
        # yet at construction time, so there's nothing to show. New Project
        # opens straight to Dataset Setup (show_dataset_setup()); otherwise
        # it stays collapsed until the user clicks a tab themselves.
        self.set_collapsed(True)

    # ------------------------------------------------------------------ #
    # Tab construction
    # ------------------------------------------------------------------ #

    def _add_tab(
        self, key: str, icon_name: str, tooltip: str, content: QWidget = None
    ) -> None:
        page = content if content is not None else _PlaceholderPage(tooltip, icon_name)
        index = self._stack.addWidget(_scroll_wrap(page))
        self._page_index[key] = index
        self._rail.add_tab(key, icon_name, tooltip)

    # ------------------------------------------------------------------ #
    # Expand / collapse
    # ------------------------------------------------------------------ #

    def _on_tab_clicked(self, key: str) -> None:
        """Handle a real user-driven (or explicitly forced) tab change.

        This is the only place that persists collapsed state -- set_collapsed()
        itself doesn't, so the constructor's mandatory initial collapse can't
        clobber whatever a prior session had saved before restore_last_state()
        gets a chance to read it.
        """
        settings = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        if key:
            self._show_tab(key)
            self.set_collapsed(False)
            settings.setValue(_SETTINGS_TAB_KEY, key)
        else:
            self.set_collapsed(True)
        settings.setValue(_SETTINGS_COLLAPSED_KEY, self._collapsed)

    def _show_tab(self, key: str) -> None:
        self._set_stack_page(key)
        self._rail.set_active(key)

    def _set_stack_page(self, key: str) -> None:
        """Pre-select *key*'s page in the stack without marking its rail
        button active -- used at construction so the panel has something
        ready to show without the rail looking like it's already expanded."""
        self._stack.setCurrentIndex(self._page_index[key])

    def set_collapsed(self, collapsed: bool) -> None:
        """Apply expand/collapse to the UI only -- does not persist.

        Callers that represent a real state change the user should get back
        next time (a tab click, restore_last_state(), show_dataset_setup())
        persist explicitly; the constructor's mandatory initial collapse must
        NOT overwrite whatever was saved from a prior session.
        """
        if self._collapsed == collapsed:
            return
        self._collapsed = collapsed
        self._stack.setVisible(not collapsed)
        if collapsed:
            self.setFixedWidth(_RAIL_WIDTH)
            self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        else:
            self.setMinimumWidth(_EXPANDED_MIN_WIDTH)
            self.setMaximumWidth(16777215)
            self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.collapsed_changed.emit(collapsed)

    def is_collapsed(self) -> bool:
        return self._collapsed

    def show_dataset_setup(self) -> None:
        """Force the Dataset Setup tab open -- called when starting a new project."""
        self._on_tab_clicked("classes")

    def restore_last_state(self) -> None:
        """Restore the last tab/expanded state -- called when an existing
        project or image folder is opened (a brand-new project instead
        calls show_dataset_setup(), which forces Dataset Setup open)."""
        settings = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        last_tab = settings.value(_SETTINGS_TAB_KEY, "classes", type=str)
        if last_tab not in self._page_index:
            last_tab = "classes"
        collapsed = settings.value(_SETTINGS_COLLAPSED_KEY, True, type=bool)
        if collapsed:
            self._set_stack_page(last_tab)
            self.set_collapsed(True)
        else:
            self._show_tab(last_tab)
            self.set_collapsed(False)

    def set_current_row(self, row: int) -> None:
        """Update the annotation-classes section for the new image."""
        self.classes.set_current_row(row)

    # ------------------------------------------------------------------ #
    # Active Tool pass-throughs
    # ------------------------------------------------------------------ #

    def set_active_tool(self, tool_key: str) -> None:
        """Update the Active Tool tab to show *tool_key*'s settings ("" = none)."""
        self.active_tool.set_active_tool(tool_key)

    def set_thickness(self, value: float) -> None:
        self.active_tool.set_thickness(value)

    def current_sam_variant(self) -> str:
        return self.active_tool.current_sam_variant()

    def sam_variant_display_name(self) -> str:
        return self.active_tool.sam_variant_display_name()

    def set_sam_status(self, text: str, color: str = "grey", italic: bool = True) -> None:
        self.active_tool.set_sam_status(text, color, italic)

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
        return self._rail.button("classes")

    def microsentry_header(self) -> QWidget:
        return self._rail.button("microsentry")

    # ------------------------------------------------------------------ #
    # Annotation mode
    # ------------------------------------------------------------------ #

    def set_annotation_mode(self, mode: str) -> None:
        """Sync the mode button state without re-emitting the signal."""
        self.classes.set_annotation_mode(mode)
