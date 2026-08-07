"""
RightPanel — VS Code-style activity bar for the AnnoMate main window.

A vertical icon rail (always visible, tooltip per tab) sits at the outer
edge of the window; clicking a tab swaps the content pane to that tab's
page and expands the panel. Clicking the active tab again collapses the
panel back down to just the rail, so the icons stay reachable without the
panel taking up canvas space.

Only "Annotation Classes" and "Microsentry" have real pages today — the
rest are icon-only placeholders for settings that currently still live in
the tool palette / viewport floating bar (see the redesign plan) and will
migrate in as separate follow-up steps.
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
    ClassesSection,
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

# Tabs with no page built yet -- icon-only stubs standing in for settings
# that will migrate in from the tool palette / viewport floating bar.
_PLACEHOLDER_TABS = (
    ("active_tool", "draw", "Active Tool"),
    ("measurement", "straighten", "Measurement & Calibration"),
    ("constraints", "warning", "Constraints"),
    ("overlays", "crop_free", "View Overlays"),
)


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

    def __init__(
        self,
        dataset_model,
        inference_model=None,
        parent: QWidget = None,
    ) -> None:
        super().__init__(parent)
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

        self._add_tab(*_PLACEHOLDER_TABS[0])

        # ---- Classes / Dataset Setup tab -- one collapsible section per
        # feature, so future dataset-setup features can join "Annotation
        # Classes" here without disturbing it. ----
        self.classes = ClassesSection(dataset_model)
        self.classes.class_selected.connect(self.class_selected)
        self.classes.annotation_mode_changed.connect(self.annotation_mode_changed)
        classes_section = _CollapsibleSection("Annotation Classes", expanded=True)
        classes_section.body_layout().setContentsMargins(0, 0, 0, 4)
        classes_section.body_layout().addWidget(self.classes)
        classes_page = _stack_sections([classes_section])
        self._add_tab("classes", "label", "Annotation Classes", classes_page)

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
        ms_section.body_layout().setContentsMargins(0, 0, 0, 4)
        ms_section.body_layout().addWidget(self._ms_section)
        ms_section.toggled.connect(
            lambda checked: QSettings(_SETTINGS_ORG, _SETTINGS_APP).setValue(
                "ui/microsentry_expanded", checked
            )
        )
        ms_page = _stack_sections([ms_section])
        self._add_tab("microsentry", "auto_awesome", "Microsentry AI", ms_page)

        for key, icon_name, title in _PLACEHOLDER_TABS[1:]:
            self._add_tab(key, icon_name, title)

        settings = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        last_tab = settings.value(_SETTINGS_TAB_KEY, "classes", type=str)
        start_collapsed = settings.value(_SETTINGS_COLLAPSED_KEY, False, type=bool)
        if last_tab not in self._page_index:
            last_tab = "classes"
        self._show_tab(last_tab)
        self.set_collapsed(start_collapsed)

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
        if key:
            self._show_tab(key)
            self.set_collapsed(False)
            QSettings(_SETTINGS_ORG, _SETTINGS_APP).setValue(_SETTINGS_TAB_KEY, key)
        else:
            self.set_collapsed(True)

    def _show_tab(self, key: str) -> None:
        self._stack.setCurrentIndex(self._page_index[key])
        self._rail.set_active(key)

    def set_collapsed(self, collapsed: bool) -> None:
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
        QSettings(_SETTINGS_ORG, _SETTINGS_APP).setValue(
            _SETTINGS_COLLAPSED_KEY, collapsed
        )
        self.collapsed_changed.emit(collapsed)

    def is_collapsed(self) -> bool:
        return self._collapsed

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
        return self._rail.button("classes")

    def microsentry_header(self) -> QWidget:
        return self._rail.button("microsentry")

    # ------------------------------------------------------------------ #
    # Annotation mode
    # ------------------------------------------------------------------ #

    def set_annotation_mode(self, mode: str) -> None:
        """Sync the mode button state without re-emitting the signal."""
        self.classes.set_annotation_mode(mode)
