import pytest
from PySide6.QtCore import QSettings

from core.states.dataset_state import DatasetState
from models.dataset_model import DatasetTableModel
from views.annomate import right_panel as right_panel_module
from views.annomate.right_panel import RightPanel

_TEST_ORG = "AnnoMateTestOrg"
_TEST_APP = "RightPanelTest"


@pytest.fixture
def isolated_settings(monkeypatch):
    """A QSettings store scoped away from the developer's real app settings."""
    monkeypatch.setattr(right_panel_module, "_SETTINGS_ORG", _TEST_ORG)
    monkeypatch.setattr(right_panel_module, "_SETTINGS_APP", _TEST_APP)
    settings = QSettings(_TEST_ORG, _TEST_APP)
    settings.clear()
    yield settings
    settings.clear()


def _make_panel(qtbot):
    dataset_model = DatasetTableModel(DatasetState())
    panel = RightPanel(dataset_model)
    qtbot.addWidget(panel)
    return panel


def test_collapsed_at_construction_with_no_prior_settings(qtbot, isolated_settings):
    """Verify the right panel starts collapsed when no project has been opened yet.

    RightPanel is always constructed before any project loads (new or
    existing), so "no project open" and "just constructed" are the same
    moment -- it should always come up collapsed here.
    """
    panel = _make_panel(qtbot)

    assert panel.is_collapsed() is True


def test_collapsed_at_construction_even_if_previously_left_expanded(
    qtbot, isolated_settings
):
    """Verify a prior session's expanded state does NOT carry over to the next launch.

    Only the last active *tab* is remembered across sessions -- whether the
    panel was left open or closed is not, since it must always start
    collapsed until the user (or a new project) explicitly opens it.
    """
    isolated_settings.setValue("ui/right_panel_active_tab", "microsentry")

    panel = _make_panel(qtbot)

    assert panel.is_collapsed() is True
    # The remembered tab still applies once the user does open it themselves.
    assert panel._stack.currentIndex() == panel._page_index["microsentry"]


def test_user_can_uncollapse_by_clicking_a_tab(qtbot, isolated_settings):
    """Verify clicking a rail tab is what uncollapses the panel."""
    panel = _make_panel(qtbot)
    assert panel.is_collapsed() is True

    panel._rail.button("classes").click()

    assert panel.is_collapsed() is False
    assert panel._stack.currentIndex() == panel._page_index["classes"]


def test_show_dataset_setup_forces_classes_tab_open(qtbot, isolated_settings):
    """Verify show_dataset_setup() opens the Dataset Setup tab regardless of current state.

    Simulates starting a new project right after launch, while the panel
    is still collapsed -- it must switch to Classes and expand.
    """
    panel = _make_panel(qtbot)
    assert panel.is_collapsed() is True

    panel.show_dataset_setup()

    assert panel.is_collapsed() is False
    assert panel._stack.currentIndex() == panel._page_index["classes"]


def test_construction_alone_never_restores_a_prior_expanded_state(
    qtbot, isolated_settings
):
    """Verify RightPanel construction by itself always stays collapsed.

    Persisted state (tab + expanded) only gets reapplied via the explicit
    restore_last_state() call below -- not automatically at construction,
    since construction always happens before any project is loaded.
    """
    panel = _make_panel(qtbot)
    panel.show_dataset_setup()  # leaves it expanded on "classes" for this session

    assert isolated_settings.value("ui/right_panel_active_tab") == "classes"
    assert isolated_settings.value("ui/right_panel_collapsed", type=bool) is False

    next_session_panel = _make_panel(qtbot)
    assert next_session_panel.is_collapsed() is True


def test_restore_last_state_reopens_a_previously_expanded_panel(
    qtbot, isolated_settings
):
    """Verify opening an existing project resumes an expanded panel, not collapsed.

    Reproduces the reported bug: open a project, expand the panel (e.g. on
    Active Tool), close and relaunch the app, then open a project again --
    it must come back expanded on that same tab, not stuck collapsed.
    """
    isolated_settings.setValue("ui/right_panel_active_tab", "active_tool")
    isolated_settings.setValue("ui/right_panel_collapsed", False)

    panel = _make_panel(qtbot)
    assert panel.is_collapsed() is True  # still collapsed right after construction

    panel.restore_last_state()

    assert panel.is_collapsed() is False
    assert panel._stack.currentIndex() == panel._page_index["active_tool"]
    assert panel._rail.button("active_tool").isChecked()


def test_restore_last_state_stays_collapsed_if_that_was_last_left(
    qtbot, isolated_settings
):
    """Verify restore_last_state() keeps the panel collapsed if the user last left it that way.

    Also guards against the rail/collapse-desync bug: the pre-selected tab
    must not be marked "active" on the rail while collapsed, or the user's
    first click on it would misread as re-collapsing an already-open tab.
    """
    isolated_settings.setValue("ui/right_panel_active_tab", "microsentry")
    isolated_settings.setValue("ui/right_panel_collapsed", True)

    panel = _make_panel(qtbot)
    panel.restore_last_state()

    assert panel.is_collapsed() is True
    assert panel._stack.currentIndex() == panel._page_index["microsentry"]

    panel._rail.button("microsentry").click()
    assert panel.is_collapsed() is False
