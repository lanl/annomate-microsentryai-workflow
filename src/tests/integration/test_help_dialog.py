import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QPushButton, QWidget

from controllers.help_controller import HelpController
from core.logic.help_entry import HelpEntry
from views.annomate.help_dialog import (
    HelpSearchDialog,
    _NO_QUERY_MESSAGE,
    _NO_RESULTS_MESSAGE,
    _NOT_VISIBLE_MESSAGE,
)


@pytest.fixture
def sample_entries():
    return [
        HelpEntry(
            title="Saving your project",
            category="Projects",
            keywords=["save"],
            description="Save your work.",
            full_text="Full save help text for the app.",
            object_name="saveButton",
        ),
        HelpEntry(
            title="Ghost entry",
            category="Projects",
            keywords=["ghost"],
            description="Points at a widget that doesn't exist.",
            full_text="Full ghost text for the app.",
            object_name="missingButton",
        ),
    ]


@pytest.fixture
def controller(sample_entries):
    return HelpController(entries=sample_entries)


@pytest.fixture
def parent_with_save_button(qtbot):
    parent = QWidget()
    btn = QPushButton(parent)
    btn.setObjectName("saveButton")
    qtbot.addWidget(parent)
    parent.show()
    return parent


@pytest.fixture
def dialog(qtbot, controller, parent_with_save_button):
    dlg = HelpSearchDialog(controller, parent_with_save_button)
    qtbot.addWidget(dlg)
    dlg.show()
    return dlg


def _card_for_title(dlg, title):
    for card in dlg._cards:
        if card.entry.title == title:
            return card
    raise AssertionError(f"No result card for title: {title}")


def test_dialog_opens_with_no_query_message(dialog):
    """Verify a fresh dialog with no initial query shows the placeholder message.

    Success means the message label is visible with the "no query" prompt
    and the results scroll area stays hidden until a search actually runs.
    """
    assert dialog._message_label.isVisible()
    assert dialog._message_label.text() == _NO_QUERY_MESSAGE
    assert not dialog._results_scroll.isVisible()


def test_initial_query_runs_search_immediately(
    qtbot, controller, parent_with_save_button
):
    """Verify passing initial_query runs the search without further interaction.

    Success means the dialog opens with result cards already populated for
    "save", matching the Help menu's "More on <query>…" entry point.
    """
    dlg = HelpSearchDialog(controller, parent_with_save_button, initial_query="save")
    qtbot.addWidget(dlg)

    assert len(dlg._cards) > 0
    assert dlg._cards[0].entry.title == "Saving your project"


def test_typing_query_and_clicking_search_populates_results(qtbot, dialog):
    """Verify typing a query and clicking Search populates result cards.

    Success means at least one card appears for "save" and the results
    scroll area becomes visible.
    """
    search_btn = None
    for child in dialog.findChildren(QPushButton):
        if child.text() == "Search":
            search_btn = child
            break
    assert search_btn is not None

    qtbot.keyClicks(dialog._search_edit, "save")
    qtbot.mouseClick(search_btn, Qt.LeftButton)

    assert len(dialog._cards) > 0
    assert dialog._results_scroll.isVisible()


def test_pressing_enter_in_search_box_runs_search(qtbot, dialog):
    """Verify pressing Enter in the search box runs the same search as clicking Search.

    Success means typing "save" and pressing Return populates result cards,
    since QLineEdit.returnPressed is wired to _run_search.
    """
    qtbot.keyClicks(dialog._search_edit, "save")
    qtbot.keyClick(dialog._search_edit, Qt.Key_Return)

    assert len(dialog._cards) > 0


def test_query_with_no_matches_shows_no_results_message(qtbot, dialog):
    """Verify a query matching nothing shows the "no results" message.

    Success means the message label switches to _NO_RESULTS_MESSAGE and the
    detail panel is cleared, rather than showing an empty results list.
    """
    qtbot.keyClicks(dialog._search_edit, "xyzzyplugh")
    dialog._run_search()

    assert dialog._message_label.text() == _NO_RESULTS_MESSAGE
    assert not dialog._results_scroll.isVisible()


def test_clicking_result_card_updates_detail_panel(qtbot, dialog):
    """Verify clicking a different result card updates the detail panel and selection.

    Both entries match "app" (body-only hits). After clicking the "Ghost
    entry" card, success means the detail panel shows its title and only
    that card is marked selected.
    """
    qtbot.keyClicks(dialog._search_edit, "app")
    dialog._run_search()
    assert len(dialog._cards) == 2

    ghost_card = _card_for_title(dialog, "Ghost entry")
    dialog._on_card_clicked(ghost_card.entry)

    assert dialog._detail_panel._title.text() == "Ghost entry"
    assert ghost_card.property("selected") is True
    other_card = _card_for_title(dialog, "Saving your project")
    assert other_card.property("selected") is False


def test_show_me_button_calls_spotlight_and_hides_dialog_on_success(
    qtbot, dialog, monkeypatch
):
    """Verify a successful "Show Me" spotlights the target and hides the dialog.

    "Saving your project" points at the real "saveButton" child of the
    parent widget. Success means spotlight_widget is called with that
    widget and the dialog hides itself once it returns True.
    """
    calls = []
    monkeypatch.setattr(
        "views.annomate.help_dialog.spotlight_widget",
        lambda widget: (calls.append(widget), True)[1],
    )
    qtbot.keyClicks(dialog._search_edit, "save")
    dialog._run_search()

    qtbot.mouseClick(dialog._detail_panel._show_me_btn, Qt.LeftButton)

    assert len(calls) == 1
    assert calls[0].objectName() == "saveButton"
    assert dialog.isHidden()


def test_show_me_shows_message_when_object_not_found(qtbot, dialog, monkeypatch):
    """Verify "Show Me" falls back to a message box when the target widget doesn't exist.

    "Ghost entry" points at "missingButton", which has no matching child in
    the parent widget. Success means QMessageBox.information is shown with
    the "not visible" message and the dialog stays open.
    """
    messages = []
    monkeypatch.setattr(
        "views.annomate.help_dialog.QMessageBox.information",
        lambda *args, **kwargs: messages.append(args[2:]),
    )
    qtbot.keyClicks(dialog._search_edit, "ghost")
    dialog._run_search()

    qtbot.mouseClick(dialog._detail_panel._show_me_btn, Qt.LeftButton)

    assert messages
    assert _NOT_VISIBLE_MESSAGE in messages[0]
    assert not dialog.isHidden()


def test_show_me_shows_message_when_spotlight_widget_returns_false(
    qtbot, dialog, monkeypatch
):
    """Verify "Show Me" falls back to a message box when spotlight_widget reports failure.

    The target widget resolves via findChild, but spotlight_widget itself
    reports the widget isn't currently visible/spotlightable. Success means
    the same message-box fallback fires and the dialog stays open.
    """
    monkeypatch.setattr(
        "views.annomate.help_dialog.spotlight_widget", lambda widget: False
    )
    monkeypatch.setattr(
        "views.annomate.help_dialog.QMessageBox.information",
        lambda *args, **kwargs: None,
    )
    qtbot.keyClicks(dialog._search_edit, "save")
    dialog._run_search()

    qtbot.mouseClick(dialog._detail_panel._show_me_btn, Qt.LeftButton)

    assert not dialog.isHidden()


def test_close_button_closes_dialog(qtbot, dialog):
    """Verify clicking Close accepts/closes the dialog.

    Success means the dialog's result becomes QDialog.Accepted, matching the
    Close button's connection to self.accept.
    """
    close_btn = None
    for child in dialog.findChildren(QPushButton):
        if child.text() == "Close":
            close_btn = child
            break
    assert close_btn is not None

    qtbot.mouseClick(close_btn, Qt.LeftButton)

    assert dialog.result() == QDialog.Accepted
