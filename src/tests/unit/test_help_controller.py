"""Unit tests for HelpController.

HelpController is QObject-based but its methods are plain synchronous calls
with no signals to wait on, so — like test_calibration_model.py's
QObject-without-signals tests — these run without qtbot.
"""

import pytest

from controllers.help_controller import HelpController
from core.logic.help_entry import HelpEntry


@pytest.fixture
def controller():
    return HelpController()


class TestSearch:
    def test_search_delegates_to_search_help(self, controller, monkeypatch):
        """Verify search() calls search_help with the query, entries, and limit.

        Monkeypatches the module-level search_help used by HelpController so
        no real ranking runs. Success means it's called with the exact
        query/entries/limit arguments and its return value is passed through
        unchanged.
        """
        captured = {}
        sentinel = ["stub-results"]

        def fake_search_help(query, entries=None, limit=10):
            captured["query"] = query
            captured["entries"] = entries
            captured["limit"] = limit
            return sentinel

        monkeypatch.setattr("controllers.help_controller.search_help", fake_search_help)

        result = controller.search("foo", limit=5)

        assert result is sentinel
        assert captured == {"query": "foo", "entries": None, "limit": 5}

    def test_search_with_injected_entries_uses_them(self, monkeypatch):
        """Verify entries passed to the constructor are forwarded to search_help.

        Constructing HelpController(entries=[...]) should make search() pass
        that exact list through instead of leaving it as None (which would
        make search_help fall back to the real HELP_ENTRIES). Success means
        the captured "entries" argument is the injected list.
        """
        sample_entries = [HelpEntry(title="Sample", category="Test")]
        controller = HelpController(entries=sample_entries)
        captured = {}

        def fake_search_help(query, entries=None, limit=10):
            captured["entries"] = entries
            return []

        monkeypatch.setattr("controllers.help_controller.search_help", fake_search_help)

        controller.search("anything")

        assert captured["entries"] is sample_entries

    def test_search_default_limit_is_ten(self, controller, monkeypatch):
        """Verify omitting limit forwards the default of 10 to search_help.

        Success means calling search() with no limit argument still passes
        limit=10 through to search_help.
        """
        captured = {}

        def fake_search_help(query, entries=None, limit=10):
            captured["limit"] = limit
            return []

        monkeypatch.setattr("controllers.help_controller.search_help", fake_search_help)

        controller.search("anything")

        assert captured["limit"] == 10
