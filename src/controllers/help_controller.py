"""
HelpController — headless business logic for the Help search dialog.

No paired Model/State exists here: search results are transient UI state,
not persistent application data (the same reasoning behind SAMController
having no model of its own).
"""

from PySide6.QtCore import QObject

from core.logic.help_search import search_help


class HelpController(QObject):
    """Runs ranked keyword/phrase searches over the Help documentation set.

    Args:
        entries: Documentation entries to search; defaults to HELP_ENTRIES
            (see core.logic.help_search.search_help) when None. Overridable
            so tests and other callers can search a fixed entry set.
    """

    def __init__(self, entries: list = None, parent: QObject = None) -> None:
        super().__init__(parent)
        self._entries = entries

    def search(self, query: str, limit: int = 10) -> list:
        """Return ranked HelpMatch results for *query* (see search_help)."""
        return search_help(query, entries=self._entries, limit=limit)
