"""Integrity checks on the real HELP_ENTRIES registry.

These run without a QApplication — help_docs.py is plain Python with no Qt
dependency. Tests check structural invariants of the registry (no duplicate
targets, required fields populated) rather than the content of any specific
entry, so they stay valid as help topics are added or edited.
"""

from core.logic.help_docs import HELP_ENTRIES


def test_no_duplicate_object_names():
    """Verify no two entries point at the same live widget.

    Two HelpEntry objects sharing an object_name would mean two "Show Me"
    targets collide on one widget, which is always a registry bug. Success
    means every non-None object_name in HELP_ENTRIES is unique.
    """
    object_names = [e.object_name for e in HELP_ENTRIES if e.object_name]
    assert len(object_names) == len(set(object_names))


def test_all_entries_have_title_and_category():
    """Verify every entry has a non-empty title and category.

    These two fields are shown as the result card's heading and badge, so a
    blank value would render as an empty label. Success means both are
    non-empty strings for every entry.
    """
    for entry in HELP_ENTRIES:
        assert isinstance(entry.title, str) and entry.title.strip()
        assert isinstance(entry.category, str) and entry.category.strip()


def test_all_entries_have_description_or_full_text():
    """Verify every entry has something to show in the results list or detail pane.

    An entry with both description and full_text blank would render as an
    empty snippet and an empty detail panel. Success means at least one of
    the two is populated for every entry.
    """
    for entry in HELP_ENTRIES:
        assert entry.description.strip() or entry.full_text.strip()


def test_registry_includes_generated_doc_entries():
    """Verify docs_index.load_doc_entries() output was actually appended.

    help_docs.py hand-authors a fixed set of entries and then appends
    docs/*.md-derived entries at import time. Generated entries are titled
    "<doc title> — <section>" (see docs_index._doc_entries), a shape no
    hand-authored entry uses. Success means at least one such entry is
    present, confirming the append actually happened rather than silently
    producing an empty list.
    """
    assert any(" — " in entry.title for entry in HELP_ENTRIES)
