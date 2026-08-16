"""Unit tests for search_help()'s tiered, synonym-aware ranking.

These run without a QApplication — HelpEntry and search_help are plain
dataclasses/functions with no Qt dependency.
"""

import pytest

from core.logic.help_entry import HelpEntry
from core.logic.help_search import HelpMatch, search_help


@pytest.fixture
def entries():
    return [
        HelpEntry(
            title="Saving your project",
            category="Projects",
            keywords=["save", "backup"],
            description="Save your work to disk.",
            full_text="Use Ctrl+S to save your project file to disk.",
            object_name="saveButton",
        ),
        HelpEntry(
            title="Opening an image folder",
            category="Projects",
            keywords=["open", "folder"],
            description="Load a folder of images.",
            full_text="Use File > Open Image Folder to load images.",
        ),
        HelpEntry(
            title="Recovering from a crash",
            category="Reference",
            keywords=[],
            description="Notes on recovery.",
            full_text=(
                "If something goes wrong, an automatic backup of your "
                "project may still be available in the recovery folder."
            ),
        ),
    ]


class TestMatchTiers:
    def test_exact_phrase_match_returns_top_result(self, entries):
        """Verify a verbatim title phrase ranks the matching entry first.

        Querying the full title text ("saving your project") should hit the
        phrase tier in the title field, which outweighs any other entry's
        weaker tier hits. Success means the top result is that entry.
        """
        results = search_help("saving your project", entries=entries)
        assert results[0].entry.title == "Saving your project"

    def test_exact_token_match(self, entries):
        """Verify a single exact keyword token matches its entry.

        The query "save" is an exact token in the first entry's keywords
        list. Success means that entry appears among the results.
        """
        results = search_help("save", entries=entries)
        titles = [m.entry.title for m in results]
        assert "Saving your project" in titles

    def test_partial_prefix_token_match(self, entries):
        """Verify a short prefix partially matches a longer field token.

        The query "sav" is a prefix of "saving" in the first entry's title.
        Success means that entry is still returned despite no exact token
        match.
        """
        results = search_help("sav", entries=entries)
        titles = [m.entry.title for m in results]
        assert "Saving your project" in titles

    def test_fuzzy_typo_token_match(self, entries):
        """Verify a typo'd token still fuzzy-matches its intended word.

        "opne" is a plausible typo of "open", which appears in the second
        entry's keywords. Success means that entry is returned even though
        "opne" never literally appears anywhere.
        """
        results = search_help("opne", entries=entries)
        titles = [m.entry.title for m in results]
        assert "Opening an image folder" in titles

    def test_synonym_token_match(self, entries):
        """Verify a synonym-only token matches an entry that never says it.

        "keep" is in the same synonym group as "save"/"backup". The query
        "keep my work" should still surface the entries that literally say
        "save"/"backup" via synonym expansion, not a literal token match.
        """
        results = search_help("keep my work", entries=entries)
        titles = [m.entry.title for m in results]
        assert "Saving your project" in titles

    def test_title_hit_outranks_body_only_hit(self, entries):
        """Verify a title-field hit ranks above a body-only hit for the same word.

        "project" appears literally in entry 1's title, and only in entry 3's
        full_text (never in its title/keywords/description). Success means
        entry 1 (title hit) scores higher than entry 3 (body-only hit).
        """
        results = search_help("project", entries=entries)
        titles = [m.entry.title for m in results]
        assert titles.index("Saving your project") < titles.index(
            "Recovering from a crash"
        )


class TestHelpMatchFields:
    def test_fields_populated(self, entries):
        """Verify a real match's HelpMatch has entry, score, and reasons set.

        Success means entry is the matched HelpEntry, score is a positive
        int, and reasons is a non-empty list of human-readable strings.
        """
        results = search_help("save", entries=entries)
        match = results[0]
        assert isinstance(match, HelpMatch)
        assert match.entry.title == "Saving your project"
        assert isinstance(match.score, int)
        assert match.score > 0
        assert match.reasons
        assert all(isinstance(r, str) for r in match.reasons)


class TestEdgeCases:
    def test_empty_query_returns_empty_list(self, entries):
        """Verify an empty query returns no results rather than everything.

        Success means search_help("") returns [].
        """
        assert search_help("", entries=entries) == []

    def test_whitespace_only_query_returns_empty_list(self, entries):
        """Verify a whitespace-only query normalizes to empty and returns [].

        Success means search_help("   ") returns [].
        """
        assert search_help("   ", entries=entries) == []

    def test_no_matches_below_min_score_returns_empty_list(self, entries):
        """Verify a query with no relation to any entry returns no results.

        A nonsense query unrelated to any title/keyword/description/body
        (and not a fuzzy/synonym match either) should score below the noise
        floor. Success means an empty list is returned.
        """
        assert search_help("xyzzyplugh", entries=entries) == []

    def test_limit_truncates_results(self, entries):
        """Verify limit caps the number of returned results, best-scored first.

        The query "backup" matches two entries; limit=1 should return only
        the single best-scoring one. Success means exactly one result comes
        back and it's the higher-ranked entry.
        """
        results = search_help("backup", entries=entries, limit=1)
        assert len(results) == 1
        assert results[0].entry.title == "Saving your project"

    def test_default_limit_is_ten(self, entries):
        """Verify the default limit caps results at 10 when omitted.

        Success means calling search_help without a limit never returns
        more than 10 matches, regardless of how many entries matched.
        """
        results = search_help("project folder backup save open", entries=entries)
        assert len(results) <= 10
