"""Unit tests for docs_index.load_doc_entries()'s markdown parsing.

These run without a QApplication — docs_index.py is plain Python with no Qt
dependency. Each test points _DOCS_DIR at a tmp_path with crafted *.md
fixtures via monkeypatch, so results depend only on the fixture content, not
on the real docs/ folder.
"""

import core.logic.docs_index as docs_index
from core.logic.docs_index import load_doc_entries


def _write_doc(tmp_path, filename: str, content: str) -> None:
    (tmp_path / filename).write_text(content, encoding="utf-8")


def test_splits_entries_by_h2_and_h3_headings(tmp_path, monkeypatch):
    """Verify a "##" section with a "###" subsection produces one combined-title entry.

    Success means exactly one entry is produced, titled "Title — SectionA:
    Sub1", with the subsection's body as its full_text.
    """
    monkeypatch.setattr(docs_index, "_DOCS_DIR", tmp_path)
    _write_doc(
        tmp_path,
        "Doc.md",
        "# Title\n\n## SectionA\n\n### Sub1\n\nBody text for sub1.\n",
    )

    entries = load_doc_entries()

    assert len(entries) == 1
    assert entries[0].title == "Title — SectionA: Sub1"
    assert "Body text for sub1." in entries[0].full_text


def test_h2_only_section_without_h3(tmp_path, monkeypatch):
    """Verify a "##" section with no "###" subsections still yields one entry.

    Success means the entry's title is "Title — SectionA" (no ": Sub" suffix)
    since there's no h3 heading to append.
    """
    monkeypatch.setattr(docs_index, "_DOCS_DIR", tmp_path)
    _write_doc(
        tmp_path,
        "Doc.md",
        "# Title\n\n## SectionA\n\nBody text without subsections.\n",
    )

    entries = load_doc_entries()

    assert len(entries) == 1
    assert entries[0].title == "Title — SectionA"


def test_extracts_bold_and_code_span_keywords(tmp_path, monkeypatch):
    """Verify **bold** and `code` spans in the body become extra keywords.

    Success means both "Bold Term" and "code_span" appear in the entry's
    keywords list, stripped of the markdown syntax around them.
    """
    monkeypatch.setattr(docs_index, "_DOCS_DIR", tmp_path)
    _write_doc(
        tmp_path,
        "Doc.md",
        "# Title\n\n## SectionA\n\nSome text with **Bold Term** and `code_span` inside.\n",
    )

    entries = load_doc_entries()

    assert "Bold Term" in entries[0].keywords
    assert "code_span" in entries[0].keywords


def test_category_override_applied_for_known_stem(tmp_path, monkeypatch):
    """Verify a file stem listed in _CATEGORY_OVERRIDES uses the override category.

    "GettingStarted.md" maps to category "Getting Started" regardless of
    its own H1 title text. Success means the entry's category is exactly
    "Getting Started".
    """
    monkeypatch.setattr(docs_index, "_DOCS_DIR", tmp_path)
    _write_doc(
        tmp_path,
        "GettingStarted.md",
        "# Getting Started Guide\n\n## SectionA\n\nBody text.\n",
    )

    entries = load_doc_entries()

    assert entries[0].category == "Getting Started"


def test_category_defaults_to_h1_title_for_unknown_stem(tmp_path, monkeypatch):
    """Verify a file stem with no override uses its own H1 text as category.

    "AnnoMate.md" isn't in _CATEGORY_OVERRIDES, so category should fall back
    to the file's own "# ..." heading text. Success means category equals
    that H1 text exactly.
    """
    monkeypatch.setattr(docs_index, "_DOCS_DIR", tmp_path)
    _write_doc(
        tmp_path,
        "AnnoMate.md",
        "# AnnoMate Guide\n\n## SectionA\n\nBody text.\n",
    )

    entries = load_doc_entries()

    assert entries[0].category == "AnnoMate Guide"


def test_text_before_first_h2_is_ignored(tmp_path, monkeypatch):
    """Verify intro text before the first "##" heading produces no entry.

    Success means only the real "##" section yields an entry — the intro
    paragraph before it is dropped, not turned into its own entry.
    """
    monkeypatch.setattr(docs_index, "_DOCS_DIR", tmp_path)
    _write_doc(
        tmp_path,
        "Doc.md",
        "# Title\n\nAn intro paragraph before any section heading.\n\n"
        "## SectionA\n\nBody text.\n",
    )

    entries = load_doc_entries()

    assert len(entries) == 1
    assert "intro paragraph" not in entries[0].full_text


def test_snippet_truncated_with_ellipsis(tmp_path, monkeypatch):
    """Verify a body longer than the snippet limit is truncated with an ellipsis.

    Success means the entry's description is shorter than the full body and
    ends with the "…" truncation marker.
    """
    monkeypatch.setattr(docs_index, "_DOCS_DIR", tmp_path)
    long_body = "word " * 60  # well over the 180-char snippet limit
    _write_doc(tmp_path, "Doc.md", f"# Title\n\n## SectionA\n\n{long_body}\n")

    entries = load_doc_entries()

    assert len(entries[0].description) < len(long_body)
    assert entries[0].description.endswith("…")


def test_missing_docs_dir_returns_empty_list(tmp_path, monkeypatch):
    """Verify a missing docs directory returns [] instead of raising.

    Success means pointing _DOCS_DIR at a nonexistent path yields an empty
    list, so the rest of Help search still works with no doc-derived
    entries rather than crashing at import time.
    """
    monkeypatch.setattr(docs_index, "_DOCS_DIR", tmp_path / "does_not_exist")

    assert load_doc_entries() == []


def test_unreadable_file_is_skipped_not_raised(tmp_path, monkeypatch):
    """Verify a file that raises OSError on read is skipped, not propagated.

    Success means load_doc_entries() returns an empty list (the one file
    present failed to read) instead of letting the OSError escape.
    """
    monkeypatch.setattr(docs_index, "_DOCS_DIR", tmp_path)
    _write_doc(tmp_path, "Doc.md", "# Title\n\n## SectionA\n\nBody text.\n")

    def _raise_oserror(self, encoding=None):
        raise OSError("permission denied")

    monkeypatch.setattr("pathlib.Path.read_text", _raise_oserror)

    assert load_doc_entries() == []
