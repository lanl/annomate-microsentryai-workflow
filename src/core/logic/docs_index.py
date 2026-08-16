"""Turns docs/*.md into searchable HelpEntry objects for the Help dialog.

Each markdown file is split into one HelpEntry per "###" subsection (or per
"##" section, for the files/sections that don't use "###"), so a search can
land on a specific paragraph of the manual rather than an entire file. Bold
text and inline code spans within a section are pulled out as extra
keywords, since those are usually the exact button/field/file names a user
would search for (e.g. **Load New**, `.annoproj`).

Runs once at import time — the docs/ folder is a handful of small files, so
there's no need to cache results across calls.
"""

import logging
import re
from pathlib import Path

from core.logic.help_entry import HelpEntry

_logger = logging.getLogger(__name__)

_DOCS_DIR = Path(__file__).resolve().parents[3] / "docs"

# Falls back to the file's own H1 title when a stem isn't listed here.
_CATEGORY_OVERRIDES = {
    "GettingStarted": "Getting Started",
}

# Contributor/architecture docs, not end-user content -- skipped so they
# don't surface in the in-app Help search.
_EXCLUDED_FILES = {"DeveloperGuide.md"}

_H1_RE = re.compile(r"^#\s+(.+)$", re.MULTILINE)
_H2_RE = re.compile(r"^##\s+(.+)$", re.MULTILINE)
_H3_RE = re.compile(r"^###\s+(.+)$", re.MULTILINE)
_LEADING_NUMBER_RE = re.compile(r"^\d+\.\s*")
_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_CODE_RE = re.compile(r"`([^`]+)`")
_MD_SYNTAX_RE = re.compile(r"[*_`#]")
_MAX_KEYWORD_LEN = 60
_SNIPPET_LEN = 180


def _clean_heading(heading: str) -> str:
    heading = _LEADING_NUMBER_RE.sub("", heading)
    return _MD_SYNTAX_RE.sub("", heading).strip()


def _extract_keywords(body: str) -> list:
    # Fenced blocks first, so a ``` ```-delimited block isn't misread as a
    # run of single-backtick inline-code spans by _CODE_RE.
    body = _FENCE_RE.sub("", body)
    keywords = {m.group(1).strip() for m in _BOLD_RE.finditer(body)}
    keywords |= {m.group(1).strip() for m in _CODE_RE.finditer(body)}
    return [k for k in keywords if k and len(k) <= _MAX_KEYWORD_LEN]


def _plain_snippet(body: str, limit: int = _SNIPPET_LEN) -> str:
    body = _FENCE_RE.sub(" ", body)
    plain = _MD_SYNTAX_RE.sub("", body)
    plain = re.sub(r"\s+", " ", plain).strip()
    if len(plain) > limit:
        plain = plain[:limit].rsplit(" ", 1)[0] + "…"
    return plain


def _split_by_heading(text: str, heading_re: "re.Pattern") -> list:
    """Return [(heading_text_or_None, body), ...] — the text before the
    first match (if non-empty) is kept with heading=None."""
    matches = list(heading_re.finditer(text))
    if not matches:
        return [(None, text)]

    chunks = []
    intro = text[: matches[0].start()].strip()
    if intro:
        chunks.append((None, intro))
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunks.append((m.group(1).strip(), text[start:end]))
    return chunks


def _doc_entries(path: Path) -> list:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        _logger.warning("Could not read doc %s: %s", path, exc)
        return []

    h1_match = _H1_RE.search(text)
    doc_title = h1_match.group(1).strip() if h1_match else path.stem
    category = _CATEGORY_OVERRIDES.get(path.stem, doc_title)

    entries = []
    for h2_heading, h2_body in _split_by_heading(text, _H2_RE):
        if h2_heading is None:
            continue  # text before the first "##" is just the doc's intro line

        clean_h2 = _clean_heading(h2_heading)
        for h3_heading, body in _split_by_heading(h2_body, _H3_RE):
            body = body.strip()
            if not body:
                continue

            if h3_heading is None:
                title = f"{doc_title} — {clean_h2}"
            else:
                title = f"{doc_title} — {clean_h2}: {_clean_heading(h3_heading)}"

            entries.append(
                HelpEntry(
                    title=title,
                    category=category,
                    keywords=_extract_keywords(body) + [doc_title, clean_h2],
                    description=_plain_snippet(body),
                    full_text=body,
                )
            )
    return entries


def load_doc_entries() -> list:
    """Return one HelpEntry per section/subsection across every docs/*.md file.

    Returns an empty list (rather than raising) if the docs folder is
    missing — e.g. a packaging config that didn't bundle it — so the rest
    of the Help search still works.
    """
    if not _DOCS_DIR.is_dir():
        _logger.warning("Docs folder not found at %s; skipping doc search", _DOCS_DIR)
        return []

    entries = []
    for path in sorted(_DOCS_DIR.glob("*.md")):
        if path.name in _EXCLUDED_FILES:
            continue
        entries.extend(_doc_entries(path))
    return entries
