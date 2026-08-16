"""HelpEntry — the schema shared by help_docs.py (curated entries) and
docs_index.py (entries generated from docs/*.md). Kept dependency-free in
its own module so those two can both import it without a circular import.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class HelpEntry:
    """A single searchable help topic.

    Args:
        title: Short heading shown as the result's title.
        category: Section/group name shown as a badge next to the title.
        keywords: Extra search terms not necessarily present in the title
            (synonyms, menu paths, abbreviations) that should still match.
            For entries tied to a live widget, this should include the
            widget's own tooltip text so the search matches on it.
        description: One- or two-sentence preview shown in the results list.
        full_text: Complete help text shown when a result is opened.
        object_name: objectName() of the live widget this entry points to
            (see views.annomate.ui_spotlight), or None if this entry has no
            single on-screen widget to highlight.
    """

    title: str
    category: str
    keywords: list = field(default_factory=list)
    description: str = ""
    full_text: str = ""
    object_name: str = None
