"""Fuzzy, synonym-aware local search over HELP_ENTRIES for the Help dialog.

No exact keyword is required: a query is matched against every entry through
five progressively looser tiers — exact phrase, exact token, partial/prefix
token, fuzzy (typo-tolerant) token, and synonym-expanded token — each scored
per field (title > keywords > description > full text) and per field-weighted
so a title hit always outranks a body-only hit. Everything runs offline with
only the standard library (difflib for fuzzy comparison).

Layout of this file:
    1. Text normalization / tokenization helpers
    2. Synonym table                              <- SYNONYM FIELDS live here
    3. Fuzzy (typo-tolerant) matching
    4. Field-weighted scoring                      <- RESULT RANKING lives here
    5. Public search_help() entry point
"""

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from core.logic.help_docs import HelpEntry, HELP_ENTRIES


@dataclass(frozen=True)
class HelpMatch:
    """One ranked search result: the entry, its score, and why it matched."""

    entry: HelpEntry
    score: int
    reasons: list  # human-readable strings, strongest signal first


# ====================================================================== #
# 1. Text normalization / tokenization
# ====================================================================== #

_WORD_RE = re.compile(r"[a-z0-9']+")

# Filler words stripped from the *query* before token-level matching, so a
# natural-language question ("how do I keep my work") scores on its content
# words ("keep", "work") instead of being diluted or falsely matched on
# "how", "do", "i", "my".
_STOPWORDS = {
    "a",
    "an",
    "the",
    "how",
    "do",
    "does",
    "did",
    "doing",
    "i",
    "im",
    "you",
    "your",
    "yours",
    "my",
    "mine",
    "me",
    "to",
    "of",
    "in",
    "on",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "can",
    "could",
    "would",
    "should",
    "will",
    "shall",
    "and",
    "or",
    "but",
    "it",
    "its",
    "this",
    "that",
    "these",
    "those",
    "with",
    "about",
    "if",
    "so",
    "get",
    "got",
    "need",
    "want",
    "please",
    "help",
    "for",
    "from",
    "at",
    "by",
    "as",
    "up",
    "out",
    "not",
    "no",
    "yes",
    "what",
    "which",
    "who",
    "where",
    "when",
    "why",
    "there",
    "here",
    "some",
    "any",
    "all",
}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _tokenize(text: str) -> list:
    return _WORD_RE.findall(text.lower())


def _content_tokens(tokens: list) -> list:
    """Drop stopwords/single letters; fall back to the raw tokens if that
    would empty out a query that was nothing but stopwords."""
    filtered = [t for t in tokens if t not in _STOPWORDS and len(t) > 1]
    return filtered or tokens


# ====================================================================== #
# 2. Synonym table — SYNONYM FIELDS
#
# Each set below is a group of interchangeable terms. Add a new set (or add
# a word to an existing one) to teach the search about a new synonym without
# touching any scoring logic. This is a *global* thesaurus layered on top of
# each HelpEntry's own `keywords` list (which can also carry entry-specific
# synonyms/phrases, e.g. "save my work").
# ====================================================================== #

_SYNONYM_GROUPS = [
    {
        "save",
        "saving",
        "saved",
        "store",
        "stored",
        "storing",
        "keep",
        "keeping",
        "write",
        "writing",
        "preserve",
        "preserving",
        "persist",
        "backup",
    },
    {
        "open",
        "opening",
        "opened",
        "load",
        "loading",
        "loaded",
        "import",
        "importing",
        "bring",
        "access",
        "launch",
        "start",
    },
    {"new", "create", "creating", "fresh", "begin"},
    {"delete", "deleting", "remove", "removing", "erase", "clear", "clearing"},
    {
        "error",
        "errors",
        "problem",
        "problems",
        "issue",
        "issues",
        "bug",
        "fail",
        "failed",
        "failure",
        "crash",
        "crashed",
        "wrong",
        "broken",
    },
    {
        "settings",
        "setting",
        "preferences",
        "preference",
        "options",
        "option",
        "configuration",
        "configure",
        "config",
    },
    {"search", "searching", "find", "finding", "look", "looking", "query"},
    {"folder", "directory", "dir", "path"},
    {"image", "images", "photo", "photos", "picture", "pictures"},
    {
        "annotation",
        "annotations",
        "label",
        "labels",
        "labeling",
        "annotate",
        "tag",
        "tags",
    },
    {"class", "classes", "category", "categories"},
    {"model", "models", "weights", "checkpoint"},
    {"measure", "measuring", "measurement", "distance", "ruler"},
    {"calibrate", "calibration", "calibrating", "scale"},
    {"segment", "segmentation", "segmenting", "mask", "masks", "outline"},
    {"project", "work", "session"},
    {"export", "exporting", "output"},
    {"shortcut", "shortcuts", "hotkey", "hotkeys", "keyboard"},
    {"documentation", "docs", "help", "guide", "manual"},
]

_SYNONYM_MAP: dict = {}
for _group in _SYNONYM_GROUPS:
    for _word in _group:
        _SYNONYM_MAP.setdefault(_word, set()).update(_group)


def _expand_with_synonyms(tokens: list) -> set:
    """Union each token with its synonym group, e.g. {'keep'} -> {'keep',
    'save', 'store', ...}. Tokens with no known synonym pass through as-is."""
    expanded = set(tokens)
    for token in tokens:
        expanded |= _SYNONYM_MAP.get(token, set())
    return expanded


# ====================================================================== #
# 3. Fuzzy (typo-tolerant) matching
# ====================================================================== #

_FUZZY_MIN_LEN = 4  # below this length, edit-distance ratios are too noisy
_FUZZY_THRESHOLD = 0.72  # calibrated against typos like "opne"->"open" (.75)


def _best_fuzzy_match(token: str, candidates) -> str:
    """Return the closest candidate word if it's a plausible typo of *token*,
    else None. Used for things like "opne" -> "open" or "documntation" ->
    "documentation"."""
    if len(token) < _FUZZY_MIN_LEN:
        return None
    best_word, best_ratio = None, 0.0
    for candidate in candidates:
        if len(candidate) < _FUZZY_MIN_LEN:
            continue
        ratio = SequenceMatcher(None, token, candidate).ratio()
        if ratio > best_ratio:
            best_word, best_ratio = candidate, ratio
    return best_word if best_ratio >= _FUZZY_THRESHOLD else None


# ====================================================================== #
# 4. Field-weighted scoring — RESULT RANKING
#
# Every query is checked against four fields per entry, each with its own
# weight table so a hit in the title always outranks the same kind of hit in
# the full body text. Within a field, match tiers are also weighted:
# whole-phrase > exact token > partial/prefix token > fuzzy token > synonym
# token — this is what lets "sav" partially match "saving" while a typo like
# "opne" still finds "open", without either drowning out a clean exact match.
# ====================================================================== #

_TITLE = "title"
_KEYWORDS = "keywords"
_DESCRIPTION = "description"
_BODY = "full text"

_FIELD_WEIGHTS = {
    _TITLE: {"phrase": 120, "exact": 40, "partial": 22, "fuzzy": 14, "synonym": 10},
    _KEYWORDS: {"phrase": 70, "exact": 24, "partial": 14, "fuzzy": 9, "synonym": 8},
    _DESCRIPTION: {"phrase": 30, "exact": 10, "partial": 6, "fuzzy": 4, "synonym": 3},
    _BODY: {"phrase": 12, "exact": 4, "partial": 3, "fuzzy": 2, "synonym": 1},
}

# Entries scoring below this are dropped as noise (e.g. a single weak
# synonym hit in the body text only) rather than shown as a "result". Low
# enough that a single keyword-level fuzzy typo match (weight 9) still
# clears it on its own.
_MIN_RESULT_SCORE = 8


def _score_field(
    field_label: str,
    field_text: str,
    query_norm: str,
    query_tokens: list,
    syn_tokens: set,
    reasons: list,
) -> int:
    weights = _FIELD_WEIGHTS[field_label]
    field_norm = _normalize(field_text)
    field_tokens = set(_tokenize(field_norm))
    score = 0

    # Tier 1: whole phrase appears verbatim (substring) in the field — this
    # alone is what makes a partial query like "sav" match "saving"/"save
    # project", since "sav" is a substring of the normalized field text.
    if len(query_norm) >= 3 and query_norm in field_norm:
        score += weights["phrase"]
        reasons.append((weights["phrase"], f"matches your search in the {field_label}"))

    extra_syn_tokens = syn_tokens - set(query_tokens)

    for token in query_tokens:
        # Tier 2: exact token match.
        if token in field_tokens:
            score += weights["exact"]
            reasons.append(
                (weights["exact"], f'"{token}" appears in the {field_label}')
            )
            continue

        # Tier 3: partial/prefix match, e.g. "sav" <-> "saving".
        partial_hit = next(
            (
                ft
                for ft in field_tokens
                if len(token) >= 3 and len(ft) >= 3 and (token in ft or ft in token)
            ),
            None,
        )
        if partial_hit:
            score += weights["partial"]
            reasons.append(
                (
                    weights["partial"],
                    f'"{token}" partially matches "{partial_hit}" in the {field_label}',
                )
            )
            continue

        # Tier 4: fuzzy/typo match, e.g. "opne" <-> "open".
        fuzzy_hit = _best_fuzzy_match(token, field_tokens)
        if fuzzy_hit:
            score += weights["fuzzy"]
            reasons.append(
                (
                    weights["fuzzy"],
                    f'"{token}" looks like "{fuzzy_hit}" in the {field_label}',
                )
            )

    # Tier 5: synonym-only tokens (present via expansion, not literally typed).
    for syn_token in extra_syn_tokens:
        if syn_token in field_tokens:
            score += weights["synonym"]
            reasons.append(
                (
                    weights["synonym"],
                    f'related term "{syn_token}" found in the {field_label}',
                )
            )

    return score


def _score_entry(
    entry: HelpEntry, query_norm: str, query_tokens: list, syn_tokens: set
):
    reasons = []
    score = 0
    score += _score_field(
        _TITLE, entry.title, query_norm, query_tokens, syn_tokens, reasons
    )
    score += _score_field(
        _KEYWORDS,
        " ".join(entry.keywords),
        query_norm,
        query_tokens,
        syn_tokens,
        reasons,
    )
    score += _score_field(
        _DESCRIPTION, entry.description, query_norm, query_tokens, syn_tokens, reasons
    )
    score += _score_field(
        _BODY, entry.full_text, query_norm, query_tokens, syn_tokens, reasons
    )

    reasons.sort(key=lambda pair: pair[0], reverse=True)
    top_reasons = [text for _weight, text in reasons[:2]]
    return score, top_reasons


# ====================================================================== #
# 5. Public entry point
# ====================================================================== #


def search_help(query: str, entries: list = None, limit: int = 10) -> list:
    """Return the top-matching help entries for *query*, best first.

    Supports partial words, typos, and natural-language phrasing (see module
    docstring) — no exact keyword is required. Weak/noise-level matches are
    dropped rather than returned, so an empty result means "nothing relevant
    found", not "nothing at all matched any character".

    Args:
        query: Raw user search text — a keyword, phrase, or question.
        entries: Documentation entries to search; defaults to HELP_ENTRIES.
        limit: Maximum number of results to return.

    Returns:
        A list of HelpMatch(entry, score, reasons), sorted by descending score.
    """
    entries = HELP_ENTRIES if entries is None else entries
    query_norm = _normalize(query)
    if not query_norm:
        return []

    query_tokens = _content_tokens(_tokenize(query_norm))
    syn_tokens = _expand_with_synonyms(query_tokens)

    matches = []
    for entry in entries:
        score, reasons = _score_entry(entry, query_norm, query_tokens, syn_tokens)
        if score >= _MIN_RESULT_SCORE:
            matches.append(HelpMatch(entry=entry, score=score, reasons=reasons))

    matches.sort(key=lambda m: m.score, reverse=True)
    return matches[:limit]
