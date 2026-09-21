"""Loads the bundled help topics (Markdown files in src/resources/help/)."""

from pathlib import Path
from typing import NamedTuple

HELP_DIR = Path(__file__).resolve().parent.parent.parent / "resources" / "help"


class HelpTopic(NamedTuple):
    title: str
    path: Path
    text: str


def load_topics(help_dir: Path = HELP_DIR) -> list:
    """Read every ``*.md`` file in *help_dir*, ordered by file name.

    The title is the first ``# `` heading; a file without one falls back to
    its file stem.

    Args:
        help_dir (Path): Folder containing the help Markdown files.

    Returns:
        list: ``HelpTopic`` entries in file-name order.
    """
    topics = []
    for path in sorted(Path(help_dir).glob("*.md")):
        text = path.read_text(encoding="utf-8")
        title = path.stem
        for line in text.splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
                break
        topics.append(HelpTopic(title, path, text))
    return topics
