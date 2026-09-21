"""HelpDialog — browsable in-app manual built from the bundled help topics."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLineEdit,
    QListWidget,
    QSplitter,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from core.utils.help_topics import HELP_DIR, load_topics


class HelpDialog(QDialog):
    """Search box and topic list on the left, rendered Markdown page on the right.

    Typing in the search box hides topics that don't contain every word
    typed, and highlights the first match on the shown page.

    Shown with ``show()`` (not ``exec()``) so it stays open beside the app.

    Args:
        topics: ``HelpTopic`` entries to display. Defaults to the bundled help.
        parent: Parent widget.
    """

    def __init__(self, topics: list = None, parent: QWidget = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Help")
        self.resize(900, 600)

        self._topics = load_topics() if topics is None else topics

        self._search = QLineEdit()
        self._search.setPlaceholderText("Search help…")
        self._search.setClearButtonEnabled(True)

        self._topic_list = QListWidget()
        for topic in self._topics:
            self._topic_list.addItem(topic.title)

        self._viewer = QTextBrowser()
        self._viewer.setSearchPaths([str(HELP_DIR)])
        self._viewer.setOpenExternalLinks(True)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(self._search)
        left_layout.addWidget(self._topic_list)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(self._viewer)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([200, 700])

        layout = QHBoxLayout(self)
        layout.addWidget(splitter)

        self._topic_list.currentRowChanged.connect(self._show_topic)
        self._search.textChanged.connect(self._apply_search)
        if self._topics:
            self._topic_list.setCurrentRow(0)

    def _apply_search(self, text: str) -> None:
        words = text.lower().split()
        first_visible = -1
        for row, topic in enumerate(self._topics):
            haystack = f"{topic.title}\n{topic.text}".lower()
            hidden = not all(word in haystack for word in words)
            self._topic_list.item(row).setHidden(hidden)
            if not hidden and first_visible == -1:
                first_visible = row

        current = self._topic_list.currentRow()
        if current == -1 or self._topic_list.item(current).isHidden():
            self._topic_list.setCurrentRow(first_visible)
        else:
            self._show_topic(current)

    def _show_topic(self, row: int) -> None:
        if not 0 <= row < len(self._topics):
            self._viewer.setMarkdown("No topics match your search.")
            return
        self._viewer.setMarkdown(self._topics[row].text)
        words = self._search.text().split()
        if words:
            self._viewer.find(words[0])
