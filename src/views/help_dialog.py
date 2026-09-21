"""HelpDialog — browsable in-app manual built from the bundled help topics."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QListWidget,
    QSplitter,
    QTextBrowser,
    QWidget,
)

from core.utils.help_topics import HELP_DIR, load_topics


class HelpDialog(QDialog):
    """Topic list on the left, rendered Markdown page on the right.

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

        self._topic_list = QListWidget()
        for topic in self._topics:
            self._topic_list.addItem(topic.title)

        self._viewer = QTextBrowser()
        self._viewer.setSearchPaths([str(HELP_DIR)])
        self._viewer.setOpenExternalLinks(True)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._topic_list)
        splitter.addWidget(self._viewer)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([200, 700])

        layout = QHBoxLayout(self)
        layout.addWidget(splitter)

        self._topic_list.currentRowChanged.connect(self._show_topic)
        if self._topics:
            self._topic_list.setCurrentRow(0)

    def _show_topic(self, row: int) -> None:
        if 0 <= row < len(self._topics):
            self._viewer.setMarkdown(self._topics[row].text)
