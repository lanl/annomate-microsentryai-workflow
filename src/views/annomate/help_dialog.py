"""HelpSearchDialog — the popup opened by the toolbar's Help item.

Search-as-you-click UI over the local HELP_ENTRIES documentation set: a
search box on top, a results list on the left, and a detail pane on the
right showing the full text of whichever result is selected. See
core.logic.help_docs for how to add new documentation entries and
core.logic.help_search for the ranking rules.
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QFrame,
    QScrollArea,
    QWidget,
    QSplitter,
    QTextBrowser,
    QMessageBox,
)

from core.logic.help_docs import HelpEntry
from core.logic.help_search import HelpMatch
from views.annomate.ui_spotlight import spotlight_widget

_NO_QUERY_MESSAGE = (
    "Type a keyword, phrase, or question above and press Search "
    "(or Enter) to look through the documentation."
)
_NO_RESULTS_MESSAGE = (
    "I couldn't find an exact match, but try searching for words like "
    "open, save, import, export, settings, or error."
)
_DETAIL_PLACEHOLDER = "Select a result on the left to view the full help topic here."
_NOT_VISIBLE_MESSAGE = (
    "That control isn't on screen right now — it may only appear once a "
    "project or image is loaded."
)


class _ResultCard(QFrame):
    """One clickable result: title, category badge, snippet, and match reason."""

    clicked = Signal(object)

    def __init__(self, match: HelpMatch, parent: QWidget = None) -> None:
        super().__init__(parent)
        self.entry = match.entry
        self.setObjectName("HelpResultCard")
        self.setCursor(Qt.PointingHandCursor)
        self.setProperty("selected", False)
        self.setStyleSheet(
            """
            QFrame#HelpResultCard {
                border-radius: 8px;
            }
            QFrame#HelpResultCard:hover {
                background: palette(alternate-base);
            }
            QFrame#HelpResultCard[selected="true"] {
                background: palette(highlight);
            }
            QFrame#HelpResultCard[selected="true"] QLabel {
                color: palette(highlighted-text);
            }
            QLabel#HelpResultTitle {
                font-size: 14px;
                font-weight: bold;
            }
            QLabel#HelpResultCategory {
                font-size: 10px;
                font-weight: bold;
                color: palette(link);
            }
            QLabel#HelpResultSnippet {
                font-size: 12px;
                color: palette(dark);
            }
            QLabel#HelpResultReason {
                font-size: 11px;
                font-style: italic;
                color: palette(dark);
            }
            """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(3)

        top_row = QHBoxLayout()
        top_row.setSpacing(8)

        title = QLabel(match.entry.title)
        title.setObjectName("HelpResultTitle")
        title.setWordWrap(True)
        top_row.addWidget(title, 1)

        category = QLabel(match.entry.category.upper())
        category.setObjectName("HelpResultCategory")
        top_row.addWidget(category, 0, Qt.AlignTop)
        layout.addLayout(top_row)

        snippet = QLabel(match.entry.description)
        snippet.setObjectName("HelpResultSnippet")
        snippet.setWordWrap(True)
        layout.addWidget(snippet)

        # "Why it matched" — surfaced from the strongest scoring signals so
        # a fuzzy/synonym/partial hit is legible instead of a black box.
        if match.reasons:
            reason = QLabel("Matched: " + "; ".join(match.reasons))
            reason.setObjectName("HelpResultReason")
            reason.setWordWrap(True)
            layout.addWidget(reason)

    def set_selected(self, selected: bool) -> None:
        self.setProperty("selected", selected)
        self.style().unpolish(self)
        self.style().polish(self)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.entry)
        super().mousePressEvent(event)


class _DetailPanel(QFrame):
    """Right-hand pane showing the full text of the selected help entry."""

    show_me_clicked = Signal(str)  # emitted with entry.object_name

    def __init__(self, parent: QWidget = None) -> None:
        super().__init__(parent)
        self.setObjectName("HelpDetailPanel")
        self.setStyleSheet(
            """
            QFrame#HelpDetailPanel {
                background: palette(base);
                border: 1px solid palette(mid);
                border-radius: 8px;
            }
            QLabel#HelpDetailCategory {
                font-size: 11px;
                font-weight: bold;
                color: palette(link);
            }
            QLabel#HelpDetailTitle {
                font-size: 17px;
                font-weight: bold;
            }
            QLabel#HelpDetailPlaceholder {
                font-size: 13px;
                font-style: italic;
                color: palette(dark);
            }
            QPushButton#HelpShowMeButton {
                background: palette(highlight);
                color: palette(highlighted-text);
                font-weight: bold;
                border: none;
                border-radius: 6px;
                min-height: 28px;
                padding-left: 16px;
                padding-right: 16px;
            }
            QPushButton#HelpShowMeButton:hover {
                background: palette(highlight);
            }
            QPushButton#HelpShowMeButton:pressed {
                padding-top: 1px;
            }
            """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(6)

        self._category = QLabel("")
        self._category.setObjectName("HelpDetailCategory")
        layout.addWidget(self._category)

        self._title = QLabel("")
        self._title.setObjectName("HelpDetailTitle")
        self._title.setWordWrap(True)
        layout.addWidget(self._title)

        self._body = QTextBrowser()
        self._body.setObjectName("HelpDetailBody")
        self._body.setFrameShape(QFrame.NoFrame)
        self._body.setStyleSheet("background: transparent; font-size: 13px;")
        layout.addWidget(self._body, 1)

        self._show_me_row = QFrame()
        self._show_me_row.setFrameShape(QFrame.HLine)
        self._show_me_row.setFrameShadow(QFrame.Sunken)
        layout.addWidget(self._show_me_row)

        show_me_row = QHBoxLayout()
        show_me_row.setContentsMargins(0, 4, 0, 0)
        show_me_row.addStretch(1)
        self._show_me_btn = QPushButton("▸  Show Me in the App")
        self._show_me_btn.setObjectName("HelpShowMeButton")
        self._show_me_btn.setCursor(Qt.PointingHandCursor)
        self._show_me_btn.clicked.connect(self._on_show_me_clicked)
        show_me_row.addWidget(self._show_me_btn)
        layout.addLayout(show_me_row)

        self._current_object_name = None
        self.clear()

    def show_entry(self, entry: HelpEntry) -> None:
        self._category.setText(entry.category.upper())
        self._title.setText(entry.title)
        self._body.setProperty("class", "")
        self._body.setMarkdown(entry.full_text)
        self._current_object_name = entry.object_name
        self._set_show_me_visible(bool(entry.object_name))

    def clear(self) -> None:
        self._category.setText("")
        self._title.setText("")
        self._body.setPlainText(_DETAIL_PLACEHOLDER)
        self._current_object_name = None
        self._set_show_me_visible(False)

    def _set_show_me_visible(self, visible: bool) -> None:
        self._show_me_row.setVisible(visible)
        self._show_me_btn.setVisible(visible)

    def _on_show_me_clicked(self) -> None:
        if self._current_object_name:
            self.show_me_clicked.emit(self._current_object_name)


class HelpSearchDialog(QDialog):
    """Modal Help window: local keyword/phrase search over app documentation.

    Args:
        help_controller: HelpController instance that runs the search.
        parent: Parent widget.
        initial_query: If given, pre-fills the search box with this text and
            runs the search immediately — used by the Help menu's inline
            "More on <query>…" entry so it opens straight to results instead
            of requiring the user to retype their query.
    """

    _RESULT_LIMIT = 10

    def __init__(
        self, help_controller, parent: QWidget = None, initial_query: str = ""
    ) -> None:
        super().__init__(parent)
        self._controller = help_controller
        self.setWindowTitle("Help")
        self.setModal(True)
        self.resize(780, 560)
        self.setMinimumSize(560, 420)

        self._cards: list = []
        self._selected_card = None

        self._build_ui()
        if initial_query:
            self._search_edit.setText(initial_query)
            self._run_search()
        else:
            self._search_edit.setFocus()

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #

    def _build_ui(self) -> None:
        self.setStyleSheet(
            """
            QLabel#HelpDialogTitle {
                font-size: 20px;
                font-weight: bold;
            }
            QLabel#HelpMessageLabel {
                font-size: 13px;
                color: palette(dark);
                padding: 12px 4px;
            }
            QLineEdit#HelpSearchEdit {
                min-height: 26px;
                padding: 4px 8px;
                border-radius: 6px;
            }
            QPushButton {
                min-height: 30px;
                padding-left: 14px;
                padding-right: 14px;
            }
            """
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(18, 16, 18, 16)
        root.setSpacing(12)

        header = QLabel("Search Help")
        header.setObjectName("HelpDialogTitle")
        root.addWidget(header)

        search_row = QHBoxLayout()
        search_row.setSpacing(8)

        self._search_edit = QLineEdit()
        self._search_edit.setObjectName("HelpSearchEdit")
        self._search_edit.setPlaceholderText(
            'Search for a topic, e.g. "export CSV" or "SAM tool"…'
        )
        self._search_edit.returnPressed.connect(self._run_search)
        search_row.addWidget(self._search_edit, 1)

        search_btn = QPushButton("Search")
        search_btn.setDefault(True)
        search_btn.clicked.connect(self._run_search)
        search_row.addWidget(search_btn)

        root.addLayout(search_row)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        splitter.addWidget(self._build_results_pane())
        self._detail_panel = _DetailPanel()
        self._detail_panel.show_me_clicked.connect(self._on_show_me)
        splitter.addWidget(self._detail_panel)
        splitter.setSizes([300, 480])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        root.addWidget(splitter, 1)

        close_row = QHBoxLayout()
        close_row.addStretch(1)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        close_row.addWidget(close_btn)
        root.addLayout(close_row)

        self._show_message(_NO_QUERY_MESSAGE)

    def _build_results_pane(self) -> QWidget:
        pane = QWidget()
        layout = QVBoxLayout(pane)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self._message_label = QLabel("")
        self._message_label.setObjectName("HelpMessageLabel")
        self._message_label.setWordWrap(True)
        self._message_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        layout.addWidget(self._message_label)

        self._results_scroll = QScrollArea()
        self._results_scroll.setWidgetResizable(True)
        self._results_scroll.setFrameShape(QFrame.NoFrame)
        self._results_scroll.setVisible(False)

        self._results_container = QWidget()
        self._results_layout = QVBoxLayout(self._results_container)
        self._results_layout.setContentsMargins(0, 0, 6, 0)
        self._results_layout.setSpacing(6)
        self._results_layout.addStretch(1)
        self._results_scroll.setWidget(self._results_container)

        layout.addWidget(self._results_scroll, 1)
        return pane

    # ------------------------------------------------------------------ #
    # Search behavior
    # ------------------------------------------------------------------ #

    def _run_search(self) -> None:
        query = self._search_edit.text()
        self._clear_results()

        if not query.strip():
            self._show_message(_NO_QUERY_MESSAGE)
            self._detail_panel.clear()
            return

        results = self._controller.search(query, limit=self._RESULT_LIMIT)
        if not results:
            self._show_message(_NO_RESULTS_MESSAGE)
            self._detail_panel.clear()
            return

        self._message_label.setVisible(False)
        self._results_scroll.setVisible(True)
        for match in results:
            card = _ResultCard(match)
            card.clicked.connect(self._on_card_clicked)
            self._results_layout.insertWidget(self._results_layout.count() - 1, card)
            self._cards.append(card)

        self._on_card_clicked(self._cards[0].entry)

    def _show_message(self, text: str) -> None:
        self._message_label.setText(text)
        self._message_label.setVisible(True)
        self._results_scroll.setVisible(False)

    def _clear_results(self) -> None:
        for card in self._cards:
            card.setParent(None)
            card.deleteLater()
        self._cards = []
        self._selected_card = None

    def _on_card_clicked(self, entry: HelpEntry) -> None:
        self._detail_panel.show_entry(entry)
        for card in self._cards:
            if card.entry is entry:
                self._select_card(card)
                break

    def _select_card(self, card: _ResultCard) -> None:
        if self._selected_card is not None:
            self._selected_card.set_selected(False)
        card.set_selected(True)
        self._selected_card = card

    def _on_show_me(self, object_name: str) -> None:
        root = self.parent()
        target = root.findChild(QWidget, object_name) if root is not None else None
        if target is not None and spotlight_widget(target):
            self.hide()
        else:
            QMessageBox.information(self, "Help", _NOT_VISIBLE_MESSAGE)
