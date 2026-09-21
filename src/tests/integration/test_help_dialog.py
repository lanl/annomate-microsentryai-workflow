from core.utils.help_topics import load_topics
from views.help_dialog import HelpDialog


def test_lists_every_topic_and_shows_the_first(qtbot):
    topics = load_topics()
    dialog = HelpDialog()
    qtbot.addWidget(dialog)

    assert dialog._topic_list.count() == len(topics)
    assert dialog._topic_list.item(0).text() == topics[0].title
    assert topics[0].title in dialog._viewer.toPlainText()


def test_selecting_a_topic_renders_its_page(qtbot):
    topics = load_topics()
    dialog = HelpDialog()
    qtbot.addWidget(dialog)

    dialog._topic_list.setCurrentRow(len(topics) - 1)

    assert topics[-1].title in dialog._viewer.toPlainText()


def test_empty_topic_list_does_not_crash(qtbot):
    dialog = HelpDialog(topics=[])
    qtbot.addWidget(dialog)

    assert dialog._topic_list.count() == 0


def _visible_titles(dialog):
    lst = dialog._topic_list
    return [
        lst.item(i).text() for i in range(lst.count()) if not lst.item(i).isHidden()
    ]


def test_search_hides_topics_without_every_word(qtbot):
    dialog = HelpDialog()
    qtbot.addWidget(dialog)

    dialog._search.setText("anomalib")

    assert _visible_titles(dialog) == ["Inference (MicroSentryAI)"]
    assert "Anomalib" in dialog._viewer.toPlainText()
    assert dialog._viewer.textCursor().selectedText().lower() == "anomalib"


def test_search_is_case_insensitive_and_needs_all_words(qtbot):
    dialog = HelpDialog()
    qtbot.addWidget(dialog)

    dialog._search.setText("ANOMALIB heatmap")
    assert _visible_titles(dialog) == ["Inference (MicroSentryAI)"]

    dialog._search.setText("anomalib zzzznotaword")
    assert _visible_titles(dialog) == []


def test_search_with_no_match_shows_message_then_clearing_restores(qtbot):
    topics = load_topics()
    dialog = HelpDialog()
    qtbot.addWidget(dialog)

    dialog._search.setText("zzzznotaword")
    assert "No topics match" in dialog._viewer.toPlainText()

    dialog._search.clear()
    assert len(_visible_titles(dialog)) == len(topics)
    assert topics[0].title in dialog._viewer.toPlainText()


def test_search_switches_away_from_a_hidden_selected_topic(qtbot):
    dialog = HelpDialog()
    qtbot.addWidget(dialog)
    assert dialog._topic_list.currentRow() == 0  # Getting Started

    dialog._search.setText("anomalib")

    assert dialog._topic_list.currentItem().text() == "Inference (MicroSentryAI)"
