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
