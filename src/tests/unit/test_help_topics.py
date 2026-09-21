from core.utils.help_topics import HELP_DIR, load_topics


def test_bundled_topics_all_have_a_title_heading():
    topics = load_topics()
    assert topics, f"no help topics found in {HELP_DIR}"
    for topic in topics:
        assert topic.text.lstrip().startswith("# "), topic.path.name
        assert topic.title != topic.path.stem, topic.path.name


def test_topics_are_ordered_by_file_name(tmp_path):
    (tmp_path / "02-b.md").write_text("# Second\n", encoding="utf-8")
    (tmp_path / "01-a.md").write_text("# First\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("ignored", encoding="utf-8")

    assert [t.title for t in load_topics(tmp_path)] == ["First", "Second"]


def test_title_falls_back_to_file_stem(tmp_path):
    (tmp_path / "untitled.md").write_text("no heading here\n", encoding="utf-8")

    assert load_topics(tmp_path)[0].title == "untitled"
