import pytest

from core.states.dataset_state import DatasetState
from models.dataset_model import DatasetTableModel
from views.annomate.sections._image_classes import ImageClassesSection


@pytest.fixture
def dataset_model(tmp_path):
    model = DatasetTableModel(DatasetState())
    model.load_folder(str(tmp_path), ["a.jpg"])
    model.add_class("scratch", (10, 20, 30))
    model.add_class("inclusion", (40, 50, 60))
    model.set_annotation_mode("image_level")
    return model


def row_names(section):
    return list(section._rows.keys())


def test_undecided_image_is_read_only_and_empty(qtbot, dataset_model):
    section = ImageClassesSection(dataset_model)
    qtbot.addWidget(section)
    section.set_current_row(0)

    assert row_names(section) == []
    assert section._empty_lbl.isVisibleTo(section) is True
    assert "Reject" in section._empty_lbl.text()
    assert section._hint_lbl.isVisibleTo(section) is False


def test_undecided_image_shows_existing_tags_read_only(qtbot, dataset_model):
    dataset_model.set_image_classes(0, ["scratch"])
    section = ImageClassesSection(dataset_model)
    qtbot.addWidget(section)
    section.set_current_row(0)

    assert row_names(section) == ["scratch"]

    # Not interactive -- clicking the row must not untag it.
    section._rows["scratch"].clicked.emit()
    assert dataset_model.get_image_classes(0) == ["scratch"]


def test_rejected_image_lists_all_classes_interactively(qtbot, dataset_model):
    dataset_model.set_review_decision(0, "reject")
    section = ImageClassesSection(dataset_model)
    qtbot.addWidget(section)
    section.set_current_row(0)

    assert sorted(row_names(section)) == ["inclusion", "scratch"]
    assert section._hint_lbl.isVisibleTo(section) is True
    assert section._empty_lbl.isVisibleTo(section) is False


def test_clicking_untagged_row_adds_tag(qtbot, dataset_model):
    dataset_model.set_review_decision(0, "reject")
    section = ImageClassesSection(dataset_model)
    qtbot.addWidget(section)
    section.set_current_row(0)

    section._rows["scratch"].clicked.emit()

    assert dataset_model.get_image_classes(0) == ["scratch"]


def test_clicking_tagged_row_removes_tag(qtbot, dataset_model):
    dataset_model.set_review_decision(0, "reject")
    dataset_model.set_image_classes(0, ["scratch"])
    section = ImageClassesSection(dataset_model)
    qtbot.addWidget(section)
    section.set_current_row(0)

    section._rows["scratch"].clicked.emit()

    assert dataset_model.get_image_classes(0) == []


def test_cannot_remove_tag_backed_by_pixel_annotations(qtbot, dataset_model, monkeypatch):
    """Untagging a class that still has pixel annotations warns and refuses.

    Mirrors the same protection the Annotation Classes panel enforces
    (classes.py's _toggle_image_tag) -- both surfaces must agree that a
    pixel-backed class can't be untagged from image-level mode alone.
    """
    warnings = []
    monkeypatch.setattr(
        "views.annomate.sections._image_classes.QMessageBox.warning",
        lambda *a, **k: warnings.append(a),
    )
    dataset_model.add_annotation(0, "scratch", [(0, 0), (1, 0), (1, 1)])
    dataset_model.set_review_decision(0, "reject")
    dataset_model.set_image_classes(0, ["scratch"])
    section = ImageClassesSection(dataset_model)
    qtbot.addWidget(section)
    section.set_current_row(0)

    section._rows["scratch"].clicked.emit()

    assert dataset_model.get_image_classes(0) == ["scratch"]
    assert len(warnings) == 1


def test_rebuild_reflects_decision_change_on_current_row(qtbot, dataset_model):
    """Row is initially undecided (read-only); rejecting it unlocks all classes."""
    section = ImageClassesSection(dataset_model)
    qtbot.addWidget(section)
    section.set_current_row(0)
    assert row_names(section) == []

    dataset_model.set_review_decision(0, "reject")

    assert sorted(row_names(section)) == ["inclusion", "scratch"]
