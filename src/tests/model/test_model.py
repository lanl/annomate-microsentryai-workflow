import pytest
from core.states.dataset_state import DatasetState
from models.dataset_model import DatasetTableModel


@pytest.fixture
def model():
    return DatasetTableModel(DatasetState())


class TestModelReset:
    def test_load_folder_emits_modelReset(self, qtbot, model, tmp_path):
        """Verify that load_folder emits the modelReset signal and produces the correct row count.

        Creates a real file in tmp_path and loads it via load_folder. The modelReset
        signal must be emitted (so connected views can refresh). Success means the
        signal fires within the timeout and rowCount equals 1.
        """
        (tmp_path / "a.jpg").touch()
        with qtbot.waitSignal(model.modelReset, timeout=1000):
            model.load_folder(str(tmp_path), ["a.jpg"])
        assert model.rowCount() == 1

    def test_row_count_after_load(self, model, tmp_path):
        """Verify that rowCount reflects the number of image files passed to load_folder.

        Loads a folder with two filenames. Success means rowCount returns 2.
        """
        model.load_folder(str(tmp_path), ["x.jpg", "y.jpg"])
        assert model.rowCount() == 2


class TestDataChanged:
    def test_add_annotation_emits_dataChanged(self, qtbot, model):
        """Verify that adding an annotation emits the dataChanged signal so views can update.

        Loads a fake folder with one image and adds an annotation. The dataChanged
        signal must fire to notify connected views. Success means the signal is emitted
        within the timeout.
        """
        model.load_folder("/fake", ["img.jpg"])
        with qtbot.waitSignal(model.dataChanged, timeout=1000):
            model.add_annotation(0, "Defect", [(0, 0), (1, 0), (1, 1)])

    def test_delete_annotation_emits_dataChanged(self, qtbot, model):
        """Verify that deleting an annotation emits the dataChanged signal.

        Adds and then deletes an annotation. The dataChanged signal must fire on
        deletion so connected views can refresh. Success means the signal is emitted
        within the timeout.
        """
        model.load_folder("/fake", ["img.jpg"])
        model.add_annotation(0, "Defect", [(0, 0), (1, 0)])
        with qtbot.waitSignal(model.dataChanged, timeout=1000):
            model.delete_annotation(0, 0)


class TestStatusColumn:
    def test_status_is_pending_initially(self, model):
        """Verify that newly loaded images have 'Pending' status in the status column.

        After loading a folder, images have no review activity, so their status should
        display as 'Pending'. Success means the data at column 1 is the string 'Pending'.
        """
        model.load_folder("/fake", ["img.jpg"])
        assert model.data(model.index(0, 1)) == "Pending"

    def test_annotation_alone_does_not_mark_reviewed(self, model):
        """Verify that adding an annotation alone does not change status to 'Reviewed'.

        A review decision is required to be considered reviewed. Annotations alone
        without an Accept or Reject decision should leave status as 'Pending'.
        """
        model.load_folder("/fake", ["img.jpg"])
        model.add_annotation(0, "Defect", [(0, 0), (1, 0), (1, 1)])
        assert model.data(model.index(0, 1)) == "Pending"

    def test_status_becomes_reviewed_after_accept(self, model):
        """Verify that an Accept decision marks an image as 'Reviewed' without annotations.

        An Accept decision alone is sufficient. Success means the status column shows
        'Reviewed' after set_review_decision is called with 'accept'.
        """
        model.load_folder("/fake", ["img.jpg"])
        model.set_review_decision(0, "accept")
        assert model.data(model.index(0, 1)) == "Reviewed"

    def test_status_becomes_reviewed_after_reject_with_annotation(self, model):
        """Verify that a Reject decision with an annotation marks an image as 'Reviewed'.

        A Reject decision requires at least one annotation to be considered reviewed.
        Success means the status column shows 'Reviewed' when both are present.
        """
        model.load_folder("/fake", ["img.jpg"])
        model.add_annotation(0, "Defect", [(0, 0), (1, 0), (1, 1)])
        model.set_review_decision(0, "reject")
        assert model.data(model.index(0, 1)) == "Reviewed"

    def test_status_pending_when_rejected_without_annotation(self, model):
        """Verify that a Reject decision without annotations leaves status as 'Pending'.

        A Reject decision alone is insufficient — annotations must also be present.
        Success means the status column shows 'Pending' when only the decision is 'reject'.
        """
        model.load_folder("/fake", ["img.jpg"])
        model.set_review_decision(0, "reject")
        assert model.data(model.index(0, 1)) == "Pending"

    def test_status_reverts_to_pending_after_last_annotation_deleted_on_reject(
        self, model
    ):
        """Verify that deleting the last annotation on a rejected image reverts status to 'Pending'.

        When a rejected image loses all its annotations, it no longer meets the reviewed
        criteria. Success means the status column shows 'Pending' after the last annotation
        is deleted.
        """
        model.load_folder("/fake", ["img.jpg"])
        model.add_annotation(0, "Defect", [(0, 0)])
        model.set_review_decision(0, "reject")
        model.delete_annotation(0, 0)
        assert model.data(model.index(0, 1)) == "Pending"

    def test_inspector_alone_does_not_mark_reviewed(self, model):
        """Verify that setting an inspector field alone does not change status to 'Reviewed'.

        Assigning an inspector without any annotations or notes is not sufficient to
        mark an image as reviewed. Success means status remains 'Pending' after only
        setting the inspector.
        """
        model.load_folder("/fake", ["img.jpg"])
        model.set_inspector(0, "Alice")
        assert model.data(model.index(0, 1)) == "Pending"


class TestQueryAPI:
    def test_get_annotations_returns_empty_list(self, model):
        """Verify that get_annotations returns an empty list for a newly loaded image.

        After loading a folder with one image and adding no annotations, the list of
        annotations should be empty. Success means get_annotations returns [].
        """
        model.load_folder("/fake", ["img.jpg"])
        assert model.get_annotations(0) == []

    def test_get_class_color_default(self, model):
        """Verify that get_class_color returns white (255, 255, 255) for unknown class names.

        When a class name is not registered, the model should return a default white
        color rather than raising a KeyError. Success means the return value is
        (255, 255, 255).
        """
        assert model.get_class_color("Unknown") == (255, 255, 255)

    def test_get_class_color_known(self, model):
        """Verify that get_class_color returns the registered color for a known class.

        After adding 'Defect' with red color, retrieving the color by the lowercased
        name 'defect' should return (255, 0, 0). Success means the stored color is
        returned correctly.
        """
        model.add_class("Defect", (255, 0, 0))
        assert model.get_class_color("defect") == (255, 0, 0)

    def test_class_visibility_defaults_visible_and_toggles(self, model):
        """Verify that a newly added class is visible by default and can be toggled.

        Adds 'Defect' and verifies it is initially visible. Toggling should flip the
        state each time. Success means visibility transitions correctly: True → False →
        True, returning the new state from toggle_class_visibility.
        """
        model.add_class("Defect", (255, 0, 0))

        assert model.is_class_visible("defect") is True
        assert model.toggle_class_visibility("defect") is False
        assert model.is_class_visible("defect") is False
        assert model.toggle_class_visibility("defect") is True

    def test_annotation_visibility_defaults_visible_and_toggles(self, model):
        """Verify that a newly added annotation is visible by default and can be toggled.

        Adds an annotation and verifies it is initially visible. Toggling should flip
        the visibility state each time. Success means the toggle transitions correctly
        and the is_annotation_visible state reflects each toggle.
        """
        model.load_folder("/fake", ["img.jpg"])
        model.add_annotation(0, "Defect", [(0, 0), (1, 0), (1, 1)])

        assert model.is_annotation_visible(0, 0) is True
        assert model.toggle_annotation_visibility(0, 0) is False
        assert model.is_annotation_visible(0, 0) is False
        assert model.toggle_annotation_visibility(0, 0) is True

    def test_get_class_annotation_count(self, model):
        """Verify that get_class_annotation_count returns the total count across all images.

        Adds two 'Defect' annotations across two images and one 'Other' annotation.
        Success means the count for 'defect' is 2, and for 'missing' (an unknown class)
        the count is 0.
        """
        model.load_folder("/fake", ["img.jpg", "other.jpg"])
        model.add_annotation(0, "Defect", [(0, 0), (1, 0), (1, 1)])
        model.add_annotation(0, "Other", [(0, 0), (1, 0), (1, 1)])
        model.add_annotation(1, "Defect", [(0, 0), (1, 0), (1, 1)])

        assert model.get_class_annotation_count("defect") == 2
        assert model.get_class_annotation_count("missing") == 0

    def test_get_inspector_empty_initially(self, model):
        """Verify that get_inspector returns an empty string for an image with no inspector set.

        Newly loaded images have no inspector assigned, so the initial value should be
        an empty string. Success means get_inspector returns ''.
        """
        model.load_folder("/fake", ["img.jpg"])
        assert model.get_inspector(0) == ""

    def test_get_note_after_set(self, model):
        """Verify that set_note and get_note round-trip the note text correctly.

        Sets a note string and immediately retrieves it. Success means the retrieved
        note matches the string that was set.
        """
        model.load_folder("/fake", ["img.jpg"])
        model.set_note(0, "Looks bad")
        assert model.get_note(0) == "Looks bad"

    def test_sort_annotations_by_area(self, model):
        """Verify that sort_annotations reorders annotations from largest area to smallest.

        Adds a small triangle (area ≈ 0.5) and a large square (area = 100). After
        sort_annotations, the larger polygon should appear first. Success means the
        first annotation has 4 vertices (the square), not 3 (the triangle).
        """
        model.load_folder("/fake", ["img.jpg"])
        model.add_annotation(0, "Defect", [(0, 0), (1, 0), (1, 1)])  # area ≈ 0.5
        model.add_annotation(
            0, "Defect", [(0, 0), (10, 0), (10, 10), (0, 10)]
        )  # area = 100
        model.sort_annotations(0)
        annos = model.get_annotations(0)
        # Large polygon should be first after sort
        assert len(annos[0]["polygon"]) == 4
