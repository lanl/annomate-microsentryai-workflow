import pytest
from core.states.dataset_state import DatasetState
from models.dataset_model import DatasetTableModel


@pytest.fixture
def model():
    return DatasetTableModel(DatasetState())


class TestModelReset:
    def test_load_folder_emits_modelReset(self, qtbot, model, tmp_path):
        (tmp_path / "a.jpg").touch()
        with qtbot.waitSignal(model.modelReset, timeout=1000):
            model.load_folder(str(tmp_path), ["a.jpg"])
        assert model.rowCount() == 1

    def test_row_count_after_load(self, model, tmp_path):
        model.load_folder(str(tmp_path), ["x.jpg", "y.jpg"])
        assert model.rowCount() == 2


class TestDataChanged:
    def test_add_annotation_emits_dataChanged(self, qtbot, model):
        model.load_folder("/fake", ["img.jpg"])
        with qtbot.waitSignal(model.dataChanged, timeout=1000):
            model.add_annotation(0, "Defect", [(0, 0), (1, 0), (1, 1)])

    def test_delete_annotation_emits_dataChanged(self, qtbot, model):
        model.load_folder("/fake", ["img.jpg"])
        model.add_annotation(0, "Defect", [(0, 0), (1, 0)])
        with qtbot.waitSignal(model.dataChanged, timeout=1000):
            model.delete_annotation(0, 0)


class TestStatusColumn:
    def test_status_is_pending_initially(self, model):
        model.load_folder("/fake", ["img.jpg"])
        assert model.data(model.index(0, 1)) == "Pending"

    def test_status_becomes_reviewed_after_annotation(self, model):
        model.load_folder("/fake", ["img.jpg"])
        model.add_annotation(0, "Defect", [(0, 0), (1, 0), (1, 1)])
        assert model.data(model.index(0, 1)) == "Reviewed"

    def test_status_reverts_to_pending_after_last_delete(self, model):
        model.load_folder("/fake", ["img.jpg"])
        model.add_annotation(0, "Defect", [(0, 0)])
        model.delete_annotation(0, 0)
        assert model.data(model.index(0, 1)) == "Pending"

    def test_inspector_marks_reviewed(self, model):
        model.load_folder("/fake", ["img.jpg"])
        model.set_inspector(0, "Alice")
        assert model.data(model.index(0, 1)) == "Reviewed"


class TestQueryAPI:
    def test_get_annotations_returns_empty_list(self, model):
        model.load_folder("/fake", ["img.jpg"])
        assert model.get_annotations(0) == []

    def test_get_class_color_default(self, model):
        assert model.get_class_color("Unknown") == (255, 255, 255)

    def test_get_class_color_known(self, model):
        assert model.get_class_color("Defect") == (255, 0, 0)

    def test_get_inspector_empty_initially(self, model):
        model.load_folder("/fake", ["img.jpg"])
        assert model.get_inspector(0) == ""

    def test_get_note_after_set(self, model):
        model.load_folder("/fake", ["img.jpg"])
        model.set_note(0, "Looks bad")
        assert model.get_note(0) == "Looks bad"

    def test_sort_annotations_by_area(self, model):
        model.load_folder("/fake", ["img.jpg"])
        model.add_annotation(0, "Defect", [(0, 0), (1, 0), (1, 1)])  # area ≈ 0.5
        model.add_annotation(
            0, "Defect", [(0, 0), (10, 0), (10, 10), (0, 10)]
        )  # area = 100
        model.sort_annotations(0)
        annos = model.get_annotations(0)
        # Large polygon should be first after sort
        assert len(annos[0]["polygon"]) == 4
