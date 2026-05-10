import pytest
from core.states.dataset_state import DatasetState
from models.dataset_model import DatasetTableModel
from controllers.io_controller import IOController
from views.annomate.window import ImageAnnotator


@pytest.fixture
def app_window(qtbot, tmp_path):
    state = DatasetState()
    model = DatasetTableModel(state)
    controller = IOController(model)
    window = ImageAnnotator(model, controller)
    qtbot.addWidget(window)

    (tmp_path / "a.jpg").touch()
    (tmp_path / "b.jpg").touch()
    controller.load_folder(str(tmp_path))
    return window, model, controller


class TestFolderLoad:
    def test_table_shows_correct_row_count(self, app_window):
        window, model, _ = app_window
        assert window.table_view.model().rowCount() == 2

    def test_lbl_img_updates_on_selection(self, qtbot, app_window):
        window, model, _ = app_window
        window.table_view.selectRow(0)
        assert "1 / 2" in window.lbl_img.text()


class TestAnnotationFlow:
    def test_polygon_finish_updates_ann_list(self, qtbot, app_window):
        window, model, _ = app_window
        window.table_view.selectRow(0)
        window.finish_polygon([(0.0, 0.0), (100.0, 0.0), (100.0, 100.0)])
        assert window.ann_list.count() == 1

    def test_delete_annotation_clears_ann_list(self, qtbot, app_window):
        window, model, _ = app_window
        window.table_view.selectRow(0)
        window.finish_polygon([(0, 0), (1, 0), (1, 1)])
        window.ann_list.setCurrentRow(0)
        window.delete_selected_annotation()
        assert window.ann_list.count() == 0
