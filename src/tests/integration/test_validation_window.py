import pytest
from unittest.mock import patch

from core.states.validation_state import ValidationState
from models.validation_model import ValidationModel
from controllers.validation_controller import ValidationController
from views.validation.window import ValidationWindow


@pytest.fixture
def validation_window(qtbot):
    state = ValidationState()
    model = ValidationModel(state)
    controller = ValidationController(model)
    window = ValidationWindow(model, controller)
    qtbot.addWidget(window)
    return window, model, controller


class TestValidationWindowInit:
    def test_generate_button_exists(self, validation_window):
        window, _, _ = validation_window
        assert window.btn_gen is not None

    def test_run_button_exists(self, validation_window):
        window, _, _ = validation_window
        assert window.btn_run is not None

    def test_buttons_enabled_on_init(self, validation_window):
        window, _, _ = validation_window
        assert window.btn_gen.isEnabled()
        assert window.btn_run.isEnabled()

    def test_path_labels_show_not_selected(self, validation_window):
        window, _, _ = validation_window
        assert window.lbl_poly.text() == "Not selected"
        assert window.lbl_json.text() == "Not selected"
        assert window.lbl_mask_out.text() == "Not selected"
        assert window.lbl_gt.text() == "Not selected"
        assert window.lbl_pred.text() == "Not selected"

    def test_progress_bar_exists(self, validation_window):
        window, _, _ = validation_window
        assert window.pbar is not None

    def test_results_feed_empty_on_init(self, validation_window):
        window, _, _ = validation_window
        assert window.results_layout.count() == 0


class TestValidationWindowUiState:
    def test_set_ui_state_false_disables_buttons(self, validation_window):
        window, _, _ = validation_window
        window._set_ui_state(False)
        assert not window.btn_gen.isEnabled()
        assert not window.btn_run.isEnabled()

    def test_set_ui_state_true_re_enables_buttons(self, validation_window):
        window, _, _ = validation_window
        window._set_ui_state(False)
        window._set_ui_state(True)
        assert window.btn_gen.isEnabled()
        assert window.btn_run.isEnabled()


class TestValidationWindowResultsFeed:
    def test_add_log_text_appends_widget(self, validation_window):
        window, _, _ = validation_window
        window._add_log_text("hello world")
        assert window.results_layout.count() == 1

    def test_add_multiple_log_entries(self, validation_window):
        window, _, _ = validation_window
        window._add_log_text("line 1")
        window._add_log_text("line 2")
        window._add_log_text("line 3")
        assert window.results_layout.count() == 3

    def test_clear_results_removes_all_widgets(self, validation_window):
        window, _, _ = validation_window
        window._add_log_text("line 1")
        window._add_log_text("line 2")
        window._clear_results()
        assert window.results_layout.count() == 0

    def test_add_result_card_appends_widget(self, validation_window, tmp_path):
        window, _, _ = validation_window
        import cv2
        import numpy as np

        img_path = str(tmp_path / "overlay.png")
        cv2.imwrite(img_path, np.ones((10, 10, 3), dtype=np.uint8) * 200)
        window._add_result_card(img_path, "Tray 001 | IoU: 75.0%", 75.0)
        assert window.results_layout.count() == 1


class TestValidationWindowRunGuards:
    def test_run_generation_without_paths_does_not_disable_buttons(
        self, validation_window
    ):
        window, _, _ = validation_window
        # Patch QMessageBox so the blocking warning dialog never appears.
        with patch("views.validation.window.QMessageBox") as mock_qmb:
            mock_qmb.warning.return_value = None
            window._run_generation()
        assert window.btn_gen.isEnabled()

    def test_run_evaluation_without_paths_does_not_disable_buttons(
        self, validation_window
    ):
        window, _, _ = validation_window
        with patch("views.validation.window.QMessageBox") as mock_qmb:
            mock_qmb.warning.return_value = None
            window._run_evaluation()
        assert window.btn_run.isEnabled()


class TestValidationWindowPathReflection:
    def test_model_path_reflected_via_label_after_selection(self, validation_window):
        window, model, _ = validation_window
        model.set_poly_path("/test/images")
        # Simulate what _select_poly does after the dialog
        window.lbl_poly.setText(model.get_poly_path())
        assert window.lbl_poly.text() == "/test/images"

    def test_mask_out_seeding_reflected_in_gt_label(self, validation_window):
        window, model, _ = validation_window
        model.set_mask_out_path("/masks")
        # Reflect seeding (mirrors _select_mask_out logic)
        gt = model.get_gt_path()
        window.lbl_gt.setText(gt)
        assert window.lbl_gt.text() == "/masks"
