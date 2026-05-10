import pytest

from core.states.validation_state import ValidationState


@pytest.fixture
def state():
    return ValidationState()


class TestValidationStateDefaults:
    def test_poly_path_empty_on_init(self, state):
        assert state.poly_path == ""

    def test_json_path_empty_on_init(self, state):
        assert state.json_path == ""

    def test_mask_out_path_empty_on_init(self, state):
        assert state.mask_out_path == ""

    def test_gt_path_empty_on_init(self, state):
        assert state.gt_path == ""

    def test_pred_path_empty_on_init(self, state):
        assert state.pred_path == ""

    def test_eval_out_path_set_by_default(self, state):
        assert state.eval_out_path != ""
        assert "evaluation_results" in state.eval_out_path


class TestValidationStateAssignment:
    def test_direct_path_assignment(self, state):
        state.poly_path = "/some/images"
        assert state.poly_path == "/some/images"

    def test_all_paths_settable(self, state):
        state.poly_path = "/images"
        state.json_path = "/data.json"
        state.mask_out_path = "/masks"
        state.gt_path = "/gt"
        state.pred_path = "/pred"
        state.eval_out_path = "/eval"
        assert state.poly_path == "/images"
        assert state.json_path == "/data.json"
        assert state.mask_out_path == "/masks"
        assert state.gt_path == "/gt"
        assert state.pred_path == "/pred"
        assert state.eval_out_path == "/eval"


class TestValidationStateClear:
    def test_clear_resets_poly_path(self, state):
        state.poly_path = "/images"
        state.clear()
        assert state.poly_path == ""

    def test_clear_resets_json_path(self, state):
        state.json_path = "/data.json"
        state.clear()
        assert state.json_path == ""

    def test_clear_resets_mask_out_path(self, state):
        state.mask_out_path = "/masks"
        state.clear()
        assert state.mask_out_path == ""

    def test_clear_resets_gt_path(self, state):
        state.gt_path = "/gt"
        state.clear()
        assert state.gt_path == ""

    def test_clear_resets_pred_path(self, state):
        state.pred_path = "/pred"
        state.clear()
        assert state.pred_path == ""

    def test_clear_preserves_eval_out_path(self, state):
        original = state.eval_out_path
        state.eval_out_path = "/custom/eval"
        state.clear()
        assert state.eval_out_path == "/custom/eval"
        assert state.eval_out_path != original or original == "/custom/eval"
