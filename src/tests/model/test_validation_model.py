import pytest

from core.states.validation_state import ValidationState
from models.validation_model import ValidationModel


@pytest.fixture
def model():
    return ValidationModel(ValidationState())


class TestValidationModelSettersAndGetters:
    def test_set_and_get_poly_path(self, model):
        model.set_poly_path("/images")
        assert model.get_poly_path() == "/images"

    def test_set_and_get_json_path(self, model):
        model.set_json_path("/data.json")
        assert model.get_json_path() == "/data.json"

    def test_set_and_get_mask_out_path(self, model):
        model.set_mask_out_path("/masks")
        assert model.get_mask_out_path() == "/masks"

    def test_set_and_get_gt_path(self, model):
        model.set_gt_path("/gt")
        assert model.get_gt_path() == "/gt"

    def test_set_and_get_pred_path(self, model):
        model.set_pred_path("/pred")
        assert model.get_pred_path() == "/pred"

    def test_set_and_get_eval_out_path(self, model):
        model.set_eval_out_path("/eval")
        assert model.get_eval_out_path() == "/eval"

    def test_eval_out_path_has_default(self, model):
        assert model.get_eval_out_path() != ""


class TestValidationModelMaskOutSeedsGtPath:
    def test_mask_out_seeds_gt_when_gt_empty(self, model):
        model.set_mask_out_path("/masks")
        assert model.get_gt_path() == "/masks"

    def test_mask_out_does_not_overwrite_existing_gt(self, model):
        model.set_gt_path("/existing_gt")
        model.set_mask_out_path("/masks")
        assert model.get_gt_path() == "/existing_gt"

    def test_mask_out_and_gt_can_differ(self, model):
        model.set_mask_out_path("/masks")
        model.set_gt_path("/different_gt")
        assert model.get_mask_out_path() == "/masks"
        assert model.get_gt_path() == "/different_gt"


class TestCanGenerate:
    def test_false_when_all_empty(self, model):
        assert model.can_generate() is False

    def test_false_when_only_poly_set(self, model):
        model.set_poly_path("/images")
        assert model.can_generate() is False

    def test_false_when_only_json_set(self, model):
        model.set_json_path("/data.json")
        assert model.can_generate() is False

    def test_false_when_only_mask_out_set(self, model):
        model.set_mask_out_path("/masks")
        assert model.can_generate() is False

    def test_false_when_poly_and_json_but_no_mask_out(self, model):
        model.set_poly_path("/images")
        model.set_json_path("/data.json")
        assert model.can_generate() is False

    def test_true_when_all_three_set(self, model):
        model.set_poly_path("/images")
        model.set_json_path("/data.json")
        model.set_mask_out_path("/masks")
        assert model.can_generate() is True


class TestCanEvaluate:
    def test_false_when_all_empty(self, model):
        assert model.can_evaluate() is False

    def test_false_when_only_gt_set(self, model):
        model.set_gt_path("/gt")
        assert model.can_evaluate() is False

    def test_false_when_only_pred_set(self, model):
        model.set_pred_path("/pred")
        assert model.can_evaluate() is False

    def test_true_when_both_set(self, model):
        model.set_gt_path("/gt")
        model.set_pred_path("/pred")
        assert model.can_evaluate() is True

    def test_true_via_mask_out_seeding(self, model):
        model.set_mask_out_path("/masks")  # seeds gt_path
        model.set_pred_path("/pred")
        assert model.can_evaluate() is True
