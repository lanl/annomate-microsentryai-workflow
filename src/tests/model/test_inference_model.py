import numpy as np
import pytest

from core.states.inference_state import InferenceState
from models.inference_model import InferenceModel


@pytest.fixture
def model():
    return InferenceModel(InferenceState())


@pytest.fixture
def score_map():
    return np.array([[0.2, 0.7], [0.4, 0.9]], dtype=np.float32)


class TestInferenceModelQueryAPI:
    def test_get_score_map_none_initially(self, model):
        assert model.get_score_map("img.jpg") is None

    def test_is_processed_false_initially(self, model):
        assert model.is_processed("img.jpg") is False

    def test_get_processed_count_zero_initially(self, model):
        assert model.get_processed_count() == 0

    def test_set_and_get_score_map(self, model, score_map):
        model.set_score_map("img.jpg", 0.7, score_map)
        stored = model.get_score_map("img.jpg")
        np.testing.assert_array_equal(stored, score_map)

    def test_is_processed_true_after_set(self, model, score_map):
        model.set_score_map("img.jpg", 0.7, score_map)
        assert model.is_processed("img.jpg") is True

    def test_get_processed_count_increments(self, model, score_map):
        model.set_score_map("a.jpg", 0.3, score_map)
        model.set_score_map("b.jpg", 0.8, score_map)
        assert model.get_processed_count() == 2

    def test_overwrite_does_not_double_count(self, model, score_map):
        model.set_score_map("img.jpg", 0.7, score_map)
        model.set_score_map("img.jpg", 0.7, score_map)
        assert model.get_processed_count() == 1

    def test_unknown_key_is_not_processed(self, model, score_map):
        model.set_score_map("a.jpg", 0.6, score_map)
        assert model.is_processed("b.jpg") is False


class TestInferenceModelClear:
    def test_clear_resets_processed_count(self, model, score_map):
        model.set_score_map("img.jpg", 0.7, score_map)
        model.clear()
        assert model.get_processed_count() == 0

    def test_clear_makes_get_return_none(self, model, score_map):
        model.set_score_map("img.jpg", 0.7, score_map)
        model.clear()
        assert model.get_score_map("img.jpg") is None

    def test_clear_makes_is_processed_false(self, model, score_map):
        model.set_score_map("img.jpg", 0.7, score_map)
        model.clear()
        assert model.is_processed("img.jpg") is False

    def test_can_store_after_clear(self, model, score_map):
        model.set_score_map("img.jpg", 0.7, score_map)
        model.clear()
        model.set_score_map("new.jpg", 0.4, score_map)
        assert model.get_processed_count() == 1
