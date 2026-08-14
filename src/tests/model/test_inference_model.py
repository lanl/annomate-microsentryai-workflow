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
        """Verify that get_score_map returns None for a filename that has not been processed.

        On a fresh InferenceModel, no images have been processed, so any filename lookup
        should return None. Success means the return value is None.
        """
        assert model.get_score_map("img.jpg") is None

    def test_is_processed_false_initially(self, model):
        """Verify that is_processed returns False for any image on a fresh model.

        Before any inference has been run, no images should be marked as processed.
        Success means is_processed returns False for any filename.
        """
        assert model.is_processed("img.jpg") is False

    def test_get_processed_count_zero_initially(self, model):
        """Verify that the processed count starts at zero on a fresh InferenceModel.

        No images have been processed yet, so the count should be 0. Success means
        get_processed_count returns 0.
        """
        assert model.get_processed_count() == 0

    def test_set_and_get_score_map(self, model, score_map):
        """Verify that a score map stored via set_score_map can be retrieved via get_score_map.

        Stores a score map array and immediately retrieves it. Success means the
        retrieved array is element-wise equal to the original.
        """
        model.set_score_map("img.jpg", 0.7, score_map)
        stored = model.get_score_map("img.jpg")
        np.testing.assert_array_equal(stored, score_map)

    def test_is_processed_true_after_set(self, model, score_map):
        """Verify that is_processed returns True after a score map is stored for an image.

        After calling set_score_map, the image should be recognized as processed.
        Success means is_processed returns True for that filename.
        """
        model.set_score_map("img.jpg", 0.7, score_map)
        assert model.is_processed("img.jpg") is True

    def test_get_processed_count_increments(self, model, score_map):
        """Verify that get_processed_count increments correctly as new images are processed.

        Stores score maps for two different images. Success means the processed count
        equals 2 after both have been stored.
        """
        model.set_score_map("a.jpg", 0.3, score_map)
        model.set_score_map("b.jpg", 0.8, score_map)
        assert model.get_processed_count() == 2

    def test_overwrite_does_not_double_count(self, model, score_map):
        """Verify that overwriting a score map for the same image does not increment the count twice.

        Storing a score map for the same filename twice should count as one processed
        image, not two. Success means get_processed_count returns 1.
        """
        model.set_score_map("img.jpg", 0.7, score_map)
        model.set_score_map("img.jpg", 0.7, score_map)
        assert model.get_processed_count() == 1

    def test_unknown_key_is_not_processed(self, model, score_map):
        """Verify that is_processed returns False for a different filename than the one stored.

        Storing a score map for 'a.jpg' should not affect the processed state of 'b.jpg'.
        Success means is_processed('b.jpg') returns False after storing only 'a.jpg'.
        """
        model.set_score_map("a.jpg", 0.6, score_map)
        assert model.is_processed("b.jpg") is False


class TestInferenceModelClear:
    def test_clear_resets_processed_count(self, model, score_map):
        """Verify that clear() resets the processed count to zero.

        After storing a score map and calling clear(), the processed image count should
        return to 0. Success means get_processed_count returns 0 after clear.
        """
        model.set_score_map("img.jpg", 0.7, score_map)
        model.clear()
        assert model.get_processed_count() == 0

    def test_clear_makes_get_return_none(self, model, score_map):
        """Verify that get_score_map returns None for previously stored images after clear.

        After clearing, score maps should no longer be accessible. Success means
        get_score_map returns None for a filename that was stored before the clear.
        """
        model.set_score_map("img.jpg", 0.7, score_map)
        model.clear()
        assert model.get_score_map("img.jpg") is None

    def test_clear_makes_is_processed_false(self, model, score_map):
        """Verify that is_processed returns False for previously processed images after clear.

        After clearing, all images should be back to an unprocessed state. Success
        means is_processed returns False for a filename that was processed before clear.
        """
        model.set_score_map("img.jpg", 0.7, score_map)
        model.clear()
        assert model.is_processed("img.jpg") is False

    def test_can_store_after_clear(self, model, score_map):
        """Verify that new score maps can be stored and counted after a clear operation.

        The model should be fully functional after clear(). Success means a new image
        stored after a clear is counted correctly (processed count equals 1).
        """
        model.set_score_map("img.jpg", 0.7, score_map)
        model.clear()
        model.set_score_map("new.jpg", 0.4, score_map)
        assert model.get_processed_count() == 1


class TestInferenceModelClearActiveHeatmaps:
    def test_clear_active_heatmaps_makes_images_reprocessable(self, model, score_map):
        """Verify clear_active_heatmaps() un-marks previously processed images.

        Reproduces the "loading a model does nothing" bug: an image already
        marked processed (e.g. from cached heatmaps loaded at project-open)
        must become is_processed()==False again after this call, so a fresh
        model load actually triggers new inference instead of finding
        everything "already done".
        """
        model.register_model("cfa", "/models/cfa.pt", "scoremaps/cfa-scoremaps.npz")
        model.switch_active_model("cfa")
        model.set_score_map("img.jpg", 0.7, score_map)
        assert model.is_processed("img.jpg") is True

        model.clear_active_heatmaps()

        assert model.is_processed("img.jpg") is False

    def test_clear_active_heatmaps_resets_dirty_flag(self, model, score_map):
        """Verify clear_active_heatmaps() leaves the dirty flag False.

        There's nothing unsaved right after clearing — dirty should only
        become True again once new inference results are stored.
        """
        model.register_model("cfa", "/models/cfa.pt", "")
        model.switch_active_model("cfa")
        model.set_score_map("img.jpg", 0.7, score_map)
        assert model.is_score_maps_dirty() is True

        model.clear_active_heatmaps()

        assert model.is_score_maps_dirty() is False

    def test_clear_active_heatmaps_does_not_touch_other_models_scores(
        self, model, score_map
    ):
        """Verify clearing the active model's heatmaps doesn't affect other models.

        Only the active model's score_maps should be discarded — scalar
        scores for every known model, and their on-disk caches, are
        untouched by this operation.
        """
        model.register_model("cfa", "/models/cfa.pt", "")
        model.switch_active_model("cfa")
        model.set_score_map("img.jpg", 0.7, score_map)

        model.register_model("efficientad", "/models/efficientad.pt", "")
        model.switch_active_model("efficientad")
        model.clear_active_heatmaps()

        model.switch_active_model("cfa")
        assert model.get_score("img.jpg") == pytest.approx(0.7)
