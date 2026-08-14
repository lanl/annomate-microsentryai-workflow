import numpy as np
import pytest

from core.states.inference_state import InferenceState


@pytest.fixture
def state():
    return InferenceState()


@pytest.fixture
def score_map():
    arr = np.array([[0.1, 0.5], [0.9, 0.3]], dtype=np.float32)
    return arr


class TestInferenceStateDefaults:
    def test_score_maps_empty_on_init(self, state):
        """Verify that a new InferenceState starts with no stored score maps.

        On initialization, score_maps should be an empty dict. Success means the
        attribute exists and equals {}.
        """
        assert state.score_maps == {}

    def test_inference_cache_empty_on_init(self, state):
        """Verify that a new InferenceState starts with an empty inference cache.

        On initialization, inference_cache should be an empty dict containing no
        previously computed scores. Success means the attribute equals {}.
        """
        assert state.inference_cache == {}

    def test_is_processed_false_for_unknown(self, state):
        """Verify that is_processed returns False for a filename that was never stored.

        Querying a filename that has never had a score map set should return False
        without raising a KeyError. Success means the return value is exactly False.
        """
        assert state.is_processed("missing.jpg") is False

    def test_get_score_map_none_for_unknown(self, state):
        """Verify that get_score_map returns None for a filename with no stored map.

        Querying a filename not present in score_maps should return None rather than
        raising a KeyError. Success means the return value is None.
        """
        assert state.get_score_map("missing.jpg") is None


class TestInferenceStateMutations:
    def test_set_score_map_stores_array(self, state, score_map):
        """Verify that set_score_map persists the numpy array and makes it retrievable.

        Stores a score map for a filename and immediately retrieves it. Success means
        get_score_map returns a non-None array that is element-wise equal to what was stored.
        """
        state.set_score_map("img.jpg", 0.7, score_map)
        stored = state.get_score_map("img.jpg")
        assert stored is not None
        np.testing.assert_array_equal(stored, score_map)

    def test_is_processed_true_after_set(self, state, score_map):
        """Verify that is_processed returns True after a score map is stored for a file.

        After calling set_score_map, the corresponding filename should be recognized
        as processed. Success means is_processed returns True for that exact filename.
        """
        state.set_score_map("img.jpg", 0.7, score_map)
        assert state.is_processed("img.jpg") is True

    def test_set_score_map_caches_passed_score(self, state, score_map):
        """Verify that the scalar score is stored in both inference_cache and scores.

        The score passed to set_score_map should be accessible via both inference_cache
        and scores dicts under the same filename key. Success means both dicts contain
        the expected score value.
        """
        state.set_score_map("img.jpg", 0.71, score_map)
        assert state.inference_cache["img.jpg"] == pytest.approx(0.71)
        assert state.scores["img.jpg"] == pytest.approx(0.71)

    def test_anomaly_label_above_threshold(self, state, score_map):
        """Verify that a score above the classification threshold is labeled ANOMALY.

        A score of 0.6 is above the default threshold (0.5), so the image should be
        classified as 'ANOMALY'. Success means labels['img.jpg'] equals 'ANOMALY'.
        """
        state.set_score_map("img.jpg", 0.6, score_map)
        assert state.labels["img.jpg"] == "ANOMALY"

    def test_normal_label_below_threshold(self, state, score_map):
        """Verify that a score below the classification threshold is labeled NORMAL.

        A score of 0.4 is below the default threshold (0.5), so the image should be
        classified as 'NORMAL'. Success means labels['img.jpg'] equals 'NORMAL'.
        """
        state.set_score_map("img.jpg", 0.4, score_map)
        assert state.labels["img.jpg"] == "NORMAL"

    def test_normal_label_at_zero(self, state):
        """Verify that a score of exactly 0.0 is classified as NORMAL without error.

        Edge case: a blank heatmap with score 0.0 should classify as 'NORMAL' and
        the cache should store 0.0. Success means no exception is raised and labels
        and cache reflect the zero score correctly.
        """
        state.set_score_map("blank.jpg", 0.0, np.zeros((4, 4), dtype=np.float32))
        assert state.inference_cache["blank.jpg"] == pytest.approx(0.0)
        assert state.labels["blank.jpg"] == "NORMAL"

    def test_multiple_images_stored_independently(self, state):
        """Verify that score maps for different filenames are stored independently.

        Stores score maps for two different files and confirms each is retrievable and
        processed independently. Success means both filenames are marked as processed
        and their cached scores are distinct and correct.
        """
        a = np.full((2, 2), 0.2, dtype=np.float32)
        b = np.full((2, 2), 0.8, dtype=np.float32)
        state.set_score_map("a.jpg", 0.2, a)
        state.set_score_map("b.jpg", 0.8, b)
        assert state.is_processed("a.jpg")
        assert state.is_processed("b.jpg")
        assert state.inference_cache["a.jpg"] == pytest.approx(0.2)
        assert state.inference_cache["b.jpg"] == pytest.approx(0.8)

    def test_overwrite_updates_cache(self, state):
        """Verify that storing a second score map for the same filename replaces the first.

        Re-running inference on the same image should overwrite the previous result.
        Success means inference_cache contains the most recently stored score (0.9),
        not the original (0.3).
        """
        state.set_score_map("img.jpg", 0.3, np.full((2, 2), 0.3, dtype=np.float32))
        state.set_score_map("img.jpg", 0.9, np.full((2, 2), 0.9, dtype=np.float32))
        assert state.inference_cache["img.jpg"] == pytest.approx(0.9)


class TestInferenceStateClear:
    def test_clear_empties_score_maps(self, state, score_map):
        """Verify that clear() removes all stored score map arrays.

        After storing a score map and calling clear(), the score_maps dict should be
        empty. Success means score_maps equals {} after the clear operation.
        """
        state.set_score_map("img.jpg", 0.7, score_map)
        state.clear()
        assert state.score_maps == {}

    def test_clear_empties_inference_cache(self, state, score_map):
        """Verify that clear() removes all entries from the inference cache.

        After storing a score map and calling clear(), inference_cache should be empty.
        Success means inference_cache equals {} after the clear operation.
        """
        state.set_score_map("img.jpg", 0.7, score_map)
        state.clear()
        assert state.inference_cache == {}

    def test_clear_empties_scores_and_labels(self, state, score_map):
        """Verify that clear() removes all entries from the scores and labels dicts.

        Both scores and labels should be empty dicts after clearing state that had
        previously stored results. Success means both dicts equal {} after clear.
        """
        state.set_score_map("img.jpg", 0.7, score_map)
        state.clear()
        assert state.scores == {}
        assert state.labels == {}

    def test_is_processed_false_after_clear(self, state, score_map):
        """Verify that is_processed returns False for a previously stored file after clear.

        A filename that was processed before clear() should not be recognized as
        processed after. Success means is_processed returns False for that filename.
        """
        state.set_score_map("img.jpg", 0.7, score_map)
        state.clear()
        assert state.is_processed("img.jpg") is False

    def test_can_store_after_clear(self, state, score_map):
        """Verify that new score maps can be stored after the state has been cleared.

        After a clear(), the state should be fully usable again for new results without
        any lingering state. Success means a new filename is recognized as processed
        after being stored following a clear.
        """
        state.set_score_map("img.jpg", 0.7, score_map)
        state.clear()
        state.set_score_map("new.jpg", 0.4, score_map)
        assert state.is_processed("new.jpg")


class TestInferenceStateMultiModel:
    def test_register_model_creates_empty_scores_entry(self, state):
        """Verify that registering a model creates an empty model_scores entry.

        Registering a new model key should make it queryable in known_models
        and give it an empty (not missing) entry in model_scores, ready to
        receive scores once it becomes active. Success means both dicts
        contain the key.
        """
        state.register_model("cfa", "/models/cfa.pt", "scoremaps/cfa-scoremaps.npz")
        assert state.known_models["cfa"]["model_path"] == "/models/cfa.pt"
        assert state.model_scores["cfa"] == {}

    def test_reregistering_with_empty_score_maps_file_keeps_existing_path(self, state):
        """Verify re-registering a known model doesn't erase its saved NPZ path.

        window.py always calls register_model(key, path, "") when loading
        model weights, since it doesn't know the NPZ path at that point. If
        this model already has a real cached score_maps_file on record (from
        a previous save), re-registering it must not clobber that path with
        an empty string — doing so would orphan its on-disk cache. Success
        means the original path survives a re-register with "".
        """
        state.register_model("cfa", "/models/cfa.pt", "scoremaps/cfa-scoremaps.npz")
        state.register_model("cfa", "/models/cfa.pt", "")
        assert (
            state.known_models["cfa"]["score_maps_file"]
            == "scoremaps/cfa-scoremaps.npz"
        )

    def test_switch_active_model_updates_active_key(self, state):
        """Verify that switching the active model updates active_model_key.

        Success means active_model_key reflects the most recently switched-to key.
        """
        state.register_model("cfa", "/models/cfa.pt", "")
        state.switch_active_model("cfa")
        assert state.active_model_key == "cfa"

    def test_set_score_map_after_switch_updates_model_scores(self, state, score_map):
        """Verify that scores set while a model is active land in model_scores too.

        Because `scores` aliases `model_scores[active_model_key]` after a switch,
        calling set_score_map should update both without any extra plumbing.
        Success means the score is visible under both `scores` and `model_scores`.
        """
        state.register_model("cfa", "/models/cfa.pt", "")
        state.switch_active_model("cfa")
        state.set_score_map("img.jpg", 0.8, score_map)
        assert state.scores["img.jpg"] == pytest.approx(0.8)
        assert state.model_scores["cfa"]["img.jpg"] == pytest.approx(0.8)

    def test_switching_models_preserves_previous_models_scores(self, state, score_map):
        """Verify that switching away from a model keeps its scores intact.

        Sets a score under model "cfa", switches to model "efficientad", then
        switches back. Success means cfa's score for img.jpg is still there —
        switching does not lose the outgoing model's cached results.
        """
        state.register_model("cfa", "/models/cfa.pt", "")
        state.switch_active_model("cfa")
        state.set_score_map("img.jpg", 0.8, score_map)

        state.register_model("efficientad", "/models/efficientad.pt", "")
        state.switch_active_model("efficientad")
        assert "img.jpg" not in state.scores  # different model, no scores yet

        state.switch_active_model("cfa")
        assert state.scores["img.jpg"] == pytest.approx(0.8)

    def test_switch_active_model_clears_score_maps(self, state, score_map):
        """Verify that switching models clears the active heatmap arrays.

        Heatmaps are never kept resident for more than one model at a time —
        the caller is responsible for loading the new model's heatmaps from
        disk afterward. Success means score_maps is empty right after a switch.
        """
        state.register_model("cfa", "/models/cfa.pt", "")
        state.switch_active_model("cfa")
        state.set_score_map("img.jpg", 0.8, score_map)

        state.register_model("efficientad", "/models/efficientad.pt", "")
        state.switch_active_model("efficientad")
        assert state.score_maps == {}

    def test_switch_active_model_resets_dirty_flag(self, state, score_map):
        """Verify that switching models clears the dirty flag.

        The newly-activated model has no unsaved in-memory changes yet (its
        heatmaps, if any, come from disk). Success means score_maps_dirty is
        False immediately after switching.
        """
        state.register_model("cfa", "/models/cfa.pt", "")
        state.switch_active_model("cfa")
        state.set_score_map("img.jpg", 0.8, score_map)
        assert state.score_maps_dirty is True

        state.register_model("efficientad", "/models/efficientad.pt", "")
        state.switch_active_model("efficientad")
        assert state.score_maps_dirty is False

    def test_clear_wipes_model_registry(self, state, score_map):
        """Verify that clear() resets the entire multi-model registry.

        clear() represents a full reset (e.g. starting a new project), not
        just deactivating the current model — it should wipe known_models,
        model_scores, and active_model_key too. Success means all three are
        empty/blank after clear().
        """
        state.register_model("cfa", "/models/cfa.pt", "")
        state.switch_active_model("cfa")
        state.set_score_map("img.jpg", 0.8, score_map)

        state.clear()
        assert state.known_models == {}
        assert state.model_scores == {}
        assert state.active_model_key == ""
