import numpy as np
import pytest

from core.logic.mask_comparator import MaskComparator


@pytest.fixture
def cmp():
    return MaskComparator()


def _solid(h, w, val=255) -> np.ndarray:
    arr = np.zeros((h, w), dtype=np.uint8)
    arr[:] = val
    return arr


def _rect(h, w, r0, r1, c0, c1) -> np.ndarray:
    arr = np.zeros((h, w), dtype=np.uint8)
    arr[r0:r1, c0:c1] = 255
    return arr


class TestCalculateMetrics:
    def test_perfect_match_gives_100_iou(self, cmp):
        mask = _rect(10, 10, 2, 8, 2, 8)
        m = cmp.calculate_metrics(mask, mask.copy())
        assert m["iou"] == pytest.approx(100.0)
        assert m["precision"] == pytest.approx(100.0)
        assert m["recall"] == pytest.approx(100.0)

    def test_no_overlap_gives_0_iou(self, cmp):
        gt = _rect(10, 10, 0, 5, 0, 10)
        pred = _rect(10, 10, 5, 10, 0, 10)
        m = cmp.calculate_metrics(gt, pred)
        assert m["iou"] == pytest.approx(0.0)
        assert m["precision"] == pytest.approx(0.0)
        assert m["recall"] == pytest.approx(0.0)

    def test_pred_subset_of_gt(self, cmp):
        gt = _solid(10, 10)  # 100 px
        pred = _rect(10, 10, 0, 5, 0, 10)  # 50 px, fully inside gt
        m = cmp.calculate_metrics(gt, pred)
        assert m["iou"] == pytest.approx(50.0)
        assert m["precision"] == pytest.approx(100.0)
        assert m["recall"] == pytest.approx(50.0)

    def test_gt_subset_of_pred(self, cmp):
        pred = _solid(10, 10)  # 100 px
        gt = _rect(10, 10, 0, 5, 0, 10)  # 50 px, fully inside pred
        m = cmp.calculate_metrics(gt, pred)
        assert m["iou"] == pytest.approx(50.0)
        assert m["precision"] == pytest.approx(50.0)
        assert m["recall"] == pytest.approx(100.0)

    def test_both_empty_returns_zeros(self, cmp):
        empty = np.zeros((10, 10), dtype=np.uint8)
        m = cmp.calculate_metrics(empty, empty)
        assert m["iou"] == pytest.approx(0.0)
        assert m["precision"] == pytest.approx(0.0)
        assert m["recall"] == pytest.approx(0.0)

    def test_area_values_are_correct(self, cmp):
        gt = _rect(10, 10, 0, 4, 0, 4)  # 16 px
        pred = _rect(10, 10, 0, 4, 0, 4)  # 16 px
        m = cmp.calculate_metrics(gt, pred)
        assert m["gt_area"] == 16
        assert m["pred_area"] == 16
        assert m["overlap_area"] == 16
        assert m["union_area"] == 16

    def test_centroid_present_when_masks_filled(self, cmp):
        mask = _rect(10, 10, 0, 10, 0, 10)
        m = cmp.calculate_metrics(mask, mask.copy())
        cx_gt, cy_gt = m["gt_centroid"]
        cx_pred, cy_pred = m["pred_centroid"]
        assert cx_gt is not None
        assert cy_gt is not None
        assert cx_pred is not None
        assert cy_pred is not None

    def test_centroid_none_when_mask_empty(self, cmp):
        gt = _rect(10, 10, 0, 5, 0, 5)
        empty = np.zeros((10, 10), dtype=np.uint8)
        m = cmp.calculate_metrics(gt, empty)
        assert m["pred_centroid"] == (None, None)

    def test_euclidean_distance_zero_for_same_mask(self, cmp):
        mask = _rect(10, 10, 2, 8, 2, 8)
        m = cmp.calculate_metrics(mask, mask.copy())
        assert m["euclidean_distance"] == pytest.approx(0.0)

    def test_euclidean_distance_none_when_pred_empty(self, cmp):
        gt = _rect(10, 10, 0, 5, 0, 5)
        empty = np.zeros((10, 10), dtype=np.uint8)
        m = cmp.calculate_metrics(gt, empty)
        assert m["euclidean_distance"] is None

    def test_metrics_dict_has_all_keys(self, cmp):
        mask = _rect(10, 10, 0, 5, 0, 5)
        m = cmp.calculate_metrics(mask, mask.copy())
        required = {
            "gt_area",
            "pred_area",
            "overlap_area",
            "union_area",
            "iou",
            "precision",
            "recall",
            "gt_centroid",
            "pred_centroid",
            "euclidean_distance",
        }
        assert required <= m.keys()


class TestGenerateComparisonViz:
    def test_output_shapes_match_input(self, cmp):
        mask = _rect(20, 30, 0, 10, 0, 15)
        m = cmp.calculate_metrics(mask, mask.copy())
        comp, overlay = cmp.generate_comparison_viz(mask, mask.copy(), m)
        assert comp.shape == (20, 30, 3)
        assert overlay.shape == (20, 30, 3)

    def test_tp_pixels_white_in_comparison_map(self, cmp):
        mask = _rect(10, 10, 0, 5, 0, 5)
        m = cmp.calculate_metrics(mask, mask.copy())
        comp, _ = cmp.generate_comparison_viz(mask, mask.copy(), m)
        # TP region: top-left 5x5 should be white [255, 255, 255]
        assert (comp[2, 2] == [255, 255, 255]).all()

    def test_fn_pixels_red_in_comparison_map(self, cmp):
        gt = _rect(10, 10, 0, 10, 0, 5)
        pred = np.zeros((10, 10), dtype=np.uint8)  # no prediction
        m = cmp.calculate_metrics(gt, pred)
        comp, _ = cmp.generate_comparison_viz(gt, pred, m)
        # FN (gt has pixel, pred does not) → red [0, 0, 255] in BGR
        assert (comp[2, 2] == [0, 0, 255]).all()

    def test_fp_pixels_green_in_comparison_map(self, cmp):
        gt = np.zeros((10, 10), dtype=np.uint8)
        pred = _rect(10, 10, 0, 10, 0, 5)
        m = cmp.calculate_metrics(gt, pred)
        comp, _ = cmp.generate_comparison_viz(gt, pred, m)
        # FP → green [0, 255, 0]
        assert (comp[2, 2] == [0, 255, 0]).all()


class TestCompareMasks:
    def test_returns_three_items(self, cmp):
        mask = _rect(10, 10, 0, 5, 0, 5)
        result = cmp.compare_masks(mask, mask.copy())
        assert len(result) == 3

    def test_third_item_is_metrics_dict(self, cmp):
        mask = _rect(10, 10, 0, 5, 0, 5)
        _, _, metrics = cmp.compare_masks(mask, mask.copy())
        assert isinstance(metrics, dict)
        assert "iou" in metrics

    def test_perfect_match_end_to_end(self, cmp):
        mask = _rect(10, 10, 2, 8, 2, 8)
        _, _, metrics = cmp.compare_masks(mask, mask.copy())
        assert metrics["iou"] == pytest.approx(100.0)
