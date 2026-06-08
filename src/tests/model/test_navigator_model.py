import numpy as np
import pytest
from PySide6.QtCore import Qt

from core.states.dataset_state import DatasetState
from core.states.inference_state import InferenceState
from models.dataset_model import DatasetTableModel
from models.inference_model import InferenceModel
from models.navigator_model import (
    NavigatorColumns,
    NavigatorSortProxyModel,
    NavigatorTableModel,
    SOURCE_ROW_ROLE,
    SORT_ROLE,
)


@pytest.fixture
def dataset_model(tmp_path):
    model = DatasetTableModel(DatasetState())
    model.load_folder(str(tmp_path), ["b.jpg", "a.jpg", "c.jpg"])
    return model


@pytest.fixture
def inference_model():
    return InferenceModel(InferenceState())


def proxy_source_rows(proxy):
    return [
        proxy.mapToSource(proxy.index(row, NavigatorColumns.IMG_ID)).row()
        for row in range(proxy.rowCount())
    ]


class TestNavigatorTableModel:
    def test_columns_and_source_row_role(self, dataset_model, inference_model):
        """Verify that NavigatorTableModel has 6 columns and exposes correct source row via SOURCE_ROW_ROLE.

        Checks the column count, the IMG_ID header text, the display value for row 1
        (which is the file 'a.jpg', so ID should be 'a'), and that SOURCE_ROW_ROLE
        returns the original source row index. Success means all four assertions pass.
        """
        model = NavigatorTableModel(dataset_model, inference_model)

        assert model.columnCount() == 6
        assert model.headerData(NavigatorColumns.IMG_ID, Qt.Horizontal) == "Img ID"
        assert model.data(model.index(1, NavigatorColumns.IMG_ID)) == "a"
        assert model.data(model.index(1, NavigatorColumns.IMG_ID), SOURCE_ROW_ROLE) == 1

    def test_annotation_and_decision_values(self, dataset_model, inference_model):
        """Verify that annotation count, review status, and review decision columns display correctly.

        Adds one annotation and sets a 'reject' review decision for the first image.
        Success means the STATUS SORT_ROLE is 1 (reviewed), ANNOTS shows '1', and
        DECISION shows 'Reject' (capitalized).
        """
        model = NavigatorTableModel(dataset_model, inference_model)

        dataset_model.add_annotation(0, "Defect", [(0, 0), (1, 0), (1, 1)])
        dataset_model.set_review_decision(0, "reject")

        assert model.data(model.index(0, NavigatorColumns.STATUS), SORT_ROLE) == 1
        assert model.data(model.index(0, NavigatorColumns.ANNOTS)) == "1"
        assert model.data(model.index(0, NavigatorColumns.DECISION)) == "Reject"

    def test_inference_values_and_missing_score(
        self, dataset_model, inference_model, tmp_path
    ):
        """Verify that inference score and class columns show values when available and empty when not.

        Stores a score map for 'a.jpg' (row 1). Success means SCORE shows '0.72',
        CLASS shows 'ANOMALY' for that row, and for row 0 (no score) SCORE is an empty
        string and SORT_ROLE is None.
        """
        inference_model.set_score_map(
            str(tmp_path / "a.jpg"), 0.72, np.zeros((2, 2), dtype=np.float32)
        )
        model = NavigatorTableModel(dataset_model, inference_model)

        assert model.data(model.index(1, NavigatorColumns.SCORE)) == "0.72"
        assert model.data(model.index(1, NavigatorColumns.CLASS)) == "ANOMALY"
        assert model.data(model.index(0, NavigatorColumns.SCORE)) == ""
        assert model.data(model.index(0, NavigatorColumns.SCORE), SORT_ROLE) is None


class TestNavigatorSortProxyModel:
    def test_img_id_sort_is_case_insensitive_with_source_rows(
        self, dataset_model, inference_model
    ):
        """Verify that sorting by image ID is case-insensitive and maps correctly to source rows.

        The dataset has 'b.jpg', 'a.jpg', 'c.jpg' in that order. Ascending sort by ID
        should order them a, b, c which maps to source rows [1, 0, 2]. Success means
        proxy_source_rows returns [1, 0, 2] after the sort.
        """
        model = NavigatorTableModel(dataset_model, inference_model)
        proxy = NavigatorSortProxyModel()
        proxy.setSourceModel(model)

        proxy.sort(NavigatorColumns.IMG_ID, Qt.AscendingOrder)

        assert proxy_source_rows(proxy) == [1, 0, 2]

    def test_numeric_annotation_sort(self, dataset_model, inference_model):
        """Verify that sorting by annotation count sorts numerically and puts highest-count image first.

        Adds 1 annotation to source row 0 and 2 annotations to source row 2. After
        descending sort by annotation count, source row 2 should be first in the proxy.
        Success means the first proxy row maps to source row 2.
        """
        dataset_model.add_annotation(0, "Defect", [(0, 0), (1, 0), (1, 1)])
        dataset_model.add_annotation(2, "Defect", [(0, 0), (1, 0), (1, 1)])
        dataset_model.add_annotation(2, "Defect", [(0, 0), (2, 0), (2, 2)])
        model = NavigatorTableModel(dataset_model, inference_model)
        proxy = NavigatorSortProxyModel()
        proxy.setSourceModel(model)

        proxy.sort(NavigatorColumns.ANNOTS, Qt.DescendingOrder)

        assert proxy_source_rows(proxy)[0] == 2

    def test_score_sort_keeps_missing_scores_last(
        self, dataset_model, inference_model, tmp_path
    ):
        """Verify that sorting by inference score always places images with no score at the end.

        Stores scores for 'b.jpg' (0.20, source row 0) and 'c.jpg' (0.90, source row 2).
        'a.jpg' (source row 1) has no score. Ascending sort should order: b(0.20),
        c(0.90), a(no score) → [0, 2, 1]. Descending: c(0.90), b(0.20), a(no score)
        → [2, 0, 1]. Success means both orderings place the unscored image last.
        """
        inference_model.set_score_map(
            str(tmp_path / "b.jpg"), 0.20, np.zeros((2, 2), dtype=np.float32)
        )
        inference_model.set_score_map(
            str(tmp_path / "c.jpg"), 0.90, np.zeros((2, 2), dtype=np.float32)
        )
        model = NavigatorTableModel(dataset_model, inference_model)
        proxy = NavigatorSortProxyModel()
        proxy.setSourceModel(model)

        proxy.sort(NavigatorColumns.SCORE, Qt.AscendingOrder)
        assert proxy_source_rows(proxy) == [0, 2, 1]

        proxy.sort(NavigatorColumns.SCORE, Qt.DescendingOrder)
        assert proxy_source_rows(proxy) == [2, 0, 1]
