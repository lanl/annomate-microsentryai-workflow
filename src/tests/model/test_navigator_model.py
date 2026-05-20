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
        model = NavigatorTableModel(dataset_model, inference_model)

        assert model.columnCount() == 6
        assert model.headerData(NavigatorColumns.IMG_ID, Qt.Horizontal) == "Img ID"
        assert model.data(model.index(1, NavigatorColumns.IMG_ID)) == "a"
        assert model.data(model.index(1, NavigatorColumns.IMG_ID), SOURCE_ROW_ROLE) == 1

    def test_annotation_and_decision_values(self, dataset_model, inference_model):
        model = NavigatorTableModel(dataset_model, inference_model)

        dataset_model.add_annotation(0, "Defect", [(0, 0), (1, 0), (1, 1)])
        dataset_model.set_review_decision(0, "reject")

        assert model.data(model.index(0, NavigatorColumns.STATUS), SORT_ROLE) == 1
        assert model.data(model.index(0, NavigatorColumns.ANNOTS)) == "1"
        assert model.data(model.index(0, NavigatorColumns.DECISION)) == "Reject"

    def test_inference_values_and_missing_score(
        self, dataset_model, inference_model, tmp_path
    ):
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
        model = NavigatorTableModel(dataset_model, inference_model)
        proxy = NavigatorSortProxyModel()
        proxy.setSourceModel(model)

        proxy.sort(NavigatorColumns.IMG_ID, Qt.AscendingOrder)

        assert proxy_source_rows(proxy) == [1, 0, 2]

    def test_numeric_annotation_sort(self, dataset_model, inference_model):
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
