from PySide6.QtCore import Qt

from core.states.dataset_state import DatasetState
from models.classes_model import (
    CLASS_NAME_ROLE,
    VISIBLE_ROLE,
    ClassColumns,
    ClassSortProxyModel,
    ClassTableModel,
)
from models.dataset_model import DatasetTableModel


def _make_model():
    dataset_model = DatasetTableModel(DatasetState())
    dataset_model.add_class("Beta", (20, 20, 20))
    dataset_model.add_class("alpha", (10, 10, 10))
    dataset_model.add_class("Gamma", (30, 30, 30))
    dataset_model.load_folder("/fake", ["one.jpg", "two.jpg"])
    dataset_model.add_annotation(0, "Beta", [(0, 0), (1, 0), (1, 1)])
    dataset_model.add_annotation(0, "Beta", [(0, 0), (2, 0), (2, 2)])
    dataset_model.add_annotation(0, "alpha", [(0, 0), (1, 0), (1, 1)])
    dataset_model.add_annotation(1, "Gamma", [(0, 0), (1, 0), (1, 1)])
    return dataset_model


def _proxy_names(proxy):
    return [
        proxy.index(row, ClassColumns.CLASS).data(CLASS_NAME_ROLE)
        for row in range(proxy.rowCount())
    ]


def test_class_rows_reflect_dataset_class_names():
    dataset_model = _make_model()
    table_model = ClassTableModel(dataset_model)

    assert table_model.rowCount() == 3
    assert [table_model.class_name(row) for row in range(3)] == [
        "Beta",
        "alpha",
        "Gamma",
    ]


def test_count_headers_are_compact_with_descriptive_tooltips():
    table_model = ClassTableModel(_make_model())

    assert table_model.headerData(ClassColumns.IMAGE, Qt.Horizontal) == "Img"
    assert table_model.headerData(ClassColumns.TOTAL, Qt.Horizontal) == "Tot"
    assert (
        table_model.headerData(ClassColumns.IMAGE, Qt.Horizontal, Qt.ToolTipRole)
        == "Class count for this image"
    )
    assert (
        table_model.headerData(ClassColumns.TOTAL, Qt.Horizontal, Qt.ToolTipRole)
        == "Class count for the whole dataset"
    )


def test_visibility_column_reflects_model_state():
    dataset_model = _make_model()
    table_model = ClassTableModel(dataset_model)
    index = table_model.index(0, ClassColumns.VISIBILITY)

    assert index.data(VISIBLE_ROLE) is True
    assert index.data(Qt.DisplayRole) == "Hide"

    dataset_model.set_class_visible("Beta", False)

    assert index.data(VISIBLE_ROLE) is False
    assert index.data(Qt.DisplayRole) == "Show"


def test_class_name_sort_is_case_insensitive():
    table_model = ClassTableModel(_make_model())
    proxy = ClassSortProxyModel()
    proxy.setSourceModel(table_model)

    proxy.sort(ClassColumns.CLASS, Qt.AscendingOrder)

    assert _proxy_names(proxy) == ["alpha", "Beta", "Gamma"]


def test_image_and_total_counts_sort_numerically():
    table_model = ClassTableModel(_make_model())
    table_model.set_current_row(0)
    proxy = ClassSortProxyModel()
    proxy.setSourceModel(table_model)

    proxy.sort(ClassColumns.IMAGE, Qt.DescendingOrder)
    assert _proxy_names(proxy) == ["Beta", "alpha", "Gamma"]

    proxy.sort(ClassColumns.TOTAL, Qt.AscendingOrder)
    assert _proxy_names(proxy) == ["alpha", "Gamma", "Beta"]


def test_proxy_index_exposes_source_class_name_after_sorting():
    table_model = ClassTableModel(_make_model())
    proxy = ClassSortProxyModel()
    proxy.setSourceModel(table_model)

    proxy.sort(ClassColumns.CLASS, Qt.DescendingOrder)
    first = proxy.index(0, ClassColumns.DELETE)

    assert first.data(CLASS_NAME_ROLE) == "Gamma"
