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
    """Verify that ClassTableModel shows one row per registered class in insertion order.

    The model has Beta, alpha, and Gamma added in that order. Success means rowCount
    is 3 and class names are returned lowercased in the same insertion order.
    """
    dataset_model = _make_model()
    table_model = ClassTableModel(dataset_model)

    assert table_model.rowCount() == 3
    assert [table_model.class_name(row) for row in range(3)] == [
        "beta",
        "alpha",
        "gamma",
    ]


def test_count_headers_are_compact_with_descriptive_tooltips():
    """Verify that the image and total count column headers are compact with full tooltips.

    The Img and Tot headers must use short abbreviations for display but expose full
    descriptions as tooltips. Success means both headers and their tooltips match the
    expected strings exactly.
    """
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
    """Verify that the visibility column accurately reflects and responds to class visibility changes.

    Initially a class is visible so the button shows 'Hide'. After hiding it,
    VISIBLE_ROLE returns False and the display switches to 'Show'. Success means both
    role data and display text update correctly when visibility is toggled.
    """
    dataset_model = _make_model()
    table_model = ClassTableModel(dataset_model)
    index = table_model.index(0, ClassColumns.VISIBILITY)

    assert index.data(VISIBLE_ROLE) is True
    assert index.data(Qt.DisplayRole) == "Hide"

    dataset_model.set_class_visible("beta", False)

    assert index.data(VISIBLE_ROLE) is False
    assert index.data(Qt.DisplayRole) == "Show"


def test_class_name_sort_is_case_insensitive():
    """Verify that sorting by class name is case-insensitive.

    Classes 'Beta', 'alpha', 'Gamma' should sort alphabetically as alpha, beta, gamma
    regardless of their original casing. Success means the proxy returns class names
    in that order after ascending sort.
    """
    table_model = ClassTableModel(_make_model())
    proxy = ClassSortProxyModel()
    proxy.setSourceModel(table_model)

    proxy.sort(ClassColumns.CLASS, Qt.AscendingOrder)

    assert _proxy_names(proxy) == ["alpha", "beta", "gamma"]


def test_image_and_total_counts_sort_numerically():
    """Verify that image and total annotation count columns sort numerically.

    Image 0 has 2 beta + 1 alpha annotations; Image 1 has 1 gamma. Sorting by
    IMAGE descending should put 'beta' (2) first, then 'alpha' (1), then 'gamma' (0).
    Sorting by TOTAL ascending should put 'alpha' (1) and 'gamma' (1) before 'beta' (2).
    Success means both sort orders return the correct class name sequences.
    """
    table_model = ClassTableModel(_make_model())
    table_model.set_current_row(0)
    proxy = ClassSortProxyModel()
    proxy.setSourceModel(table_model)

    proxy.sort(ClassColumns.IMAGE, Qt.DescendingOrder)
    assert _proxy_names(proxy) == ["beta", "alpha", "gamma"]

    proxy.sort(ClassColumns.TOTAL, Qt.AscendingOrder)
    assert _proxy_names(proxy) == ["alpha", "gamma", "beta"]


def test_proxy_index_exposes_source_class_name_after_sorting():
    """Verify that CLASS_NAME_ROLE on any proxy column returns the correct source class name after sorting.

    After sorting descending by class name, 'gamma' should be first in the proxy.
    Querying CLASS_NAME_ROLE from the DELETE column of the first row should still
    return 'gamma'. Success means the role is correctly exposed from any column.
    """
    table_model = ClassTableModel(_make_model())
    proxy = ClassSortProxyModel()
    proxy.setSourceModel(table_model)

    proxy.sort(ClassColumns.CLASS, Qt.DescendingOrder)
    first = proxy.index(0, ClassColumns.DELETE)

    assert first.data(CLASS_NAME_ROLE) == "gamma"
