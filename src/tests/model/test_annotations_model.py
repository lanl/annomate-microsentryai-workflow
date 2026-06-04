from PySide6.QtCore import Qt

from core.states.calibration_state import CalibrationState
from core.states.dataset_state import DatasetState
from models.annotations_model import (
    ANNOTATION_INDEX_ROLE,
    VISIBLE_ROLE,
    AnnotationColumns,
    AnnotationSortProxyModel,
    AnnotationTableModel,
)
from models.calibration_model import CalibrationModel
from models.dataset_model import DatasetTableModel


def _make_model():
    dataset_model = DatasetTableModel(DatasetState())
    dataset_model.add_class("Beta", (20, 20, 20))
    dataset_model.add_class("alpha", (10, 10, 10))
    dataset_model.add_class("Gamma", (30, 30, 30))
    dataset_model.load_folder("/fake", ["one.jpg"])
    dataset_model.add_annotation(0, "Beta", [(0, 0), (1, 0), (1, 1)])
    dataset_model.add_annotation(0, "alpha", [(0, 0), (2, 0), (2, 2), (0, 2)])
    dataset_model.add_annotation(0, "Gamma", [(0, 0), (3, 0), (3, 3), (1, 4), (0, 3)])
    return dataset_model


def _proxy_indices(proxy):
    return [
        proxy.index(row, AnnotationColumns.CLASS).data(ANNOTATION_INDEX_ROLE)
        for row in range(proxy.rowCount())
    ]


def test_annotation_rows_reflect_current_image_annotations():
    table_model = AnnotationTableModel(_make_model())
    table_model.set_current_row(0)

    assert table_model.rowCount() == 3
    assert table_model.index(0, AnnotationColumns.CLASS).data() == "beta"
    assert table_model.index(1, AnnotationColumns.VERTICES).data() == "4"
    assert table_model.headerData(AnnotationColumns.AREA, Qt.Horizontal) == "Area (px)"
    assert table_model.index(1, AnnotationColumns.AREA).data() == "4"


def test_visibility_column_reflects_annotation_state():
    dataset_model = _make_model()
    table_model = AnnotationTableModel(dataset_model)
    table_model.set_current_row(0)
    index = table_model.index(0, AnnotationColumns.VISIBILITY)

    assert index.data(VISIBLE_ROLE) is True
    assert index.data(Qt.DisplayRole) == "Hide"

    dataset_model.set_annotation_visible(0, 0, False)

    assert (
        table_model.index(0, AnnotationColumns.VISIBILITY).data(VISIBLE_ROLE) is False
    )
    assert (
        table_model.index(0, AnnotationColumns.VISIBILITY).data(Qt.DisplayRole)
        == "Show"
    )


def test_class_name_sort_is_case_insensitive():
    table_model = AnnotationTableModel(_make_model())
    table_model.set_current_row(0)
    proxy = AnnotationSortProxyModel()
    proxy.setSourceModel(table_model)

    proxy.sort(AnnotationColumns.CLASS, Qt.AscendingOrder)

    assert _proxy_indices(proxy) == [1, 0, 2]


def test_vertex_count_sorts_numerically():
    table_model = AnnotationTableModel(_make_model())
    table_model.set_current_row(0)
    proxy = AnnotationSortProxyModel()
    proxy.setSourceModel(table_model)

    proxy.sort(AnnotationColumns.VERTICES, Qt.DescendingOrder)

    assert _proxy_indices(proxy) == [2, 1, 0]


def test_area_sorts_numerically():
    table_model = AnnotationTableModel(_make_model())
    table_model.set_current_row(0)
    proxy = AnnotationSortProxyModel()
    proxy.setSourceModel(table_model)

    proxy.sort(AnnotationColumns.AREA, Qt.DescendingOrder)

    assert _proxy_indices(proxy) == [2, 1, 0]


def test_area_recalculates_with_calibration():
    calibration_model = CalibrationModel(CalibrationState())
    table_model = AnnotationTableModel(_make_model(), calibration_model)
    table_model.set_current_row(0)

    assert table_model.headerData(AnnotationColumns.AREA, Qt.Horizontal) == "Area (px)"
    assert table_model.index(1, AnnotationColumns.AREA).data() == "4"

    calibration_model.set_calib_points((0.0, 0.0), (100.0, 0.0))
    assert calibration_model.apply_calibration(5.0, "mm")

    assert table_model.headerData(AnnotationColumns.AREA, Qt.Horizontal) == "Area (mm)"
    assert table_model.index(1, AnnotationColumns.AREA).data() == "0.01"


def test_nonzero_subunit_area_does_not_round_to_zero():
    calibration_model = CalibrationModel(CalibrationState())
    table_model = AnnotationTableModel(_make_model(), calibration_model)
    table_model.set_current_row(0)

    calibration_model.set_calib_points((0.0, 0.0), (100.0, 0.0))
    assert calibration_model.apply_calibration(5.0, "mm")

    assert table_model.index(0, AnnotationColumns.AREA).data() != "0"


def test_area_rounds_to_whole_numbers_before_scientific_notation():
    dataset_model = DatasetTableModel(DatasetState())
    dataset_model.add_class("Beta", (20, 20, 20))
    dataset_model.load_folder("/fake", ["one.jpg"])
    dataset_model.add_annotation(0, "Beta", [(0, 0), (123456, 0), (123456, 1), (0, 1)])
    table_model = AnnotationTableModel(dataset_model)
    table_model.set_current_row(0)

    assert table_model.index(0, AnnotationColumns.AREA).data() == "123456"


def test_area_switches_to_scientific_notation_after_six_digits():
    dataset_model = DatasetTableModel(DatasetState())
    dataset_model.add_class("Beta", (20, 20, 20))
    dataset_model.load_folder("/fake", ["one.jpg"])
    dataset_model.add_annotation(
        0, "Beta", [(0, 0), (1234567, 0), (1234567, 1), (0, 1)]
    )
    table_model = AnnotationTableModel(dataset_model)
    table_model.set_current_row(0)

    assert table_model.index(0, AnnotationColumns.AREA).data() == "1.23457e+06"


def test_class_column_updates_source_annotation():
    dataset_model = _make_model()
    table_model = AnnotationTableModel(dataset_model)
    table_model.set_current_row(0)

    assert table_model.setData(
        table_model.index(0, AnnotationColumns.CLASS), "gamma", Qt.EditRole
    )

    assert dataset_model.get_annotations(0)[0]["category_name"] == "gamma"
