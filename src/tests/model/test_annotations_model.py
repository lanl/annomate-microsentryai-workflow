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
    """Verify that AnnotationTableModel displays the correct rows and cell values for the current image.

    Sets the current row to the first image which has three annotations. Success means
    rowCount is 3, class names are lowercase, vertex count is displayed as a string,
    and area is shown in pixel units with the correct header.
    """
    table_model = AnnotationTableModel(_make_model())
    table_model.set_current_row(0)

    assert table_model.rowCount() == 3
    assert table_model.index(0, AnnotationColumns.CLASS).data() == "beta"
    assert table_model.index(1, AnnotationColumns.VERTICES).data() == "4"
    assert table_model.headerData(AnnotationColumns.AREA, Qt.Horizontal) == "Area (px)"
    assert table_model.index(1, AnnotationColumns.AREA).data() == "4"


def test_visibility_column_reflects_annotation_state():
    """Verify that the visibility column accurately reflects and responds to annotation visibility changes.

    Initially the annotation is visible so the button shows 'Hide'. After hiding it,
    VISIBLE_ROLE should return False and the display text should change to 'Show'.
    Success means both the role data and display text update correctly on change.
    """
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
    """Verify that sorting by class name is case-insensitive (alpha before Beta before Gamma).

    The three annotations have classes 'Beta', 'alpha', and 'Gamma'. Ascending sort by
    class should order them as alpha (idx 1), Beta (idx 0), Gamma (idx 2). Success means
    the proxy returns source annotation indices in that order.
    """
    table_model = AnnotationTableModel(_make_model())
    table_model.set_current_row(0)
    proxy = AnnotationSortProxyModel()
    proxy.setSourceModel(table_model)

    proxy.sort(AnnotationColumns.CLASS, Qt.AscendingOrder)

    assert _proxy_indices(proxy) == [1, 0, 2]


def test_vertex_count_sorts_numerically():
    """Verify that sorting by vertex count sorts numerically, not lexicographically.

    The annotations have 3, 4, and 5 vertices. Descending sort should order them
    5-vertex (idx 2), 4-vertex (idx 1), 3-vertex (idx 0). Success means the proxy
    returns source indices in descending vertex-count order.
    """
    table_model = AnnotationTableModel(_make_model())
    table_model.set_current_row(0)
    proxy = AnnotationSortProxyModel()
    proxy.setSourceModel(table_model)

    proxy.sort(AnnotationColumns.VERTICES, Qt.DescendingOrder)

    assert _proxy_indices(proxy) == [2, 1, 0]


def test_area_sorts_numerically():
    """Verify that sorting by area sorts numerically in descending order.

    The annotations have progressively larger areas. Descending sort should order
    them from largest to smallest. Success means the proxy returns source annotation
    indices in descending area order [2, 1, 0].
    """
    table_model = AnnotationTableModel(_make_model())
    table_model.set_current_row(0)
    proxy = AnnotationSortProxyModel()
    proxy.setSourceModel(table_model)

    proxy.sort(AnnotationColumns.AREA, Qt.DescendingOrder)

    assert _proxy_indices(proxy) == [2, 1, 0]


def test_area_recalculates_with_calibration():
    """Verify that the area column updates its unit header and values when calibration is applied.

    Initially area is shown in pixels. After applying a 5mm/100px calibration, the
    header should switch to 'Area (mm)' and the area value should be converted to
    mm² using the scale factor. Success means both header and area value update correctly.
    """
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
    """Verify that very small calibrated area values are not incorrectly rounded to zero.

    After calibration with a large scale factor, small polygons may produce sub-unit
    areas. The display should show a meaningful non-zero value rather than rounding
    down to '0'. Success means the displayed area is not the string '0'.
    """
    calibration_model = CalibrationModel(CalibrationState())
    table_model = AnnotationTableModel(_make_model(), calibration_model)
    table_model.set_current_row(0)

    calibration_model.set_calib_points((0.0, 0.0), (100.0, 0.0))
    assert calibration_model.apply_calibration(5.0, "mm")

    assert table_model.index(0, AnnotationColumns.AREA).data() != "0"


def test_area_rounds_to_whole_numbers_before_scientific_notation():
    """Verify that large areas within 6 significant digits are displayed as whole numbers.

    An area of 123456 should be shown as '123456', not in scientific notation. The
    threshold for switching to scientific notation should be above 6 digits. Success
    means the displayed string equals '123456'.
    """
    dataset_model = DatasetTableModel(DatasetState())
    dataset_model.add_class("Beta", (20, 20, 20))
    dataset_model.load_folder("/fake", ["one.jpg"])
    dataset_model.add_annotation(0, "Beta", [(0, 0), (123456, 0), (123456, 1), (0, 1)])
    table_model = AnnotationTableModel(dataset_model)
    table_model.set_current_row(0)

    assert table_model.index(0, AnnotationColumns.AREA).data() == "123456"


def test_area_switches_to_scientific_notation_after_six_digits():
    """Verify that areas exceeding 6 significant digits are displayed in scientific notation.

    An area of 1234567 (7 significant digits) should be shown as '1.23457e+06' to
    prevent the column from becoming too wide. Success means the displayed string
    matches the expected scientific notation format.
    """
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
    """Verify that editing the class column via setData updates the underlying annotation in the dataset.

    Uses setData with Qt.EditRole on the class column to change an annotation's
    category to 'gamma'. Success means the dataset model reflects the updated
    category_name after the edit.
    """
    dataset_model = _make_model()
    table_model = AnnotationTableModel(dataset_model)
    table_model.set_current_row(0)

    assert table_model.setData(
        table_model.index(0, AnnotationColumns.CLASS), "gamma", Qt.EditRole
    )

    assert dataset_model.get_annotations(0)[0]["category_name"] == "gamma"
