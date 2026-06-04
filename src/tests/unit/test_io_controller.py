import csv
import pytest
from core.states.dataset_state import DatasetState
from models.dataset_model import DatasetTableModel
from controllers.io_controller import IOController


@pytest.fixture
def setup(tmp_path):
    state = DatasetState()
    model = DatasetTableModel(state)
    controller = IOController(model)
    return model, controller, tmp_path


class TestLoadFolder:
    def test_loads_images_sorted(self, setup, tmp_path):
        model, controller, _ = setup
        for name in ["c.jpg", "a.png", "b.bmp"]:
            (tmp_path / name).touch()
        controller.load_folder(str(tmp_path))
        assert model.rowCount() == 3
        assert model.get_annotations(0) == []  # "a.png" is row 0 after sort

    def test_ignores_non_image_files(self, setup, tmp_path):
        model, controller, _ = setup
        (tmp_path / "report.txt").touch()
        (tmp_path / "data.csv").touch()
        (tmp_path / "photo.jpg").touch()
        controller.load_folder(str(tmp_path))
        assert model.rowCount() == 1

    def test_empty_folder_produces_zero_rows(self, setup, tmp_path):
        model, controller, _ = setup
        controller.load_folder(str(tmp_path))
        assert model.rowCount() == 0


class TestExportCsv:
    def test_csv_has_correct_columns(self, setup, tmp_path):
        model, controller, _ = setup
        (tmp_path / "img.jpg").touch()
        controller.load_folder(str(tmp_path))
        model.set_inspector(0, "Bob")
        out_path = str(tmp_path / "out.csv")
        controller.export_csv(out_path)
        with open(out_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert rows[0]["inspector"] == "Bob"
        assert rows[0]["image_name"] == "img.jpg"


class TestAnnotationClassFiles:
    def test_import_simple_class_file_adds_classes_in_order(self, setup, tmp_path):
        model, controller, _ = setup
        class_file = tmp_path / "classes.txt"
        class_file.write_text("inclusion\nvoid\ncrack\n", encoding="utf-8")

        msg = controller.import_annotation_classes(str(class_file))

        assert model.get_class_names() == ["inclusion", "void", "crack"]
        assert msg == "Imported 3 class(es), skipped 0 duplicate(s)."

    def test_import_ignores_blank_lines_whitespace_and_comments(self, setup, tmp_path):
        model, controller, _ = setup
        class_file = tmp_path / "classes.txt"
        class_file.write_text(
            "\n  inclusion  \n# comment\n   # another comment\nvoid\n",
            encoding="utf-8",
        )

        controller.import_annotation_classes(str(class_file))

        assert model.get_class_names() == ["inclusion", "void"]

    def test_import_skips_existing_classes_and_keeps_colors(self, setup, tmp_path):
        model, controller, _ = setup
        model.add_class("void", (12, 34, 56))
        class_file = tmp_path / "classes.txt"
        class_file.write_text("void\ncrack\nvoid\n", encoding="utf-8")

        msg = controller.import_annotation_classes(str(class_file))

        assert model.get_class_names() == ["void", "crack"]
        assert model.get_class_color("void") == (12, 34, 56)
        assert msg == "Imported 1 class(es), skipped 2 duplicate(s)."

    def test_export_writes_one_class_per_line_in_order(self, setup, tmp_path):
        model, controller, _ = setup
        model.add_class("inclusion", (255, 0, 0))
        model.add_class("void", (0, 200, 0))
        out_dir = tmp_path / "project"

        controller.export_annotation_classes(str(out_dir))

        class_file = out_dir / "annotation_classes.txt"
        assert class_file.read_text(encoding="utf-8").splitlines() == [
            "inclusion",
            "void",
        ]

    def test_export_creates_output_directory_and_returns_path(self, setup, tmp_path):
        model, controller, _ = setup
        model.add_class("inclusion", (255, 0, 0))
        out_dir = tmp_path / "missing" / "project"

        msg = controller.export_annotation_classes(str(out_dir))

        class_file = out_dir / "annotation_classes.txt"
        assert class_file.exists()
        assert str(class_file) in msg
