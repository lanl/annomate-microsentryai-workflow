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
        """Verify that load_folder discovers all image files and loads them in sorted order.

        Creates three image files with different extensions in an unsorted order and
        loads the folder. Success means all three appear as rows and the first row
        ('a.png' after alphabetical sort) has no annotations.
        """
        model, controller, _ = setup
        for name in ["c.jpg", "a.png", "b.bmp"]:
            (tmp_path / name).touch()
        controller.load_folder(str(tmp_path))
        assert model.rowCount() == 3
        assert model.get_annotations(0) == []  # "a.png" is row 0 after sort

    def test_ignores_non_image_files(self, setup, tmp_path):
        """Verify that load_folder skips non-image file types.

        Creates two non-image files (.txt, .csv) and one image file (.jpg) in the
        folder. Only the image file should appear in the model. Success means
        rowCount() is 1.
        """
        model, controller, _ = setup
        (tmp_path / "report.txt").touch()
        (tmp_path / "data.csv").touch()
        (tmp_path / "photo.jpg").touch()
        controller.load_folder(str(tmp_path))
        assert model.rowCount() == 1

    def test_empty_folder_produces_zero_rows(self, setup, tmp_path):
        """Verify that loading an empty folder results in an empty model.

        With no files present in the target directory, load_folder should not raise
        an error and the model should have zero rows. Success means rowCount() is 0.
        """
        model, controller, _ = setup
        controller.load_folder(str(tmp_path))
        assert model.rowCount() == 0


class TestExportCsv:
    def test_csv_has_correct_columns(self, setup, tmp_path):
        """Verify that export_csv writes a file with the expected column names and values.

        Loads a folder with one image, sets an inspector name, and exports to CSV.
        The resulting file should be parseable as a dict and contain 'inspector' and
        'image_name' columns with the correct values. Success means both field values
        match what was set in the model.
        """
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
        """Verify that import_annotation_classes reads all class names in file order.

        Imports a text file with three class names, one per line. Success means the
        model's class list matches the file order exactly and the return message
        reports the correct count of imported and skipped classes.
        """
        model, controller, _ = setup
        class_file = tmp_path / "classes.txt"
        class_file.write_text("inclusion\nvoid\ncrack\n", encoding="utf-8")

        msg = controller.import_annotation_classes(str(class_file))

        assert model.get_class_names() == ["inclusion", "void", "crack"]
        assert msg == "Imported 3 class(es), skipped 0 duplicate(s)."

    def test_import_ignores_blank_lines_whitespace_and_comments(self, setup, tmp_path):
        """Verify that import_annotation_classes skips blank lines, extra whitespace, and comments.

        A class file may contain blank lines, leading/trailing spaces, and lines
        starting with '#'. Only the two valid class names should be imported. Success
        means the model contains exactly 'inclusion' and 'void'.
        """
        model, controller, _ = setup
        class_file = tmp_path / "classes.txt"
        class_file.write_text(
            "\n  inclusion  \n# comment\n   # another comment\nvoid\n",
            encoding="utf-8",
        )

        controller.import_annotation_classes(str(class_file))

        assert model.get_class_names() == ["inclusion", "void"]

    def test_import_skips_existing_classes_and_keeps_colors(self, setup, tmp_path):
        """Verify that importing a class that already exists preserves its original color.

        Pre-adds 'void' with a specific color, then imports a file that lists 'void'
        twice and 'crack' once. Duplicates should be skipped and the pre-existing
        color preserved. Success means void is not duplicated, crack is added, and
        void's color is unchanged.
        """
        model, controller, _ = setup
        model.add_class("void", (12, 34, 56))
        class_file = tmp_path / "classes.txt"
        class_file.write_text("void\ncrack\nvoid\n", encoding="utf-8")

        msg = controller.import_annotation_classes(str(class_file))

        assert model.get_class_names() == ["void", "crack"]
        assert model.get_class_color("void") == (12, 34, 56)
        assert msg == "Imported 1 class(es), skipped 2 duplicate(s)."

    def test_export_writes_one_class_per_line_in_order(self, setup, tmp_path):
        """Verify that export_annotation_classes writes class names in insertion order.

        Adds two classes and exports to an output directory. The resulting
        annotation_classes.txt file should contain one class name per line in the
        same order they were added. Success means file contents match expected lines.
        """
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
        """Verify that export_annotation_classes creates missing directories and returns the file path.

        Exports to a nested directory that does not yet exist. The controller should
        create all necessary parent directories and return a message containing the
        full path of the written file. Success means the file exists and its path is
        included in the returned message.
        """
        model, controller, _ = setup
        model.add_class("inclusion", (255, 0, 0))
        out_dir = tmp_path / "missing" / "project"

        msg = controller.export_annotation_classes(str(out_dir))

        class_file = out_dir / "annotation_classes.txt"
        assert class_file.exists()
        assert str(class_file) in msg
