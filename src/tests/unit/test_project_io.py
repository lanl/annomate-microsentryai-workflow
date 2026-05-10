import json
import pytest
import numpy as np
from pathlib import Path

from core.persistence.project_io import ProjectIO
from core.states.dataset_state import DatasetState
from core.states.inference_state import InferenceState
from core.states.validation_state import ValidationState


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #


@pytest.fixture
def pio():
    return ProjectIO()


@pytest.fixture
def dataset_state():
    return DatasetState()


@pytest.fixture
def inference_state():
    return InferenceState()


@pytest.fixture
def validation_state():
    return ValidationState()


def _make_dataset(tmp_path):
    """Return a populated DatasetState with one image file on disk."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    # Write a minimal valid 1×1 PNG so PIL can read its size
    from PIL import Image as PILImage

    PILImage.new("RGB", (320, 240)).save(img_dir / "img001.jpg")

    state = DatasetState()
    state.image_dir = str(img_dir)
    state.image_files = ["img001.jpg"]
    state.add_class("Defect", (255, 0, 0))
    state.add_annotation("img001.jpg", "Defect", [(10, 20), (50, 20), (50, 60)])
    state.set_inspector("img001.jpg", "Alice")
    state.set_note("img001.jpg", "test note")
    return state


# ------------------------------------------------------------------ #
# Polygon serialization helpers
# ------------------------------------------------------------------ #


class TestPolyHelpers:
    def test_poly_to_coco_seg_round_trip(self, pio):
        poly = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]
        seg = pio._poly_to_coco_seg(poly)
        assert seg == [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]
        assert pio._coco_seg_to_poly(seg) == poly

    def test_coco_seg_to_poly_flat_list(self, pio):
        # Some COCO files store the flat list without the outer wrapper
        flat = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        result = pio._coco_seg_to_poly(flat)
        assert result == [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]

    def test_empty_segmentation_returns_empty(self, pio):
        assert pio._coco_seg_to_poly([]) == []

    def test_coco_annotation_has_required_keys(self, pio):
        poly = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]
        ann = pio._build_coco_annotation(1, 1, 1, poly)
        for key in (
            "id",
            "image_id",
            "category_id",
            "segmentation",
            "area",
            "bbox",
            "iscrowd",
        ):
            assert key in ann
        assert ann["iscrowd"] == 0
        assert ann["area"] > 0
        assert len(ann["bbox"]) == 4


# ------------------------------------------------------------------ #
# NPZ key sanitization
# ------------------------------------------------------------------ #


class TestNpzKeys:
    def test_round_trip_dot_and_slash(self, pio):
        fname = "subdir/image.001.jpg"
        key = pio._filename_to_npz_key(fname)
        assert "." not in key
        assert "/" not in key
        assert pio._npz_key_to_filename(key) == fname

    def test_plain_name_unchanged_by_round_trip(self, pio):
        fname = "plain_name"
        assert pio._npz_key_to_filename(pio._filename_to_npz_key(fname)) == fname


# ------------------------------------------------------------------ #
# Image size reading
# ------------------------------------------------------------------ #


class TestReadImageSize:
    def test_reads_real_image(self, pio, tmp_path):
        from PIL import Image as PILImage

        img_path = tmp_path / "test.png"
        PILImage.new("RGB", (100, 200)).save(img_path)
        assert pio._read_image_size(str(img_path)) == (100, 200)

    def test_missing_image_returns_zeros(self, pio, tmp_path):
        assert pio._read_image_size(str(tmp_path / "missing.jpg")) == (0, 0)


# ------------------------------------------------------------------ #
# COCO export / import
# ------------------------------------------------------------------ #


class TestCocoRoundTrip:
    def test_export_produces_valid_json(self, pio, tmp_path):
        ds = _make_dataset(tmp_path)
        coco_path = str(tmp_path / "out.coco.json")
        pio.export_coco(coco_path, ds)

        data = json.loads((tmp_path / "out.coco.json").read_text())
        assert "images" in data
        assert "annotations" in data
        assert "categories" in data
        assert len(data["images"]) == 1
        assert len(data["annotations"]) == 1
        assert data["images"][0]["file_name"] == "img001.jpg"
        assert data["images"][0]["width"] == 320

    def test_export_bbox_is_coco_format(self, pio, tmp_path):
        ds = _make_dataset(tmp_path)
        coco_path = str(tmp_path / "out.coco.json")
        pio.export_coco(coco_path, ds)
        data = json.loads((tmp_path / "out.coco.json").read_text())
        bbox = data["annotations"][0]["bbox"]
        assert len(bbox) == 4
        # [x_min, y_min, width, height] — all non-negative
        assert all(v >= 0 for v in bbox)

    def test_import_restores_annotations(self, pio, tmp_path):
        ds_src = _make_dataset(tmp_path)
        coco_path = str(tmp_path / "out.coco.json")
        pio.export_coco(coco_path, ds_src)

        ds_dst = DatasetState()
        pio.import_coco(coco_path, ds_dst)

        assert "Defect" in ds_dst.class_names
        assert "img001.jpg" in ds_dst.annotations
        poly = ds_dst.annotations["img001.jpg"][0]["polygon"]
        assert len(poly) == 3

    def test_import_merges_classes(self, pio, tmp_path):
        ds = _make_dataset(tmp_path)
        coco_path = str(tmp_path / "out.coco.json")
        pio.export_coco(coco_path, ds)

        ds_dst = DatasetState()
        ds_dst.add_class("Existing", (0, 255, 0))
        pio.import_coco(coco_path, ds_dst)

        assert "Existing" in ds_dst.class_names
        assert "Defect" in ds_dst.class_names


# ------------------------------------------------------------------ #
# Full project save / load round-trip
# ------------------------------------------------------------------ #


class TestProjectRoundTrip:
    def test_save_creates_required_files(self, pio, tmp_path):
        ds = _make_dataset(tmp_path)
        vs = ValidationState()
        vs.poly_path = "/some/path"
        inf = InferenceState()

        proj_dir = str(tmp_path / "proj")
        pio.save_project(proj_dir, "myproject", ds, vs, inf)

        assert (tmp_path / "proj" / "myproject.annoproj").exists()
        assert (tmp_path / "proj" / "annotations.coco.json").exists()

    def test_round_trip_restores_class_names(self, pio, tmp_path):
        ds = _make_dataset(tmp_path)
        proj_dir = str(tmp_path / "proj")
        path = pio.save_project(
            proj_dir, "myproject", ds, ValidationState(), InferenceState()
        )

        data = pio.load_project(path)
        ds2 = DatasetState()
        ds2.image_dir = ds.image_dir
        ds2.image_files = list(ds.image_files)
        pio.apply_project_to_states(data, ds2, ValidationState(), InferenceState())

        assert "Defect" in ds2.class_names
        assert ds2.class_colors["Defect"] == (255, 0, 0)

    def test_round_trip_restores_annotations(self, pio, tmp_path):
        ds = _make_dataset(tmp_path)
        proj_dir = str(tmp_path / "proj")
        path = pio.save_project(
            proj_dir, "myproject", ds, ValidationState(), InferenceState()
        )

        data = pio.load_project(path)
        ds2 = DatasetState()
        ds2.image_dir = ds.image_dir
        ds2.image_files = list(ds.image_files)
        pio.apply_project_to_states(data, ds2, ValidationState(), InferenceState())

        assert "img001.jpg" in ds2.annotations
        assert len(ds2.annotations["img001.jpg"]) == 1

    def test_round_trip_restores_inspector_and_note(self, pio, tmp_path):
        ds = _make_dataset(tmp_path)
        proj_dir = str(tmp_path / "proj")
        path = pio.save_project(
            proj_dir, "myproject", ds, ValidationState(), InferenceState()
        )

        data = pio.load_project(path)
        ds2 = DatasetState()
        ds2.image_dir = ds.image_dir
        ds2.image_files = list(ds.image_files)
        vs2 = ValidationState()
        pio.apply_project_to_states(data, ds2, vs2, InferenceState())

        assert ds2.inspectors.get("img001.jpg") == "Alice"
        assert ds2.notes.get("img001.jpg") == "test note"

    def test_round_trip_restores_validation_paths(self, pio, tmp_path):
        ds = _make_dataset(tmp_path)
        vs = ValidationState()
        vs.poly_path = "/poly"
        vs.json_path = "/ann.json"
        proj_dir = str(tmp_path / "proj")
        path = pio.save_project(proj_dir, "myproject", ds, vs, InferenceState())

        data = pio.load_project(path)
        vs2 = ValidationState()
        pio.apply_project_to_states(data, DatasetState(), vs2, InferenceState())

        assert vs2.poly_path == "/poly"
        assert vs2.json_path == "/ann.json"

    def test_round_trip_restores_inference_cache(self, pio, tmp_path):
        ds = _make_dataset(tmp_path)
        inf = InferenceState()
        inf.inference_cache = {"img001.jpg": 0.87}
        proj_dir = str(tmp_path / "proj")
        path = pio.save_project(proj_dir, "myproject", ds, ValidationState(), inf)

        data = pio.load_project(path)
        inf2 = InferenceState()
        pio.apply_project_to_states(data, DatasetState(), ValidationState(), inf2)

        assert inf2.inference_cache.get("img001.jpg") == pytest.approx(0.87)

    def test_score_maps_saved_and_restored(self, pio, tmp_path):
        ds = _make_dataset(tmp_path)
        inf = InferenceState()
        arr = np.array([[0.1, 0.9], [0.5, 0.3]], dtype=np.float32)
        inf.score_maps["img001.jpg"] = arr
        inf.inference_cache["img001.jpg"] = float(arr.max())

        proj_dir = str(tmp_path / "proj")
        path = pio.save_project(
            proj_dir, "myproject", ds, ValidationState(), inf, save_score_maps=True
        )
        assert (tmp_path / "proj" / "scoremaps.npz").exists()

        data = pio.load_project(path)
        inf2 = InferenceState()
        pio.apply_project_to_states(data, DatasetState(), ValidationState(), inf2)

        assert "img001.jpg" in inf2.score_maps
        np.testing.assert_array_almost_equal(inf2.score_maps["img001.jpg"], arr)

    def test_skip_score_maps_flag(self, pio, tmp_path):
        ds = _make_dataset(tmp_path)
        inf = InferenceState()
        inf.score_maps["img001.jpg"] = np.zeros((4, 4), dtype=np.float32)

        proj_dir = str(tmp_path / "proj")
        pio.save_project(
            proj_dir, "myproject", ds, ValidationState(), inf, save_score_maps=False
        )

        assert not (tmp_path / "proj" / "scoremaps.npz").exists()

    def test_missing_coco_file_does_not_crash(self, pio, tmp_path):
        ds = _make_dataset(tmp_path)
        proj_dir = str(tmp_path / "proj")
        path = pio.save_project(
            proj_dir, "myproject", ds, ValidationState(), InferenceState()
        )

        # Remove the COCO file to simulate a corrupted/moved project
        (tmp_path / "proj" / "annotations.coco.json").unlink()

        data = pio.load_project(path)
        ds2 = DatasetState()
        ds2.image_dir = ds.image_dir
        ds2.image_files = list(ds.image_files)
        # Should not raise — annotations simply remain empty
        pio.apply_project_to_states(data, ds2, ValidationState(), InferenceState())
        assert ds2.annotations == {}

    def test_relative_path_resolution(self, pio, tmp_path):
        ds = _make_dataset(tmp_path)
        proj_dir = str(tmp_path / "proj")
        path = pio.save_project(
            proj_dir, "myproject", ds, ValidationState(), InferenceState()
        )

        # Manually strip the absolute key to force relative-path fallback
        data = pio.load_project(path)
        data["annotations_file_abs"] = "/nonexistent/path/annotations.coco.json"
        resolved = pio._resolve_coco_path(path, data)
        assert resolved.endswith("annotations.coco.json")
        assert "proj" in resolved

    def test_project_schema_version_present(self, pio, tmp_path):
        ds = _make_dataset(tmp_path)
        proj_dir = str(tmp_path / "proj")

        # 1. Capture the returned path
        path = pio.save_project(
            proj_dir, "myproject", ds, ValidationState(), InferenceState()
        )

        # 2. Use the returned path to read the file!
        # (Wrapping in Path() just in case pio.save_project returns a string instead of a pathlib.Path object)
        raw = json.loads(Path(path).read_text())

        assert raw["version"] == ProjectIO.SCHEMA_VERSION
        assert "created_at" in raw
        assert "modified_at" in raw
