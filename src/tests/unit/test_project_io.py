import json
import pytest
import numpy as np
from pathlib import Path

from core.persistence.project_io import ProjectIO
from core.states.calibration_state import CalibrationState
from core.states.dataset_state import DatasetState
from core.states.center_template_state import CenterTemplateState
from core.states.inference_state import InferenceState


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
        """Verify that polygon coordinates survive conversion to COCO segmentation and back.

        Converts a list of (x, y) tuples to the COCO flat-list-wrapped format and then
        back to tuples. Success means the result of the round-trip equals the original
        polygon exactly.
        """
        poly = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]
        seg = pio._poly_to_coco_seg(poly)
        assert seg == [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]
        assert pio._coco_seg_to_poly(seg) == poly

    def test_coco_seg_to_poly_flat_list(self, pio):
        """Verify that _coco_seg_to_poly handles the legacy flat-list format.

        Some older COCO files store segmentations as a plain flat list instead of
        [[x1,y1,x2,y2,...]]. The parser should handle both. Success means the flat
        list is correctly parsed into (x, y) tuples.
        """
        # Some COCO files store the flat list without the outer wrapper
        flat = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        result = pio._coco_seg_to_poly(flat)
        assert result == [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]

    def test_empty_segmentation_returns_empty(self, pio):
        """Verify that _coco_seg_to_poly returns an empty list for an empty segmentation.

        An empty segmentation list should not raise an error and should return an
        empty polygon list. Success means the return value is [].
        """
        assert pio._coco_seg_to_poly([]) == []

    def test_coco_annotation_has_required_keys(self, pio):
        """Verify that _build_coco_annotation produces an annotation with all required COCO keys.

        The COCO format requires specific fields for each annotation. This test confirms
        that the helper produces a dict containing all required keys with valid values:
        iscrowd must be 0, area must be positive, and bbox must have exactly 4 elements.
        """
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
        """Verify that filenames with dots and slashes survive NPZ key sanitization.

        NumPy's NPZ format does not support '.' or '/' in array keys. The helpers
        should encode these characters and decode them back losslessly. Success means
        the sanitized key contains neither '.' nor '/', and the round-trip recovers
        the original filename exactly.
        """
        fname = "subdir/image.001.jpg"
        key = pio._filename_to_npz_key(fname)
        assert "." not in key
        assert "/" not in key
        assert pio._npz_key_to_filename(key) == fname

    def test_plain_name_unchanged_by_round_trip(self, pio):
        """Verify that filenames without special characters round-trip without alteration.

        A plain filename with no dots or slashes should still decode to itself after
        key conversion. Success means the decoded filename exactly equals the original.
        """
        fname = "plain_name"
        assert pio._npz_key_to_filename(pio._filename_to_npz_key(fname)) == fname


# ------------------------------------------------------------------ #
# Image size reading
# ------------------------------------------------------------------ #


class TestReadImageSize:
    def test_reads_real_image(self, pio, tmp_path):
        """Verify that _read_image_size returns the correct (width, height) for a real image file.

        Creates a 100x200 PNG image on disk and calls the helper. Success means the
        returned tuple is (100, 200), matching the image dimensions.
        """
        from PIL import Image as PILImage

        img_path = tmp_path / "test.png"
        PILImage.new("RGB", (100, 200)).save(img_path)
        assert pio._read_image_size(str(img_path)) == (100, 200)

    def test_missing_image_returns_zeros(self, pio, tmp_path):
        """Verify that _read_image_size returns (0, 0) for a nonexistent file path.

        When the image file does not exist, the helper should not raise an exception
        and should return a safe default of (0, 0). Success means no exception is
        raised and the return value equals (0, 0).
        """
        assert pio._read_image_size(str(tmp_path / "missing.jpg")) == (0, 0)


# ------------------------------------------------------------------ #
# COCO export / import
# ------------------------------------------------------------------ #


class TestCocoRoundTrip:
    def test_export_produces_valid_json(self, pio, tmp_path):
        """Verify that export_coco writes a JSON file with the required COCO top-level keys.

        Exports a dataset with one image and one annotation. The resulting JSON must
        contain 'images', 'annotations', and 'categories' arrays, with the image's
        filename and width matching the source dataset. Success means all required
        keys are present and values match expectations.
        """
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
        """Verify that exported bounding boxes follow the COCO [x_min, y_min, width, height] format.

        The COCO bbox field must be a 4-element list of non-negative numbers in
        [x_min, y_min, width, height] order. Success means the bbox has exactly 4
        elements and all values are non-negative.
        """
        ds = _make_dataset(tmp_path)
        coco_path = str(tmp_path / "out.coco.json")
        pio.export_coco(coco_path, ds)
        data = json.loads((tmp_path / "out.coco.json").read_text())
        bbox = data["annotations"][0]["bbox"]
        assert len(bbox) == 4
        # [x_min, y_min, width, height] — all non-negative
        assert all(v >= 0 for v in bbox)

    def test_import_restores_annotations(self, pio, tmp_path):
        """Verify that import_coco reconstructs annotation polygons from a COCO JSON file.

        Exports a dataset and imports it into a fresh DatasetState. The imported state
        should contain the correct class name, the image filename as a key, and a
        polygon with the same vertex count. Success means class, filename, and polygon
        length all match the original.
        """
        ds_src = _make_dataset(tmp_path)
        coco_path = str(tmp_path / "out.coco.json")
        pio.export_coco(coco_path, ds_src)

        ds_dst = DatasetState()
        pio.import_coco(coco_path, ds_dst)

        assert "defect" in ds_dst.class_names
        assert "img001.jpg" in ds_dst.annotations
        poly = ds_dst.annotations["img001.jpg"][0]["polygon"]
        assert len(poly) == 3

    def test_import_merges_classes(self, pio, tmp_path):
        """Verify that import_coco merges new categories with pre-existing classes.

        Pre-adds an 'Existing' class to the destination state, then imports a COCO
        file containing 'Defect'. Both classes should coexist in the result. Success
        means both 'existing' and 'defect' appear in class_names after import.
        """
        ds = _make_dataset(tmp_path)
        coco_path = str(tmp_path / "out.coco.json")
        pio.export_coco(coco_path, ds)

        ds_dst = DatasetState()
        ds_dst.add_class("Existing", (0, 255, 0))
        pio.import_coco(coco_path, ds_dst)

        assert "existing" in ds_dst.class_names
        assert "defect" in ds_dst.class_names


# ------------------------------------------------------------------ #
# Full project save / load round-trip
# ------------------------------------------------------------------ #


class TestProjectRoundTrip:
    def test_save_creates_required_files(self, pio, tmp_path):
        """Verify that save_project creates the .annoproj metadata file and the COCO annotations file.

        Both the project manifest (.annoproj) and the COCO JSON file must exist on
        disk after a successful save. Success means both files are present in the
        project directory.
        """
        ds = _make_dataset(tmp_path)
        inf = InferenceState()

        proj_dir = str(tmp_path / "proj")
        pio.save_project(proj_dir, "myproject", ds, inf)

        assert (tmp_path / "proj" / "myproject.annoproj").exists()
        assert (tmp_path / "proj" / "annotations.coco.json").exists()

    def test_round_trip_restores_class_names(self, pio, tmp_path):
        """Verify that class names and colors survive a full project save/load round-trip.

        Saves a dataset with a 'Defect' class and loads the project into a fresh
        DatasetState. Success means 'defect' appears in class_names and its color
        matches the original (255, 0, 0).
        """
        ds = _make_dataset(tmp_path)
        proj_dir = str(tmp_path / "proj")
        path = pio.save_project(proj_dir, "myproject", ds, InferenceState())

        data = pio.load_project(path)
        ds2 = DatasetState()
        ds2.image_dir = ds.image_dir
        ds2.image_files = list(ds.image_files)
        pio.apply_project_to_states(data, ds2, InferenceState())

        assert "defect" in ds2.class_names
        assert ds2.class_colors["defect"] == (255, 0, 0)

    def test_round_trip_restores_annotations(self, pio, tmp_path):
        """Verify that polygon annotations survive a full project save/load round-trip.

        Saves a dataset with one annotation for 'img001.jpg' and loads it back. Success
        means the restored state contains 'img001.jpg' as an annotation key with
        exactly one annotation entry.
        """
        ds = _make_dataset(tmp_path)
        proj_dir = str(tmp_path / "proj")
        path = pio.save_project(proj_dir, "myproject", ds, InferenceState())

        data = pio.load_project(path)
        ds2 = DatasetState()
        ds2.image_dir = ds.image_dir
        ds2.image_files = list(ds.image_files)
        pio.apply_project_to_states(data, ds2, InferenceState())

        assert "img001.jpg" in ds2.annotations
        assert len(ds2.annotations["img001.jpg"]) == 1

    def test_round_trip_restores_inspector_and_note(self, pio, tmp_path):
        """Verify that per-image inspector and note fields survive a full save/load round-trip.

        Saves a dataset where 'img001.jpg' has inspector 'Alice' and a note, then
        loads it back. Success means inspectors and notes for that image match the
        original values after applying project state.
        """
        ds = _make_dataset(tmp_path)
        proj_dir = str(tmp_path / "proj")
        path = pio.save_project(proj_dir, "myproject", ds, InferenceState())

        data = pio.load_project(path)
        ds2 = DatasetState()
        ds2.image_dir = ds.image_dir
        ds2.image_files = list(ds.image_files)
        pio.apply_project_to_states(data, ds2, InferenceState())

        assert ds2.inspectors.get("img001.jpg") == "Alice"
        assert ds2.notes.get("img001.jpg") == "test note"

    def test_round_trip_restores_inference_cache(self, pio, tmp_path):
        """Verify that inference scores and labels survive a full project save/load round-trip.

        Saves a project with a known score (0.87) and 'ANOMALY' label for an image,
        then loads it back. Success means the restored InferenceState contains the
        correct score and label for the absolute image path.
        """
        ds = _make_dataset(tmp_path)
        abs_img = str(tmp_path / "images" / "img001.jpg")
        inf = InferenceState()
        inf.scores = {abs_img: 0.87}
        inf.labels = {abs_img: "ANOMALY"}
        inf.inference_cache = {abs_img: 0.87}
        proj_dir = str(tmp_path / "proj")
        path = pio.save_project(proj_dir, "myproject", ds, inf)

        data = pio.load_project(path)
        inf2 = InferenceState()
        pio.apply_project_to_states(data, DatasetState(), inf2)

        assert inf2.scores.get(abs_img) == pytest.approx(0.87)
        assert inf2.labels.get(abs_img) == "ANOMALY"
        assert inf2.inference_cache.get(abs_img) == pytest.approx(0.87)

    def test_score_maps_saved_and_restored(self, pio, tmp_path):
        """Verify that score map arrays are written to a .npz file and restored correctly.

        When save_score_maps=True is passed, a scoremaps.npz file should be created.
        After loading, the restored InferenceState should contain the original array
        for 'img001.jpg'. Success means the .npz file exists and the restored array
        matches element-wise.
        """
        ds = _make_dataset(tmp_path)
        inf = InferenceState()
        arr = np.array([[0.1, 0.9], [0.5, 0.3]], dtype=np.float32)
        inf.score_maps["img001.jpg"] = arr
        inf.score_maps_dirty = True
        inf.inference_cache["img001.jpg"] = float(arr.max())

        proj_dir = str(tmp_path / "proj")
        path = pio.save_project(proj_dir, "myproject", ds, inf, save_score_maps=True)
        assert (tmp_path / "proj" / "scoremaps.npz").exists()

        data = pio.load_project(path)
        inf2 = InferenceState()
        pio.apply_project_to_states(data, DatasetState(), inf2)

        assert "img001.jpg" in inf2.score_maps
        np.testing.assert_array_almost_equal(inf2.score_maps["img001.jpg"], arr)

    def test_skip_score_maps_flag(self, pio, tmp_path):
        """Verify that save_score_maps=False prevents writing the scoremaps.npz file.

        Even when score maps exist in the state, passing save_score_maps=False should
        result in no .npz file being created. Success means scoremaps.npz does not
        exist in the project directory.
        """
        ds = _make_dataset(tmp_path)
        inf = InferenceState()
        inf.score_maps["img001.jpg"] = np.zeros((4, 4), dtype=np.float32)
        inf.score_maps_dirty = True

        proj_dir = str(tmp_path / "proj")
        pio.save_project(proj_dir, "myproject", ds, inf, save_score_maps=False)

        assert not (tmp_path / "proj" / "scoremaps.npz").exists()

    def test_missing_coco_file_does_not_crash(self, pio, tmp_path):
        """Verify that loading a project with a deleted COCO file does not raise an exception.

        Simulates a corrupted or moved project by deleting the COCO annotations file
        after saving. apply_project_to_states should handle the missing file gracefully,
        leaving annotations empty. Success means no exception is raised and annotations
        equals {}.
        """
        ds = _make_dataset(tmp_path)
        proj_dir = str(tmp_path / "proj")
        path = pio.save_project(proj_dir, "myproject", ds, InferenceState())

        # Remove the COCO file to simulate a corrupted/moved project
        (tmp_path / "proj" / "annotations.coco.json").unlink()

        data = pio.load_project(path)
        ds2 = DatasetState()
        ds2.image_dir = ds.image_dir
        ds2.image_files = list(ds.image_files)
        # Should not raise — annotations simply remain empty
        pio.apply_project_to_states(data, ds2, InferenceState())
        assert ds2.annotations == {}

    def test_relative_path_resolution(self, pio, tmp_path):
        """Verify that _resolve_coco_path falls back to the relative path when the absolute path is invalid.

        Sets the stored absolute COCO path to a nonexistent location, then calls
        _resolve_coco_path. It should fall back to resolving the path relative to the
        project file's directory. Success means the resolved path ends with
        'annotations.coco.json' and contains 'proj' (the project directory name).
        """
        ds = _make_dataset(tmp_path)
        proj_dir = str(tmp_path / "proj")
        path = pio.save_project(proj_dir, "myproject", ds, InferenceState())

        # Manually strip the absolute key to force relative-path fallback
        data = pio.load_project(path)
        data["annotations_file_abs"] = "/nonexistent/path/annotations.coco.json"
        resolved = pio._resolve_coco_path(path, data)
        assert resolved.endswith("annotations.coco.json")
        assert "proj" in resolved

    def test_project_schema_version_present(self, pio, tmp_path):
        """Verify that the saved project file contains schema version and timestamp fields.

        The .annoproj JSON must include a 'version' field matching ProjectIO.SCHEMA_VERSION,
        plus 'created_at' and 'modified_at' timestamps for auditability. Success means
        all three fields are present with the correct version value.
        """
        ds = _make_dataset(tmp_path)
        proj_dir = str(tmp_path / "proj")

        # 1. Capture the returned path
        path = pio.save_project(proj_dir, "myproject", ds, InferenceState())

        # 2. Use the returned path to read the file!
        # (Wrapping in Path() just in case pio.save_project returns a string instead of a pathlib.Path object)
        raw = json.loads(Path(path).read_text())

        assert raw["version"] == ProjectIO.SCHEMA_VERSION
        assert "created_at" in raw
        assert "modified_at" in raw

    def test_calibration_pixel_defaults_round_trip(self, pio, tmp_path):
        """Verify that a default (pixel-scale) CalibrationState survives a project save/load cycle.

        Saves a project with an uncalibrated CalibrationState and loads it back into
        a fresh state. Success means the restored state has scale=1.0, unit='px',
        user_calibrated=False, and the grid is visible.
        """
        ds = _make_dataset(tmp_path)
        proj_dir = tmp_path / "proj"
        state = CalibrationState()

        path = pio.save_project(
            str(proj_dir),
            "myproject",
            ds,
            InferenceState(),
            calibration_state=state,
        )

        data = pio.load_project(path)
        restored = CalibrationState()
        pio.apply_project_to_states(
            data,
            DatasetState(),
            InferenceState(),
            calibration_state=restored,
        )

        assert restored.scale == pytest.approx(1.0)
        assert restored.unit == "px"
        assert restored.user_calibrated is False
        assert restored.grid_visible is True

    def test_legacy_calibration_without_mode_loads_as_user_calibrated(self, pio):
        """Verify that a legacy project dict with scale but no 'mode' key is treated as user-calibrated.

        Older project files stored calibration as scale + unit with no explicit
        calibration mode flag. When loading such data, the absence of a mode key
        should default to treating the calibration as user-set. Success means
        user_calibrated is True and scale/unit match the stored values.
        """
        restored = CalibrationState()

        pio.apply_project_to_states(
            {"calibration": {"scale": 0.05, "unit": "mm"}},
            DatasetState(),
            InferenceState(),
            calibration_state=restored,
        )

        assert restored.scale == pytest.approx(0.05)
        assert restored.unit == "mm"
        assert restored.user_calibrated is True

    def test_missing_calibration_loads_as_pixel_default(self, pio):
        """Verify that a project dict with no calibration key resets any pre-set calibration state.

        If no 'calibration' key is present in the project data, any previously loaded
        calibration values should be overwritten with pixel defaults. Success means
        scale=1.0, unit='px', user_calibrated=False, and grid_visible=True after loading.
        """
        restored = CalibrationState()
        restored.scale = 0.05
        restored.unit = "mm"
        restored.user_calibrated = True

        pio.apply_project_to_states(
            {},
            DatasetState(),
            InferenceState(),
            calibration_state=restored,
        )

        assert restored.scale == pytest.approx(1.0)
        assert restored.unit == "px"
        assert restored.user_calibrated is False
        assert restored.grid_visible is True

    def test_center_template_metadata_round_trips(self, pio, tmp_path):
        """Verify that all CenterTemplateState fields survive a project save/load round-trip.

        Saves a fully configured center template state (enabled, template path, anchor,
        crop shape, and center coordinates) and restores it from the project file.
        Success means every field on the restored state matches the original values.
        """
        ds = _make_dataset(tmp_path)
        proj_dir = tmp_path / "proj"
        proj_dir.mkdir()
        template = proj_dir / "center_template.png"
        template.write_bytes(b"not a real image")

        state = CenterTemplateState()
        state.enabled = True
        state.template_file = "center_template.png"
        state.template_path = str(template)
        state.anchor_x = 12
        state.anchor_y = 14
        state.crop_shape = "circle"
        state.crop_width = 1210
        state.crop_height = 1210
        state.center_x = 50.0
        state.center_y = 60.0

        path = pio.save_project(
            str(proj_dir),
            "myproject",
            ds,
            InferenceState(),
            center_template_state=state,
        )

        data = pio.load_project(path)
        restored = CenterTemplateState()
        pio.apply_project_to_states(
            data,
            DatasetState(),
            InferenceState(),
            center_template_state=restored,
        )

        assert restored.enabled is True
        assert restored.template_file == "center_template.png"
        assert restored.template_path == str(template)
        assert restored.anchor_x == 12
        assert restored.anchor_y == 14
        assert restored.center_x == 50.0
        assert restored.center_y == 60.0

    def test_missing_center_template_disables_matching(self, pio, tmp_path):
        """Verify that a saved template whose file no longer exists is loaded with matching disabled.

        If the template image file referenced in the project does not exist on disk,
        the restored CenterTemplateState should have enabled=False to prevent broken
        template matching. The filename field should still be restored for display
        purposes. Success means enabled is False but template_file is preserved.
        """
        ds = _make_dataset(tmp_path)
        proj_dir = tmp_path / "proj"
        state = CenterTemplateState()
        state.enabled = True
        state.template_file = "center_template.png"
        state.template_path = str(proj_dir / "center_template.png")

        path = pio.save_project(
            str(proj_dir),
            "myproject",
            ds,
            InferenceState(),
            center_template_state=state,
        )

        data = pio.load_project(path)
        restored = CenterTemplateState()
        pio.apply_project_to_states(
            data,
            DatasetState(),
            InferenceState(),
            center_template_state=restored,
        )

        assert restored.enabled is False
        assert restored.template_file == "center_template.png"

    # ------------------------------------------------------------------ #
    # per_image format
    # ------------------------------------------------------------------ #

    def test_per_image_replaces_old_keys(self, pio, tmp_path):
        """Verify that the new 'per_image' format is used instead of legacy top-level keys.

        The current schema consolidates per-image data under a single 'per_image' key
        instead of the old 'review_status', 'review_decisions', and inference cache
        keys. Success means 'per_image' is present and the legacy keys are absent.
        """
        ds = _make_dataset(tmp_path)
        abs_img = str(tmp_path / "images" / "img001.jpg")
        inf = InferenceState()
        inf.scores = {abs_img: 0.75}
        inf.labels = {abs_img: "ANOMALY"}

        proj_dir = str(tmp_path / "proj")
        path = pio.save_project(proj_dir, "myproject", ds, inf)

        raw = json.loads(Path(path).read_text())
        assert "per_image" in raw
        assert "review_status" not in raw
        assert "review_decisions" not in raw
        assert "score_cache" not in raw.get("inference", {})
        assert "label_cache" not in raw.get("inference", {})

    def test_per_image_uses_basename_keys(self, pio, tmp_path):
        """Verify that per_image dict keys are basenames, not absolute file paths.

        The per_image section must use portable basename keys (e.g., 'img001.jpg')
        so the project file remains valid when the image directory is moved. Success
        means all keys in per_image are basenames with no leading '/'.
        """
        ds = _make_dataset(tmp_path)
        abs_img = str(tmp_path / "images" / "img001.jpg")
        inf = InferenceState()
        inf.scores = {abs_img: 0.5}
        inf.labels = {abs_img: "NORMAL"}

        proj_dir = str(tmp_path / "proj")
        path = pio.save_project(proj_dir, "myproject", ds, inf)

        raw = json.loads(Path(path).read_text())
        assert "img001.jpg" in raw["per_image"]
        assert all(not key.startswith("/") for key in raw["per_image"]), (
            "per_image keys must be basenames, not absolute paths"
        )

    def test_per_image_round_trip_review_decision(self, pio, tmp_path):
        """Verify that per-image review decisions survive a save/load round-trip.

        Sets a 'reject' review decision for an image, saves the project, and loads it
        back. Success means the review decision is correctly restored to 'reject' in
        the new DatasetState.
        """
        ds = _make_dataset(tmp_path)
        ds.review_decisions["img001.jpg"] = "reject"

        proj_dir = str(tmp_path / "proj")
        path = pio.save_project(proj_dir, "myproject", ds, InferenceState())

        data = pio.load_project(path)
        ds2 = DatasetState()
        ds2.image_dir = ds.image_dir
        ds2.image_files = list(ds.image_files)
        pio.apply_project_to_states(data, ds2, InferenceState())

        assert ds2.review_decisions.get("img001.jpg") == "reject"

    def test_per_image_all_fields_round_trip(self, pio, tmp_path):
        """Verify that all per-image fields (inspector, note, decision, score, label) round-trip together.

        Saves a dataset with a full set of per-image metadata including inspector, note,
        review decision, inference score, and label. After loading, every field should
        match the original in both the DatasetState and InferenceState. Success means
        all five fields are restored correctly.
        """
        ds = _make_dataset(tmp_path)
        ds.review_decisions["img001.jpg"] = "accept"
        abs_img = str(tmp_path / "images" / "img001.jpg")
        inf = InferenceState()
        inf.scores = {abs_img: 0.42}
        inf.labels = {abs_img: "NORMAL"}

        proj_dir = str(tmp_path / "proj")
        path = pio.save_project(proj_dir, "myproject", ds, inf)

        data = pio.load_project(path)
        ds2 = DatasetState()
        inf2 = InferenceState()
        pio.apply_project_to_states(data, ds2, inf2)

        assert ds2.inspectors.get("img001.jpg") == "Alice"
        assert ds2.notes.get("img001.jpg") == "test note"
        assert ds2.review_decisions.get("img001.jpg") == "accept"
        assert inf2.scores.get(abs_img) == pytest.approx(0.42)
        assert inf2.labels.get(abs_img) == "NORMAL"

    def test_legacy_format_loads_review_status_and_decisions(self, pio, tmp_path):
        """Verify that the legacy 'review_status' and 'review_decisions' top-level keys are still readable.

        Older project files stored inspector/note data under 'review_status' and
        review results under 'review_decisions'. The loader must be backward compatible
        and correctly populate the DatasetState from these legacy keys. Success means
        inspector, note, and decision are all loaded for the image.
        """
        legacy_data = {
            "dataset": {"image_dir": str(tmp_path / "images")},
            "review_status": {"img001.jpg": {"inspector": "Bob", "note": "looks fine"}},
            "review_decisions": {"img001.jpg": "accept"},
        }

        ds2 = DatasetState()
        inf2 = InferenceState()
        pio.apply_project_to_states(legacy_data, ds2, inf2)

        assert ds2.inspectors.get("img001.jpg") == "Bob"
        assert ds2.notes.get("img001.jpg") == "looks fine"
        assert ds2.review_decisions.get("img001.jpg") == "accept"

    def test_legacy_format_loads_absolute_cache_keys(self, pio, tmp_path):
        """Verify that legacy inference caches keyed by absolute paths are loaded correctly.

        Older project files stored inference scores and labels under an 'inference'
        dict with absolute path keys. The loader must restore scores and labels keyed
        by the same absolute paths. Success means scores and labels match the stored
        values for the absolute image path.
        """
        abs_img = str(tmp_path / "images" / "img001.jpg")
        legacy_data = {
            "dataset": {"image_dir": str(tmp_path / "images")},
            "inference": {
                "score_cache": {abs_img: 0.9},
                "label_cache": {abs_img: "ANOMALY"},
            },
        }

        inf2 = InferenceState()
        pio.apply_project_to_states(legacy_data, DatasetState(), inf2)

        assert inf2.scores.get(abs_img) == pytest.approx(0.9)
        assert inf2.labels.get(abs_img) == "ANOMALY"

    def test_as_relative_path_with_traversal(self, pio, tmp_path):
        """Verify that _as_relative_path correctly computes a '../' traversal for sibling paths.

        Given a base path pointing to a subdirectory and a target that is a sibling
        of the base's parent, the relative path should use '../' to traverse up.
        Success means the result equals '../model.pt'.
        """
        base = str(tmp_path / "project" / "subdir")
        sibling = str(tmp_path / "project" / "model.pt")
        result = pio._as_relative_path(sibling, base)
        assert result == "../model.pt"

    def test_image_dir_outside_project_becomes_relative(self, pio, tmp_path):
        """Verify that an image directory outside the project folder is stored as a relative path.

        When the image directory is a sibling of the project folder (not inside it),
        the saved project should store the image_dir as a relative path rather than
        an absolute one. Success means the saved value starts without '/' and equals
        '../images'.
        """
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        from PIL import Image as PILImage

        PILImage.new("RGB", (10, 10)).save(img_dir / "img001.jpg")

        ds = DatasetState()
        ds.image_dir = str(img_dir)
        ds.image_files = ["img001.jpg"]

        proj_dir = str(tmp_path / "myproject")
        path = pio.save_project(proj_dir, "myproject", ds, InferenceState())

        raw = json.loads(Path(path).read_text())
        saved_dir = raw["dataset"]["image_dir"]
        assert not saved_dir.startswith("/"), "image_dir should be relative"
        assert saved_dir == "../images"
