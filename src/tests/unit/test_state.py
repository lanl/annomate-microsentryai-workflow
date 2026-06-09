class TestAnnotationCRUD:
    def test_add_annotation(self, state):
        """Verify that add_annotation stores a new annotation with the correct category and visibility.

        Adds a polygon annotation for 'Defect' on an image. Success means the
        annotations dict contains exactly one entry for 'img.jpg', with category name
        normalized to lowercase and visible flag set to True.
        """
        state.image_files = ["img.jpg"]
        state.add_annotation("img.jpg", "Defect", [(0, 0), (1, 0), (1, 1)])
        assert len(state.annotations["img.jpg"]) == 1
        assert state.annotations["img.jpg"][0]["category_name"] == "defect"
        assert state.annotations["img.jpg"][0]["visible"] is True

    def test_delete_annotation(self, state):
        """Verify that delete_annotation removes the annotation at the specified index.

        Pre-populates one annotation and deletes it by index 0. Success means the
        annotation list for 'img.jpg' is empty after deletion.
        """
        state.annotations["img.jpg"] = [
            {"category_name": "Defect", "polygon": [(0, 0)]}
        ]
        state.delete_annotation("img.jpg", 0)
        assert state.annotations["img.jpg"] == []

    def test_delete_out_of_bounds_is_safe(self, state):
        """Verify that delete_annotation does not raise when given an out-of-bounds index.

        Attempting to delete an annotation at index 99 from an empty list should not
        raise an IndexError or any other exception. Success means no exception is raised.
        """
        state.annotations["img.jpg"] = []
        state.delete_annotation("img.jpg", 99)

    def test_update_annotation_points(self, state):
        """Verify that update_annotation_points replaces the polygon vertices for a specific annotation.

        Pre-populates an annotation with two points and updates it with three new points.
        Success means the annotation's polygon list is replaced with the new points.
        """
        state.annotations["img.jpg"] = [
            {"category_name": "Defect", "polygon": [(0, 0), (1, 0)]}
        ]
        new_pts = [(5, 5), (6, 5), (6, 6)]
        state.update_annotation_points("img.jpg", 0, new_pts)
        assert state.annotations["img.jpg"][0]["polygon"] == new_pts

    def test_set_annotation_visible(self, state):
        """Verify that set_annotation_visible changes the visibility flag of a specific annotation.

        Adds an annotation (default visible=True) and then hides it. Success means
        is_annotation_visible returns False after the visibility is set to False.
        """
        state.add_annotation("img.jpg", "Defect", [(0, 0), (1, 0), (1, 1)])

        state.set_annotation_visible("img.jpg", 0, False)

        assert state.is_annotation_visible("img.jpg", 0) is False


class TestClassRegistry:
    def test_class_names_empty_on_init(self, state):
        """Verify that a fresh DatasetState has no registered annotation classes.

        On initialization, class_names, class_colors, and class_visibility should all
        be empty collections. Success means all three are empty.
        """
        assert state.class_names == []
        assert state.class_colors == {}
        assert state.class_visibility == {}

    def test_add_class(self, state):
        """Verify that add_class registers a class with its color and defaults to visible.

        Adds the class 'Crack' with a specific color. Success means the class name
        (lowercased) appears in class_names, its color is stored, and it is visible.
        """
        state.add_class("Crack", (200, 100, 0))
        assert "crack" in state.class_names
        assert state.class_colors["crack"] == (200, 100, 0)
        assert state.is_class_visible("crack") is True

    def test_add_duplicate_class_is_idempotent(self, state):
        """Verify that adding a class that already exists does not create a duplicate entry.

        Adding 'Defect' twice should result in only one 'defect' entry in class_names.
        Success means class_names.count('defect') equals 1.
        """
        state.add_class("Defect", (255, 0, 0))
        state.add_class("Defect", (0, 0, 0))
        assert state.class_names.count("defect") == 1

    def test_delete_class_removes_annotations(self, state):
        """Verify that deleting a class also removes all annotations belonging to that class.

        Pre-populates annotations for 'crack' and 'defect', then deletes 'crack'.
        Success means 'crack' no longer appears in class_visibility or any annotation's
        category_name, while 'defect' annotations are untouched.
        """
        state.add_class("Crack", (200, 100, 0))
        state.set_class_visible("crack", False)
        state.annotations["img.jpg"] = [
            {"category_name": "crack", "polygon": [(0, 0)]},
            {"category_name": "defect", "polygon": [(1, 1)]},
        ]
        state.delete_class("crack")
        remaining = state.annotations["img.jpg"]
        assert "crack" not in state.class_visibility
        assert all(a["category_name"] != "crack" for a in remaining)
        assert any(a["category_name"] == "defect" for a in remaining)

    def test_set_class_visible(self, state):
        """Verify that set_class_visible changes the visibility of a class.

        Adds 'Crack' (defaults to visible) and hides it. Success means
        is_class_visible returns False after visibility is set to False.
        """
        state.add_class("Crack", (200, 100, 0))

        state.set_class_visible("crack", False)

        assert state.is_class_visible("crack") is False


class TestClear:
    def test_clear_resets_folder_data(self, state):
        """Verify that clear() resets image directory, file list, and annotations.

        Sets image_dir, image_files, and a pre-populated annotation, then calls clear().
        Success means image_dir is empty string, image_files is empty, and annotations
        dict is empty after the clear.
        """
        state.image_dir = "/some/path"
        state.image_files = ["a.jpg"]
        state.annotations["a.jpg"] = [{"category_name": "X", "polygon": []}]
        state.clear()
        assert state.image_dir == ""
        assert state.image_files == []
        assert state.annotations == {}

    def test_clear_preserves_class_registry(self, state):
        """Verify that clear() preserves class names and colors so they survive a folder reload.

        Classes and colors are project-level data that should not be wiped when loading
        a new folder. Success means a class added before clear() is still present with
        its original color after the clear.
        """
        state.add_class("Custom", (10, 20, 30))
        state.clear()
        assert "custom" in state.class_names
        assert state.class_colors["custom"] == (10, 20, 30)


class TestIsReviewed:
    def test_not_reviewed_when_empty(self, state):
        """Verify that is_reviewed returns False for an image with no annotations, notes, or inspector.

        An image with no associated data should not be considered reviewed. Success
        means is_reviewed returns False for a filename with nothing stored.
        """
        assert not state.is_reviewed("img.jpg")

    def test_reviewed_when_annotation_exists(self, state):
        """Verify that an image with at least one annotation is considered reviewed.

        Adding any annotation to an image should mark it as reviewed. Success means
        is_reviewed returns True after add_annotation is called.
        """
        state.add_annotation("img.jpg", "Defect", [(0, 0)])
        assert state.is_reviewed("img.jpg")

    def test_not_reviewed_when_only_inspector_set(self, state):
        """Verify that setting an inspector alone does not mark an image as reviewed.

        An inspector field on its own is not sufficient to classify an image as
        reviewed — actual annotations or a note must also be present. Success means
        is_reviewed returns False when only the inspector is set.
        """
        state.set_inspector("img.jpg", "Alice")
        assert not state.is_reviewed("img.jpg")

