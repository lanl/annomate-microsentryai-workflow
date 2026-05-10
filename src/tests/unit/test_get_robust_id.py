from controllers.validation_controller import get_robust_id


class TestGetRobustIdTrayFormat:
    """Filenames following 'Tray_images_Index' convention."""

    def test_numeric_tray_and_index(self):
        assert get_robust_id("118_images_003_01-25-26-20-43-41_poly.jpg") == "118_003"

    def test_different_tray_and_index(self):
        assert get_robust_id("5_images_042_timestamp_poly.png") == "5_042"


class TestGetRobustIdClassIndexFormat:
    """Filenames following 'class_index_timestamp' convention."""

    def test_three_digit_index(self):
        assert get_robust_id("hole_003_02-16-26-01-41-33_poly.jpg") == "003"

    def test_four_digit_index(self):
        assert get_robust_id("defect_0042_extra.png") == "0042"


class TestGetRobustIdBinaryMaskFormat:
    """Filenames produced by the mask generation step."""

    def test_binary_mask_filename(self):
        assert get_robust_id("003_binary_mask.png") == "003"

    def test_eval_output_filename_first_underscore_match(self):
        # _(\d{3,})_ matches _118_ first, so returns "118"
        assert get_robust_id("eval_118_003.png") == "118"


class TestGetRobustIdFallbacks:
    def test_two_long_numbers_joined(self):
        result = get_robust_id("abc123def456.jpg")
        assert result == "123_456"

    def test_one_long_number_returned(self):
        result = get_robust_id("sample123.jpg")
        assert result == "123"

    def test_first_short_number_used_as_last_resort(self):
        result = get_robust_id("ab1cd.jpg")
        assert result == "1"

    def test_no_numbers_returns_stem(self):
        result = get_robust_id("nodigits.jpg")
        assert result == "nodigits"
