import numpy as np

from core.logic.center_template import extract_template, locate_center


def _pattern(width=20, height=16):
    y, x = np.mgrid[:height, :width]
    return np.dstack(
        [
            (x * 7 + y * 3) % 255,
            (x * 5 + y * 11) % 255,
            (x * 13 + y * 17) % 255,
        ]
    ).astype(np.uint8)


def test_extract_template_returns_expected_anchor():
    """Verify that extract_template crops the correct region and computes the right anchor offset.

    Places a known pattern at pixel (40, 32) in an 80x100 image, then extracts a 20x16
    template centered on (50, 40). The anchor offset should be half the template size
    (10, 8) and the extracted pixels should exactly match the placed pattern. Success
    means anchor_x == 10, anchor_y == 8, shape matches, and pixel values are identical.
    """
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    pattern = _pattern()
    image[32:48, 40:60] = pattern

    template, anchor_x, anchor_y = extract_template(image, 50, 40, 20, 16)

    assert anchor_x == 10
    assert anchor_y == 8
    assert template.shape == (16, 20, 3)
    np.testing.assert_array_equal(template, pattern)


def test_extract_template_clips_at_image_border():
    """Verify that extract_template clips the crop window to the actual image bounds.

    Requests a 20x20 template centered near the edge of a 12x14 image, which would
    extend beyond the image boundary. The returned template should be clipped to the
    full image size, and the anchor should equal the requested center coordinates.
    Success means the template shape is (12, 14, 3) and anchor equals the center point.
    """
    image = np.zeros((12, 14, 3), dtype=np.uint8)

    template, anchor_x, anchor_y = extract_template(image, 7, 6, 20, 20)

    assert template.shape == (12, 14, 3)
    assert anchor_x == 7
    assert anchor_y == 6


def test_locate_center_matches_expected_center():
    """Verify that locate_center finds the correct center coordinates in a clean image.

    Places a known pattern at a specific location and runs template matching. The
    returned (center_x, center_y) must match the expected center of the placed pattern
    after applying the anchor offset. Success means exact coordinate match and score > 0.99.
    """
    template = _pattern()
    image = np.zeros((180, 220, 3), dtype=np.uint8)
    image[122:138, 140:160] = template

    center_x, center_y, score = locate_center(image, template, 10, 8)

    assert center_x == 150
    assert center_y == 130
    assert score > 0.99


def test_locate_center_applies_anchor_offset():
    """Verify that locate_center correctly applies a non-centered anchor offset.

    Uses an anchor of (3, 5) instead of the template center, so the reported center
    should be offset accordingly from the template match position. Success means the
    returned coordinates reflect the anchor-adjusted center and the match score is high.
    """
    template = _pattern()
    image = np.zeros((90, 120, 3), dtype=np.uint8)
    image[40:56, 30:50] = template

    center_x, center_y, score = locate_center(image, template, 3, 5)

    assert center_x == 33
    assert center_y == 45
    assert score > 0.99


def test_locate_center_returns_score_for_noisy_image():
    """Verify that locate_center always returns a valid score even when the template is not present.

    Runs template matching against a random noise image where the template has no match.
    The function should still return a float score in the valid [-1.0, 1.0] range without
    raising an exception. Success means score is a float within the expected bounds.
    """
    rng = np.random.default_rng(42)
    template = _pattern()
    image = rng.integers(0, 255, size=(90, 120, 3), dtype=np.uint8)

    _center_x, _center_y, score = locate_center(image, template, 10, 8)

    assert isinstance(score, float)
    assert -1.0 <= score <= 1.0
