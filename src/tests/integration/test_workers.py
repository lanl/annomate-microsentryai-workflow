"""
Tests for asynchronous QThread background workers.
Ensures threads handle execution and exceptions safely without deadlocking.
"""

import os
import json
import cv2
import numpy as np
import pytest

from controllers.validation_controller import MaskGenWorker, EvaluationWorker


class TestMaskGenWorker:
    """Test the thread responsible for rendering ground-truth JSON to binary masks."""

    @pytest.fixture
    def setup_worker_env(self, tmp_path):
        """Creates dummy directories, images, and JSON annotations for the worker."""
        img_dir = tmp_path / "images"
        out_dir = tmp_path / "masks"
        img_dir.mkdir()
        out_dir.mkdir()

        # Create dummy image
        img_path = img_dir / "123_test.jpg"
        cv2.imwrite(str(img_path), np.zeros((50, 50, 3), dtype=np.uint8))

        # Create dummy JSON
        json_path = tmp_path / "annotations.json"
        dummy_data = {
            "images": {
                "123_test.jpg": {
                    "annotations": [{"polygon": [[0, 0], [10, 0], [10, 10], [0, 10]]}]
                }
            }
        }
        with open(json_path, "w") as f:
            json.dump(dummy_data, f)

        return str(img_dir), str(json_path), str(out_dir)

    def test_mask_generation_success(self, qtbot, setup_worker_env):
        """Verify normal execution parses JSON, renders masks, and emits finished."""
        # Arrange
        img_dir, json_path, out_dir = setup_worker_env
        worker = MaskGenWorker(img_dir, json_path, out_dir)

        # Act & Assert
        with qtbot.waitSignal(worker.finished, timeout=3000):
            worker.start()

        worker.wait()  # CRITICAL: Block until the OS thread completely exits

        # Assert side-effects (mask file created)
        assert len(os.listdir(out_dir)) == 1, "One binary mask should be generated."

    def test_worker_handles_invalid_json_gracefully(self, qtbot, tmp_path):
        """Verify the worker catches JSON parsing errors and emits finished without crashing."""
        # Arrange
        img_dir = tmp_path / "images"
        out_dir = tmp_path / "masks"
        img_dir.mkdir()
        out_dir.mkdir()

        bad_json = tmp_path / "bad.json"
        with open(bad_json, "w") as f:
            f.write("{ INVALID JSON DATA }")

        worker = MaskGenWorker(str(img_dir), str(bad_json), str(out_dir))

        # Collect signals manually to avoid MultiSignalBlocker issues
        emitted_logs = []
        worker.log_message.connect(emitted_logs.append)

        # Act & Assert
        with qtbot.waitSignal(worker.finished, timeout=2000):
            worker.start()

        worker.wait()  # CRITICAL: Prevent QThread teardown exception

        # Verify it logged an error rather than raising an unhandled exception
        assert any(
            "Critical Error loading JSON" in msg for msg in emitted_logs
        ), "Should log JSON failure."


class TestEvaluationWorker:
    """Test the thread responsible for comparing GT and Pred masks and calculating IoU."""

    @pytest.fixture
    def setup_eval_env(self, tmp_path):
        gt_dir = tmp_path / "gt"
        pred_dir = tmp_path / "pred"
        out_dir = tmp_path / "out"
        gt_dir.mkdir()
        pred_dir.mkdir()
        out_dir.mkdir()

        # Create identical GT and Pred masks (Should yield 100% IoU)
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[10:40, 10:40] = 255

        cv2.imwrite(str(gt_dir / "118_mask.png"), mask)
        cv2.imwrite(str(pred_dir / "118_mask.png"), mask)

        return str(gt_dir), str(pred_dir), str(out_dir)

    def test_evaluation_success(self, qtbot, setup_eval_env):
        """Verify evaluation processes matching pairs and calculates metrics."""
        # Arrange
        gt_dir, pred_dir, out_dir = setup_eval_env
        worker = EvaluationWorker(gt_dir, pred_dir, out_dir)

        emitted_ious = []
        worker.match_found.connect(lambda path, text, iou: emitted_ious.append(iou))

        # Act & Assert
        with qtbot.waitSignal(worker.finished, timeout=3000):
            worker.start()

        worker.wait()  # CRITICAL: Prevent QThread teardown exception

        # Assert
        assert len(emitted_ious) > 0, "At least one evaluation match should be emitted."
        assert emitted_ious[0] == 100.0, "Identical masks should result in 100% IoU."
