"""Tests for optical flow validation, mask handling, and coordinate scaling."""

from __future__ import annotations

import unittest
import warnings

import cv2
import numpy as np

from kp_detection.detectors.shi_tomashi.parameter import ShiTomashiParameters
from optical_flow import (
    FarnebackFlow,
    FarnebackParameters,
    FarnebackResult,
    LucasKanadeFlow,
    LucasKanadeParameters,
    LucasKanadeResult,
    OpticalFlowValidator,
)


class TestOpticalFlowValidation(unittest.TestCase):
    def setUp(self) -> None:
        self.farneback = FarnebackFlow(params=FarnebackParameters())
        self.lucas_kanade = LucasKanadeFlow(
            params=LucasKanadeParameters(is_SparseRLOF=False),
            keypoint_params=ShiTomashiParameters(max_corners=20),
        )

    def test_raises_on_shape_mismatch(self) -> None:
        source_image = np.zeros((40, 40), dtype=np.uint8)
        target_image = np.zeros((40, 50), dtype=np.uint8)

        with self.assertRaises(ValueError):
            self.farneback.run(source_image, target_image)

        with self.assertRaises(ValueError):
            self.lucas_kanade.run(source_image, target_image)

    def test_raises_on_invalid_mask_shape(self) -> None:
        source_image = np.zeros((40, 40), dtype=np.uint8)
        target_image = np.zeros((40, 40), dtype=np.uint8)
        mask = np.ones((30, 40), dtype=np.uint8)

        with self.assertRaises(ValueError):
            self.farneback.run(source_image, target_image, mask=mask)

        with self.assertRaises(ValueError):
            self.lucas_kanade.run(source_image, target_image, mask=mask)

    def test_accepts_color_images(self) -> None:
        source_image = np.zeros((64, 64, 3), dtype=np.uint8)
        target_image = source_image.copy()
        cv2.rectangle(source_image, (10, 10), (50, 50), (255, 255, 255), -1)
        cv2.rectangle(target_image, (12, 10), (52, 50), (255, 255, 255), -1)

        result = self.farneback.run(source_image, target_image)
        self.assertEqual(result.flow.shape, (64, 64, 2))


class TestOpticalFlowMaskHandling(unittest.TestCase):
    def test_mask_is_not_mutated_by_prepare_detection_mask(self) -> None:
        mask = np.zeros((80, 80), dtype=np.uint8)
        mask[20:60, 20:60] = 1
        original_mask = mask.copy()

        prepared_mask = OpticalFlowValidator.prepare_detection_mask(mask)

        np.testing.assert_array_equal(mask, original_mask)
        self.assertIsNot(prepared_mask, mask)
        self.assertEqual(prepared_mask.dtype, np.uint8)
        self.assertEqual(int(prepared_mask.max()), 255)


class TestFarnebackResultScale(unittest.TestCase):
    def test_scale_restores_flow_dimensions_and_magnitude(self) -> None:
        flow = np.full((50, 80, 2), 2.0, dtype=np.float32)
        result = FarnebackResult(flow=flow)
        params = FarnebackParameters(scale_factor=0.5)

        result.scale(original_shape=(100, 160), params=params)

        self.assertEqual(result.flow.shape, (100, 160, 2))
        np.testing.assert_allclose(result.flow, 4.0)

    def test_identity_when_scale_factor_is_one(self) -> None:
        flow = np.ones((40, 60, 2), dtype=np.float32)
        result = FarnebackResult(flow=flow.copy())
        params = FarnebackParameters(scale_factor=1.0)

        result.scale(original_shape=(40, 60), params=params)

        np.testing.assert_array_equal(result.flow, 1.0)


class TestLucasKanadeResultScale(unittest.TestCase):
    def test_scale_restores_keypoint_coordinates(self) -> None:
        source_keypoints = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
        target_keypoints = np.array([[11.0, 21.0], [31.0, 41.0]], dtype=np.float32)
        status = np.ones((2, 1), dtype=np.uint8)
        error = np.zeros((2, 1), dtype=np.float32)
        result = LucasKanadeResult(
            source_keypoints=source_keypoints.copy(),
            target_keypoints=target_keypoints.copy(),
            status=status,
            error=error,
        )
        params = LucasKanadeParameters(scale_factor=0.5)

        result.scale(params)

        np.testing.assert_allclose(
            result.source_keypoints,
            np.array([[20.0, 40.0], [60.0, 80.0]], dtype=np.float32),
        )
        np.testing.assert_allclose(
            result.target_keypoints,
            np.array([[22.0, 42.0], [62.0, 82.0]], dtype=np.float32),
        )

    def test_identity_when_scale_factor_is_one(self) -> None:
        source_keypoints = np.array([[1.0, 2.0]], dtype=np.float32)
        target_keypoints = np.array([[3.0, 4.0]], dtype=np.float32)
        status = np.ones((1, 1), dtype=np.uint8)
        error = np.zeros((1, 1), dtype=np.float32)
        result = LucasKanadeResult(
            source_keypoints=source_keypoints.copy(),
            target_keypoints=target_keypoints.copy(),
            status=status,
            error=error,
        )
        params = LucasKanadeParameters(scale_factor=1.0)

        result.scale(params)

        np.testing.assert_array_equal(result.source_keypoints, source_keypoints)
        np.testing.assert_array_equal(result.target_keypoints, target_keypoints)


class TestLucasKanadeResultStatistics(unittest.TestCase):
    def test_zero_tracked_keypoints_return_zeros_with_warning(self) -> None:
        source_keypoints = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        target_keypoints = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
        status = np.zeros((2, 1), dtype=np.uint8)
        error = np.ones((2, 1), dtype=np.float32)

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            result = LucasKanadeResult(
                source_keypoints=source_keypoints,
                target_keypoints=target_keypoints,
                status=status,
                error=error,
            )

        self.assertEqual(len(caught_warnings), 1)
        self.assertEqual(caught_warnings[0].category, UserWarning)
        self.assertIn("all status values are 0", str(caught_warnings[0].message))

        np.testing.assert_allclose(result.mean_flow, np.zeros(2, dtype=np.float32))
        self.assertEqual(result.mean_l2_norm, 0.0)
        np.testing.assert_allclose(result.flow_std, np.zeros(2, dtype=np.float32))
        self.assertEqual(len(result), 0)

    def test_partial_tracking_statistics_use_successful_keypoints_only(self) -> None:
        source_keypoints = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
        target_keypoints = np.array([[2.0, 0.0], [99.0, 99.0]], dtype=np.float32)
        status = np.array([[1], [0]], dtype=np.uint8)
        error = np.zeros((2, 1), dtype=np.float32)
        result = LucasKanadeResult(
            source_keypoints=source_keypoints,
            target_keypoints=target_keypoints,
            status=status,
            error=error,
        )

        np.testing.assert_allclose(result.mean_flow, np.array([2.0, 0.0], dtype=np.float32))
        self.assertEqual(result.mean_l2_norm, 2.0)
        np.testing.assert_allclose(result.flow_std, np.zeros(2, dtype=np.float32))
        self.assertEqual(len(result), 1)


if __name__ == "__main__":
    unittest.main()
