"""Tests for keypoint result coordinate scaling."""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from kp_detection import KPDetectionMethod
from kp_detection.results import ArrayKPDetectionResult, KPDetectionResult


class TestDetectionResultScaleCoordinates(unittest.TestCase):
    def test_cv2_keypoint_scales_pt_and_size(self) -> None:
        kp = cv2.KeyPoint(x=10.0, y=20.0, size=4.0, angle=30.0)
        result = KPDetectionResult(
            method=KPDetectionMethod.SIFT,
            keypoints=[kp],
            descriptors=None,
        )
        result.scale_coordinates(2.0)
        scaled = result.keypoints[0]
        self.assertAlmostEqual(scaled.pt[0], 20.0)
        self.assertAlmostEqual(scaled.pt[1], 40.0)
        self.assertAlmostEqual(scaled.size, 8.0)
        self.assertAlmostEqual(scaled.angle, 30.0)

    def test_array_preserves_n1x2_layout(self) -> None:
        coords = np.array([[[1.0, 2.0]], [[3.0, 4.0]]], dtype=np.float32)
        result = ArrayKPDetectionResult(
            method=KPDetectionMethod.SHI_TOMASHI,
            keypoints=coords,
            descriptors=None,
        )
        result.scale_coordinates(0.5)
        np.testing.assert_allclose(
            result.keypoints,
            np.array([[[0.5, 1.0]], [[1.5, 2.0]]], dtype=np.float32),
        )

    def test_raises_for_non_positive_factor(self) -> None:
        result = ArrayKPDetectionResult(
            method=KPDetectionMethod.SIFT,
            keypoints=np.zeros((0, 2), dtype=np.float32),
            descriptors=None,
        )
        with self.assertRaises(ValueError):
            result.scale_coordinates(0.0)


if __name__ == "__main__":
    unittest.main()
