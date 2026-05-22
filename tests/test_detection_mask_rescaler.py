"""Tests for detection mask rescaling."""

from __future__ import annotations

import unittest
import warnings

import numpy as np

from kp_detection import KPDetectionMethod
from kp_detection.parameter import KPDetectionParameters
from kp_detection.detectors.agast import AGASTDetector
from kp_detection.detectors.mser import MSERDetector
from kp_detection.detectors.simple_blob import SimpleBlobDetector
from kp_detection.detectors.shi_tomashi.parameter import ShiTomashiParameters
from kp_detection.detectors.harris.parameter import HarrisParameters


class TestMaskRescaler(unittest.TestCase):
    def test_identity_when_scale_factor_is_one(self) -> None:
        params = KPDetectionParameters(
            method=KPDetectionMethod.SIFT,
            scale_factor=1.0,
        )
        mask = np.ones((40, 60), dtype=np.uint8)
        np.testing.assert_array_equal(params.mask_rescaler(mask), mask)

    def test_resizes_mask_with_nearest_neighbor(self) -> None:
        params = KPDetectionParameters(
            method=KPDetectionMethod.SIFT,
            scale_factor=0.5,
        )
        mask = np.zeros((100, 200), dtype=np.uint8)
        mask[10:90, 30:170] = 255
        scaled = params.mask_rescaler(mask)
        self.assertEqual(scaled.shape, (50, 100))
        self.assertGreater(scaled.max(), 0)
        self.assertEqual(scaled.min(), 0)

    def test_subclass_inherits_mask_rescaler(self) -> None:
        shi_params = ShiTomashiParameters(scale_factor=2.0)
        mask = np.ones((10, 10), dtype=np.uint8)
        scaled = shi_params.mask_rescaler(mask)
        self.assertEqual(scaled.shape, (20, 20))

        harris_params = HarrisParameters(scale_factor=0.25)
        scaled_harris = harris_params.mask_rescaler(mask)
        self.assertEqual(scaled_harris.shape, (2, 2))


class TestUnusedMaskWarning(unittest.TestCase):
    def test_agast_warns_when_mask_provided(self) -> None:
        params = KPDetectionParameters(method=KPDetectionMethod.AGAST)
        detector = AGASTDetector(params=params)
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        mask = np.ones((32, 32), dtype=np.uint8)
        with self.assertWarns(UserWarning):
            detector.detect(img, mask=mask)

    def test_mser_warns_when_mask_provided(self) -> None:
        params = KPDetectionParameters(method=KPDetectionMethod.MSER)
        detector = MSERDetector(params=params)
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        mask = np.ones((64, 64), dtype=np.uint8)
        with self.assertWarns(UserWarning):
            detector.detect(img, mask=mask)

    def test_simple_blob_warns_when_mask_provided(self) -> None:
        params = KPDetectionParameters(method=KPDetectionMethod.SIMPLE_BLOB)
        detector = SimpleBlobDetector(params=params)
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        mask = np.ones((32, 32), dtype=np.uint8)
        with self.assertWarns(UserWarning):
            detector.detect(img, mask=mask)


if __name__ == "__main__":
    unittest.main()
