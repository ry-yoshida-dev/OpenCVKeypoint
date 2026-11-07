"""Smoke tests using synthetic descriptors (no image files)."""

from __future__ import annotations

import unittest

import numpy as np

from kp_detection import KPDetectionMethod
from kp_detection.results import ArrayKPDetectionResult
from kp_matching import (
    FLANNParameters,
    KPMatchCommonParameters,
    KPMatchingParameters,
    KPMatchingProcessor,
    KPMatchMethod,
    RatioTestParameters,
)


def _random_descriptors(
    n: int, dim: int, seed: int, *, gallery_scale: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((n, dim), dtype=np.float64).astype(np.float32)
    g = gallery_scale * rng.standard_normal((n, dim), dtype=np.float64).astype(
        np.float32
    )
    return q, g


def _random_detection(
    n: int, dim: int, seed: int, *, gallery_scale: float = 1.0
) -> tuple[ArrayKPDetectionResult, ArrayKPDetectionResult]:
    rng = np.random.default_rng(seed + 1)
    q_coords = rng.uniform(0.0, 512.0, size=(n, 2)).astype(np.float32)
    g_coords = rng.uniform(0.0, 512.0, size=(n, 2)).astype(np.float32)
    q_desc, g_desc = _random_descriptors(n, dim, seed, gallery_scale=gallery_scale)
    query = ArrayKPDetectionResult(
        method=KPDetectionMethod.SIFT,
        keypoints=q_coords,
        descriptors=q_desc,
    )
    gallery = ArrayKPDetectionResult(
        method=KPDetectionMethod.SIFT,
        keypoints=g_coords,
        descriptors=g_desc,
    )
    return query, gallery


class TestKPMatchingRandomDescriptors(unittest.TestCase):
    @staticmethod
    def _default_matching_params() -> KPMatchingParameters:
        return KPMatchingParameters(
            common_params=KPMatchCommonParameters(
                detection_method=KPDetectionMethod.SIFT,
                method=KPMatchMethod.KNN,
                knn=2,
            ),
            ratio_test_params=RatioTestParameters(),
            flann_params=FLANNParameters(),
        )

    def test_knn_match_returns_match_result(self) -> None:
        params = self._default_matching_params()
        proc = KPMatchingProcessor(params)
        desc1, desc2 = _random_descriptors(20, 32, seed=0)
        result = proc.match(desc1, desc2)
        self.assertGreaterEqual(len(result), 0)
        for m in result:
            self.assertGreaterEqual(m.queryIdx, 0)
            self.assertGreaterEqual(m.trainIdx, 0)

    def test_bf_match_without_ratio_test(self) -> None:
        params = KPMatchingParameters(
            common_params=KPMatchCommonParameters(
                detection_method=KPDetectionMethod.SIFT,
                method=KPMatchMethod.BF,
                knn=1,
            ),
            ratio_test_params=RatioTestParameters(is_enabled=False),
            flann_params=FLANNParameters(),
        )
        proc = KPMatchingProcessor(params)
        desc1, desc2 = _random_descriptors(15, 64, seed=1)
        result = proc.match(desc1, desc2)
        self.assertGreaterEqual(len(result), 0)

    def test_run_pipeline_with_synthetic_detections(self) -> None:
        params = self._default_matching_params()
        proc = KPMatchingProcessor(params)
        query, gallery = _random_detection(12, 32, seed=2, gallery_scale=0.3)
        paired = proc.run_pipeline(query, gallery)
        self.assertGreaterEqual(len(paired.match_result), 0)
        if len(paired.match_result) > 0:
            qm = paired.query_matched_coordinates
            gm = paired.gallery_matched_coordinates
            self.assertEqual(qm.shape[0], len(paired.match_result))
            self.assertEqual(gm.shape[0], len(paired.match_result))

    def test_run_pipeline_raises_without_descriptors(self) -> None:
        params = self._default_matching_params()
        proc = KPMatchingProcessor(params)
        rng = np.random.default_rng(3)
        coords = rng.uniform(0.0, 100.0, size=(5, 2)).astype(np.float32)
        empty = ArrayKPDetectionResult(
            method=KPDetectionMethod.SIFT,
            keypoints=coords,
            descriptors=None,
        )
        full, _ = _random_detection(5, 32, seed=4)
        with self.assertRaises(ValueError):
            _ = proc.run_pipeline(empty, full)
        with self.assertRaises(ValueError):
            _ = proc.run_pipeline(full, empty)


if __name__ == "__main__":
    _ = unittest.main()
