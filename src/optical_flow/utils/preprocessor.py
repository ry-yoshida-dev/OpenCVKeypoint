from __future__ import annotations

import cv2
import numpy as np


class OpticalFlowPreprocessor:
    """Image preprocessing helpers for optical flow processors."""

    @staticmethod
    def to_grayscale(image: np.ndarray) -> np.ndarray:
        """
        Convert an image to a single-channel grayscale array.

        Parameters
        ----------
        image: np.ndarray
            Input image with shape ``(H, W)`` or ``(H, W, C)``.

        Returns
        -------
        np.ndarray
            Grayscale image with shape ``(H, W)``.

        Raises
        ------
        ValueError
            If the input rank or channel count is unsupported.
        """
        if image.ndim == 2:
            return image
        if image.ndim == 3:
            channel_count = image.shape[2]
            if channel_count == 1:
                return image[:, :, 0]
            if channel_count == 3:
                return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if channel_count == 4:
                return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            raise ValueError(
                f"Unsupported channel count for grayscale conversion: {channel_count}"
            )
        raise ValueError(
            f"image must have shape (H, W) or (H, W, C), got {image.shape}"
        )

    @staticmethod
    def prepare_detection_mask(mask: np.ndarray | None) -> np.ndarray | None:
        """
        Copy a mask and scale 0/1 values to 0/255 for OpenCV feature detectors.

        Parameters
        ----------
        mask: np.ndarray | None
            Optional mask with shape ``(H, W)``.

        Returns
        -------
        np.ndarray | None
            A copied mask suitable for keypoint detection, or ``None``.
        """
        if mask is None:
            return None

        prepared_mask = mask.copy()
        if np.max(prepared_mask) <= 1:
            prepared_mask = (prepared_mask * 255).astype(np.uint8)
        return prepared_mask
