from __future__ import annotations

import numpy as np


class OpticalFlowValidator:
    """Static validation helpers for optical flow inputs."""

    @staticmethod
    def validate_image_pair(
        source_image: np.ndarray,
        target_image: np.ndarray,
    ) -> None:
        """
        Validate that two images can be used as an optical flow pair.

        Parameters
        ----------
        source_image: np.ndarray
            Source image with shape ``(H, W)`` or ``(H, W, C)``.
        target_image: np.ndarray
            Target image with shape ``(H, W)`` or ``(H, W, C)``.

        Raises
        ------
        ValueError
            If either image has an invalid rank or the spatial shapes differ.
        """
        if source_image.ndim not in (2, 3):
            raise ValueError(
                f"source_image must have shape (H, W) or (H, W, C), got {source_image.shape}"
            )
        if target_image.ndim not in (2, 3):
            raise ValueError(
                f"target_image must have shape (H, W) or (H, W, C), got {target_image.shape}"
            )
        if source_image.shape[:2] != target_image.shape[:2]:
            raise ValueError(
                "source_image and target_image must share the same spatial shape: "
                f"{source_image.shape[:2]} != {target_image.shape[:2]}"
            )

    @staticmethod
    def validate_mask(
        mask: np.ndarray,
        image_shape: tuple[int, int],
    ) -> None:
        """
        Validate that a mask matches the spatial shape of an image pair.

        Parameters
        ----------
        mask: np.ndarray
            Mask with shape ``(H, W)``.
        image_shape: tuple[int, int]
            ``(height, width)`` of the source and target images.

        Raises
        ------
        ValueError
            If the mask is not 2D or its shape does not match ``image_shape``.
        """
        if mask.ndim != 2:
            raise ValueError(f"mask must be a 2D array, got shape {mask.shape}")
        if mask.shape[:2] != image_shape:
            raise ValueError(
                "mask spatial shape must match the input images: "
                f"{mask.shape[:2]} != {image_shape}"
            )

    @staticmethod
    def validate_run_inputs(
        source_image: np.ndarray,
        target_image: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> None:
        """
        Validate images and an optional mask before optical flow computation.

        Parameters
        ----------
        source_image: np.ndarray
            Source image in original-image coordinates.
        target_image: np.ndarray
            Target image in original-image coordinates.
        mask: np.ndarray | None
            Optional mask in original-image coordinates.

        Raises
        ------
        ValueError
            If the image pair or mask is invalid.
        """
        OpticalFlowValidator.validate_image_pair(source_image, target_image)
        if mask is not None:
            OpticalFlowValidator.validate_mask(mask, source_image.shape[:2])
