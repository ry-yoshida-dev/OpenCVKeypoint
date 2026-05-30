from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

import cv2
import numpy as np

from .method import OpticalFlowMethod
from .parameter import OpticalFlowParameters

ParamsT = TypeVar("ParamsT", bound=OpticalFlowParameters)
ResultT = TypeVar("ResultT")


@dataclass(repr=False, eq=False)
class OpticalFlowProcessor(ABC, Generic[ParamsT, ResultT]):
    """
    Base class for optical flow processors.

    Type parameter ParamsT is the concrete parameters dataclass for this processor.
    Type parameter ResultT is the return type of run().

    Attributes:
    ----------
    params: ParamsT
        Parameters for this optical flow processor.
    """
    params: ParamsT

    @property
    @abstractmethod
    def method(self) -> OpticalFlowMethod:
        """
        Return the optical flow method enum for this processor.

        Returns:
        ----------
        OpticalFlowMethod
            The optical flow method.
        """

    def rescale_image(self, image: np.ndarray) -> np.ndarray:
        """
        Resize an image according to ``params.scale_factor``.

        Parameters
        ----------
        image: np.ndarray
            Input image with shape ``(H, W)`` or ``(H, W, C)``.

        Returns
        -------
        np.ndarray
            Resized image. Returns ``image`` unchanged when ``params.scale_factor == 1.0``.
        """
        if self.params.scale_factor == 1.0:
            return image

        height, width = image.shape[:2]
        new_width = int(width * self.params.scale_factor)
        new_height = int(height * self.params.scale_factor)
        if new_width <= 0 or new_height <= 0:
            raise ValueError(
                f"scale_factor={self.params.scale_factor} produces invalid size "
                f"({new_height}, {new_width}) from input shape {image.shape}"
            )

        return cv2.resize(
            image,
            (new_width, new_height),
            interpolation=self.params.interpolation_flag,
        )

    def rescale_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Resize a mask according to ``params.scale_factor``.

        Parameters
        ----------
        mask: np.ndarray
            Boolean or numeric mask with shape ``(H, W)``.

        Returns
        -------
        np.ndarray
            Resized mask. Returns ``mask`` unchanged when ``params.scale_factor == 1.0``.
        """
        if self.params.scale_factor == 1.0:
            return mask

        if mask.ndim != 2:
            raise ValueError(f"mask must be a 2D array, got shape {mask.shape}")

        height, width = mask.shape[:2]
        new_width = int(width * self.params.scale_factor)
        new_height = int(height * self.params.scale_factor)
        if new_width <= 0 or new_height <= 0:
            raise ValueError(
                f"scale_factor={self.params.scale_factor} produces invalid mask size "
                f"({new_height}, {new_width}) from input shape {mask.shape}"
            )

        return cv2.resize(mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply rescaling and method-specific preprocessing to an image.

        Parameters
        ----------
        image: np.ndarray
            Input image in original-image coordinates.

        Returns
        -------
        np.ndarray
            Preprocessed image ready for optical flow computation.
        """
        return self._preprocess_image(self.rescale_image(image))

    @abstractmethod
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply method-specific preprocessing to a rescaled image.

        Parameters
        ----------
        image: np.ndarray
            Rescaled input image.

        Returns
        -------
        np.ndarray
            Image ready for optical flow computation.
        """

    def preprocess_mask(self, mask: np.ndarray | None) -> np.ndarray | None:
        """
        Preprocess a mask before optical flow computation.

        Parameters
        ----------
        mask: np.ndarray | None
            Optional mask in original-image coordinates.

        Returns
        -------
        np.ndarray | None
            Resized mask, or ``None`` when ``mask`` is ``None``.
        """
        if mask is None:
            return None
        return self.rescale_mask(mask)

    @abstractmethod
    def run(
        self,
        source_image: np.ndarray,
        target_image: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> ResultT:
        """
        Compute optical flow from source_image to target_image.

        Parameters
        ----------
        source_image: np.ndarray
            The source image.
        target_image: np.ndarray
            The target image.
        mask: np.ndarray | None
            Optional mask (0 = ignore, 1 = include).

        Returns
        -------
        ResultT
            Processor-specific optical flow result.
        """

    @abstractmethod
    def __str__(self) -> str:
        """
        Return a string representation of the processor.

        Returns:
        ----------
        str
            The string representation of the processor.
        """
