from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any, Type

if TYPE_CHECKING:
    from .processor import OpticalFlowProcessor


class OpticalFlowMethod(Enum):
    """
    OpticalFlowMethod is the method of the optical flow processor.

    Attributes:
    ----------
    FARNEBACK: Farneback method.
    LUCAS_KANADE: Lucas-Kanade method.
    SPARSE_LUCAS_KANADE: Sparse Lucas-Kanade method.
    """
    FARNEBACK = "Farneback"
    LUCAS_KANADE = "Lucas-Kanade"
    SPARSE_LUCAS_KANADE = "SparseLucas-Kanade"

    @property
    def processor_class(self) -> Type[OpticalFlowProcessor[Any, Any]]:
        """
        Get the processor class for the method.

        Returns:
        ----------
        Type[OpticalFlowProcessor[Any, Any]]
            The processor class for the method.
        """
        match self:
            case OpticalFlowMethod.FARNEBACK:
                from .farneback import FarnebackFlow
                return FarnebackFlow
            case OpticalFlowMethod.LUCAS_KANADE | OpticalFlowMethod.SPARSE_LUCAS_KANADE:
                from .lucas_canade import LucasKanadeFlow
                return LucasKanadeFlow
