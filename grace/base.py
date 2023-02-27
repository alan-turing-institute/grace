from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass
class DetectionNode:
    """A detection node object from a detection module.

    Parameters
    ----------
    x : float
    y : float
    features : array
    label : int
    object_idx : int
    """

    x: float
    y: float
    features: np.ndarray
    label: int
    object_idx: int = 0

    def asdict(self) -> DetectionNode:
        return dataclasses.asdict(self)
