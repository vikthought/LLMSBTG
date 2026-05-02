"""
Abstract dataset interface.

Every data source (OH16230, DANDI NWB, future EEG, etc.) implements this
interface so that estimators, evaluation, and figure code work identically
regardless of the underlying format.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np


class Dataset(ABC):
    """Abstract base for all datasets."""

    @abstractmethod
    def load_segments(
        self,
        stimulus: Optional[str] = None,
    ) -> List[np.ndarray]:
        """
        Load time-series data as a list of (T_u, n) arrays.

        Each element corresponds to one worm / recording segment.
        If *stimulus* is given, return only data for that stimulus.
        """
        ...

    @abstractmethod
    def neuron_names(self) -> List[str]:
        """Ordered list of neuron identifiers (length n)."""
        ...

    @abstractmethod
    def fps(self) -> float:
        """Sampling rate in Hz."""
        ...

    @abstractmethod
    def available_stimuli(self) -> List[str]:
        """List of stimulus labels present in this dataset."""
        ...

    @property
    def name(self) -> str:
        """Short human-readable dataset name."""
        return self.__class__.__name__
