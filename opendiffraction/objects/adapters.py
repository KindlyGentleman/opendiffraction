"""Optional dataset adapters for external vision benchmarks (EMNIST, CIFAR-100).

These adapters provide drop-in compatibility with the original DPR training setup
when the datasets are downloaded locally. They are NOT required for the core
OpenDiffraction release, which is 100% procedural and free of third-party licensing constraints.
"""

from __future__ import annotations

import os
from typing import Any
import numpy as np


class OptionalExternalAdapter:
    """Safely checks for and wraps optional external source datasets."""

    def __init__(self, root: str = "./external_data") -> None:
        self.root = root

    def is_available(self, dataset_name: str) -> bool:
        """Check if torchvision and dataset files exist locally."""
        try:
            import torchvision  # noqa: F401
            return os.path.exists(os.path.join(self.root, dataset_name))
        except ImportError:
            return False

    def load_sample(self, dataset_name: str, index: int) -> np.ndarray:
        """Load external sample if available, or raise RuntimeError."""
        if not self.is_available(dataset_name):
            raise RuntimeError(
                f"Dataset '{dataset_name}' not found in {self.root}. "
                "OpenDiffraction core uses procedural generation and does not require external data."
            )
        # Placeholder for external loading logic
        return np.zeros((64, 64), dtype=np.float32)
