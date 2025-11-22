"""Dataset utilities for the quantum variational classifier.

This module creates a simple one-dimensional dataset where the label is
`1` when the feature `x` is less than `0.5`, and `0` otherwise. Optional
noise can be added to demonstrate robustness. The functions return NumPy
arrays that are easy to feed into both the quantum and classical models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class DatasetConfig:
    """Configuration options for generating the synthetic dataset.

    Attributes:
        n_samples: Number of samples to create.
        noise: Standard deviation of Gaussian noise added to each feature.
        seed: Optional random seed for reproducibility.
    """

    n_samples: int = 200
    noise: float = 0.02
    seed: int | None = 7


def generate_dataset(config: DatasetConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Generate the synthetic dataset for binary classification.

    The rule is intentionally simple so that the QNN and the classical
    baseline can learn it quickly:

    * If ``x < 0.5`` the label is ``1``.
    * Otherwise the label is ``0``.

    Args:
        config: Settings that describe the dataset size and optional noise.

    Returns:
        Tuple of features ``x`` with shape ``(n_samples, 1)`` and labels
        ``y`` with shape ``(n_samples,)``.
    """

    if config.seed is not None:
        np.random.seed(config.seed)

    # Draw x values uniformly from the range [0, 1].
    x = np.random.rand(config.n_samples, 1)

    if config.noise > 0:
        # Add small Gaussian noise to make the task slightly less perfect.
        x = x + np.random.normal(loc=0.0, scale=config.noise, size=x.shape)
        # Keep values in [0, 1] after noise.
        x = np.clip(x, 0.0, 1.0)

    # Apply the simple rule to generate labels.
    y = (x[:, 0] < 0.5).astype(int)
    return x, y
