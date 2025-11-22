"""Tests for dataset generation."""

import numpy as np

from app.dataset import DatasetConfig, generate_dataset


def test_binary_rule_applied():
    config = DatasetConfig(n_samples=10, noise=0.0, seed=0)
    x, y = generate_dataset(config)
    expected = (x[:, 0] < 0.5).astype(int)
    assert np.array_equal(y, expected)
