"""Tests for the quantum model outputs."""

import numpy as np

from app.qnn_model import QNNConfig, QuantumVariationalClassifier


def test_probability_range():
    model = QuantumVariationalClassifier(QNNConfig(seed=0))
    x = np.linspace(0, 1, 5).reshape(-1, 1)
    probs = model.predict_proba(x)
    assert np.all(probs >= 0) and np.all(probs <= 1)
