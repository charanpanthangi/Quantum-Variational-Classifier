"""Tests for the QNN training loop."""

import numpy as np

from app.dataset import DatasetConfig, generate_dataset
from app.qnn_model import QNNConfig, QuantumVariationalClassifier
from app.trainer import TrainerConfig, train_qnn


def test_training_reduces_loss():
    x, y = generate_dataset(DatasetConfig(n_samples=20, noise=0.0, seed=1))
    model = QuantumVariationalClassifier(QNNConfig(seed=0))
    config = TrainerConfig(epochs=5, learning_rate=0.2, log_interval=10)
    _, losses = train_qnn(model, x, y, config)
    assert losses[-1] <= losses[0]
