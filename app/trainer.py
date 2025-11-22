"""Training utilities for the quantum variational classifier.

The training loop uses PennyLane's built-in gradient descent optimizer to
minimize the binary cross-entropy loss between predicted probabilities and
true labels. Progress is logged to keep learners aware of how the loss and
accuracy change over time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pennylane as qml

from .qnn_model import (
    QuantumVariationalClassifier,
    binary_cross_entropy,
    accuracy_score,
)


@dataclass
class TrainerConfig:
    """Configuration options for the training process."""

    epochs: int = 80
    learning_rate: float = 0.1
    log_interval: int = 10


def train_qnn(
    model: QuantumVariationalClassifier, x: np.ndarray, y: np.ndarray, config: TrainerConfig
) -> Tuple[QuantumVariationalClassifier, List[float]]:
    """Train the quantum model using gradient descent.

    Args:
        model: The quantum classifier instance to train.
        x: Features with shape ``(n_samples, 1)``.
        y: Binary labels with shape ``(n_samples,)``.
        config: Hyperparameters for optimization.

    Returns:
        The trained model and a list of loss values recorded each epoch.
    """

    opt = qml.GradientDescentOptimizer(stepsize=config.learning_rate)
    loss_history: List[float] = []

    def cost_fn(weights: np.ndarray) -> float:
        # Compute predictions using the proposed weights.
        model.set_parameters(weights)
        y_pred = model.predict_proba(x)
        return binary_cross_entropy(y, y_pred)

    weights = model.parameters()
    for epoch in range(1, config.epochs + 1):
        weights, current_loss = opt.step_and_cost(cost_fn, weights)
        model.set_parameters(weights)
        loss_history.append(current_loss)

        if epoch % config.log_interval == 0 or epoch == 1 or epoch == config.epochs:
            preds = model.predict(x)
            acc = accuracy_score(y, preds)
            print(
                f"Epoch {epoch:03d} | Loss: {current_loss:.4f} | Accuracy: {acc:.3f}"
            )

    return model, loss_history
