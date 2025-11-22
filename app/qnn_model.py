"""Quantum Variational Classifier built with PennyLane.

The model encodes a single feature ``x`` using a rotation around the X-axis
and then applies trainable RX, RY, and RZ gates. The circuit measures the
expectation value of the Pauli-Z operator and converts it to a probability
that the label is 1. This simple design keeps the math light while
highlighting how quantum circuits can represent learnable decision boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pennylane as qml


@dataclass
class QNNConfig:
    """Configuration for the quantum model.

    Attributes:
        n_qubits: Number of qubits in the circuit. For this demo we use 1.
        seed: Optional random seed for weight initialization.
    """

    n_qubits: int = 1
    seed: int | None = 42


class QuantumVariationalClassifier:
    """Small quantum classifier with trainable rotation gates."""

    def __init__(self, config: QNNConfig):
        self.config = config
        if self.config.seed is not None:
            np.random.seed(self.config.seed)

        # Initialize weights for RX, RY, and RZ gates.
        self.weights = np.random.uniform(low=-np.pi, high=np.pi, size=(3,))

        # Use the default.qubit device since no hardware is required.
        self.device = qml.device("default.qubit", wires=self.config.n_qubits)

        # Build the QNode that will serve as the forward pass of our model.
        self.circuit: Callable[[float, np.ndarray], float] = qml.QNode(
            self._circuit, self.device
        )

    def _circuit(self, x: float, weights: np.ndarray) -> float:
        """Quantum circuit that returns the expectation value.

        Args:
            x: Single input feature value.
            weights: Trainable parameters for RX, RY, and RZ gates.

        Returns:
            Expectation value of Pauli-Z measurement on the last qubit.
        """

        # Encode the classical input into the quantum state via rotation.
        qml.RX(np.pi * x, wires=0)

        # Apply learnable single-qubit rotations. More complex entanglement
        # could be added here for multi-qubit setups.
        qml.RX(weights[0], wires=0)
        qml.RY(weights[1], wires=0)
        qml.RZ(weights[2], wires=0)

        # Measure the Z expectation value which lies between -1 and 1.
        return qml.expval(qml.PauliZ(0))

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Predict probabilities for inputs using the quantum circuit.

        Args:
            x: Array of shape ``(n_samples, 1)`` containing feature values.

        Returns:
            Array of probabilities for the positive class with shape
            ``(n_samples,)``.
        """

        probabilities = []
        for value in x[:, 0]:
            # Convert expectation (range [-1, 1]) to probability [0, 1].
            expectation = self.circuit(value, self.weights)
            probability = (1 - expectation) / 2
            probabilities.append(probability)
        return np.array(probabilities)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict binary labels by thresholding the probability at 0.5."""

        return (self.predict_proba(x) >= 0.5).astype(int)

    def parameters(self) -> np.ndarray:
        """Return the current trainable parameters."""

        return self.weights

    def set_parameters(self, new_weights: np.ndarray) -> None:
        """Update the trainable parameters in-place."""

        self.weights = new_weights


# Utility loss function used by the training loop.
def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute binary cross-entropy loss with numerical stability.

    Args:
        y_true: Ground truth labels of shape ``(n_samples,)``.
        y_pred: Predicted probabilities of shape ``(n_samples,)``.

    Returns:
        Mean binary cross-entropy loss as a float.
    """

    eps = 1e-9
    y_pred = np.clip(y_pred, eps, 1 - eps)
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return float(loss)


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute classification accuracy."""

    return float(np.mean(y_true == y_pred))
