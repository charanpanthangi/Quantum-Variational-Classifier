"""Command-line interface to train the QNN and baseline models.

Running this script will:
1. Generate the synthetic dataset.
2. Train the quantum variational classifier.
3. Train a classical logistic regression baseline.
4. Report accuracies.
5. Save SVG plots for decision boundaries and training loss.
"""

from __future__ import annotations

import argparse

import numpy as np

from .classical_baseline import train_logistic_regression
from .dataset import DatasetConfig, generate_dataset
from .plots import EXAMPLES_DIR, save_decision_boundary, save_loss_curve
from .qnn_model import QuantumVariationalClassifier, QNNConfig
from .trainer import TrainerConfig, train_qnn


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Train a quantum variational classifier.")
    parser.add_argument("--n-samples", type=int, default=150, help="Number of data samples.")
    parser.add_argument("--epochs", type=int, default=80, help="Training epochs for the QNN.")
    parser.add_argument(
        "--learning-rate", type=float, default=0.1, help="Learning rate for gradient descent."
    )
    parser.add_argument("--noise", type=float, default=0.02, help="Standard deviation of noise.")
    return parser.parse_args()


def main() -> None:
    """Entry point for running the training pipeline."""

    args = parse_args()

    # 1. Prepare dataset.
    data_config = DatasetConfig(n_samples=args.n_samples, noise=args.noise)
    x, y = generate_dataset(data_config)

    # 2. Train quantum model.
    qnn = QuantumVariationalClassifier(QNNConfig())
    trainer_config = TrainerConfig(epochs=args.epochs, learning_rate=args.learning_rate)
    qnn, losses = train_qnn(qnn, x, y, trainer_config)
    qnn_preds = qnn.predict(x)

    # 3. Train classical baseline.
    baseline_result = train_logistic_regression(x, y)

    # 4. Report accuracies.
    qnn_acc = np.mean(qnn_preds == y)
    print(f"QNN accuracy: {qnn_acc:.3f}")
    print(f"Logistic regression accuracy: {baseline_result.accuracy:.3f}")

    # 5. Save SVG plots.
    qnn_plot = save_decision_boundary(
        x,
        y_true=y,
        y_pred=qnn_preds,
        filename=EXAMPLES_DIR / "decision_boundary_qnn.svg",
        title="QNN decision boundary",
    )
    classical_plot = save_decision_boundary(
        x,
        y_true=y,
        y_pred=baseline_result.model.predict(x),
        filename=EXAMPLES_DIR / "decision_boundary_classical.svg",
        title="Classical logistic regression decision boundary",
    )
    loss_plot = save_loss_curve(losses, EXAMPLES_DIR / "training_loss_qnn.svg")

    print("SVG plots saved:")
    print(f"- {qnn_plot}")
    print(f"- {classical_plot}")
    print(f"- {loss_plot}")


if __name__ == "__main__":
    main()
