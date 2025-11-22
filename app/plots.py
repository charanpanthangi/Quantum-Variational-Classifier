"""Plotting utilities that save lightweight SVG visuals.

All plots are saved as SVG files to avoid GitHub's warning about binary files
like PNG. SVGs are text-based, easy to version, and render clearly in the
repository diff viewer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

# Ensure Matplotlib uses a clean, minimal style.
plt.style.use("seaborn-v0_8-whitegrid")


EXAMPLES_DIR = Path("examples")


def _prepare_dir(path: Path) -> None:
    """Create parent directories for a given path if they do not exist."""

    path.parent.mkdir(parents=True, exist_ok=True)


def save_decision_boundary(
    x: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    filename: Path,
    title: str,
) -> Path:
    """Plot a decision boundary scatter plot and save as SVG.

    Args:
        x: Input features with shape ``(n_samples, 1)``.
        y_true: Ground truth labels.
        y_pred: Predicted labels to compare.
        filename: Path to save the SVG file.
        title: Title for the plot.

    Returns:
        Path to the saved SVG file.
    """

    _prepare_dir(filename)

    plt.figure(figsize=(4, 3))
    plt.scatter(x[:, 0], y_true, color="#1f77b4", label="True labels", s=20)
    plt.scatter(x[:, 0], y_pred, color="#ff7f0e", label="Model predictions", s=20)
    plt.axvline(0.5, color="gray", linestyle="--", linewidth=1, label="Rule x=0.5")
    plt.xlabel("Feature x")
    plt.ylabel("Label")
    plt.title(title)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    # Save in SVG format only to keep repository diffs text-friendly.
    plt.savefig(filename, format="svg")
    plt.close()
    return filename


def save_loss_curve(losses: Iterable[float], filename: Path) -> Path:
    """Plot training loss over epochs and save as SVG."""

    _prepare_dir(filename)
    plt.figure(figsize=(4, 3))
    plt.plot(range(1, len(losses) + 1), losses, marker="o", markersize=3)
    plt.xlabel("Epoch")
    plt.ylabel("Binary cross-entropy loss")
    plt.title("QNN training loss")
    plt.tight_layout()
    plt.savefig(filename, format="svg")
    plt.close()
    return filename
