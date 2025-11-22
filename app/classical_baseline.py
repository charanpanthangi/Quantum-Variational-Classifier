"""Classical logistic regression baseline for comparison with the QNN.

The baseline uses scikit-learn's `LogisticRegression` to learn the same
simple rule as the quantum model. This gives readers a familiar reference
point for expected performance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


@dataclass
class BaselineResult:
    """Container for fitted model results."""

    model: LogisticRegression
    accuracy: float


def train_logistic_regression(x: np.ndarray, y: np.ndarray) -> BaselineResult:
    """Train a logistic regression classifier.

    Args:
        x: Features with shape ``(n_samples, 1)``.
        y: Binary labels with shape ``(n_samples,)``.

    Returns:
        BaselineResult containing the trained model and accuracy.
    """

    clf = LogisticRegression()
    clf.fit(x, y)
    preds = clf.predict(x)
    acc = accuracy_score(y, preds)
    return BaselineResult(model=clf, accuracy=float(acc))
