from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
)


def seed_everything(seed: int = 42) -> None:
    """Seed common libraries used by the project."""
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        return


def quadratic_weighted_kappa(
    y_true: np.ndarray | list[int],
    y_pred: np.ndarray | list[int],
    num_classes: int = 5,
) -> float:
    """Compute the Kaggle metric for the ordinal adoption-speed target."""
    return float(
        cohen_kappa_score(
            y_true,
            y_pred,
            labels=np.arange(num_classes),
            weights="quadratic",
        )
    )


def classification_metrics(
    y_true: np.ndarray | list[int],
    y_pred: np.ndarray | list[int],
    num_classes: int = 5,
) -> dict[str, Any]:
    """Return the core classification metrics used across experiments."""
    labels = list(range(num_classes))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "qwk": quadratic_weighted_kappa(y_true, y_pred, num_classes=num_classes),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
    }


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def plot_class_distribution(
    labels: np.ndarray | list[int],
    title: str = "Adoption Speed Distribution",
) -> Any:
    """Plot target class counts and return the matplotlib axis."""
    import matplotlib.pyplot as plt
    import pandas as pd

    counts = pd.Series(labels).value_counts().sort_index()
    axis = counts.plot(kind="bar", color="#4c78a8", edgecolor="black")
    axis.set_title(title)
    axis.set_xlabel("AdoptionSpeed")
    axis.set_ylabel("Count")
    plt.tight_layout()
    return axis


def plot_confusion_matrix(
    y_true: np.ndarray | list[int],
    y_pred: np.ndarray | list[int],
    title: str = "Confusion Matrix",
    num_classes: int = 5,
) -> Any:
    """Plot a labeled confusion matrix and return the matplotlib axis."""
    import matplotlib.pyplot as plt

    matrix = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    fig, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=axis)
    axis.set_title(title)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_xticks(range(num_classes))
    axis.set_yticks(range(num_classes))

    for row in range(num_classes):
        for col in range(num_classes):
            axis.text(col, row, matrix[row, col], ha="center", va="center", color="black")

    plt.tight_layout()
    return axis
