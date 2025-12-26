from __future__ import annotations

from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


def compute_confusion_matrix(
    y_true: List[str],
    y_pred: List[str],
    label_list: List[str],
    normalize: Optional[str] = None,
) -> np.ndarray:
    """
    Computes a confusion matrix.

    normalize:
        None        -> raw counts
        'true'      -> normalize by true labels (row-wise)
        'pred'      -> normalize by predictions (column-wise)
        'all'       -> normalize by total samples
    """
    cm = confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        labels=label_list,
        normalize=normalize,
    )
    return cm


def plot_confusion_matrix(
    cm: np.ndarray,
    label_list: List[str],
    title: str = "Confusion Matrix",
    figsize: tuple = (8, 6),
    cmap: str = "Blues",
    save_path: Optional[str] = None,
) -> None:
    """
    Plots a confusion matrix.
    """

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(label_list)),
        yticks=np.arange(len(label_list)),
        xticklabels=label_list,
        yticklabels=label_list,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]:.2f}" if cm.dtype != int else str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
