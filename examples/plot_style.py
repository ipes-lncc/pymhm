"""Shared scientific axis formatting for reproducible publication figures."""

from typing import Any

import numpy as np
from matplotlib.ticker import NullFormatter


def set_refinement_ticks(
    axis: Any, values: Any, labels: Any = None, *, max_labels: int | None = None
) -> None:
    """Label measured positions, optionally selecting a bounded number of axis labels.

    Selection affects tick labels only: every measured value remains in the
    plotted series. The first and last positions are retained when the bound
    allows at least two labels. Logarithmic minor ticks remain unlabelled.
    """
    positions = np.asarray(values, dtype=float)
    if labels is None:
        labels = [f"{value:g}" for value in positions]
    if max_labels is not None:
        if max_labels < 1:
            raise ValueError("max_labels must be positive")
        indices = np.unique(np.linspace(0, len(positions) - 1, max_labels, dtype=int))
        positions = positions[indices]
        labels = [labels[index] for index in indices]
    axis.set_xticks(positions, labels=labels)
    axis.xaxis.set_minor_formatter(NullFormatter())
