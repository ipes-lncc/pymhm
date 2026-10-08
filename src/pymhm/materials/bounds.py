"""Declared certified material bounds, separate from integration-point checks."""

from typing import Any

import numpy as np

from pymhm.materials.cartesian import CartesianCellField
from pymhm.materials.evaluation import tensor_values


def ellipticity_lower_bound(material: Any, supplied: Any) -> float:
    """Infer a constant-material bound or require a certified callback bound."""
    if supplied is None:
        if isinstance(material, CartesianCellField):
            data = material.values
            supplied = np.min(data) if data.ndim == 2 else np.min(np.linalg.eigvalsh(data))
        elif callable(material):
            raise ValueError("material callbacks require a certified ellipticity_lower_bound")
        else:
            supplied = np.linalg.eigvalsh(tensor_values(material, np.zeros((1, 2)))).min()
    value = np.asarray(supplied)
    if value.shape != () or np.iscomplexobj(value) or not np.isfinite(value) or value <= 0:
        raise ValueError("ellipticity_lower_bound must be a finite positive scalar")
    return float(value)
