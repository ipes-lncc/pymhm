"""Componentwise physical data in an explicitly signed vector triangular trace."""

from functools import partial
from typing import Any

import numpy as np

from pymhm.core.validation import FloatArray
from pymhm.fem.traces.triangle_3d import TriangularSkeleton, tetra_boundary_data
from pymhm.materials.evaluation import vector_values_3d


def _component(points: FloatArray, *, datum: Any, component: int) -> FloatArray:
    """Evaluate one vector component through the shared three-dimensional validator."""
    return vector_values_3d(datum, points)[:, component]


def vector_boundary_components_3d(
    skeleton: TriangularSkeleton,
    dirichlet: Any,
    traction: dict[int, Any],
    components: dict[int, dict[int, Any]],
    order: int,
) -> tuple[FloatArray, dict[int, float]]:
    """Project physical outward traction and assemble all remaining velocity moments."""
    exterior = set(skeleton.mesh.boundary_faces)
    if not set(traction).issubset(exterior) or not set(components).issubset(exterior):
        raise ValueError("traction keys must identify exterior faces")
    if set(traction) & set(components):
        raise ValueError("full and componentwise traction must not overlap")
    for values in components.values():
        if (
            not isinstance(values, dict)
            or not values
            or any(
                isinstance(component, (bool, np.bool_)) or component not in (0, 1, 2)
                for component in values
            )
        ):
            raise ValueError(
                "traction_components needs nonempty dictionaries with components 0, 1 or 2"
            )
    load = np.zeros((skeleton.size, 3))
    fixed = {}
    for component in range(3):
        prescribed = {
            face: partial(_component, datum=value, component=component)
            for face, value in traction.items()
        }
        prescribed.update(
            {face: values[component] for face, values in components.items() if component in values}
        )
        load[:, component], scalar_fixed = tetra_boundary_data(
            skeleton, partial(_component, datum=dirichlet, component=component), prescribed, order
        )
        fixed.update({3 * int(index) + component: -value for index, value in scalar_fixed.items()})
    return load.ravel(), fixed
