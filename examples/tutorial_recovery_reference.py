"""Acquire an independent classical reference from caller-declared UFL forms.

Meshes, spaces, physical fields, boundary nodes and the mathematical operator
remain explicit caller inputs. Native space conversion, essential elimination,
full-field integration and sampling reuse their existing owners.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from examples.introduction.scalar import dirichlet_solve, native_scalar_space
from examples.introduction.transport import scalar_error_norms
from pymhm.core.equations import compile_form
from pymhm.fem.scalar.triangle import nodal_space
from pymhm.meshes.triangle import TriangleMesh
from pymhm.postprocessing.fields import DiscreteField
from pymhm.postprocessing.nodal import nodal_field


def classical_reference_control(
    mesh: TriangleMesh,
    degree: int,
    variational_forms: Callable[[Any, Any], tuple[Any, Any]],
    boundary_nodes: Callable[[np.ndarray], np.ndarray],
    exact_pressure: Callable[[np.ndarray], np.ndarray],
    exact_gradient: Callable[[np.ndarray], np.ndarray],
) -> tuple[dict[str, Any], DiscreteField]:
    """Solve caller-supplied scalar forms with homogeneous essential pressure.

    ``boundary_nodes`` returns the portable nodal indices at which pressure is
    zero. The native ordering is mapped before elimination. Physical flux norms
    use unit diffusion and two independent rules, with 12 and 16 points per
    Duffy coordinate. No MHM matrix, basis or coefficient vector is reused.
    """
    domain, space, mapping = native_scalar_space(mesh, degree)
    bilinear, linear = variational_forms(domain, space)
    matrix, load = compile_form(bilinear), compile_form(linear)
    _, points = nodal_space(mesh, degree)
    boundary = boundary_nodes(points)
    values = dirichlet_solve(matrix, load, mapping[boundary], np.zeros(len(boundary)))[mapping]
    errors = [
        scalar_error_norms(
            (mesh,), (values,), degree, exact_pressure, exact_gradient, 1.0, order=order
        )
        for order in (12, 16)
    ]
    difference = max(
        abs(errors[0][key] - errors[1][key]) / max(errors[1][key], np.finfo(float).tiny)
        for key in ("scalar_l2", "flux_l2")
    )
    if difference > 1e-7:
        raise ValueError("independent classical physical-error quadrature has not stabilized")
    return {
        "unknowns": len(points),
        "fine_cells": len(mesh.cells),
        "mesh_diameter": float(np.max(mesh.lengths)),
        "degree": degree,
        "quadrature_orders": [12, 16],
        "quadrature_errors": errors,
        "quadrature_relative_difference": difference,
        **errors[-1],
    }, DiscreteField(nodal_field("pressure", mesh, degree), values)


def classical_display_samples(
    field: DiscreteField,
    mesh: TriangleMesh,
    comparison_macro: TriangleMesh,
    exact_pressure: Callable[[np.ndarray], np.ndarray],
    exact_gradient: Callable[[np.ndarray], np.ndarray],
) -> dict[str, Any]:
    """Sample the native global reference with the requested comparison macro mesh."""
    points = mesh.points[mesh.cells].mean(axis=1)
    pressure, gradient = field.values_and_gradient(points, cells=np.arange(len(mesh.cells)))
    analytical = exact_pressure(points)
    physical = -exact_gradient(points)
    numerical = -gradient
    return {
        "points": mesh.points.tolist(),
        "cells": mesh.cells.tolist(),
        "macro_points": comparison_macro.points.tolist(),
        "macro_faces": comparison_macro.faces.tolist(),
        "pressure": pressure.tolist(),
        "pressure_error": (pressure - analytical).tolist(),
        "flux_magnitude": np.linalg.norm(numerical, axis=1).tolist(),
        "flux_error": np.linalg.norm(numerical - physical, axis=1).tolist(),
    }
