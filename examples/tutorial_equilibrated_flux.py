"""Observe strict RT0 equilibration of explicitly declared primal equations.

The notebook owns every mathematical form, mesh and degree choice. This module
only calls the package's existing reconstruction, integration and sampling
owners. Its scalar diagnostics and centroid samples contain no coefficient
vectors intended for field replay.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from pymhm.fem.scalar.operators import rt0_evaluate
from pymhm.postprocessing.solutions import DarcySolution
from pymhm.recovery.equilibrated import EquilibratedFlux, equilibrate_flux


def equilibrated_diagnostics(
    solution: DarcySolution,
    exact_pressure: Callable[[np.ndarray], np.ndarray],
    exact_gradient: Callable[[np.ndarray], np.ndarray],
) -> tuple[dict[str, Any], EquilibratedFlux]:
    """Measure physical vector errors twice and verify every fine-cell balance.

    The caller supplies unit diffusion, homogeneous Dirichlet pressure and P0
    skeletal segments aligned with the fine boundary edges. The RT0 algorithm
    itself checks its source/trace compatibility and constrained local equation.
    """
    recovered = equilibrate_flux(solution)

    def physical_flux(points: np.ndarray) -> np.ndarray:
        """Use the independently differentiated exact pressure."""
        return -exact_gradient(points)

    errors = [
        {
            "pressure_l2": solution.l2_error(exact_pressure, order=order),
            "raw_flux_l2": solution.flux_l2_error(physical_flux, order=order),
            "equilibrated_flux_l2": recovered.l2_error(physical_flux, order=order),
        }
        for order in (12, 16)
    ]
    discrepancy = max(
        abs(errors[0][name] - errors[1][name]) / max(errors[1][name], np.finfo(float).tiny)
        for name in errors[0]
    )
    balance = max(float(np.max(abs(values))) for values in recovered.conservation_residuals())
    original = float(solution.hybrid.raw_residual)
    if discrepancy > 1e-7:
        raise ValueError("physical RT0 error quadrature has not stabilized")
    if max(balance, original) > 1e-10:
        raise ValueError("original equations or strict fine-cell balance failed")
    macro = solution.skeleton.mesh
    return {
        "macro_cells": len(macro.cells),
        "macro_diameter": float(np.max(macro.lengths)),
        "local_degree": solution.degree,
        "trace_degree": max(max(face.degrees) for face in solution.skeleton.faces),
        "quadrature_orders": [12, 16],
        "quadrature_errors": errors,
        "quadrature_relative_difference": discrepancy,
        **errors[-1],
        "fine_cell_balance_defect": balance,
        "original_equation_residual": original,
        "basis_digests": [field.basis_digest for field in solution.hybrid.field("pressure")],
    }, recovered


def equilibrated_display_samples(
    solution: DarcySolution,
    recovered: EquilibratedFlux,
    exact_pressure: Callable[[np.ndarray], np.ndarray],
    exact_gradient: Callable[[np.ndarray], np.ndarray],
) -> dict[str, Any]:
    """Sample the actual equilibrated RT0 field on separate incident fine cells."""
    panels = []
    bary = np.full((1, 3), 1 / 3)
    for mesh, field, flux in zip(
        solution.local_meshes,
        solution.hybrid.field("pressure"),
        recovered.coefficients,
        strict=True,
    ):
        points = mesh.points[mesh.cells].mean(axis=1)
        pressure = field.evaluate(points, cells=np.arange(len(mesh.cells)))
        numerical = rt0_evaluate(mesh, flux, bary)[:, 0]
        analytical = -exact_gradient(points)
        panels.append(
            {
                "points": mesh.points.tolist(),
                "cells": mesh.cells.tolist(),
                "analytical_pressure": exact_pressure(points).tolist(),
                "pressure": pressure.tolist(),
                "pressure_error": (pressure - exact_pressure(points)).tolist(),
                "analytical_flux_magnitude": np.linalg.norm(analytical, axis=1).tolist(),
                "recovered_flux_magnitude": np.linalg.norm(numerical, axis=1).tolist(),
                "recovered_flux_error": np.linalg.norm(numerical - analytical, axis=1).tolist(),
            }
        )
    macro = solution.skeleton.mesh
    return {
        "macro_points": macro.points.tolist(),
        "macro_faces": macro.faces.tolist(),
        "panels": panels,
    }
