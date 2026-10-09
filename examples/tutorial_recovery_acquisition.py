"""Measure user-declared Darcy recovery studies through shared numerical owners.

This helper selects no PDE or approximation spaces. Notebook cells provide
their equations, source, analytical fields, meshes and refinement decisions.
Only scalar diagnostics and display samples are persisted; coefficient-vector
replay is outside this record's contract.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from pymhm.estimators.darcy import DarcyEstimator, estimate_darcy_error
from pymhm.fem.hdiv.rt import rt_evaluate
from pymhm.io.provenance import current_source_manifest, file_digest
from pymhm.postprocessing.solutions import DarcySolution


def recovery_diagnostics(
    solution: DarcySolution,
    exact_pressure: Callable[[np.ndarray], np.ndarray],
    exact_gradient: Callable[[np.ndarray], np.ndarray],
    *,
    reconstruction_degree: int = 2,
    quadrature_orders: tuple[int, int] = (12, 16),
) -> tuple[dict[str, Any], DarcyEstimator]:
    """Integrate recovery errors twice and report conservation and effectivity.

    Flux is the physical vector ``-grad(p)`` for the explicitly required unit
    diffusion. The estimator uses the higher quadrature order. The projected
    divergence is its continuous macro-local L2 projection, not raw divergence.
    No independent fine-cell balance is imposed by canonical RT moments.
    """
    first, second = quadrature_orders
    if not 0 < first < second:
        raise ValueError("two increasing positive quadrature orders are required")
    estimate = estimate_darcy_error(
        solution,
        homogeneous_dirichlet=True,
        degree=reconstruction_degree,
        quadrature_order=second,
    )
    recovered = estimate.reconstructed_flux

    def physical_flux(points: np.ndarray) -> np.ndarray:
        """Differentiate the independently supplied analytical pressure."""
        return -exact_gradient(points)

    errors = []
    for order in quadrature_orders:
        errors.append(
            {
                "pressure_l2": solution.l2_error(exact_pressure, order=order),
                "raw_flux_l2": estimate.energy_error(exact_gradient, order=order),
                "recovered_flux_l2": recovered.flux_l2_error(physical_flux, order=order),
                "projected_divergence_l2": recovered.projected_divergence_l2_error(
                    solution.source, order=order
                ),
                "raw_divergence_l2": recovered.divergence_l2_error(solution.source, order=order),
                "conforming_pressure_l2": estimate.potential.l2_error(exact_pressure, order=order),
            }
        )
    discrepancy = max(
        abs(errors[0][name] - errors[1][name]) / max(errors[1][name], np.finfo(float).tiny)
        for name in errors[0]
    )
    if discrepancy > 1e-7:
        raise ValueError("physical error quadrature has not stabilized")
    energy = estimate.energy_error(exact_gradient, order=second)
    normal = max(float(np.max(abs(row))) for row in recovered.normal_flux_residuals())
    continuous = max(float(np.max(abs(row))) for row in recovered.continuous_moment_residuals())
    raw_residual = float(solution.hybrid.raw_residual)
    if max(normal, continuous, raw_residual) > 1e-10:
        raise ValueError("the original equations or recovered conservation identity failed")
    if estimate.total < energy:
        raise ValueError(
            "the evaluated estimator does not bound the independently integrated error"
        )
    mesh = solution.skeleton.mesh
    row = {
        "macro_cells": len(mesh.cells),
        "macro_diameter": float(np.max(mesh.lengths)),
        "trace_dofs": solution.skeleton.size,
        "local_degree": solution.degree,
        "trace_degree": max(max(face.degrees) for face in solution.skeleton.faces),
        "reconstruction_degree": reconstruction_degree,
        "quadrature_orders": list(quadrature_orders),
        "quadrature_relative_difference": float(discrepancy),
        "quadrature_errors": errors,
        **errors[-1],
        "energy_error": energy,
        "estimator": estimate.total,
        "effectivity": estimate.total / energy,
        "eta_1": float(np.linalg.norm(estimate.flux_defect)),
        "eta_2": float(np.linalg.norm(estimate.nonconformity)),
        "eta_3": float(np.linalg.norm(estimate.divergence_defect)),
        "eta_osc": float(np.linalg.norm(estimate.oscillation)),
        "normal_moment_defect": normal,
        "continuous_moment_defect": continuous,
        "fine_cell_balance_defect_not_imposed": max(
            float(np.max(abs(values))) for values in recovered.fine_conservation_residuals()
        ),
        "original_equation_residual": raw_residual,
        "basis_digests": [field.basis_digest for field in solution.hybrid.field("pressure")],
    }
    return row, estimate


def recovery_display_samples(
    solution: DarcySolution,
    estimate: DarcyEstimator,
    exact_pressure: Callable[[np.ndarray], np.ndarray],
    exact_gradient: Callable[[np.ndarray], np.ndarray],
) -> dict[str, Any]:
    """Sample pressure and RT flux independently on every fine triangle.

    Each cell's own centroid and coefficient vector retain the two sides of
    macro interfaces. These display samples never replace integrated norms.
    """
    panels = []
    bary = np.full((1, 3), 1 / 3)
    fields = solution.hybrid.field("pressure")
    for mesh, field, coefficients in zip(
        solution.local_meshes, fields, estimate.reconstructed_flux.flux, strict=True
    ):
        points = mesh.points[mesh.cells].mean(axis=1)
        pressure = field.evaluate(points, cells=np.arange(len(mesh.cells)))
        recovered, _ = rt_evaluate(mesh, coefficients, estimate.reconstructed_flux.degree, bary)
        recovered = recovered[:, 0]
        analytical = exact_pressure(points)
        flux = -exact_gradient(points)
        panels.append(
            {
                "points": mesh.points.tolist(),
                "cells": mesh.cells.tolist(),
                "analytical_pressure": analytical.tolist(),
                "pressure": pressure.tolist(),
                "pressure_error": (pressure - analytical).tolist(),
                "analytical_flux_magnitude": np.linalg.norm(flux, axis=1).tolist(),
                "recovered_flux_magnitude": np.linalg.norm(recovered, axis=1).tolist(),
                "recovered_flux_error": np.linalg.norm(recovered - flux, axis=1).tolist(),
            }
        )
    macro = solution.skeleton.mesh
    return {
        "macro_points": macro.points.tolist(),
        "macro_faces": macro.faces.tolist(),
        "panels": panels,
    }


def recovery_provenance(root: Path, source_paths: Sequence[Path]) -> dict[str, str]:
    """Snapshot the numerical source closure before an acquisition starts."""
    sources = [*sorted((root / "src/pymhm").rglob("*.py")), *source_paths]
    identities = {str(item.relative_to(root)): file_digest(item) for item in sources}
    return current_source_manifest(identities, packages=("pymhm", "dolfinx", "ufl", "basix"))


def notebook_numerical_digest(path: Path) -> str:
    """Identify executable cells except the checksum-only workspace bootstrap.

    Updating a companion URL cannot change the equations, studies, data or
    plotting cells. All those cells remain in this digest. The complete executed
    notebook retains its separate source-manifest fingerprint.
    """
    document = json.loads(path.read_text(encoding="utf-8"))
    source = [
        "".join(cell["source"])
        for cell in document["cells"]
        if cell["cell_type"] == "code"
        and not (
            "workspace_from_archive" in "".join(cell["source"])
            and "COMPANION_SHA256" in "".join(cell["source"])
        )
    ]
    return sha256(json.dumps(source, ensure_ascii=False).encode()).hexdigest()


def read_recovery_record(path: Path, *, notebook: Path) -> dict[str, Any]:
    """Read attributed observations and verify the numerical notebook cells."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("numerical_notebook_sha256") != notebook_numerical_digest(notebook):
        raise ValueError("the recovery record belongs to different numerical notebook cells")
    return data


def write_recovery_record(
    path: Path,
    *,
    root: Path,
    source_paths: Sequence[Path],
    source_manifest: dict[str, str],
    data: dict[str, Any],
) -> None:
    """Reject changed sources, then archive scalar diagnostics and native versions.

    ``source_manifest`` must be the snapshot taken before the first solve. The
    executed notebook and acquisition dependencies are supplied explicitly;
    package owners are included independently. No coefficient vector is replayed.
    """
    if source_manifest != recovery_provenance(root, source_paths):
        raise RuntimeError("recovery acquisition sources changed during execution")
    versions = {
        name: version(name)
        for name in ("pymhm", "fenics-dolfinx", "fenics-ufl", "fenics-basix", "numpy", "scipy")
    }
    notebooks = [item for item in source_paths if item.suffix == ".ipynb"]
    if len(notebooks) != 1:
        raise ValueError("declare exactly one numerical notebook source")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                **data,
                "source_manifest": source_manifest,
                "versions": versions,
                "numerical_notebook_sha256": notebook_numerical_digest(notebooks[0]),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
