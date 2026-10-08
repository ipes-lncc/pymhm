"""Initial macro refinement on an explicitly smaller primary-data Marmousi crop."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from numpy.polynomial import polynomial
from numpy.polynomial.legendre import leggauss

from examples.helmholtz_basis_archive import basis_payload
from examples.helmholtz_trace_family import verify_helmholtz_solution
from examples.marmousi_data import load_marmousi_crop
from examples.minimal_wave_convergence import (
    ROOT,
    digest,
    quadrature_change,
    require_original,
    write,
)
from examples.tutorial_helmholtz_equations import solve_acoustic
from pymhm.fem.scalar.quadrilateral import cardinal_polynomials, qk_space
from pymhm.fem.traces.helmholtz import helmholtz_skeleton
from pymhm.meshes.cartesian import CartesianMacroMesh


def reference_values(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Evaluate the archived ascending-power cardinal factors in x-fast Q3 order."""
    x = np.column_stack([polynomial.polyval(points[:, 0], row) for row in matrix])
    y = np.column_stack([polynomial.polyval(points[:, 1], row) for row in matrix])
    return (y[:, :, None] * x[:, None, :]).reshape(len(points), -1)


def physical_norm(
    values: np.ndarray,
    matrix: np.ndarray,
    target: np.ndarray | None = None,
    target_matrix: np.ndarray | None = None,
    order: int = 4,
) -> float:
    """Integrate complex pressure L2 on the fixed2.5m fine-cell partition, without smoothing."""
    t, w = leggauss(order)
    reference = np.asarray([(x, y) for y in (t + 1) / 2 for x in (t + 1) / 2])
    weights = np.outer(w / 2, w / 2).ravel() * 2.5**2
    actual = np.einsum("qi,ti->tq", reference_values(matrix, reference), values.reshape(-1, 16))
    if target is not None:
        if target_matrix is None or target.shape != values.shape:
            raise ValueError("A same-grid target pressure and its own archived basis are required")
        actual -= np.einsum(
            "qi,ti->tq", reference_values(target_matrix, reference), target.reshape(-1, 16)
        )
    return float(np.sqrt(np.sum(weights * abs(actual) ** 2, dtype=np.longdouble)))


def acquire(output: Path) -> dict[str, Any]:
    """Keep primary SI data and source coordinates; declare the160x80m crop and H80/40/20."""
    material = load_marmousi_crop(
        ROOT / "build/datasets/marmousi", origin=(8315.0, 515.0), shape=(32, 16)
    )
    rows, held = [], []
    for n in (2, 4, 8):
        started = perf_counter()
        mesh = CartesianMacroMesh(n, n // 2, (0, 160, 0, 80))
        refinement = 64 // n
        skeleton = helmholtz_skeleton(mesh, 40 * np.pi, degree=1)
        absorbing = {
            int(face): 0j
            for face in mesh.boundary_faces
            if not np.all(mesh.points[mesh.faces[face], 1] == 0)
        }
        solution = solve_acoustic(
            mesh,
            omega=40 * np.pi,
            density=material.density,
            bulk_modulus=material.bulk_modulus,
            point_sources=((80, 50, 1.0),),
            absorbing=absorbing,
            skeleton=skeleton,
            degree=3,
            local_refinement=refinement,
            quadrature_order=8,
            backend="serial",
            workers=1,
        )
        checks = verify_helmholtz_solution(solution)
        require_original(checks)
        values: np.ndarray = np.empty((32, 64, 16), dtype=complex)
        occupied: np.ndarray = np.zeros((32, 64), dtype=bool)
        for fine, pressure in zip(solution.local_meshes, solution.pressure, strict=True):
            dofs, _ = qk_space(fine, 3)
            indices = np.rint(fine.points[fine.cells[:, 0]] / 2.5).astype(int)
            x, y = indices.T
            if np.any(occupied[y, x]):
                raise ValueError("Marmousi fine-cell physical ownership is duplicated")
            occupied[y, x] = True
            values[y, x] = pressure[dofs]
        if not occupied.all():
            raise ValueError("Marmousi fine-cell coverage is incomplete")
        matrix = np.asarray(cardinal_polynomials(3))
        low, high = physical_norm(values, matrix, order=4), physical_norm(values, matrix, order=5)
        sensitivity = quadrature_change({"pressure_l2": low}, {"pressure_l2": high})
        if sensitivity > 1e-10:
            raise ArithmeticError("Marmousi pressure norm quadrature is unresolved")
        field = output / f"macro-n{n}-fields.npz"
        np.savez_compressed(
            field,
            pressure=np.asarray(solution.pressure),
            fine_pressure=values,
            trace=solution.trace,
            actual_cardinal_ascending_power_matrix=matrix,
            macro_points=mesh.points,
            macro_cells=mesh.cells,
            macro_faces=mesh.faces,
            local_degree=np.asarray(3),
            local_refinement=np.asarray(refinement),
            local_points=np.asarray([fine.points for fine in solution.local_meshes]),
            local_cells=np.asarray([fine.cells for fine in solution.local_meshes]),
            **basis_payload(skeleton),
        )
        rows.append(
            {
                "level": n,
                "H_m": 160 / n,
                "macro_shape": [n, n // 2],
                "norms": {"pressure_l2": high},
                "original_equations": checks,
                "quadrature_orders": [4, 5],
                "quadrature_relative_change": sensitivity,
                "archive": field.name,
                "archive_sha256": digest(field),
                "elapsed_seconds": perf_counter() - started,
            }
        )
        held.append((values, matrix))
        write(output / "progress.json", {"rows": rows})
        print(f"Marmousi crop H={160 / n:g}: {rows[-1]['elapsed_seconds']:.3f}s", flush=True)
    for row, (values, matrix) in zip(rows[:-1], held[:-1], strict=True):
        low = physical_norm(values, matrix, *held[-1], order=4)
        high = physical_norm(values, matrix, *held[-1], order=5)
        if quadrature_change({"pressure_l2": low}, {"pressure_l2": high}) > 1e-10:
            raise ArithmeticError("Marmousi pressure increment quadrature is unresolved")
        row["norms"]["pressure_increment_l2"] = high
    return {
        "reference": "Chaumont-Frelet and Valentin (2020), doi:10.1137/19M1255616",
        "scope": (
            "Explicit smaller primary-data crop pilot; "
            "initial macro refinement on a fixed fine mesh"
        ),
        "material": material.provenance,
        "configuration": {
            "crop_bounds_m": [0, 160, 0, 80],
            "crop_primary_origin_m": [8315, 515],
            "crop_material_pixels": [32, 16],
            "material_pixel_m": 5,
            "fine_cell_m": 2.5,
            "original_selected_crop_extent_m": [10240, 2560],
            "crop_offset_within_selected_domain_m": [4920, 0],
            "source_crop_coordinates_m": [80, 50],
            "source_original_selected_coordinates_m": [5000, 50],
            "point_source_strength": 1,
            "omega_rad_s": 40 * np.pi,
            "local_space": "Q3",
            "trace_space": "P1",
            "assembly_order": 8,
            "boundary": "Top weak Dirichlet zero; three remaining sides outgoing absorption zero",
        },
        "basis_contract": (
            "Actual cached Q3 cardinal matrices, executed face basis, "
            "geometry and original coefficients retained"
        ),
        "rows": rows,
        "plot_fields": ["pressure_l2", "pressure_increment_l2"],
        "level_label": "Macro resolution in x",
        "norm_label": "Pressure L2 norm / increment",
        "figure_title": "Marmousi160x80m crop: initial macro refinement",
        "exact_solution_available": False,
        "finest_numerical_reference_macro_shape": [8, 4],
        "reference_refinement_verified": False,
        "asymptotic_convergence_verified": False,
        "limitations": [
            "The crop changes lateral/bottom boundary locations; "
            "the full selected paper domain remains outside this pilot",
            "Fine spacing is fixed at2.5m; the study measures macro trace restriction "
            "and does not establish continuum spatial convergence",
            "The finest MHM field is a numerical comparison level; "
            "it is not an exact or resolved reference",
            "Primary SEG data are identified; the historical article arrays remain unidentified",
        ],
    }
