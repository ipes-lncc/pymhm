"""Persist complete point-well fields with their executed evaluation matrices.

Pressure and flux coefficients retain separate one-sided macro fields. Replay
uses the archived matrices and maps; trace/coarse coordinates alone do not
replay a corrected physical field. The archive describes a specified discrete
Dirac-well case, not a finite-energy exact PDE solution.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from pymhm._legacy.models.darcy.primal import DarcySolution, _DarcyLocalFactory
from pymhm.core.validation import positive_int
from pymhm.fem.scalar.operators import rt0_evaluate, triangle_quadrature
from pymhm.fem.scalar.triangle import tabulate
from pymhm.linalg.linear import LinearSolveError, _accurate_residual
from pymhm.materials.evaluation import tensor_values


def matrix_digest(array: np.ndarray) -> str:
    """Digest an executed matrix including its shape, dtype and contiguous bytes."""
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode())
    digest.update(array.dtype.str.encode())
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def point_field_arrays(solution: DarcySolution, order: int = 4) -> tuple[dict, dict]:
    """Archive P2 or RT0/P0 fields on the original two-subdivision local meshes.

    The positive Duffy rule integrates squared P2 pressure and RT0 or raw P2
    flux exactly for the fitted piecewise-constant point-case materials.
    Original volume rows and free pressure-continuity moments use the original
    point functional, zero exterior flux and physical zero-mean pressure.
    The full residual is normalized by the original volumetric point load and
    must satisfy 1e-10 independently of the reduced system's residual.
    """
    if positive_int(order, "order") < 3:
        raise ValueError("point-field norm quadrature requires order at least three")
    if (solution.formulation, solution.degree) not in (("primal", 2), ("mixed", 1)):
        raise ValueError("point archives require primal P2 or mixed RT0/P0")
    macro = solution.skeleton.mesh
    meshes = solution.local_meshes
    if len(meshes) != len(macro.cells) or any(len(mesh.cells) != 4 for mesh in meshes):
        raise ValueError("point archives require two local subdivisions per macroedge")
    if any(len(space.degrees) != 1 or space.degrees[0] for space in solution.skeleton.faces):
        raise ValueError("point archives require one constant trace per macroface")
    if len(solution.point_sources) != len(meshes):
        raise ValueError("point archives require the original point-source allocations")
    for mesh, pressure, flux, field in zip(
        meshes, solution.pressure, solution.flux, solution.hybrid.fields, strict=True
    ):
        expected_pressure = (
            field
            if solution.formulation == "primal"
            else field[len(mesh.faces) : len(mesh.faces) + len(mesh.cells)]
        )
        if not np.array_equal(pressure, expected_pressure):
            raise ValueError("physical pressure coefficients differ from the checked hybrid field")
        if solution.formulation == "mixed" and not np.array_equal(flux, field[: len(mesh.faces)]):
            raise ValueError("physical flux coefficients differ from the checked hybrid field")
    bary, weights = triangle_quadrature(order)
    point_offsets = np.r_[0, np.cumsum([len(mesh.points) for mesh in meshes])]
    pressure_offsets = np.r_[0, np.cumsum([len(p) for p in solution.pressure])]
    flux_offsets = np.r_[0, np.cumsum([len(q) for q in solution.flux])]
    arrays: dict[str, Any] = dict(
        macro_points=macro.points,
        macro_cells=macro.cells,
        macro_faces=macro.faces,
        macro_cell_faces=macro.cell_faces,
        macro_normals=macro.normals,
        macro_signs=macro.signs,
        points=np.concatenate([mesh.points for mesh in meshes]),
        cells=np.concatenate([mesh.cells + point_offsets[i] for i, mesh in enumerate(meshes)]),
        macro_cell=np.repeat(np.arange(len(meshes)), 4),
        point_offsets=point_offsets,
        pressure_offsets=pressure_offsets,
        pressure_coefficients=np.concatenate(solution.pressure),
        flux_offsets=flux_offsets,
        hybrid_fields=np.concatenate(solution.hybrid.fields),
        hybrid_field_offsets=np.r_[0, np.cumsum([len(field) for field in solution.hybrid.fields])],
        quadrature_barycentric=bary,
        quadrature_weights=weights,
        trace=solution.hybrid.trace,
        point_sources=np.concatenate(solution.point_sources),
        point_source_offsets=np.r_[0, np.cumsum([len(part) for part in solution.point_sources])],
    )
    sampled_pressure, sampled_flux, coordinates, pressure_dofs, gradients = [], [], [], [], []
    flux_basis, flux_dofs, tensors = [], [], []
    for i, (mesh, pressure, flux) in enumerate(
        zip(meshes, solution.pressure, solution.flux, strict=True)
    ):
        x = np.einsum("qi,tij->tqj", bary, mesh.points[mesh.cells])
        if solution.formulation == "primal":
            dofs, nodes, values, derivative, _ = tabulate(mesh, 2, bary)
            coordinates.append(nodes)
            pressure_dofs.append(dofs + pressure_offsets[i])
            gradients.append(derivative)
            coefficient = tensor_values(solution.permeability, x.reshape(-1, 2)).reshape(
                *x.shape[:2], 2, 2
            )
            tensors.append(coefficient)
            sampled_pressure.append(pressure[dofs] @ values.T)
            gradient = np.einsum("ti,tqia->tqa", pressure[dofs], derivative)
            sampled_flux.append(-np.einsum("tqab,tqb->tqa", coefficient, gradient))
        else:
            units = np.eye(len(flux))
            # Each column executes the same physical evaluator used for this
            # coefficient vector. Its face signs are included in the matrix.
            basis = np.stack([rt0_evaluate(mesh, unit, bary) for unit in units], axis=2)
            flux_basis.append(basis)
            flux_dofs.append(
                np.broadcast_to(np.arange(len(flux)) + flux_offsets[i], (4, len(flux)))
            )
            sampled_pressure.append(np.broadcast_to(pressure[:, None], (4, len(bary))))
            sampled_flux.append(rt0_evaluate(mesh, flux, bary))
    if solution.formulation == "primal":
        arrays.update(
            centroid_flux_samples=np.concatenate(solution.flux),
            pressure_nodes=np.concatenate(coordinates),
            pressure_cell_dofs=np.concatenate(pressure_dofs),
            pressure_basis_values=values,
            pressure_basis_gradients=np.concatenate(gradients),
            permeability_tensors=np.concatenate(tensors),
        )
    else:
        arrays.update(
            flux_coefficients=np.concatenate(solution.flux),
            flux_basis_values=np.concatenate(flux_basis),
            flux_cell_dofs=np.concatenate(flux_dofs),
            pressure_cell_dofs=np.arange(len(arrays["cells"])),
        )
    arrays["pressure_quadrature"] = np.concatenate(sampled_pressure)
    arrays["flux_quadrature"] = np.concatenate(sampled_flux)
    factory = _DarcyLocalFactory(
        macro,
        solution.skeleton,
        solution.permeability,
        solution.source,
        solution.point_sources,
        2,
        tuple(meshes),
        solution.degree,
        solution.formulation,
        solution.quadrature_order,
    )
    weak = np.zeros(solution.skeleton.size, dtype=np.longdouble)
    norm_squared, load_squared, mean = np.longdouble(0), np.longdouble(0), np.longdouble(0)
    kernels, constraints = [], []
    for cell, field in enumerate(solution.hybrid.fields):
        assembly = factory(cell)
        problem = assembly.problem
        operator = sparse.hstack((problem.matrix, sparse.csr_matrix(problem.coupling))).tocsr()
        defect = _accurate_residual(
            operator, problem.load, np.r_[field, solution.hybrid.trace[problem.trace_dofs]]
        )
        norm_squared += np.sum(defect**2)
        load_squared += np.sum(problem.load.astype(np.longdouble) ** 2)
        np.add.at(weak, problem.trace_dofs, problem.test_coupling.astype(np.longdouble).T @ field)
        mean += assembly.metadata[1].astype(np.longdouble) @ field
        kernels.append(problem.kernel)
        constraints.append(problem.constraints)
    fixed = np.concatenate([solution.skeleton.dofs(int(face)) for face in macro.boundary_faces])
    free = np.setdiff1d(np.arange(solution.skeleton.size), fixed)
    if np.any(solution.hybrid.trace[fixed] != 0):
        raise ValueError("point archive requires zero exterior physical flux")
    residual = float(np.sqrt(norm_squared + np.sum(weak[free] ** 2)) / np.sqrt(load_squared))
    if not np.isfinite(residual) or residual > 1e-10:
        raise LinearSolveError(f"original point-well saddle residual {residual:.6e} exceeds 1e-10")
    if abs(mean) > 1e-10:
        raise LinearSolveError("point-well physical pressure integral exceeds 1e-10")
    arrays.update(
        declared_kernels=np.concatenate(kernels), declared_constraints=np.concatenate(constraints)
    )
    basis_keys = [
        key
        for key in arrays
        if "basis" in key
        or key
        in (
            "declared_kernels",
            "declared_constraints",
            "pressure_cell_dofs",
            "flux_cell_dofs",
            "pressure_nodes",
            "permeability_tensors",
            "macro_faces",
            "macro_cell_faces",
            "macro_normals",
            "macro_signs",
            "points",
            "cells",
            "macro_points",
            "macro_cells",
            "macro_cell",
            "pressure_offsets",
            "flux_offsets",
            "hybrid_field_offsets",
            "point_sources",
            "point_source_offsets",
            "quadrature_barycentric",
            "quadrature_weights",
        )
    ]
    record = dict(
        formulation=solution.formulation,
        degree=solution.degree,
        local_refinement=2,
        trace_degree=0,
        trace_segments=1,
        norm_order=order,
        macro_cells=len(macro.cells),
        fine_cells=len(arrays["cells"]),
        original_saddle_relative_load_residual=residual,
        physical_pressure_integral=float(mean),
        basis_sha256={key: matrix_digest(arrays[key]) for key in basis_keys},
        kernel_convention=(
            "Declared constant pressure and normalized physical mean moments "
            "from the original Darcy assembler"
        ),
        coefficient_convention=(
            "Complete nodal P2 pressure or physical RT0 face-integral/P0 pressure "
            "coefficients and full original hybrid fields including mixed boundary "
            "pressure multipliers; independent fields per macrocell"
        ),
        replay_convention=(
            "Archived executed evaluation matrices and coefficient maps; "
            "coarse coordinates alone do not replay complete fields"
        ),
    )
    return arrays, record


def replay_point_fields(
    arrays: dict[str, np.ndarray], record: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Replay physical quadrature fields using only digested archived bases/maps."""
    for key, digest in record["basis_sha256"].items():
        if matrix_digest(arrays[key]) != digest:
            raise ValueError(f"archived point-field basis digest differs: {key}")
    pressure = arrays["pressure_coefficients"][arrays["pressure_cell_dofs"]]
    if record["formulation"] == "primal":
        gradient = np.einsum("ti,tqia->tqa", pressure, arrays["pressure_basis_gradients"])
        return pressure @ arrays["pressure_basis_values"].T, -np.einsum(
            "tqab,tqb->tqa", arrays["permeability_tensors"], gradient
        )
    flux = np.einsum(
        "tqia,ti->tqa",
        arrays["flux_basis_values"],
        arrays["flux_coefficients"][arrays["flux_cell_dofs"]],
    )
    return np.broadcast_to(pressure[:, None], arrays["pressure_quadrature"].shape), flux


def write_point_archive(path: Path, arrays: dict[str, np.ndarray]) -> str:
    """Flush the complete field/basis NPZ before atomic replacement and return SHA."""
    import os

    temporary = path.with_suffix(".npz.pending")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, allow_pickle=False, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_point_record(path: Path, record: dict) -> None:
    """Flush finite scientific metadata before atomically replacing its JSON record."""
    import os

    payload = json.dumps(record, indent=2, allow_nan=False) + "\n"
    temporary = path.with_suffix(".json.pending")
    with temporary.open("w") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
