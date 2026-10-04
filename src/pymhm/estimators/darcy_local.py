"""Nested local energy differences at a fixed physical MHM skeleton trace."""

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from pymhm._legacy.models.darcy.primal import DarcySolution
from pymhm.core.contracts import LocalProblem
from pymhm.core.validation import FloatArray, positive_int
from pymhm.execution.cpu import map_local
from pymhm.fem.quadrature.material import material_triangle_quadrature
from pymhm.fem.scalar.operators import p1_geometry
from pymhm.fem.scalar.triangle import (
    element_tabulate,
    nodal_space,
    reference_basis,
    scalar_operators,
    trace_coupling,
)
from pymhm.fem.traces.interval import SkeletonSpace
from pymhm.materials.evaluation import tensor_values
from pymhm.meshes.refinement import refine_triangles
from pymhm.meshes.triangle import TriangleMesh


@dataclass(frozen=True)
class LocalDarcyRefinementEstimate:
    """Energy differences between nested local solves with the same macro flux.

    This is a computed refinement difference, not an error bound without a
    separately justified saturation estimate. Local means do not enter its norm.
    """

    local_squared: FloatArray
    local_meshes: tuple[TriangleMesh, ...]
    pressure: tuple[FloatArray, ...]

    @property
    def total(self) -> float:
        """Return the square root of the sum of all local energy differences."""
        return float(np.sqrt(np.sum(self.local_squared)))


def estimate_darcy_local_refinement(
    solution: DarcySolution,
    *,
    quadrature_order: int | None = None,
    local_solver: str = "scipy",
    refinement_precision: Literal["double", "extended"] = "double",
    backend: Literal["serial", "thread", "process"] = "serial",
    workers: int | None = None,
) -> LocalDarcyRefinementEstimate:
    """Red-refine every local triangle and solve with unchanged physical face data.

    The finite-element degree, source, permeability and skeletal coefficients
    remain fixed. The enriched local pressure has zero mean; the energy norm
    only compares gradients. Return the difference in the genuine broken
    ``K`` energy norm using material-cut quadrature and exact parent-cell
    ancestry. Point sources and mixed pressure spaces are outside this contract.
    No reconstructed-divergence term is used as a surrogate local-error norm.
    ``backend`` distributes independent local solves in cell order. Process
    workers use spawn and require picklable source/material callbacks; all
    native factors are constructed and released inside their owning worker.
    """
    if solution.formulation != "primal" or any(len(part) for part in solution.point_sources):
        raise ValueError("local energy refinement requires primal Darcy with an L2 source")
    order = (
        max(solution.degree + 2, solution.quadrature_order)
        if quadrature_order is None
        else positive_int(quadrature_order, "quadrature_order", solution.degree + 2)
    )
    factory = _LocalRefinementFactory(
        solution.skeleton,
        solution.local_meshes,
        solution.pressure,
        solution.hybrid.trace,
        solution.permeability,
        solution.source,
        solution.degree,
        order,
        local_solver,
        refinement_precision,
    )
    results = map_local(
        factory, range(len(solution.local_meshes)), backend=backend, workers=workers
    )
    return LocalDarcyRefinementEstimate(
        np.asarray([part[0] for part in results]),
        tuple(part[1] for part in results),
        tuple(part[2] for part in results),
    )


@dataclass(frozen=True)
class _LocalRefinementFactory:
    """Transfer only physical data to workers and create each local factor there."""

    skeleton: SkeletonSpace
    local_meshes: tuple[TriangleMesh, ...]
    pressure: tuple[FloatArray, ...]
    trace: FloatArray
    permeability: Any
    source: Any
    degree: int
    order: int
    local_solver: str
    refinement_precision: Literal["double", "extended"]

    def __call__(self, cell: int) -> tuple[float, TriangleMesh, FloatArray]:
        """Return one nested energy integral, refined mesh and local pressure."""
        original, coefficients = self.local_meshes[cell], self.pressure[cell]
        refinement = refine_triangles(original, np.ones(len(original.cells), dtype=bool))
        fine = refinement.mesh
        matrix, mass, force = scalar_operators(
            fine,
            self.degree,
            diffusion=self.permeability,
            source=self.source,
            order=self.order,
        )
        coupling = trace_coupling(self.skeleton.mesh, cell, fine, self.skeleton, self.degree)
        kernel = np.ones((len(force), 1))
        problem = LocalProblem(
            matrix, coupling, force, self.skeleton.cell_dofs(cell), kernel, mass @ kernel
        )
        response = problem.condense(
            solver=self.local_solver, refinement_precision=self.refinement_precision
        )
        pressure = response.reconstruct(self.trace[self.skeleton.cell_dofs(cell)], np.zeros(1))
        bary, weights, material = material_triangle_quadrature(fine, self.permeability, self.order)
        dofs, _, _, gradient, _ = element_tabulate(fine, self.degree, bary)
        enriched_gradient = np.einsum("ti,tqia->tqa", pressure[dofs], gradient)
        points = np.einsum("tqi,tia->tqa", bary, fine.points[fine.cells])
        vertices = original.points[original.cells[refinement.parent_cells]]
        inverse = np.linalg.inv((vertices[:, 1:] - vertices[:, :1]).swapaxes(1, 2))
        coordinates = np.einsum("tab,tqb->tqa", inverse, points - vertices[:, :1])
        old_bary = np.concatenate((1 - coordinates.sum(axis=2, keepdims=True), coordinates), axis=2)
        derivative = reference_basis(self.degree, old_bary.reshape(-1, 3))[1]
        derivative = derivative.reshape(*old_bary.shape[:2], -1, 3)
        old_dofs, _ = nodal_space(original, self.degree)
        geometry = p1_geometry(original)[0][refinement.parent_cells]
        old_gradient = np.einsum(
            "tqib,tba,ti->tqa",
            derivative,
            geometry,
            coefficients[old_dofs[refinement.parent_cells]],
        )
        difference = enriched_gradient - old_gradient
        tensors = tensor_values(material, points.reshape(-1, 2)).reshape(*points.shape[:2], 2, 2)
        squared = np.einsum(
            "t,tq,tqa,tqab,tqb->", fine.areas, weights, difference, tensors, difference
        )
        return squared, fine, pressure
