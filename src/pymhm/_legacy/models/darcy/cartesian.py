"""Cartesian quadrilateral MHM with continuous tensor-product Qk local pressures.

The skeleton uses the same signed physical normal flux as triangular Darcy.
Only axis-aligned Cartesian rectangles are supported; curved and bilinear
non-affine quadrilaterals require a different geometric transformation.
"""

from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np

from pymhm.core.contracts import HybridSolution, LocalAssembly, LocalProblem
from pymhm.core.system import HybridSystem
from pymhm.core.validation import FloatArray, positive_int
from pymhm.fem.scalar.operators import boundary_data
from pymhm.fem.scalar.quadrilateral import (
    _cardinal_values as _cardinal_values,
)
from pymhm.fem.scalar.quadrilateral import (
    _cardinals as _cardinals,
)
from pymhm.fem.scalar.quadrilateral import (
    _cartesian_rectangle_quadrature as _cartesian_rectangle_quadrature,
)
from pymhm.fem.scalar.quadrilateral import (
    _grid_resolves_material as _grid_resolves_material,
)
from pymhm.fem.scalar.quadrilateral import (
    qk_basis as qk_basis,
)
from pymhm.fem.scalar.quadrilateral import (
    qk_space as qk_space,
)
from pymhm.fem.scalar.quadrilateral import (
    quadrilateral_operators as quadrilateral_operators,
)
from pymhm.fem.scalar.quadrilateral import (
    quadrilateral_quadrature as quadrilateral_quadrature,
)
from pymhm.fem.scalar.quadrilateral import (
    quadrilateral_trace_coupling as quadrilateral_trace_coupling,
)
from pymhm.fem.traces.interval import SkeletonSpace
from pymhm.materials.evaluation import scalar_values, tensor_values, vector_values
from pymhm.meshes.cartesian import (
    CartesianMacroMesh as CartesianMacroMesh,
)
from pymhm.meshes.cartesian import (
    _refinement as _refinement,
)
from pymhm.meshes.triangle import TriangleMesh


@dataclass(frozen=True)
class _QuadTask:
    """Picklable geometry and coefficients for one complete worker-local solve."""

    mesh: CartesianMacroMesh
    cell: int
    refinement: tuple[int, int]
    skeleton: SkeletonSpace
    degree: int
    permeability: Any
    source: Any
    order: int


def _assemble_quad(task: _QuadTask) -> LocalAssembly:
    """Build one original Qk Neumann problem and return reconstruction metadata."""
    fine = task.mesh.submesh(task.cell, task.refinement)
    matrix, mass, load = quadrilateral_operators(
        fine, task.degree, permeability=task.permeability, source=task.source, order=task.order
    )
    coupling = quadrilateral_trace_coupling(task.mesh, task.cell, fine, task.skeleton, task.degree)
    kernel = np.ones((matrix.shape[0], 1))
    constraints = mass @ kernel
    return LocalAssembly(
        LocalProblem(
            matrix, coupling, load, task.skeleton.cell_dofs(task.cell), kernel, constraints
        ),
        (fine, constraints[:, 0]),
    )


@dataclass(frozen=True)
class QuadrilateralDarcySolution:
    """Broken continuous-Qk pressure and conservative macro normal-flux trace.

    ``flux`` stores samples of the raw physical field −K grad p at fine-cell
    centers. It is generally not H(div)-conforming. ``evaluate`` preserves
    distinct values on neighboring macrocells, without averaging.
    """

    skeleton: SkeletonSpace
    local_meshes: tuple[CartesianMacroMesh, ...]
    pressure: tuple[FloatArray, ...]
    flux: tuple[FloatArray, ...]
    hybrid: HybridSolution
    permeability: Any
    source: Any
    degree: int
    quadrature_order: int

    def evaluate(self, cell: int, points: Any) -> tuple[FloatArray, FloatArray]:
        """Sample pressure and physical flux inside one specified macrocell.

        At fine-cell interfaces the cell on the positive coordinate side is
        selected, except on the outer boundary. No interpolation across a
        permeability jump or a macroface is performed.
        """
        positive_int(cell, "cell", 0)
        if cell >= len(self.local_meshes):
            raise ValueError("cell index outside mesh")
        if np.iscomplexobj(points):
            raise ValueError("sample points must be real")
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise ValueError("sample points must be finite with shape (n,2)")
        mesh = self.local_meshes[cell]
        x0, x1, y0, y1 = mesh.bounds
        coordinates = (points - [x0, y0]) / mesh.spacing
        counts = np.array([mesh.nx, mesh.ny], dtype=int)
        if np.any(coordinates < -1e-10) or np.any(coordinates > counts + 1e-10):
            raise ValueError("sample points lie outside the selected macrocell")
        nearest = np.rint(coordinates)
        coordinates = np.where(
            np.isclose(coordinates, nearest, rtol=0, atol=32 * np.finfo(float).eps * counts),
            nearest,
            coordinates,
        )
        indices = np.clip(np.floor(coordinates).astype(int), 0, counts - 1)
        reference = np.clip(coordinates - indices, 0, 1)
        width = mesh.nx * self.degree + 1
        offsets = np.array(
            [j * width + i for j in range(self.degree + 1) for i in range(self.degree + 1)]
        )
        origins = (indices[:, 1] * width + indices[:, 0]) * self.degree
        basis, gradients = qk_basis(self.degree, reference)
        coefficients = self.pressure[cell][origins[:, None] + offsets]
        values = np.einsum("qi,qi->q", coefficients, basis)
        grad = np.einsum("qi,qia->qa", coefficients, gradients / mesh.spacing)
        # A discontinuous material must use the same one-sided fine cell as
        # the polynomial gradient, including at a macrocell upper boundary.
        centers = np.array([x0, y0]) + (indices + 0.5) * mesh.spacing
        on_interface = np.isclose(reference, 0, rtol=0, atol=32 * np.finfo(float).eps) | (
            np.isclose(reference, 1, rtol=0, atol=32 * np.finfo(float).eps)
        )
        material_points = np.where(on_interface, np.nextafter(points, centers), points)
        flux = -np.einsum("qab,qb->qa", tensor_values(self.permeability, material_points), grad)
        return values, flux

    def _error(self, exact: Any, order: int, flux: bool) -> float:
        """Integrate one pressure or vector-flux error without cosmetic averaging."""
        reference, weights = quadrilateral_quadrature(order)
        total = 0.0
        evaluator = vector_values if flux else scalar_values
        for cell, mesh in enumerate(self.local_meshes):
            for begin in range(0, len(mesh.cells), 256):
                origins = mesh.points[mesh.cells[begin : begin + 256, 0]]
                points = origins[:, None, :] + reference[None, :, :] * mesh.spacing
                flat = points.reshape(-1, 2)
                values = self.evaluate(cell, flat)[int(flux)]
                difference = values - evaluator(exact, flat)
                squares = np.sum(difference**2, axis=1) if flux else difference**2
                total += float(
                    np.prod(mesh.spacing) * np.sum(squares.reshape(-1, len(weights)) @ weights)
                )
        return float(np.sqrt(total))

    def l2_error(self, exact: Any, order: int = 6) -> float:
        """Integrate the pressure L2 error with independently chosen quadrature."""
        return self._error(exact, order, False)

    def flux_l2_error(self, exact: Any, order: int = 6) -> float:
        """Integrate the raw physical-flux L2 error with independent quadrature."""
        return self._error(exact, order, True)

    def conservation_residuals(self, order: int | None = None) -> FloatArray:
        """Return integrated macro trace outflow minus volume source per cell."""
        reference, weights = quadrilateral_quadrature(
            self.quadrature_order if order is None else order
        )
        residuals = []
        for cell, fine in enumerate(self.local_meshes):
            flux = 0.0
            for side, face in enumerate(self.skeleton.mesh.cell_faces[cell]):
                space = self.skeleton.faces[face]
                t, w = space.quadrature(max(space.degrees) + 2)
                flux += (
                    self.skeleton.mesh.signs[cell, side]
                    * self.skeleton.mesh.lengths[face]
                    * ((w @ space.evaluate(t)) @ self.hybrid.trace[self.skeleton.dofs(int(face))])
                )
            integral = 0.0
            for begin in range(0, len(fine.cells), 256):
                points = fine.points[fine.cells[begin : begin + 256, 0], None, :] + (
                    reference[None, :, :] * fine.spacing
                )
                values = scalar_values(self.source, points.reshape(-1, 2))
                integral += float(
                    np.prod(fine.spacing) * np.sum(values.reshape(-1, len(weights)) @ weights)
                )
            residuals.append(float(flux) - integral)
        return np.asarray(residuals)


def solve_darcy_quadrilateral(
    mesh: CartesianMacroMesh,
    *,
    permeability: Any = 1.0,
    source: Any = 0.0,
    dirichlet: Any = 0.0,
    neumann: dict[int, Any] | None = None,
    skeleton: SkeletonSpace | None = None,
    local_refinement: int | tuple[int, int] = 4,
    degree: int = 1,
    quadrature_order: int = 4,
    mean_pressure: float = 0.0,
    solver: str = "scipy",
    local_solver: str = "scipy",
    backend: Literal["serial", "thread", "process"] = "serial",
    workers: int | None = None,
) -> QuadrilateralDarcySolution:
    """Solve Darcy MHM on Cartesian macrorectangles with continuous Qk locals.

    Dirichlet pressure is imposed weakly on faces not listed in ``neumann``;
    Neumann values are outward physical normal fluxes. Pure Neumann problems
    require compatibility and impose the physical mean pressure. The local
    refinement may differ between x and y, independently of trace segments.
    Assembly and condensation both execute inside the selected CPU workers.
    Process execution requires picklable coefficient callbacks and the usual
    guarded script entry point. Sparse local operators reuse one factorization
    for every source and trace right-hand side.
    """
    if not isinstance(mesh, CartesianMacroMesh):
        raise TypeError("quadrilateral Darcy requires CartesianMacroMesh")
    degree = positive_int(degree, "degree")
    order = max(positive_int(quadrature_order, "quadrature_order"), degree + 1)
    refinement = _refinement(local_refinement)
    skeleton = SkeletonSpace(cast(TriangleMesh, mesh)) if skeleton is None else skeleton
    if skeleton.mesh is not mesh or skeleton.components != 1:
        raise ValueError("quadrilateral Darcy requires a scalar skeleton on the supplied mesh")
    boundary, fixed = boundary_data(skeleton, dirichlet, neumann, order=order)
    tasks = [
        _QuadTask(mesh, cell, refinement, skeleton, degree, permeability, source, order)
        for cell in range(len(mesh.cells))
    ]
    system = HybridSystem.from_local_factory(
        _assemble_quad,
        tasks,
        boundary_load=boundary,
        local_solver=local_solver,
        backend=backend,
        workers=workers,
    )
    local_meshes, means = zip(*system.local_metadata, strict=True)
    pure_neumann = neumann is not None and set(neumann) == set(mesh.boundary_faces)
    constraints = (
        [system.mean_constraint(means, mean_pressure * float(np.sum(mesh.areas)))]
        if pure_neumann
        else None
    )
    hybrid = system.solve(solver=solver, fixed=fixed, constraints=constraints)
    solution = QuadrilateralDarcySolution(
        skeleton,
        tuple(local_meshes),
        hybrid.fields,
        (),
        hybrid,
        permeability,
        source,
        degree,
        order,
    )
    flux = tuple(
        solution.evaluate(cell, fine.points[fine.cells].mean(axis=1))[1]
        for cell, fine in enumerate(local_meshes)
    )
    return QuadrilateralDarcySolution(
        skeleton,
        tuple(local_meshes),
        hybrid.fields,
        flux,
        hybrid,
        permeability,
        source,
        degree,
        order,
    )
