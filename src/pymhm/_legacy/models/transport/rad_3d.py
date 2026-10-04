"""Conservative reaction-advection-diffusion on tetrahedra with polynomial face traces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy import sparse

from pymhm.core.contracts import HybridSolution, LocalAssembly, LocalProblem
from pymhm.core.system import HybridSystem
from pymhm.core.validation import FloatArray, positive_int
from pymhm.fem.scalar.operators import triangle_quadrature
from pymhm.fem.scalar.tetrahedron import (
    tetra_element_tabulate,
    tetra_nodal_space,
    tetra_tabulate,
    tetrahedron_quadrature,
)
from pymhm.fem.traces.triangle_3d import TriangularSkeleton, _boundary, tetra_trace_coupling
from pymhm.linalg.linear import solve_linear
from pymhm.materials.evaluation import scalar_values_3d, tensor_values_3d
from pymhm.materials.evaluation import (
    vector_values_3d as vector_values_3d,
)
from pymhm.meshes.tetrahedron import TetraMesh, _dyadic


def _coefficient_derivatives(
    points: FloatArray,
    diffusion: Any,
    velocity: Any,
    velocity_divergence: Any,
    diffusion_divergence: Any,
    stabilization: str,
) -> tuple[Any, Any]:
    """Require declared variable derivatives and reject contradictory constant data."""
    if callable(velocity) and velocity_divergence is None:
        raise ValueError("variable velocity requires velocity_divergence")
    if stabilization == "supg" and callable(diffusion) and diffusion_divergence is None:
        raise ValueError("SUPG with variable diffusion requires diffusion_divergence")
    div_beta = 0.0 if velocity_divergence is None else velocity_divergence
    div_tensor = (0.0, 0.0, 0.0) if diffusion_divergence is None else diffusion_divergence
    if not callable(velocity) and np.any(scalar_values_3d(div_beta, points)):
        raise ValueError("constant velocity must have zero divergence")
    if not callable(diffusion) and np.any(vector_values_3d(div_tensor, points)):
        raise ValueError("constant diffusion must have zero divergence")
    return div_beta, div_tensor


def tetra_rad_operators(
    mesh: TetraMesh,
    degree: int,
    *,
    diffusion: Any = 1.0,
    velocity: Any = (0.0, 0.0, 0.0),
    reaction: Any = 0.0,
    source: Any = 0.0,
    velocity_divergence: Any = None,
    diffusion_divergence: Any = None,
    stabilization: str = "galerkin",
    order: int = 6,
) -> tuple[Any, FloatArray, FloatArray, bool]:
    """Assemble skew conservative RAD and optional full-residual SUPG.

    The associated boundary flux is ``(-K grad(u)+beta*u/2).n``. The source
    equation is ``-div(K grad(u))+div(beta*u)+reaction*u=f``. Coercivity requires
    ``reaction+div(beta)/2 >= 0``; the implementation checks integration points.
    Returned physical moments integrate the nodal functions over the domain.
    """
    if stabilization not in ("galerkin", "supg"):
        raise ValueError("stabilization must be galerkin or supg")
    velocity_divergence, diffusion_divergence = _coefficient_derivatives(
        mesh.points,
        diffusion,
        velocity,
        velocity_divergence,
        diffusion_divergence,
        stabilization,
    )
    bary, weights = tetrahedron_quadrature(max(positive_int(order, "order"), degree + 2))
    dofs, nodes, basis, gradient, hessian = tetra_element_tabulate(mesh, degree, bary)
    points = np.einsum("qi,tij->tqj", bary, mesh.points[mesh.cells])
    shape = points.shape[:2]
    flat = points.reshape(-1, 3)
    tensor = tensor_values_3d(diffusion, flat).reshape(*shape, 3, 3)
    beta = vector_values_3d(velocity, flat).reshape(*shape, 3)
    div_beta = scalar_values_3d(velocity_divergence, flat).reshape(shape)
    c = scalar_values_3d(reaction, flat).reshape(shape)
    effective = c + div_beta / 2
    if np.any(effective < 0):
        raise ValueError("reaction+div(velocity)/2 must be nonnegative")
    force = scalar_values_3d(source, flat).reshape(shape)
    blocks = np.einsum("t,q,tqia,tqab,tqjb->tij", mesh.volumes, weights, gradient, tensor, gradient)
    blocks += np.einsum("t,q,qi,qj,tq->tij", mesh.volumes, weights, basis, basis, effective)
    streamline = np.einsum("tqa,tqia->tqi", beta, gradient)
    transport = np.einsum("t,q,qi,tqj->tij", mesh.volumes, weights, basis, streamline)
    blocks += (transport - transport.swapaxes(1, 2)) / 2
    load = np.einsum("t,q,qi,tq->ti", mesh.volumes, weights, basis, force)
    if stabilization == "supg":
        div_tensor = vector_values_3d(diffusion_divergence, flat).reshape(*shape, 3)
        strong = (
            -np.einsum("tqab,tqiab->tqi", tensor, hessian)
            - np.einsum("tqa,tqia->tqi", div_tensor, gradient)
            + streamline
            + (c + div_beta)[:, :, None] * basis[None]
        )
        corners = mesh.points[mesh.cells]
        diameter = np.max(
            np.linalg.norm(corners[:, :, None] - corners[:, None, :], axis=-1), axis=(1, 2)
        )[:, None]
        tau = 1 / np.sqrt(
            (2 * np.linalg.norm(beta, axis=-1) / diameter) ** 2
            + (4 * np.linalg.eigvalsh(tensor)[..., -1] / diameter**2) ** 2
            + (c + div_beta) ** 2
        )
        blocks += np.einsum("t,q,tqi,tqj,tq->tij", mesh.volumes, weights, streamline, strong, tau)
        load += np.einsum("t,q,tqi,tq,tq->ti", mesh.volumes, weights, streamline, force, tau)
    rows = np.broadcast_to(dofs[:, :, None], blocks.shape).ravel()
    columns = np.broadcast_to(dofs[:, None, :], blocks.shape).ravel()
    matrix = sparse.coo_matrix(
        (blocks.ravel(), (rows, columns)), shape=(len(nodes), len(nodes))
    ).tocsc()
    assembled = np.zeros(len(nodes))
    moments = np.zeros(len(nodes))
    np.add.at(assembled, dofs, load)
    np.add.at(moments, dofs, mesh.volumes[:, None] * (weights @ basis))
    return matrix, assembled, moments, not np.any(beta) and not np.any(effective)


@dataclass(frozen=True)
class _RADFactory:
    """Picklable local assembly for independent three-dimensional macro problems."""

    mesh: TetraMesh
    skeleton: TriangularSkeleton
    refinement: int
    degree: int
    coefficients: dict[str, Any]
    coarse_space: Literal["constants", "kernel"] = "constants"

    def __call__(self, cell: int) -> LocalAssembly:
        """Retain the physical constant, including nonzero or arbitrarily small transport."""
        fine = self.mesh.submesh(cell, self.refinement)
        matrix, load, moments, pure = tetra_rad_operators(fine, self.degree, **self.coefficients)
        coupling = tetra_trace_coupling(self.mesh, cell, fine, self.skeleton, self.degree)
        constant = np.ones((len(load), 1))
        retained = {"kernel": constant} if pure else {"coarse_basis": constant}
        constraints = moments[:, None]
        if self.coarse_space == "kernel" and not pure:
            bary, _ = tetrahedron_quadrature(self.coefficients["order"])
            points = np.einsum("qi,tia->tqa", bary, fine.points[fine.cells]).reshape(-1, 3)
            kernel = (
                not np.any(scalar_values_3d(self.coefficients["reaction"], points))
                and not np.any(scalar_values_3d(self.coefficients["velocity_divergence"], points))
                and _boundary_tangent_3d(
                    TriangularSkeleton(fine),
                    self.coefficients["velocity"],
                    self.coefficients["order"],
                )
            )
            retained = {"kernel": constant} if kernel else {}
            if not kernel:
                constraints = np.empty((len(load), 0))
        problem = LocalProblem(
            matrix,
            coupling,
            load,
            self.skeleton.cell_dofs(cell),
            constraints=constraints,
            **retained,
        )
        return LocalAssembly(problem, (fine, moments))


@dataclass(frozen=True)
class RAD3DSolution:
    """Broken tetrahedral scalar field and full conservative physical flux.

    ``evaluate`` returns ``u`` and ``-K grad(u)+beta*u``. This physical flux
    differs from the skeletal Robin multiplier by ``beta*u/2``.
    """

    local_meshes: tuple[TetraMesh, ...]
    values: tuple[FloatArray, ...]
    degree: int
    diffusion: Any
    velocity: Any
    hybrid: HybridSolution | None = None
    skeleton: TriangularSkeleton | None = None
    residual: float = 0.0

    def evaluate(self, cell: int, bary: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Evaluate scalar and physical flux without averaging across macro interfaces."""
        cell = positive_int(cell, "cell", 0)
        if cell >= len(self.local_meshes):
            raise ValueError("cell outside local meshes")
        fine = self.local_meshes[cell]
        dofs, _, basis, gradient = tetra_tabulate(fine, self.degree, bary)
        value = self.values[cell][dofs] @ basis.T
        derivative = np.einsum("ti,tqia->tqa", self.values[cell][dofs], gradient)
        points = np.einsum("qi,tij->tqj", bary, fine.points[fine.cells])
        tensor = tensor_values_3d(self.diffusion, points.reshape(-1, 3)).reshape(
            *points.shape[:2], 3, 3
        )
        beta = vector_values_3d(self.velocity, points.reshape(-1, 3)).reshape(points.shape)
        return value, -np.einsum("tqab,tqb->tqa", tensor, derivative) + beta * value[:, :, None]

    def l2_error(self, exact: Any, order: int = 7) -> float:
        """Integrate the scalar error with a positive tetrahedral rule."""
        bary, weights = tetrahedron_quadrature(order)
        total = 0.0
        for cell, fine in enumerate(self.local_meshes):
            value = self.evaluate(cell, bary)[0]
            points = np.einsum("qi,tij->tqj", bary, fine.points[fine.cells])
            expected = scalar_values_3d(exact, points.reshape(-1, 3)).reshape(value.shape)
            total += float(fine.volumes @ ((value - expected) ** 2 @ weights))
        return float(np.sqrt(total))

    def h1_seminorm_error(self, exact_gradient: Any, order: int = 7) -> float:
        """Integrate the complete broken physical gradient error."""
        bary, weights = tetrahedron_quadrature(order)
        total = 0.0
        for cell, fine in enumerate(self.local_meshes):
            dofs, _, _, gradient = tetra_tabulate(fine, self.degree, bary)
            values = np.einsum("ti,tqia->tqa", self.values[cell][dofs], gradient)
            points = np.einsum("qi,tij->tqj", bary, fine.points[fine.cells])
            expected = vector_values_3d(exact_gradient, points.reshape(-1, 3)).reshape(values.shape)
            total += float(fine.volumes @ (np.sum((values - expected) ** 2, axis=-1) @ weights))
        return float(np.sqrt(total))

    def flux_l2_error(self, exact_flux: Any, order: int = 7) -> float:
        """Integrate the complete conservative physical flux error."""
        bary, weights = tetrahedron_quadrature(order)
        total = 0.0
        for cell, fine in enumerate(self.local_meshes):
            values = self.evaluate(cell, bary)[1]
            points = np.einsum("qi,tij->tqj", bary, fine.points[fine.cells])
            expected = vector_values_3d(exact_flux, points.reshape(-1, 3)).reshape(values.shape)
            total += float(fine.volumes @ (np.sum((values - expected) ** 2, axis=-1) @ weights))
        return float(np.sqrt(total))


def _boundary_tangent_3d(skeleton: TriangularSkeleton, velocity: Any, order: int) -> bool:
    """Test external tangency relative to componentwise floating-point cancellation scales."""
    bary, _ = triangle_quadrature(max(8, order))
    for face in skeleton.mesh.boundary_faces:
        vertices = skeleton.mesh.points[skeleton.mesh.faces[face]]
        # Anchored affine evaluation preserves coordinates constant on a face.
        points = vertices[0] + bary[:, 1:] @ (vertices[1:] - vertices[0])
        beta = vector_values_3d(velocity, points)
        normal = skeleton.mesh.normals[face]
        scale = float(np.max(np.abs(beta) @ np.abs(normal)))
        if np.any(np.abs(beta @ normal) > 64 * np.finfo(float).eps * scale):
            return False
    return True


def solve_rad_3d(
    mesh: TetraMesh,
    *,
    diffusion: Any = 1.0,
    velocity: Any = (0.0, 0.0, 0.0),
    reaction: Any = 0.0,
    source: Any = 0.0,
    dirichlet: Any = 0.0,
    neumann: dict[int, Any] | None = None,
    skeleton: TriangularSkeleton | None = None,
    local_refinement: int = 2,
    degree: int = 4,
    quadrature_order: int = 6,
    velocity_divergence: Any = None,
    diffusion_divergence: Any = None,
    stabilization: str = "galerkin",
    mean_value: float = 0.0,
    solver: str = "scipy",
    local_solver: str = "scipy",
    backend: Literal["serial", "thread", "process"] = "serial",
    workers: int | None = None,
    coarse_space: Literal["constants", "kernel"] = "constants",
    global_refinement_precision: Literal["double", "extended"] = "double",
) -> RAD3DSolution:
    """Solve conservative RAD using continuous P1–P4 local tetrahedral fields.

    The default skeleton has an independent P1 polynomial on each macroface.
    ``neumann`` prescribes outward ``(-K grad(u)+beta*u/2).n``. Other exterior
    faces receive weak Dirichlet moments. Variable velocity requires its
    divergence; SUPG with variable diffusion also requires ``div(K)``. A mean
    is permitted only with all-Robin data, zero reaction, divergence-free
    exterior-tangent velocity and a representable global constant mode.
    ``coarse_space='kernel'`` retains only genuine constant null modes, giving
    the selective mixed/primal formulation; the default retains every constant
    to handle near-singular local operators without a numerical rank threshold.
    """
    refinement = _dyadic(local_refinement, "local_refinement")
    if coarse_space not in ("constants", "kernel"):
        raise ValueError("coarse_space must be constants or kernel")
    tetra_nodal_space(mesh, degree)
    order = max(positive_int(quadrature_order, "quadrature_order"), degree + 2)
    div_beta, div_tensor = _coefficient_derivatives(
        mesh.points,
        diffusion,
        velocity,
        velocity_divergence,
        diffusion_divergence,
        stabilization,
    )
    if not np.isfinite(mean_value):
        raise ValueError("mean_value must be finite")
    skeleton = TriangularSkeleton(mesh, degree=1) if skeleton is None else skeleton
    if skeleton.mesh is not mesh:
        raise ValueError("skeleton must belong to the supplied tetrahedral mesh")
    if np.any(skeleton.subdivisions > refinement):
        raise ValueError("local refinement must resolve every skeleton subdivision")
    natural = {} if neumann is None else dict(neumann)
    boundary, fixed = _boundary(skeleton, dirichlet, natural, order)
    coefficients = dict(
        diffusion=diffusion,
        velocity=velocity,
        reaction=reaction,
        source=source,
        velocity_divergence=div_beta,
        diffusion_divergence=div_tensor,
        stabilization=stabilization,
        order=order,
    )
    factory = _RADFactory(mesh, skeleton, refinement, degree, coefficients, coarse_space)
    system = HybridSystem.from_local_factory(
        factory,
        range(len(mesh.cells)),
        boundary_load=boundary,
        local_solver=local_solver,
        backend=backend,
        workers=workers,
    )
    constraints = []
    bary, _ = tetrahedron_quadrature(order)
    all_points = np.einsum("qi,tij->tqj", bary, mesh.points[mesh.cells]).reshape(-1, 3)
    gauge = (
        set(natural) == set(mesh.boundary_faces)
        and not np.any(scalar_values_3d(reaction, all_points))
        and not np.any(scalar_values_3d(div_beta, all_points))
        and _boundary_tangent_3d(skeleton, velocity, order)
    )
    if gauge:
        constraints.append(
            system.mean_constraint(
                [data[1] for data in system.local_metadata], mean_value * float(mesh.volumes.sum())
            )
        )
    elif mean_value != 0:
        raise ValueError(
            "mean_value requires all-Robin zero-reaction divergence-free tangential data"
        )
    result = system.solve(
        solver=solver,
        fixed=fixed,
        constraints=constraints,
        refinement_precision=global_refinement_precision,
    )
    return RAD3DSolution(
        tuple(data[0] for data in system.local_metadata),
        result.fields,
        degree,
        diffusion,
        velocity,
        result,
        skeleton,
        result.residual,
    )


def solve_rad_3d_conforming(
    mesh: TetraMesh,
    *,
    degree: int = 2,
    dirichlet: Any = 0.0,
    solver: str = "scipy",
    **coefficients: Any,
) -> RAD3DSolution:
    """Solve the classical continuous-Pk Galerkin/SUPG problem with strong exterior Dirichlet.

    This separate global assembly has no macroface restriction or local
    condensation. Boundary elimination lifts all prescribed nodal values into
    the original equations. Its discretization error must be checked by refinement.
    """
    matrix, load, _, _ = tetra_rad_operators(mesh, degree, **coefficients)
    dofs, nodes = tetra_nodal_space(mesh, degree)
    boundary_nodes: set[int] = set()
    for face in mesh.boundary_faces:
        cell = int(mesh.face_cells[face, 0])
        vertices = mesh.points[mesh.cells[cell]]
        reference = (nodes[dofs[cell]] - vertices[0]) @ np.linalg.inv(
            (vertices[1:] - vertices[0]).T
        ).T
        bary = np.column_stack((1 - reference.sum(axis=1), reference))
        opposite = int(np.flatnonzero(mesh.cell_faces[cell] == face)[0])
        boundary_nodes.update(dofs[cell][np.abs(bary[:, opposite]) < 1e-12])
    fixed = np.array(sorted(boundary_nodes), dtype=int)
    free = np.setdiff1d(np.arange(len(nodes)), fixed)
    values = np.zeros(len(nodes))
    values[fixed] = scalar_values_3d(dirichlet, nodes[fixed])
    rhs = (load - matrix @ values)[free]
    if len(free):
        values[free] = solve_linear(matrix[free][:, free], rhs, solver=solver)
    residual = np.linalg.norm((matrix @ values - load)[free]) / max(
        np.linalg.norm(rhs) + np.linalg.norm(abs(matrix[free][:, free]) @ abs(values[free])),
        np.finfo(float).tiny,
    )
    return RAD3DSolution(
        (mesh,),
        (values,),
        degree,
        coefficients.get("diffusion", 1.0),
        coefficients.get("velocity", (0.0, 0.0, 0.0)),
        residual=float(residual),
    )
