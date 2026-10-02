"""Canonical RT moment reconstruction of primal MHM Darcy fluxes.

This is the face/volume construction of Barrenechea et al. (2026), equations
(4.9) and (5.1). It is distinct from an energy-minimizing equilibrated flux:
its divergence agrees with the source against continuous local test functions,
not necessarily against individual discontinuous fine-cell constants.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
from numpy.polynomial.legendre import leggauss, legvander
from scipy import sparse

from pymhm.cut_cells import (
    cartesian_edge_quadrature,
    cartesian_trace_values,
    material_triangle_quadrature,
)
from pymhm.darcy import DarcySolution
from pymhm.darcy_rt import rt_trace_map
from pymhm.elements import (
    p1_geometry,
    scalar_values,
    tensor_values,
    triangle_quadrature,
    vector_values,
)
from pymhm.lagrange import nodal_space, reference_basis, tabulate
from pymhm.mesh import FloatArray, IntArray, SkeletonSpace, TriangleMesh, positive_int
from pymhm.parallel import map_local
from pymhm.planar_material import PlanarMaterial
from pymhm.reservoir import CartesianCellField
from pymhm.rt import rt_degree, rt_evaluate, rt_interior_tests
from pymhm.solvers import solve_linear

RawFlux = Callable[[FloatArray, IntArray], Any]


def _boundary_map(
    skeleton: SkeletonSpace, cell: int, mesh: TriangleMesh, degree: int
) -> FloatArray:
    """Check fine-edge alignment and use the shared arbitrary-order RT normal map."""
    covered = np.zeros(len(mesh.boundary_faces), dtype=int)
    coarse = skeleton.mesh
    for face in coarse.cell_faces[cell]:
        space = skeleton.faces[face]
        if max(space.degrees) > degree:
            raise ValueError("RT degree must contain every skeleton polynomial degree")
        start, end = coarse.points[coarse.faces[face]]
        tangent = end - start
        coordinates = mesh.points[mesh.faces[mesh.boundary_faces]]
        t = (coordinates - start) @ tangent / (tangent @ tangent)
        on_face = np.all(
            np.abs(coordinates - (start + t[..., None] * tangent)) <= 1e-12, axis=(1, 2)
        )
        on_face &= (t.min(axis=1) >= -1e-12) & (t.max(axis=1) <= 1 + 1e-12)
        covered += on_face
        if not np.any(on_face):
            raise ValueError("local mesh boundary must cover every macroface")
        for breakpoint in space.breaks:
            if not np.any(np.isclose(t[on_face], breakpoint, atol=1e-12, rtol=0)):
                raise ValueError("skeleton breaks must align with fine boundary edges")
    if not np.all(covered == 1):
        raise ValueError("every local boundary edge must lie on exactly one macroface")
    return rt_trace_map(coarse, cell, mesh, skeleton, degree)


def _continuous_basis(
    mesh: TriangleMesh, degree: int, bary: FloatArray
) -> tuple[IntArray, FloatArray, int]:
    """Return the macroconstant or continuous piecewise-polynomial test space."""
    if degree == 0:
        return np.zeros((len(mesh.cells), 1), dtype=np.int64), np.ones((len(bary), 1)), 1
    dofs, nodes, basis, _, _ = tabulate(mesh, degree, bary)
    return dofs, basis, len(nodes)


@dataclass(frozen=True)
class MomentFluxSolution:
    """An H(div) RT flux with explicit continuous-test conservation diagnostics."""

    skeleton: SkeletonSpace
    trace: FloatArray
    local_meshes: tuple[TriangleMesh, ...]
    flux: tuple[FloatArray, ...]
    degree: int
    source: Any
    quadrature_order: int

    def flux_l2_error(self, exact: Any, order: int = 6) -> float:
        """Integrate the physical vector-flux error without post-smoothing."""
        bary, weights = triangle_quadrature(order)
        total = 0.0
        for mesh, coefficients in zip(self.local_meshes, self.flux, strict=True):
            points = np.einsum("qi,tij->tqj", bary, mesh.points[mesh.cells])
            values, _ = rt_evaluate(mesh, coefficients, self.degree, bary)
            difference = values - vector_values(exact, points.reshape(-1, 2)).reshape(values.shape)
            total += float(mesh.areas @ (np.sum(difference**2, axis=2) @ weights))
        return float(np.sqrt(total))

    def divergence_l2_error(self, exact: Any, order: int = 6) -> float:
        """Integrate raw divergence error, without projecting onto continuous tests."""
        bary, weights = triangle_quadrature(order)
        total = 0.0
        for mesh, coefficients in zip(self.local_meshes, self.flux, strict=True):
            points = np.einsum("qi,tij->tqj", bary, mesh.points[mesh.cells])
            _, divergence = rt_evaluate(mesh, coefficients, self.degree, bary)
            difference = divergence - scalar_values(exact, points.reshape(-1, 2)).reshape(
                divergence.shape
            )
            total += float(mesh.areas @ (difference**2 @ weights))
        return float(np.sqrt(total))

    def continuous_moment_residuals(self) -> tuple[FloatArray, ...]:
        """Return moments of div(q)-f against C0 P_m on each macrocell.

        For m=0 this test space contains one macroconstant, not separate fine
        constants. The identity requires that the primal local test space
        contain this continuous space and that its variational equations hold.
        """
        bary, weights = triangle_quadrature(self.quadrature_order)
        residuals = []
        for mesh, coefficients in zip(self.local_meshes, self.flux, strict=True):
            points = np.einsum("qi,tij->tqj", bary, mesh.points[mesh.cells])
            _, divergence = rt_evaluate(mesh, coefficients, self.degree, bary)
            source = scalar_values(self.source, points.reshape(-1, 2)).reshape(divergence.shape)
            dofs, basis, size = _continuous_basis(mesh, self.degree, bary)
            local = np.einsum("q,qi,tq,t->ti", weights, basis, divergence - source, mesh.areas)
            residuals.append(np.bincount(dofs.ravel(), weights=local.ravel(), minlength=size))
        return tuple(residuals)

    def fine_conservation_residuals(self) -> tuple[FloatArray, ...]:
        """Measure fine-cell integral defects; this reconstruction does not impose them."""
        bary, weights = triangle_quadrature(self.quadrature_order)
        residuals = []
        for mesh, coefficients in zip(self.local_meshes, self.flux, strict=True):
            points = np.einsum("qi,tij->tqj", bary, mesh.points[mesh.cells])
            _, divergence = rt_evaluate(mesh, coefficients, self.degree, bary)
            source = scalar_values(self.source, points.reshape(-1, 2)).reshape(divergence.shape)
            residuals.append(mesh.areas * ((divergence - source) @ weights))
        return tuple(residuals)

    def conservation_residuals(self) -> FloatArray:
        """Return macrocell integrated flux-minus-source defects."""
        return np.array([values.sum() for values in self.fine_conservation_residuals()])

    def normal_flux_residuals(self) -> tuple[FloatArray, ...]:
        """Compare every RT boundary normal moment with the skeletal multiplier."""
        result = []
        for cell, (mesh, coefficients) in enumerate(zip(self.local_meshes, self.flux, strict=True)):
            dofs = (
                (self.degree + 1) * mesh.boundary_faces[:, None] + np.arange(self.degree + 1)
            ).ravel()
            mapping = _boundary_map(self.skeleton, cell, mesh, self.degree)
            result.append(coefficients[dofs] - mapping @ self.trace[self.skeleton.cell_dofs(cell)])
        return tuple(result)

    def continuous_divergence_projection(self) -> tuple[FloatArray, ...]:
        """Compute the macro-local L2 projection of divergence onto continuous P_m."""
        bary, weights = triangle_quadrature(self.quadrature_order)
        results = []
        for mesh, coefficients in zip(self.local_meshes, self.flux, strict=True):
            _, divergence = rt_evaluate(mesh, coefficients, self.degree, bary)
            dofs, basis, size = _continuous_basis(mesh, self.degree, bary)
            width = basis.shape[1]
            blocks = np.einsum("q,qi,qj,t->tij", weights, basis, basis, mesh.areas)
            matrix = sparse.coo_matrix(
                (
                    blocks.ravel(),
                    (np.repeat(dofs, width, axis=1).ravel(), np.tile(dofs, (1, width)).ravel()),
                ),
                shape=(size, size),
            ).tocsc()
            moments = np.einsum("q,qi,tq,t->ti", weights, basis, divergence, mesh.areas)
            load = np.bincount(dofs.ravel(), weights=moments.ravel(), minlength=size)
            results.append(solve_linear(matrix, load))
        return tuple(results)

    def projected_divergence_l2_error(self, exact: Any, order: int = 6) -> float:
        """Integrate error of the continuous projection, distinct from raw divergence."""
        projection = self.continuous_divergence_projection()
        bary, weights = triangle_quadrature(order)
        total = 0.0
        for mesh, coefficients in zip(self.local_meshes, projection, strict=True):
            dofs, basis, _ = _continuous_basis(mesh, self.degree, bary)
            values = coefficients[dofs] @ basis.T
            points = np.einsum("qi,tij->tqj", bary, mesh.points[mesh.cells])
            difference = values - scalar_values(exact, points.reshape(-1, 2)).reshape(values.shape)
            total += float(mesh.areas @ (difference**2 @ weights))
        return float(np.sqrt(total))


def _batched_flux_values(evaluate: RawFlux, points: FloatArray, owners: IntArray) -> FloatArray:
    """Bound temporary finite-element tabulation to 4096 explicitly owned points."""
    values = np.empty_like(points)
    for start in range(0, len(points), 4096):
        part = slice(start, start + 4096)
        values[part] = vector_values(evaluate(points[part], owners[part]), points[part])
    return values


def _cut_face_moments(
    mesh: TriangleMesh,
    interior: IntArray,
    evaluate: RawFlux,
    material: CartesianCellField,
    degree: int,
    order: int,
) -> FloatArray:
    """Integrate arithmetic one-sided flux averages in bounded batches of cut faces."""
    moments = np.empty((len(interior), degree + 1))
    for first in range(0, len(interior), 64):
        faces = interior[first : first + 64]
        point_parts, weight_parts, basis_parts = [], [], []
        for face in faces:
            start, end = mesh.points[mesh.faces[face]]
            tangent = end - start
            center = mesh.points[mesh.cells[mesh.face_cells[face, 0]]].mean(axis=0)
            points, weights, _ = cartesian_edge_quadrature(
                start, end, material, order, interior_point=center
            )
            parameter = (points - start) @ tangent / (tangent @ tangent)
            point_parts.append(points)
            weight_parts.append(weights * mesh.lengths[face])
            basis_parts.append(legvander(2 * parameter - 1, degree))
        counts = np.array([len(part) for part in point_parts])
        starts = np.r_[0, np.cumsum(counts)[:-1]]
        points = np.concatenate(point_parts)
        weighted_basis = np.concatenate(basis_parts) * np.concatenate(weight_parts)[:, None]
        normals = np.repeat(mesh.normals[faces], counts, axis=0)
        average = np.zeros_like(points)
        for neighbor in range(2):
            owners = np.repeat(mesh.face_cells[faces, neighbor], counts)
            average += _batched_flux_values(evaluate, points, owners) / 2
        normal_flux = np.einsum("pa,pa->p", average, normals)
        moments[first : first + len(faces)] = np.add.reduceat(
            weighted_basis * normal_flux[:, None], starts, axis=0
        )
    return moments


@dataclass(frozen=True)
class _MomentFactory:
    """Evaluate one macrocell's face and volume moments inside its worker."""

    skeleton: SkeletonSpace
    trace: FloatArray
    meshes: tuple[TriangleMesh, ...]
    evaluators: tuple[RawFlux, ...]
    m: int
    order: int
    material: Any
    parameter: FloatArray
    legendre: FloatArray
    w: FloatArray

    def __call__(self, cell: int) -> FloatArray:
        """Return canonical RT coefficients with the unchanged quadrature ordering."""
        skeleton, trace, m, order = self.skeleton, self.trace, self.m, self.order
        mesh, evaluate, material = self.meshes[cell], self.evaluators[cell], self.material
        parameter, legendre, w = self.parameter, self.legendre, self.w
        mapping = _boundary_map(skeleton, cell, mesh, m)
        size = (m + 1) * len(mesh.faces) + m * (m + 1) * len(mesh.cells)
        coefficients = np.zeros(size)
        rows = ((m + 1) * mesh.boundary_faces[:, None] + np.arange(m + 1)).ravel()
        coefficients[rows] = mapping @ trace[skeleton.cell_dofs(cell)]
        interior = np.flatnonzero(mesh.face_cells[:, 1] >= 0)
        if len(interior) and isinstance(material, CartesianCellField):
            moments = _cut_face_moments(mesh, interior, evaluate, material, m, order)
            dofs = ((m + 1) * interior[:, None] + np.arange(m + 1)).ravel()
            coefficients[dofs] = moments.ravel()
        elif len(interior):
            vertices = mesh.points[mesh.faces[interior]]
            points = vertices[:, :1] + parameter[None, :, None] * (
                vertices[:, 1:] - vertices[:, :1]
            )
            flat = points.reshape(-1, 2)
            average = np.zeros_like(points)
            for neighbor in range(2):
                indices = np.repeat(mesh.face_cells[interior, neighbor], len(parameter))
                average += vector_values(evaluate(flat, indices), flat).reshape(points.shape) / 2
            normal = np.einsum("fqa,fa->fq", average, mesh.normals[interior])
            moments = np.einsum("q,qi,fq,f->fi", w / 2, legendre, normal, mesh.lengths[interior])
            dofs = ((m + 1) * interior[:, None] + np.arange(m + 1)).ravel()
            coefficients[dofs] = moments.ravel()
        if m:
            bary, weights, _ = material_triangle_quadrature(mesh, material, order)
            vertices = mesh.points[mesh.cells]
            points = np.einsum("tqi,tij->tqj", bary, vertices)
            flat = points.reshape(-1, 2)
            values = vector_values(
                evaluate(flat, np.repeat(np.arange(len(mesh.cells)), bary.shape[1])), flat
            )
            inverse = np.linalg.inv((vertices[:, 1:] - vertices[:, :1]).swapaxes(1, 2))
            pullback = np.einsum("tab,tqb->tqa", inverse, values.reshape(points.shape))
            tests = rt_interior_tests(m, bary.reshape(-1, 3)[:, 1:]).reshape(*bary.shape[:2], -1)
            moments = np.einsum("tq,tqi,tqa,t->tia", weights, tests, pullback, mesh.areas)
            coefficients[(m + 1) * len(mesh.faces) :] = moments.ravel()
        return coefficients


def reconstruct_flux_moments(
    skeleton: SkeletonSpace,
    trace: FloatArray,
    local_meshes: Sequence[TriangleMesh],
    raw_fluxes: Sequence[RawFlux],
    *,
    degree: int = 1,
    source: Any = 0.0,
    quadrature_order: int = 6,
    material: Any = None,
    backend: Literal["serial", "thread", "process"] = "serial",
    workers: int | None = None,
) -> MomentFluxSolution:
    """Reconstruct RT0/RT1/RT2 from signed skeleton and one-sided primal fluxes.

    Each macrocell supplies ``raw_flux(points, fine_cell_indices)`` returning
    physical vectors with shape (points,2). This contract makes traces at
    discontinuous material interfaces explicit. Boundary normal moments come
    from the skeleton; interior normal moments use the arithmetic average of
    both one-sided fluxes; volume moments match the original primal flux.
    A ``CartesianCellField`` supplied through ``material`` partitions interior
    faces and volumes at its pixel interfaces before integrating those moments.

    The source is used only in diagnostics. No fine-cell equilibrium equation
    or optimization is added. Trace degree must be <= RT degree and every
    skeleton break must align with fine boundary edges. Process execution requires
    picklable raw-flux evaluators; native primal evaluators support this contract.
    """
    m = rt_degree(degree)
    order = positive_int(quadrature_order, "quadrature_order", m + 2)
    meshes, evaluators = tuple(local_meshes), tuple(raw_fluxes)
    if skeleton.components != 1:
        raise ValueError("flux reconstruction requires a scalar skeleton")
    if len(meshes) != len(skeleton.mesh.cells) or len(evaluators) != len(meshes):
        raise ValueError("provide one local mesh and raw-flux evaluator per macrocell")
    if not all(isinstance(mesh, TriangleMesh) for mesh in meshes) or not all(
        map(callable, evaluators)
    ):
        raise TypeError("local meshes must be TriangleMesh and raw-flux evaluators callable")
    if np.iscomplexobj(trace):
        raise ValueError("trace must be real")
    trace = np.array(trace, dtype=float, copy=True)
    if trace.shape != (skeleton.size,) or not np.isfinite(trace).all():
        raise ValueError("trace must be finite with the skeleton size")
    x, w = leggauss(order)
    parameter, legendre = (x + 1) / 2, cast(FloatArray, legvander(x, m))
    factory = _MomentFactory(
        skeleton, trace, meshes, evaluators, m, order, material, parameter, legendre, w
    )
    fluxes = map_local(factory, range(len(meshes)), backend=backend, workers=workers)
    return MomentFluxSolution(skeleton, trace, meshes, tuple(fluxes), m, source, order)


@dataclass(frozen=True)
class _PrimalFlux:
    """Picklable one-sided gradient evaluator with precomputed fine-cell geometry."""

    degree: int
    material: Any
    pressure: FloatArray
    dofs: IntArray
    vertices: FloatArray
    inverse: FloatArray
    bary_gradient: FloatArray

    def __call__(self, points: FloatArray, indices: IntArray) -> FloatArray:
        """Evaluate physical -K grad(p_h) in explicitly identified fine triangles."""
        inverse, vertices = self.inverse, self.vertices
        pressure, dofs, bary_gradient = self.pressure, self.dofs, self.bary_gradient
        reference = np.einsum("pab,pb->pa", inverse[indices], points - vertices[indices, 0])
        bary = np.column_stack((1 - reference.sum(axis=1), reference))
        _, derivative, _ = reference_basis(self.degree, bary)
        gradients = np.einsum("pin,pna->pia", derivative, bary_gradient[indices])
        gradient = np.einsum("pi,pia->pa", pressure[dofs[indices]], gradients)
        material = self.material
        if isinstance(material, CartesianCellField):
            material = cartesian_trace_values(material, points, vertices[indices].mean(axis=1))
        elif isinstance(material, PlanarMaterial):
            material = material.trace_values(points, vertices[indices].mean(axis=1))
        return -np.einsum("pab,pb->pa", tensor_values(material, points), gradient)


def _primal_flux(solution: DarcySolution, cell: int) -> RawFlux:
    """Build a one-sided finite-element gradient evaluator for a primal macrocell."""
    mesh, pressure = solution.local_meshes[cell], solution.pressure[cell]
    dofs, _ = nodal_space(mesh, solution.degree)
    vertices = mesh.points[mesh.cells]
    inverse = np.linalg.inv((vertices[:, 1:] - vertices[:, :1]).swapaxes(1, 2))
    bary_gradient, _ = p1_geometry(mesh)
    return _PrimalFlux(
        solution.degree, solution.permeability, pressure, dofs, vertices, inverse, bary_gradient
    )


def reconstruct_darcy_moments(
    solution: DarcySolution,
    *,
    degree: int = 1,
    quadrature_order: int | None = None,
    raw_fluxes: Sequence[RawFlux] | None = None,
    backend: Literal["serial", "thread", "process"] = "serial",
    workers: int | None = None,
) -> MomentFluxSolution:
    """Apply the RT moment reconstruction to a native primal Pk Darcy solution.

    Require m<=k so continuous P_m tests belong to the primal space. This
    identity alone is weaker than the paper's optimal-estimate hypothesis
    k>=ell+2 in two dimensions, where ell is the skeleton degree.

    CartesianCellField interfaces use exact geometric cuts and one-sided pixel
    values from the incident triangle. For other callbacks, K is evaluated at
    physical face points. If K has different one-sided values there, provide ``raw_fluxes`` with
    explicit fine-cell indices; the routine never infers a material limit by
    perturbing coordinates. Discontinuities solely at macrofaces use the
    skeletal boundary data and do not require such an override.
    """
    m = rt_degree(degree)
    if solution.formulation != "primal":
        raise ValueError("moment reconstruction requires a primal Darcy solution")
    if m > solution.degree:
        raise ValueError(
            "RT degree must not exceed the primal degree for continuous-test conservation"
        )
    order = (
        max(solution.quadrature_order, solution.degree + 2, m + 2)
        if quadrature_order is None
        else quadrature_order
    )
    evaluators = (
        tuple(_primal_flux(solution, cell) for cell in range(len(solution.local_meshes)))
        if raw_fluxes is None
        else raw_fluxes
    )
    return reconstruct_flux_moments(
        solution.skeleton,
        solution.hybrid.trace,
        solution.local_meshes,
        evaluators,
        degree=m,
        source=solution.source,
        quadrature_order=order,
        material=solution.permeability,
        backend=backend,
        workers=workers,
    )
