"""Tensor-product Raviart--Thomas elements with independent interior enrichment.

On each rectangle, normal traces have degree k and interior vector functions
have RT order s=k+n. The pressure is Q_s. Face DOFs are integral Legendre
moments; cell DOFs are coefficients of zero-normal-trace polynomial bubbles.
The construction implements the face/interior separation used by the mixed
MHM family of Duran et al. (2019), without copying a reference implementation.
"""

from dataclasses import dataclass
from functools import cache
from typing import Any, cast

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy import sparse

from pymhm.element_backends import (
    ReferenceElementSpec,
    create_reference_element,
    interpolate_reference,
    legendre_values,
    reference_interpolation_points,
)
from pymhm.elements import boundary_data, scalar_values, tensor_values, vector_values
from pymhm.hdiv_reference import vector_tabulation
from pymhm.hybrid import HybridSolution, HybridSystem, LocalProblem
from pymhm.mesh import FaceSpace, FloatArray, IntArray, SkeletonSpace, TriangleMesh, positive_int
from pymhm.offline import condense_cached
from pymhm.quadrilateral import (
    CartesianMacroMesh,
    _cartesian_rectangle_quadrature,
    _grid_resolves_material,
    quadrilateral_quadrature,
)
from pymhm.reservoir import CartesianCellField


@cache
def _reference_map(degree: int, enriched_degree: int) -> FloatArray:
    """Express declared face lifts and bubble coordinates in native rectangular RT."""
    k, s = degree, enriched_degree
    element = create_reference_element(
        ReferenceElementSpec("RT", "quadrilateral", s + 1, lagrange_variant="legendre")
    )
    points = reference_interpolation_points(element)
    x, y = points.T
    lx, ly = legendre_values(2 * x - 1, s), legendre_values(2 * y - 1, s)
    values = np.zeros((len(points), 4 * (k + 1) + 2 * s * (s + 1), 2))
    for edge in range(4):
        parameter = (x, y, 1 - x, 1 - y)[edge]
        moments = legendre_values(2 * parameter - 1, k) * (2 * np.arange(k + 1) + 1)
        indices = slice(edge * (k + 1), (edge + 1) * (k + 1))
        axis = 1 if edge in (0, 2) else 0
        values[:, indices, axis] = (y - 1, x, y, x - 1)[edge][:, None] * moments
    offset = 4 * (k + 1)
    for axis in range(2):
        for b in range(s + 1 if axis == 0 else s):
            for a in range(s if axis == 0 else s + 1):
                t = x if axis == 0 else y
                values[:, offset, axis] = t * (1 - t) * lx[:, a] * ly[:, b]
                offset += 1
    result = interpolate_reference(element, values)
    result.setflags(write=False)
    return result


def tensor_rt_basis(
    mesh: CartesianMacroMesh, degree: int, enrichment: int, points: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Evaluate globally oriented RT vectors, divergence and modal Q_s pressure.

    ``points`` are reference coordinates in [0,1]^2, either shared (q,2) or
    cell dependent (cells,q,2). Vector/divergence arrays
    include a leading cell axis. The pressure basis is cell independent and
    ordered with x degree varying fastest. The Piola transform preserves all
    normal moments even on anisotropic rectangles.
    """
    k = positive_int(degree, "RT degree", 0)
    s = k + positive_int(enrichment, "interior enrichment", 0)
    raw = np.asarray(points)
    if (
        np.iscomplexobj(raw)
        or raw.ndim not in (2, 3)
        or raw.shape[-1] != 2
        or (raw.ndim == 3 and raw.shape[0] != len(mesh.cells))
        or not np.isfinite(raw).all()
    ):
        raise ValueError("reference points must be finite real pairs")
    points = np.asarray(raw, dtype=float).reshape(-1, 2)
    x, y = points.T
    lx, ly = legendre_values(2 * x - 1, s), legendre_values(2 * y - 1, s)
    width = 4 * (k + 1) + 2 * s * (s + 1)
    native, native_divergence = vector_tabulation("RT", "quadrilateral", s + 1, points)
    transform = _reference_map(k, s)
    values = np.einsum("qia,ij->qja", native, transform)
    divergence = native_divergence @ transform
    orientation = np.ones((len(mesh.cells), width))
    orientation[:, : 4 * (k + 1)] = (mesh.signs[:, :, None] ** np.arange(1, k + 2)).reshape(
        len(mesh.cells), -1
    )
    pressure = (ly[:, :, None] * lx[:, None, :]).reshape(len(points), -1)
    if raw.ndim == 3:
        values = values.reshape(len(mesh.cells), -1, width, 2)
        divergence = divergence.reshape(len(mesh.cells), -1, width)
        pressure = pressure.reshape(*raw.shape[:2], -1)
    else:
        values, divergence = values[None], divergence[None]
    physical = values * mesh.spacing[None, None, None, :] / np.prod(mesh.spacing)
    physical = physical * orientation[:, None, :, None]
    div = divergence * orientation[:, None, :] / np.prod(mesh.spacing)
    return physical, div, pressure


def tensor_rt_dofs(mesh: CartesianMacroMesh, degree: int, enrichment: int) -> IntArray:
    """Map conforming face moments and independent cell bubbles to global DOFs."""
    k = positive_int(degree, "RT degree", 0)
    s = k + positive_int(enrichment, "interior enrichment", 0)
    face = ((k + 1) * mesh.cell_faces[:, :, None] + np.arange(k + 1)).reshape(len(mesh.cells), -1)
    interior = (
        (k + 1) * len(mesh.faces)
        + 2 * s * (s + 1) * np.arange(len(mesh.cells))[:, None]
        + np.arange(2 * s * (s + 1))
    )
    return np.column_stack((face, interior))


def _trace_map(
    mesh: CartesianMacroMesh,
    cell: int,
    fine: CartesianMacroMesh,
    skeleton: SkeletonSpace,
    degree: int,
) -> FloatArray:
    """Integrate each oriented macro flux against fine-edge Legendre moments."""
    x, w = leggauss(degree + 2)
    t = (x + 1) / 2
    width = sum(skeleton.faces[face].size for face in mesh.cell_faces[cell])
    mapping = np.zeros(((degree + 1) * len(fine.boundary_faces), width))
    offset = 0
    for side, face_id in enumerate(mesh.cell_faces[cell]):
        start, end = mesh.points[mesh.faces[face_id]]
        tangent = end - start
        for row, edge in enumerate(fine.boundary_faces):
            coordinates = fine.points[fine.faces[edge]]
            interval = (coordinates - start) @ tangent / (tangent @ tangent)
            if (
                not np.allclose(
                    coordinates, start + interval[:, None] * tangent, rtol=0, atol=1e-12
                )
                or min(interval) < -1e-12
                or max(interval) > 1 + 1e-12
            ):
                continue
            parameter = np.clip(interval[0] + t * (interval[1] - interval[0]), 0, 1)
            space = skeleton.faces[face_id]
            mapping[(degree + 1) * row : (degree + 1) * (row + 1), offset : offset + space.size] = (
                fine.lengths[edge]
                * mesh.signs[cell, side]
                * legendre_values(x, degree).T
                @ (w[:, None] / 2 * space.evaluate(parameter))
            )
        offset += skeleton.faces[face_id].size
    return mapping


def _material_rule(
    mesh: CartesianMacroMesh, permeability: Any, order: int
) -> tuple[FloatArray, FloatArray]:
    """Return cellwise normalized quadrature, resolving declared material interfaces."""
    if isinstance(permeability, CartesianCellField) and not _grid_resolves_material(
        mesh, permeability
    ):
        return _cartesian_rectangle_quadrature(
            mesh.points[mesh.cells[:, 0]], mesh.spacing, permeability, order
        )
    points, weights = quadrilateral_quadrature(order)
    return (
        np.broadcast_to(points, (len(mesh.cells), *points.shape)),
        np.broadcast_to(weights, (len(mesh.cells), len(weights))),
    )


def _operators(
    mesh: CartesianMacroMesh,
    degree: int,
    enrichment: int,
    permeability: Any,
    source: Any,
    order: int,
) -> tuple[Any, Any, FloatArray]:
    """Assemble mixed inverse-permeability mass and divergence moments."""
    points, weights = _material_rule(mesh, permeability, order)
    basis, div, pressure = tensor_rt_basis(mesh, degree, enrichment, points)
    physical = mesh.points[mesh.cells[:, 0], None] + points * mesh.spacing
    inverse = np.linalg.inv(tensor_values(permeability, physical.reshape(-1, 2))).reshape(
        len(mesh.cells), -1, 2, 2
    )
    blocks = np.einsum("t,tq,tqia,tqab,tqjb->tij", mesh.areas, weights, basis, inverse, basis)
    dofs = tensor_rt_dofs(mesh, degree, enrichment)
    nq = int(dofs.max()) + 1
    npres = len(mesh.cells) * pressure.shape[-1]
    width = dofs.shape[1]
    mass = sparse.coo_matrix(
        (
            blocks.ravel(),
            (np.repeat(dofs, width, axis=1).ravel(), np.tile(dofs, (1, width)).ravel()),
        ),
        shape=(nq, nq),
    ).tocsc()
    blocks = np.einsum("t,tq,tqi,tqj->tij", mesh.areas, weights, pressure, div)
    pids = np.arange(npres).reshape(len(mesh.cells), -1)
    divergence = sparse.coo_matrix(
        (
            blocks.ravel(),
            (
                np.repeat(pids, width, axis=1).ravel(),
                np.tile(dofs, (1, pressure.shape[-1])).ravel(),
            ),
        ),
        shape=(npres, nq),
    ).tocsc()
    force = scalar_values(source, physical.reshape(-1, 2)).reshape(physical.shape[:2])
    load = np.einsum("t,tq,tqi,tq->ti", mesh.areas, weights, pressure, force).ravel()
    return mass, divergence, load


@dataclass(frozen=True)
class TensorRTDarcySolution:
    """Enriched rectangular RT flux, modal Q_s pressure and physical skeleton flux."""

    skeleton: SkeletonSpace
    local_meshes: tuple[CartesianMacroMesh, ...]
    pressure: tuple[FloatArray, ...]
    flux: tuple[FloatArray, ...]
    hybrid: HybridSolution
    degree: int
    enrichment: int
    permeability: Any
    source: Any
    quadrature_order: int

    def evaluate(self, cell: int, points: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Evaluate pressure, physical flux and divergence on every local rectangle."""
        fine = self.local_meshes[cell]
        basis, div, pressure = tensor_rt_basis(fine, self.degree, self.enrichment, points)
        coefficients = self.flux[cell][tensor_rt_dofs(fine, self.degree, self.enrichment)]
        return (
            np.einsum(
                "ti,tqi->tq",
                self.pressure[cell],
                np.broadcast_to(pressure, (*basis.shape[:2], pressure.shape[-1])),
            ),
            np.einsum("tqia,ti->tqa", basis, coefficients),
            np.einsum("tqi,ti->tq", div, coefficients),
        )

    def errors(self, pressure: Any, flux: Any, divergence: Any, order: int = 8) -> dict[str, float]:
        """Integrate three independent physical L2 errors without smoothing interfaces."""
        errors = np.zeros(3)
        for cell, fine in enumerate(self.local_meshes):
            points, weights = _material_rule(fine, self.permeability, order)
            physical = fine.points[fine.cells[:, 0], None] + points * fine.spacing
            flat = physical.reshape(-1, 2)
            p, q, d = self.evaluate(cell, points)
            errors[0] += fine.areas @ (
                np.sum((p - scalar_values(pressure, flat).reshape(p.shape)) ** 2 * weights, axis=1)
            )
            errors[1] += fine.areas @ (
                np.sum(
                    np.sum((q - vector_values(flux, flat).reshape(q.shape)) ** 2, axis=-1)
                    * weights,
                    axis=1,
                )
            )
            errors[2] += fine.areas @ (
                np.sum(
                    (d - scalar_values(divergence, flat).reshape(d.shape)) ** 2 * weights, axis=1
                )
            )
        return dict(zip(("pressure_l2", "flux_l2", "divergence_l2"), np.sqrt(errors), strict=True))

    def equilibrium_residuals(self) -> tuple[FloatArray, ...]:
        """Return all Q_s moments of div(q)-f in each local fine cell."""
        residuals = []
        for cell, fine in enumerate(self.local_meshes):
            points, weights = _material_rule(fine, self.permeability, self.quadrature_order)
            physical = fine.points[fine.cells[:, 0], None] + points * fine.spacing
            _, _, pressure = tensor_rt_basis(fine, self.degree, self.enrichment, points)
            d = self.evaluate(cell, points)[2]
            force = scalar_values(self.source, physical.reshape(-1, 2)).reshape(d.shape)
            residuals.append(np.einsum("t,tq,tqi,tq->ti", fine.areas, weights, pressure, d - force))
        return tuple(residuals)

    def normal_flux_residuals(self) -> tuple[FloatArray, ...]:
        """Return all fine-boundary moments of q.n minus the represented macro trace."""
        residuals = []
        for cell, fine in enumerate(self.local_meshes):
            ids = (
                (self.degree + 1) * fine.boundary_faces[:, None] + np.arange(self.degree + 1)
            ).ravel()
            mapping = _trace_map(
                cast(CartesianMacroMesh, self.skeleton.mesh), cell, fine, self.skeleton, self.degree
            )
            residuals.append(
                self.flux[cell][ids] - mapping @ self.hybrid.trace[self.skeleton.cell_dofs(cell)]
            )
        return tuple(residuals)


def solve_darcy_tensor_rt(
    mesh: CartesianMacroMesh,
    *,
    degree: int = 1,
    enrichment: int = 0,
    permeability: Any = 1.0,
    source: Any = 0.0,
    dirichlet: Any = 0.0,
    neumann: dict[int, Any] | None = None,
    skeleton: SkeletonSpace | None = None,
    local_refinement: int = 2,
    quadrature_order: int = 6,
    mean_pressure: float = 0.0,
    solver: str = "scipy",
    local_solver: str = "scipy",
    reuse_operators: bool = True,
) -> TensorRTDarcySolution:
    """Solve Darcy with RT face degree k and interior degree k+n on rectangles.

    The DG pressure degree is s=k+n in each coordinate. Skeleton trace degrees
    may differ by face, but cannot exceed k and must align with fine edges.
    Pressure data are imposed weakly; Neumann data are outward physical flux.
    Pure Neumann problems use one global pressure mean and retain the local
    constant pressure mode. No projected or averaged permeability is introduced.
    """
    k = positive_int(degree, "RT degree", 0)
    enrichment = positive_int(enrichment, "interior enrichment", 0)
    s = k + enrichment
    r = positive_int(local_refinement, "local_refinement")
    order = positive_int(quadrature_order, "quadrature order", s + 2)
    skeleton = (
        SkeletonSpace(cast(TriangleMesh, mesh), tuple(FaceSpace.uniform(k) for _ in mesh.faces))
        if skeleton is None
        else skeleton
    )
    if skeleton.mesh is not mesh or skeleton.components != 1:
        raise ValueError("tensor RT requires a scalar skeleton on the supplied mesh")
    for face in skeleton.faces:
        if max(face.degrees) > k or not np.allclose(
            np.asarray(face.breaks) * r, np.rint(np.asarray(face.breaks) * r), rtol=0, atol=1e-12
        ):
            raise ValueError(
                "RT trace degree must not exceed k and breaks must align with fine edges"
            )
    boundary, fixed = boundary_data(skeleton, dirichlet, neumann)
    problems, meshes, means = [], [], []
    for cell in range(len(mesh.cells)):
        fine = mesh.submesh(cell, r)
        mass, divergence, force = _operators(fine, k, enrichment, permeability, source, order)
        nq, npres = mass.shape[0], divergence.shape[0]
        ids = ((k + 1) * fine.boundary_faces[:, None] + np.arange(k + 1)).ravel()
        nb = len(ids)
        selector = sparse.coo_matrix((np.ones(nb), (ids, np.arange(nb))), shape=(nq, nb)).tocsc()
        matrix = sparse.bmat(
            [[mass, -divergence.T, selector], [-divergence, None, None], [selector.T, None, None]],
            format="csc",
        )
        mapping = _trace_map(mesh, cell, fine, skeleton, k)
        coupling = np.zeros((nq + npres + nb, mapping.shape[1]))
        coupling[-nb:] = -mapping
        constant = np.zeros(npres)
        constant[:: (s + 1) ** 2] = 1
        kernel = np.r_[
            np.zeros(nq), constant, np.tile(np.r_[1.0, np.zeros(k)], len(fine.boundary_faces))
        ][:, None]
        weights = np.r_[np.zeros(nq), constant * np.repeat(fine.areas, (s + 1) ** 2), np.zeros(nb)]
        problems.append(
            LocalProblem(
                matrix,
                coupling,
                np.r_[np.zeros(nq), -force, np.zeros(nb)],
                skeleton.cell_dofs(cell),
                kernel,
                weights[:, None],
            )
        )
        meshes.append(fine)
        means.append(weights)
    system = (
        HybridSystem.from_responses(
            condense_cached(problems, solver=local_solver), boundary_load=-boundary
        )
        if reuse_operators
        else HybridSystem(problems, boundary_load=-boundary, local_solver=local_solver)
    )
    constraints = (
        [system.mean_constraint(means, mean_pressure * sum(mesh.areas))]
        if neumann is not None and set(neumann) == set(mesh.boundary_faces)
        else None
    )
    hybrid = system.solve(solver=solver, fixed=fixed, constraints=constraints)
    pressures, fluxes = [], []
    for fine, values in zip(meshes, hybrid.fields, strict=True):
        nq = int(tensor_rt_dofs(fine, k, enrichment).max()) + 1
        fluxes.append(values[:nq])
        pressures.append(
            values[nq : nq + len(fine.cells) * (s + 1) ** 2].reshape(len(fine.cells), -1)
        )
    return TensorRTDarcySolution(
        skeleton,
        tuple(meshes),
        tuple(pressures),
        tuple(fluxes),
        hybrid,
        k,
        enrichment,
        permeability,
        source,
        order,
    )
