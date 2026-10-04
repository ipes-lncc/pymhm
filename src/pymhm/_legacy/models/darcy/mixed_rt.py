"""Mixed Darcy with Raviart--Thomas fluxes and discontinuous nodal pressure."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy import sparse

from pymhm._legacy.models.darcy._mixed import (
    normal_flux_blocks,
    triangle_mixed_operators,
    triangle_pressure_basis,
    triangle_pressure_integrals,
)
from pymhm.core.contracts import HybridSolution, LocalAssembly, LocalProblem
from pymhm.core.system import HybridSystem
from pymhm.core.validation import FloatArray, positive_int
from pymhm.fem.hdiv.rt import rt_basis, rt_degree, rt_dofs, rt_evaluate
from pymhm.fem.quadrature.material import material_triangle_quadrature
from pymhm.fem.reference import legendre_values
from pymhm.fem.scalar.operators import boundary_data
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace
from pymhm.linalg.linear import solve_linear
from pymhm.materials.evaluation import scalar_values, vector_values
from pymhm.meshes.triangle import TriangleMesh


def pressure_basis(degree: int, bary: FloatArray) -> FloatArray:
    """Tabulate discontinuous nodal Pm pressure, using a constant for RT0."""
    return triangle_pressure_basis(rt_degree(degree), bary)


def rt_operators(
    mesh: TriangleMesh, degree: int, permeability: Any = 1.0, source: Any = 0.0, order: int = 5
) -> tuple[Any, Any, FloatArray, FloatArray]:
    """Assemble inverse-diffusion mass, DG-Pm divergence/load and pressure integral weights."""
    m = rt_degree(degree)
    bary, weights, material = material_triangle_quadrature(mesh, permeability, max(order, m + 3))
    values, divergence = rt_basis(mesh, m, bary)
    pressure = pressure_basis(m, bary)
    dofs = rt_dofs(mesh, m)
    nq = (m + 1) * len(mesh.faces) + m * (m + 1) * len(mesh.cells)
    mass, D, load = triangle_mixed_operators(
        mesh, bary, weights, material, source, values, divergence, pressure, dofs, nq
    )
    moment = triangle_pressure_integrals(mesh, weights, pressure)
    return mass, D, load, moment


def rt_trace_map(
    coarse: TriangleMesh, cell: int, fine: TriangleMesh, skeleton: SkeletonSpace, degree: int
) -> FloatArray:
    """Integrate canonical RT normal moments of the oriented macroface flux density."""
    m = rt_degree(degree)
    count = m + 1
    result = np.zeros((count * len(fine.boundary_faces), len(skeleton.cell_dofs(cell))))
    x, w = leggauss(m + 3)
    parameter = (x + 1) / 2
    basis = legendre_values(x, m)
    offset = 0
    for side, face in enumerate(coarse.cell_faces[cell]):
        space = skeleton.faces[face]
        start, end = coarse.points[coarse.faces[face]]
        tangent = end - start
        for index, edge in enumerate(fine.boundary_faces):
            vertices = fine.points[fine.faces[edge]]
            t = (vertices - start) @ tangent / (tangent @ tangent)
            if (
                not np.allclose(vertices, start + t[:, None] * tangent, atol=1e-12, rtol=0)
                or min(t) < -1e-12
                or max(t) > 1 + 1e-12
            ):
                continue
            evaluation = np.clip(t[0] + parameter * (t[1] - t[0]), 0, 1)
            result[count * index : count * (index + 1), offset : offset + space.size] = (
                fine.lengths[edge]
                * coarse.signs[cell, side]
                * (basis.T @ (w[:, None] / 2 * space.evaluate(evaluation)))
            )
        offset += space.size
    return result


@dataclass(frozen=True)
class _RTFactory:
    """Build an RT local mixed Neumann system with boundary-pressure multipliers."""

    mesh: TriangleMesh
    skeleton: SkeletonSpace
    degree: int
    refinement: int
    permeability: Any
    source: Any
    order: int

    def __call__(self, cell: int) -> LocalAssembly:
        """Assemble the flux-prescribing local saddle and joint pressure null mode."""
        m = self.degree
        fine = self.mesh.submesh(cell, self.refinement)
        M, D, f, moment = rt_operators(fine, m, self.permeability, self.source, self.order)
        nq, npres = M.shape[0], D.shape[0]
        count = m + 1
        nb = count * len(fine.boundary_faces)
        indices = (count * fine.boundary_faces[:, None] + np.arange(count)).ravel()
        mapping = rt_trace_map(self.mesh, cell, fine, self.skeleton, m)
        A, B, load = normal_flux_blocks(M, D, f, indices, mapping)
        boundary_constant = np.zeros((len(fine.boundary_faces), count))
        boundary_constant[:, 0] = 1
        kernel = np.r_[np.zeros(nq), np.ones(npres), boundary_constant.ravel()][:, None]
        weights = np.r_[np.zeros(nq), moment, np.zeros(nb)]
        problem = LocalProblem(
            A,
            B,
            load,
            self.skeleton.cell_dofs(cell),
            kernel=kernel,
            constraints=weights[:, None],
        )
        return LocalAssembly(problem, (fine, weights, nq, npres))


@dataclass(frozen=True)
class RTDarcySolution:
    """Locally H(div)-conforming RTm flux and discontinuous nodal Pm pressure.

    Mathematical degree m>=0 corresponds to Basix RT element degree m+1.
    MHM traces prescribe physical normal flux, while classical solutions use a
    globally conforming fine mesh. The latter have no skeletal multiplier.
    """

    local_meshes: tuple[TriangleMesh, ...]
    pressure: tuple[FloatArray, ...]
    flux: tuple[FloatArray, ...]
    degree: int
    permeability: Any
    source: Any
    quadrature_order: int
    skeleton: SkeletonSpace | None = None
    hybrid: HybridSolution | None = None
    residual: float = 0.0

    def l2_error(self, exact: Any, order: int = 6) -> float:
        """Integrate the discontinuous pressure error without centroidal substitution."""
        total = 0.0
        for mesh, p in zip(self.local_meshes, self.pressure, strict=True):
            bary, w, _ = material_triangle_quadrature(mesh, self.permeability, order)
            points = np.einsum("tqi,tij->tqj", bary, mesh.points[mesh.cells])
            value = np.einsum("ti,tqi->tq", p, pressure_basis(self.degree, bary))
            error = value - scalar_values(exact, points.reshape(-1, 2)).reshape(w.shape)
            total += float(mesh.areas @ np.sum(w * error**2, axis=1))
        return float(np.sqrt(total))

    def flux_l2_error(self, exact: Any, order: int = 6) -> float:
        """Integrate the complete physical H(div) vector error."""
        total = 0.0
        for mesh, q in zip(self.local_meshes, self.flux, strict=True):
            bary, w, _ = material_triangle_quadrature(mesh, self.permeability, order)
            points = np.einsum("tqi,tij->tqj", bary, mesh.points[mesh.cells])
            value = rt_evaluate(mesh, q, self.degree, bary)[0]
            error = value - vector_values(exact, points.reshape(-1, 2)).reshape(value.shape)
            total += float(mesh.areas @ np.sum(w * np.sum(error**2, axis=2), axis=1))
        return float(np.sqrt(total))

    def divergence_l2_error(self, exact: Any, order: int = 6) -> float:
        """Integrate the full divergence error, distinguishing projection from pointwise balance."""
        total = 0.0
        for mesh, q in zip(self.local_meshes, self.flux, strict=True):
            bary, weights, _ = material_triangle_quadrature(mesh, self.permeability, order)
            points = np.einsum("tqi,tij->tqj", bary, mesh.points[mesh.cells])
            divergence = rt_evaluate(mesh, q, self.degree, bary)[1]
            exact_values = scalar_values(exact, points.reshape(-1, 2)).reshape(weights.shape)
            total += float(mesh.areas @ np.sum(weights * (divergence - exact_values) ** 2, axis=1))
        return float(np.sqrt(total))

    def fine_equilibrium_residuals(self) -> tuple[FloatArray, ...]:
        """Return every DG-Pm moment of div(q)-f on each fine triangle."""
        result = []
        for mesh, q in zip(self.local_meshes, self.flux, strict=True):
            bary, w, _ = material_triangle_quadrature(
                mesh, self.permeability, self.quadrature_order
            )
            points = np.einsum("tqi,tij->tqj", bary, mesh.points[mesh.cells])
            divergence = rt_evaluate(mesh, q, self.degree, bary)[1]
            defect = divergence - scalar_values(self.source, points.reshape(-1, 2)).reshape(w.shape)
            result.append(
                np.einsum(
                    "tq,tqi,tq,t->ti", w, pressure_basis(self.degree, bary), defect, mesh.areas
                )
            )
        return tuple(result)

    def fine_conservation_residuals(self) -> tuple[FloatArray, ...]:
        """Return integrated fine-cell flux/source imbalance via the partition of unity."""
        return tuple(moment.sum(axis=1) for moment in self.fine_equilibrium_residuals())

    def conservation_residuals(self) -> FloatArray:
        """Sum fine-cell physical balances over each local mesh/macrocell."""
        return np.array([defect.sum() for defect in self.fine_conservation_residuals()])

    def normal_flux_residuals(self) -> tuple[FloatArray, ...]:
        """Compare all local exterior RT normal moments with the physical MHM trace."""
        if self.skeleton is None or self.hybrid is None:
            return ()
        defects = []
        count = self.degree + 1
        for cell, (mesh, q) in enumerate(zip(self.local_meshes, self.flux, strict=True)):
            indices = (count * mesh.boundary_faces[:, None] + np.arange(count)).ravel()
            mapping = rt_trace_map(self.skeleton.mesh, cell, mesh, self.skeleton, self.degree)
            defects.append(q[indices] - mapping @ self.hybrid.trace[self.skeleton.cell_dofs(cell)])
        return tuple(defects)


def solve_darcy_rt(
    mesh: TriangleMesh,
    *,
    degree: int = 2,
    permeability: Any = 1.0,
    source: Any = 0.0,
    dirichlet: Any = 0.0,
    neumann: dict[int, Any] | None = None,
    skeleton: SkeletonSpace | None = None,
    local_refinement: int = 2,
    quadrature_order: int = 5,
    mean_pressure: float = 0.0,
    solver: str = "scipy",
    local_solver: str = "scipy",
    backend: Literal["serial", "thread", "process"] = "serial",
    workers: int | None = None,
) -> RTDarcySolution:
    """Solve MHM Darcy using RTm/Pm locals, m>=0, and aligned physical flux traces.

    Skeletal degrees may not exceed m; segment breaks must coincide with fine
    boundary vertices. Normal flux is prescribed through exact moment constraints,
    with one joint constant-pressure/boundary-pressure kernel retained per cell.
    All DG-Pm divergence moments are enforced independently, including the cell
    conservation equation. Cartesian material fields use geometric cut quadrature.
    """
    m = rt_degree(degree)
    refinement = positive_int(local_refinement, "local_refinement")
    order = max(positive_int(quadrature_order, "quadrature_order"), m + 3)
    skeleton = (
        SkeletonSpace(mesh, tuple(FaceSpace.uniform(m) for _ in mesh.faces))
        if skeleton is None
        else skeleton
    )
    if skeleton.mesh is not mesh or skeleton.components != 1:
        raise ValueError("RT Darcy requires a scalar skeleton on the supplied mesh")
    for face in skeleton.faces:
        if max(face.degrees) > m or not np.allclose(
            np.asarray(face.breaks) * refinement,
            np.round(np.asarray(face.breaks) * refinement),
            rtol=0,
            atol=1e-12,
        ):
            raise ValueError(
                "RT trace degrees exceed m or segment breaks are not aligned with fine edges"
            )
    boundary, fixed = boundary_data(skeleton, dirichlet, neumann, order=order)
    factory = _RTFactory(mesh, skeleton, m, refinement, permeability, source, order)
    system = HybridSystem.from_local_factory(
        factory,
        range(len(mesh.cells)),
        boundary_load=-boundary,
        local_solver=local_solver,
        backend=backend,
        workers=workers,
    )
    moments = [metadata[1] for metadata in system.local_metadata]
    gauges = (
        [system.mean_constraint(moments, mean_pressure * mesh.areas.sum())]
        if neumann is not None and set(neumann) == set(mesh.boundary_faces)
        else None
    )
    hybrid = system.solve(solver=solver, fixed=fixed, constraints=gauges)
    pressure = []
    flux = []
    for metadata, field in zip(system.local_metadata, hybrid.fields, strict=True):
        fine, _, nq, npres = metadata
        flux.append(field[:nq])
        pressure.append(field[nq : nq + npres].reshape(len(fine.cells), -1))
    return RTDarcySolution(
        tuple(metadata[0] for metadata in system.local_metadata),
        tuple(pressure),
        tuple(flux),
        m,
        permeability,
        source,
        order,
        skeleton,
        hybrid,
        hybrid.residual,
    )


def solve_darcy_rt_conforming(
    mesh: TriangleMesh,
    *,
    degree: int = 2,
    permeability: Any = 1.0,
    source: Any = 0.0,
    dirichlet: Any = 0.0,
    neumann: dict[int, Any] | None = None,
    quadrature_order: int = 5,
    mean_pressure: float = 0.0,
    solver: str = "scipy",
) -> RTDarcySolution:
    """Solve the classical globally conforming RTm/Pm mixed problem without macro restrictions.

    Pressure is natural weak data; outward normal flux moments are essential.
    The unknowns are fine-mesh flux/pressure coefficients. A pure Neumann solve
    adds only the physical mean gauge, then checks the original free equations.
    This is a classical numerical reference, not an exact analytical solution.
    """
    m = rt_degree(degree)
    order = max(positive_int(quadrature_order, "quadrature_order"), m + 3)
    neumann = {} if neumann is None else neumann
    if any(face not in mesh.boundary_faces for face in neumann):
        raise ValueError("Neumann data require external faces")
    M, D, f, moment = rt_operators(mesh, m, permeability, source, order)
    nq = M.shape[0]
    A = sparse.bmat([[M, -D.T], [-D, None]], format="csc")
    b = np.r_[np.zeros(nq), -f]
    x, w = leggauss(order)
    basis = legendre_values(x, m)
    fixed: dict[int, float] = {}
    lengths = mesh.lengths
    for face in mesh.boundary_faces:
        a, z = mesh.points[mesh.faces[face]]
        points = a + (x[:, None] + 1) / 2 * (z - a)
        data = scalar_values(neumann.get(int(face), dirichlet), points)
        moments = (w * data / 2) @ basis
        indices = (m + 1) * face + np.arange(m + 1)
        if face in neumann:
            fixed.update(
                (int(i), float(v)) for i, v in zip(indices, lengths[face] * moments, strict=True)
            )
        else:
            b[indices] = -(2 * np.arange(m + 1) + 1) * moments
    values = np.zeros(len(b))
    known = np.array(sorted(fixed), dtype=int)
    values[known] = [fixed[i] for i in known]
    free = np.setdiff1d(np.arange(len(b)), known)
    physical = A[free][:, free]
    rhs = b[free] - A[free][:, known] @ values[known]
    matrix = physical
    if len(neumann) == len(mesh.boundary_faces):
        weights = np.r_[np.zeros(nq), moment][free]
        matrix = sparse.bmat(
            [
                [physical, sparse.csc_matrix(weights[:, None])],
                [sparse.csc_matrix(weights[None]), None],
            ],
            format="csc",
        )
        solve_rhs = np.r_[rhs, mean_pressure * mesh.areas.sum()]
    else:
        solve_rhs = rhs
    values[free] = solve_linear(matrix, solve_rhs, solver=solver)[: len(free)]
    residual = float(
        np.linalg.norm(physical @ values[free] - rhs)
        / max(
            np.linalg.norm(rhs),
            np.linalg.norm(abs(physical) @ np.abs(values[free])),
            np.finfo(float).tiny,
        )
    )
    if residual > 1e-8:
        raise ValueError("incompatible data: pressure gauge changed the physical equations")
    return RTDarcySolution(
        (mesh,),
        (values[nq:].reshape(len(mesh.cells), -1),),
        (values[:nq],),
        m,
        permeability,
        source,
        order,
        residual=residual,
    )
