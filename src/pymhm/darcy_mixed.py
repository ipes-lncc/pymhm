"""Polynomial BDM Darcy fluxes with independent normal and interior enrichment."""

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy import sparse

from pymhm.bdm import bdm2_basis, bdm2_evaluate, bdm2_trace_map
from pymhm.bdm_family import BDMFamily
from pymhm.cut_cells import material_triangle_quadrature
from pymhm.elements import (
    boundary_data,
    scalar_values,
    tensor_values,
    vector_values,
)
from pymhm.hybrid import HybridSolution, HybridSystem, LocalProblem
from pymhm.lagrange import reference_basis
from pymhm.mesh import FaceSpace, FloatArray, SkeletonSpace, TriangleMesh, positive_int

_BDM2 = BDMFamily()


def _pressure_basis(family: BDMFamily, bary: FloatArray) -> FloatArray:
    """Return a cardinal complete pressure space with partition of unity."""
    degree = family.polynomial_degree - 1
    if degree == 0:
        return np.ones((*bary.shape[:-1], 1))
    basis = reference_basis(degree, bary.reshape(-1, 3))[0]
    return basis.reshape(*bary.shape[:-1], basis.shape[-1])


def _basis(
    family: BDMFamily, mesh: TriangleMesh, bary: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Retain the historical BDM2 coordinates and use general canonical moments otherwise."""
    return bdm2_basis(mesh, bary) if family == BDMFamily() else family.basis(mesh, bary)


def _evaluate(
    family: BDMFamily, mesh: TriangleMesh, flux: FloatArray, bary: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Evaluate vectors in the same executed coordinate convention as assembly."""
    return (
        bdm2_evaluate(mesh, flux, bary)
        if family == BDMFamily()
        else family.evaluate(mesh, flux, bary)
    )


def _trace_map(
    family: BDMFamily, mesh: TriangleMesh, cell: int, fine: TriangleMesh, skeleton: SkeletonSpace
) -> FloatArray:
    """Map the physical multiplier into the family's oriented boundary moments."""
    return (
        bdm2_trace_map(mesh, cell, fine, skeleton)
        if family == BDMFamily()
        else family.trace_map(mesh, cell, fine, skeleton)
    )


def _operators(
    mesh: TriangleMesh, permeability: Any, source: Any, order: int, family: BDMFamily = _BDM2
) -> tuple[Any, Any, FloatArray]:
    """Assemble inverse-permeability mass and the complete divergence pressure moments."""
    bary, weights, material = material_triangle_quadrature(mesh, permeability, order)
    basis, div = _basis(family, mesh, bary)
    scalar = _pressure_basis(family, bary)
    points = np.einsum("tqi,tij->tqj", bary, mesh.points[mesh.cells])
    inverse = np.linalg.inv(tensor_values(material, points.reshape(-1, 2))).reshape(
        *weights.shape, 2, 2
    )
    blocks = np.einsum("tq,tqia,tqab,tqjb,t->tij", weights, basis, inverse, basis, mesh.areas)
    dofs = family.dofs(mesh)
    width, pressure_count = family.local_size, scalar.shape[-1]
    nq, npres = family.size(mesh), pressure_count * len(mesh.cells)
    mass = sparse.coo_matrix(
        (
            blocks.ravel(),
            (np.repeat(dofs, width, axis=1).ravel(), np.tile(dofs, (1, width)).ravel()),
        ),
        shape=(nq, nq),
    ).tocsc()
    div_blocks = np.einsum("tq,tqi,tqj,t->tij", weights, scalar, div, mesh.areas)
    divergence = sparse.coo_matrix(
        (
            div_blocks.ravel(),
            (
                np.repeat(np.arange(npres).reshape(-1, pressure_count), width, axis=1).ravel(),
                np.tile(dofs, (1, pressure_count)).ravel(),
            ),
        ),
        shape=(npres, nq),
    ).tocsc()
    load = np.einsum(
        "tq,tqi,tq,t->ti",
        weights,
        scalar,
        scalar_values(source, points.reshape(-1, 2)).reshape(points.shape[:2]),
        mesh.areas,
    ).ravel()
    return mass, divergence, load


@dataclass(frozen=True)
class BDMDarcySolution:
    """Conforming BDM(k,n) flux and complete discontinuous pressure from a MHM solve.

    Flux stores integrated normal moments and cell moments as specified in
    ``pymhm.bdm_family``. The default BDM2/P1 uses the equivalent coordinates
    in ``pymhm.bdm``. Pressure has degree k+n-1 and cardinal coefficients.
    The skeletal multiplier is the physical flux q.n in the macro orientation.
    """

    skeleton: SkeletonSpace
    local_meshes: tuple[TriangleMesh, ...]
    pressure: tuple[FloatArray, ...]
    flux: tuple[FloatArray, ...]
    hybrid: HybridSolution
    permeability: Any
    source: Any
    quadrature_order: int
    family: BDMFamily = BDMFamily()

    def l2_error(self, exact: Any, order: int = 5) -> float:
        """Integrate broken pressure error against an analytical pressure."""
        total = 0.0
        for mesh, pressure in zip(self.local_meshes, self.pressure, strict=True):
            bary, weights, _ = material_triangle_quadrature(mesh, self.permeability, order)
            points = np.einsum("tqi,tij->tqj", bary, mesh.points[mesh.cells])
            values = np.einsum("ti,tqi->tq", pressure, _pressure_basis(self.family, bary))
            error = values - scalar_values(exact, points.reshape(-1, 2)).reshape(values.shape)
            total += float(mesh.areas @ np.sum(error**2 * weights, axis=1))
        return float(np.sqrt(total))

    def flux_l2_error(self, exact: Any, order: int = 5) -> float:
        """Integrate physical BDM flux error without gradient recovery or smoothing."""
        total = 0.0
        for mesh, flux in zip(self.local_meshes, self.flux, strict=True):
            bary, weights, _ = material_triangle_quadrature(mesh, self.permeability, order)
            points = np.einsum("tqi,tij->tqj", bary, mesh.points[mesh.cells])
            values, _ = _evaluate(self.family, mesh, flux, bary)
            error = values - vector_values(exact, points.reshape(-1, 2)).reshape(values.shape)
            total += float(mesh.areas @ np.sum(np.sum(error**2, axis=2) * weights, axis=1))
        return float(np.sqrt(total))

    def divergence_l2_error(self, exact: Any, order: int = 5) -> float:
        """Integrate divergence error; exact=source measures strong equilibrium error."""
        total = 0.0
        for mesh, flux in zip(self.local_meshes, self.flux, strict=True):
            bary, weights, _ = material_triangle_quadrature(mesh, self.permeability, order)
            points = np.einsum("tqi,tij->tqj", bary, mesh.points[mesh.cells])
            _, values = _evaluate(self.family, mesh, flux, bary)
            error = values - scalar_values(exact, points.reshape(-1, 2)).reshape(values.shape)
            total += float(mesh.areas @ np.sum(error**2 * weights, axis=1))
        return float(np.sqrt(total))

    def fine_equilibrium_residuals(self) -> tuple[FloatArray, ...]:
        """Return all complete pressure moments of div(q)-f in every fine triangle."""
        residuals = []
        for mesh, flux in zip(self.local_meshes, self.flux, strict=True):
            bary, weights, _ = material_triangle_quadrature(
                mesh, self.permeability, self.quadrature_order
            )
            points = np.einsum("tqi,tij->tqj", bary, mesh.points[mesh.cells])
            _, divergence = _evaluate(self.family, mesh, flux, bary)
            source = scalar_values(self.source, points.reshape(-1, 2)).reshape(divergence.shape)
            residuals.append(
                np.einsum(
                    "tq,tqi,tq,t->ti",
                    weights,
                    _pressure_basis(self.family, bary),
                    divergence - source,
                    mesh.areas,
                )
            )
        return tuple(residuals)

    def fine_conservation_residuals(self) -> tuple[FloatArray, ...]:
        """Return integrated div(q)-f over each fine cell, using assembly quadrature."""
        return tuple(residual.sum(axis=1) for residual in self.fine_equilibrium_residuals())

    def conservation_residuals(self) -> FloatArray:
        """Return macro flux minus source integrals, with the signed skeleton flux."""
        defects = []
        coarse = self.skeleton.mesh
        for cell, fine in enumerate(self.local_meshes):
            bary, weights, _ = material_triangle_quadrature(
                fine, self.permeability, self.quadrature_order
            )
            points = np.einsum("tqi,tij->tqj", bary, fine.points[fine.cells])
            source = scalar_values(self.source, points.reshape(-1, 2)).reshape(points.shape[:2])
            defect = -float(fine.areas @ np.sum(source * weights, axis=1))
            for side, face in enumerate(coarse.cell_faces[cell]):
                parameter, w = self.skeleton.faces[face].quadrature(max(4, self.family.degree + 1))
                values = self.skeleton.faces[face].evaluate(parameter)
                defect += (
                    coarse.signs[cell, side]
                    * coarse.lengths[face]
                    * float(w @ values @ self.hybrid.trace[self.skeleton.dofs(int(face))])
                )
            defects.append(defect)
        return np.asarray(defects)

    def normal_flux_residuals(self) -> tuple[FloatArray, ...]:
        """Return all boundary normal-moment differences between BDM flux and skeleton."""
        defects = []
        for cell, (fine, flux) in enumerate(zip(self.local_meshes, self.flux, strict=True)):
            count = self.family.degree + 1
            dofs = (count * fine.boundary_faces[:, None] + np.arange(count)).ravel()
            mapping = _trace_map(self.family, self.skeleton.mesh, cell, fine, self.skeleton)
            defects.append(flux[dofs] - mapping @ self.hybrid.trace[self.skeleton.cell_dofs(cell)])
        return tuple(defects)


def solve_darcy_bdm(
    mesh: TriangleMesh,
    *,
    permeability: Any = 1.0,
    source: Any = 0.0,
    dirichlet: Any = 0.0,
    neumann: dict[int, Any] | None = None,
    skeleton: SkeletonSpace | None = None,
    local_refinement: int = 2,
    degree: int = 2,
    enrichment: int = 0,
    quadrature_order: int = 4,
    mean_pressure: float = 0.0,
    solver: str = "scipy",
    local_solver: str = "scipy",
    backend: Literal["serial", "thread", "process"] = "serial",
    workers: int | None = None,
) -> BDMDarcySolution:
    """Solve div(q)=f, q=-K grad(p) with BDM(k,n) flux and DG(k+n-1) pressure.

    ``degree=k`` controls the physical normal polynomial; ``enrichment=n``
    retains all zero-normal interior modes of BDM(k+n). Default k=2,n=0
    recovers BDM2/P1. Default skeletal traces are P1 on every macroface.
    Traces up to Pk and piecewise partitions are supported when every break
    aligns with a fine edge. Quadrature is raised to at least k+n+2.
    Permeability accepts scalar, SPD tensor or physical-coordinate callbacks.
    Neumann data are outward physical flux; remaining boundary faces prescribe
    pressure weakly. Pure Neumann data require compatibility and use a global
    mean-pressure gauge. A quadratic pressure generally has a nonzero P1 error
    even when its linear flux is reproduced exactly.
    """
    family = BDMFamily(degree, enrichment)
    refinement = positive_int(local_refinement, "local_refinement")
    positive_int(quadrature_order, "quadrature_order", 3)
    skeleton = (
        SkeletonSpace(mesh, tuple(FaceSpace.uniform(1) for _ in mesh.faces))
        if skeleton is None
        else skeleton
    )
    if skeleton.mesh is not mesh or skeleton.components != 1:
        raise ValueError("BDM Darcy requires a scalar skeleton on the supplied mesh")
    family.validate_trace(skeleton, refinement)
    quadrature_order = max(quadrature_order, family.polynomial_degree + 2)
    boundary_load, fixed = boundary_data(skeleton, dirichlet, neumann)
    problems, local_meshes, mean_weights = [], [], []
    for cell in range(len(mesh.cells)):
        fine = mesh.submesh(cell, refinement)
        mass, divergence, force = _operators(fine, permeability, source, quadrature_order, family)
        count = family.degree + 1
        nq, npres, nb = mass.shape[0], divergence.shape[0], count * len(fine.boundary_faces)
        dofs = (count * fine.boundary_faces[:, None] + np.arange(count)).ravel()
        selector = sparse.coo_matrix((np.ones(nb), (dofs, np.arange(nb))), shape=(nq, nb)).tocsc()
        matrix = sparse.bmat(
            [
                [mass, -divergence.T, selector],
                [-divergence, sparse.csc_matrix((npres, npres)), sparse.csc_matrix((npres, nb))],
                [selector.T, sparse.csc_matrix((nb, npres)), sparse.csc_matrix((nb, nb))],
            ],
            format="csc",
        )
        mapping = _trace_map(family, mesh, cell, fine, skeleton)
        coupling = np.zeros((nq + npres + nb, mapping.shape[1]))
        coupling[-nb:] = -mapping
        kernel = np.r_[
            np.zeros(nq), np.ones(npres), np.tile(np.r_[1.0, np.zeros(count - 1)], nb // count)
        ][:, None]
        bary, weights, _ = material_triangle_quadrature(fine, permeability, quadrature_order)
        moments = np.einsum(
            "tq,tqi,t->ti", weights, _pressure_basis(family, bary), fine.areas
        ).ravel()
        constraints = np.r_[np.zeros(nq), moments, np.zeros(nb)][:, None]
        problems.append(
            LocalProblem(
                matrix,
                coupling,
                np.r_[np.zeros(nq), -force, np.zeros(nb)],
                skeleton.cell_dofs(cell),
                kernel,
                constraints,
            )
        )
        mean_weights.append(constraints[:, 0])
        local_meshes.append(fine)
    system = HybridSystem(
        problems,
        boundary_load=-boundary_load,
        local_solver=local_solver,
        backend=backend,
        workers=workers,
    )
    pure_neumann = neumann is not None and set(neumann) == set(mesh.boundary_faces)
    gauges = (
        [system.mean_constraint(mean_weights, mean_pressure * float(mesh.areas.sum()))]
        if pure_neumann
        else None
    )
    hybrid = system.solve(solver=solver, fixed=fixed, constraints=gauges)
    pressure, fluxes = [], []
    for fine, field in zip(local_meshes, hybrid.fields, strict=True):
        nq = family.size(fine)
        count = family.polynomial_degree * (family.polynomial_degree + 1) // 2
        fluxes.append(field[:nq])
        pressure.append(field[nq : nq + count * len(fine.cells)].reshape(-1, count))
    return BDMDarcySolution(
        skeleton,
        tuple(local_meshes),
        tuple(pressure),
        tuple(fluxes),
        hybrid,
        permeability,
        source,
        quadrature_order,
        family,
    )
