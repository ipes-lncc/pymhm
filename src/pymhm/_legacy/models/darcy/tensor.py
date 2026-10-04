"""Historical Darcy solve and coefficient records on enriched rectangular RT spaces."""

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from scipy import sparse

from pymhm.core.contracts import HybridSolution, LocalProblem
from pymhm.core.offline import condense_cached
from pymhm.core.system import HybridSystem
from pymhm.core.validation import FloatArray, positive_int
from pymhm.fem.hdiv.tensor_rt import (
    _material_rule,
    _operators,
    _trace_map,
    tensor_rt_basis,
    tensor_rt_dofs,
)
from pymhm.fem.scalar.operators import boundary_data
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace
from pymhm.materials.evaluation import scalar_values, vector_values
from pymhm.meshes.cartesian import CartesianMacroMesh
from pymhm.meshes.triangle import TriangleMesh


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
