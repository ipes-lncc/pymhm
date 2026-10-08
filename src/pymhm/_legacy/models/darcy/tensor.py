"""Historical Darcy solve and coefficient records on enriched rectangular RT spaces."""

from typing import Any, cast

import numpy as np
from scipy import sparse

from pymhm.core.contracts import LocalProblem
from pymhm.core.offline import condense_cached
from pymhm.core.system import HybridSystem
from pymhm.core.validation import positive_int
from pymhm.fem.hdiv.tensor_rt import (
    _operators,
    _trace_map,
    tensor_rt_dofs,
)
from pymhm.fem.scalar.operators import boundary_data
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace
from pymhm.meshes.cartesian import CartesianMacroMesh
from pymhm.meshes.triangle import TriangleMesh
from pymhm.postprocessing.solutions import TensorRTDarcySolution as TensorRTDarcySolution


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
