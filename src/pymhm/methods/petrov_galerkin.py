"""Petrov-Galerkin MHM with residual face enrichment for two-dimensional Darcy.

Equations (27)--(34) of Fernando, Martins, Pereira and Valentin (2023) use
the conormal A grad(p).n. Here all stored multipliers use physical Darcy flux
q.n=-A grad(p).n. Only the enriched multiplier is macro conservative.
"""

from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
from scipy import sparse

from pymhm._legacy.models.darcy.primal import _DarcyLocalFactory
from pymhm.core.contracts import LocalAssembly
from pymhm.core.system import HybridSystem
from pymhm.core.validation import positive_int
from pymhm.fem.scalar.operators import boundary_data
from pymhm.fem.traces.interval import SkeletonSpace
from pymhm.fem.traces.jump import FaceJumpForm as _compat_FacePenalty
from pymhm.fem.traces.jump import face_jump_form as _penalty
from pymhm.fem.traces.jump import scalar_trace_matrix as _compat_trace_matrix
from pymhm.linalg.linear import solve_linear
from pymhm.materials.evaluation import tensor_values
from pymhm.meshes.polygonal import PolygonMesh
from pymhm.meshes.triangle import TriangleMesh
from pymhm.methods.robin import _ellipticity
from pymhm.postprocessing.solutions import PGMHMSolution as PGMHMSolution


@dataclass(frozen=True)
class _PGFactory:
    """Reuse the original Darcy local assembly and enforce the declared ellipticity bound."""

    darcy: _DarcyLocalFactory
    lower: float

    def __call__(self, cell: int) -> LocalAssembly:
        """Return the unchanged mean-constrained Darcy problem and its physical metadata."""
        item = self.darcy(cell)
        fine = item.metadata[0]
        minimum = np.linalg.eigvalsh(tensor_values(self.darcy.permeability, fine.points)).min()
        if minimum < self.lower * (1 - 64 * np.finfo(float).eps):
            raise ValueError("ellipticity_lower_bound exceeds a sampled material eigenvalue")
        return item


def solve_pgmhm(
    mesh: TriangleMesh | PolygonMesh,
    *,
    stabilization_parameter: float,
    permeability: Any = 1.0,
    source: Any = 0.0,
    dirichlet: Any = 0.0,
    neumann: dict[int, Any] | None = None,
    skeleton: SkeletonSpace | None = None,
    degree: int = 2,
    local_refinement: int = 1,
    local_meshes: tuple[TriangleMesh, ...] | None = None,
    quadrature_order: int = 6,
    ellipticity_lower_bound: float | None = None,
    mean_pressure: float = 0.0,
    solver: str = "scipy",
    local_solver: str = "scipy",
    local_refinement_precision: Literal["double", "extended"] = "double",
    backend: Literal["serial", "thread", "process"] = "serial",
    workers: int | None = None,
) -> PGMHMSolution:
    """Solve equations (27)--(34), including the conservative residual enrichment.

    ``stabilization_parameter=alpha`` is explicit: the theorem requires alpha
    sufficiently small but does not provide a universal numerical threshold.
    The weight is alpha*K_min/(2*H_E), using the full macroface length, not
    the lengths of refined skeleton segments. The local degree must satisfy
    k>=ell+2. The stated uniform theorem assumes ell>=1; ell=0 is included
    among the article's numerical experiments. Rank/residual checks establish
    neither a uniform inf-sup bound nor parameter-independent conditioning.

    Dirichlet faces contribute jumps against their prescribed trace. Physical
    Neumann faces are fixed essentially and receive no jump penalty or flux
    enrichment. Compatible pure Neumann data use the physical mean-pressure
    gauge; the local enrichment has zero volume mean. The boundary projection
    in equation (32) is evaluated on the common fine-trace partition so that
    its Pk moments agree with the Dirichlet functional in equation (29).
    ``local_refinement_precision="extended"`` retains residual-correction
    digits in both the local response and its residual enrichment. It leaves
    the operator and residual tolerances unchanged and requires a platform
    with a wider long-double type, as in :func:`pymhm.linalg.linear.solve_linear`.
    """
    if not isinstance(mesh, (TriangleMesh, PolygonMesh)):
        raise TypeError("PGMHM requires triangular or polygonal two-dimensional macrocells")
    if local_refinement_precision not in {"double", "extended"}:
        raise ValueError("local_refinement_precision must be double or extended")
    degree = positive_int(degree, "degree")
    refinement = positive_int(local_refinement, "local_refinement")
    order = max(positive_int(quadrature_order, "quadrature_order"), degree + 2)
    alpha = np.asarray(stabilization_parameter)
    if alpha.shape != () or np.iscomplexobj(alpha) or not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("stabilization_parameter must be a finite positive scalar")
    lower = _ellipticity(permeability, ellipticity_lower_bound)
    geometry = cast(TriangleMesh, mesh)
    skeleton = SkeletonSpace(geometry) if skeleton is None else skeleton
    if skeleton.mesh is not mesh or skeleton.components != 1:
        raise ValueError("PGMHM requires a scalar skeleton on the supplied mesh")
    if degree < max(max(space.degrees) for space in skeleton.faces) + 2:
        raise ValueError("PGMHM requires local degree k >= trace degree ell + 2")
    _, fixed = boundary_data(skeleton, 0.0, neumann, order=order)
    neumann_faces = set() if neumann is None else set(neumann)
    pure_neumann = neumann_faces == set(mesh.boundary_faces)
    if not np.isfinite(mean_pressure) or (mean_pressure != 0 and not pure_neumann):
        raise ValueError("a finite nonzero mean_pressure is only valid for pure Neumann data")
    if local_meshes is not None:
        from pymhm.meshes.refinement import validate_submesh

        if len(local_meshes) != len(mesh.cells):
            raise ValueError("one local mesh is required for each macrocell")
        for cell, fine in enumerate(local_meshes):
            validate_submesh(geometry, cell, fine)
    factory = _PGFactory(
        _DarcyLocalFactory(
            geometry,
            skeleton,
            permeability,
            source,
            tuple(np.empty((0, 3)) for _ in mesh.cells),
            refinement,
            local_meshes,
            degree,
            "primal",
            order,
        ),
        lower,
    )
    system = HybridSystem.from_local_factory(
        factory,
        range(len(mesh.cells)),
        local_solver=local_solver,
        local_refinement_precision=local_refinement_precision,
        backend=backend,
        workers=workers,
    )
    penalties = tuple(
        _penalty(system, skeleton, face, degree, order, float(alpha), lower, dirichlet)
        for face in range(len(mesh.faces))
        if face not in neumann_faces
    )
    rows: list[int] = []
    columns: list[int] = []
    entries: list[float] = []
    for data in penalties:
        if len(data.sides) == 1:
            # Both appearances of g use the same common fine-trace partition.
            # In particular, continuous piecewise polynomial data may have a
            # derivative jump within a macroface and cannot use one Gauss rule.
            boundary = skeleton.faces[data.face].evaluate(data.parameter).T @ (
                data.weights * data.prescribed
            )
            dofs = skeleton.dofs(data.face)
            system.rhs[dofs] -= boundary
            system.load_scale[dofs] += abs(boundary)
        weighted = data.weights * data.coefficient
        block = data.jump.T @ (weighted[:, None] * data.jump)
        load = data.jump.T @ (weighted * (data.prescribed - data.source_jump))
        rows.extend(np.repeat(data.indices, len(data.indices)))
        columns.extend(np.tile(data.indices, len(data.indices)))
        entries.extend(block.ravel())
        np.add.at(system.rhs, data.indices, load)
        np.add.at(system.load_scale, data.indices, abs(load))
    system.matrix = (
        system.matrix
        + sparse.coo_matrix((entries, (rows, columns)), shape=system.matrix.shape).tocsc()
    )
    constraints = None
    if pure_neumann:
        weights = [item[1] for item in system.local_metadata]
        constraints = [system.mean_constraint(weights, mean_pressure * float(np.sum(mesh.areas)))]
    hybrid = system.solve(solver=solver, fixed=fixed, constraints=constraints)
    coefficients = np.r_[hybrid.trace, np.concatenate(hybrid.coarse)]
    extra = [
        np.zeros(len(response.problem.load), dtype=base.dtype)
        for response, base in zip(system.responses, hybrid.fields, strict=True)
    ]
    for data in penalties:
        jump = data.jump @ coefficients[data.indices] + data.source_jump - data.prescribed
        flux_enrichment = -data.coefficient * jump
        for cell, sign, evaluation in data.sides:
            extra[cell] += sign * (evaluation.T @ (data.weights * flux_enrichment))
    enriched = []
    for response, base, load in zip(system.responses, hybrid.fields, extra, strict=True):
        problem = response.problem.with_load(-load)
        operator, rhs = problem.condensation_system()
        correction = solve_linear(
            operator,
            rhs[:, 0],
            solver=local_solver,
            refinement_precision=local_refinement_precision,
        )[: len(load)]
        enriched.append(base + correction)
    return PGMHMSolution(
        skeleton,
        tuple(item[0] for item in system.local_metadata),
        hybrid.fields,
        tuple(enriched),
        hybrid,
        system,
        permeability,
        source,
        degree,
        order,
        float(alpha),
        lower,
        penalties,
        tuple(extra),
    )


_trace_matrix = _compat_trace_matrix
_FacePenalty = _compat_FacePenalty
