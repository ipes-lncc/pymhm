"""Tetrahedral AFW mixed MHM elasticity with H(div) stress and weak symmetry in 3D."""

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy import sparse

from pymhm._legacy.models.darcy.hdiv_3d import _trace_mapping
from pymhm._legacy.models.elasticity.boundary import require_compatible_displacement_flux
from pymhm._legacy.models.elasticity.stress_forms_3d import (
    boundary_selector,
    mixed_elasticity_operators_3d,
    rigid_coefficients,
)
from pymhm.core.contracts import LocalAssembly, LocalProblem
from pymhm.core.system import HybridSystem
from pymhm.core.validation import positive_int
from pymhm.fem.hdiv.family_3d import (
    HDiv3DFamily,
)
from pymhm.fem.traces.traction_3d import TractionSkeleton3D as TractionSkeleton3D
from pymhm.fem.traces.traction_3d import traction_boundary_data as _boundary
from pymhm.meshes.mixed import AffineMixedMesh
from pymhm.postprocessing.stress_3d import MixedElasticity3DSolution as MixedElasticity3DSolution


@dataclass(frozen=True)
class _Factory:
    """Assemble and retain the six exact local rigid modes on independent subdomains."""

    skeleton: TractionSkeleton3D
    family: HDiv3DFamily
    refinement: int
    lame_lambda: Any
    lame_mu: Any
    source: Any
    compliance: Any
    order: int

    def __call__(self, cell: int) -> LocalAssembly:
        """Return the complete Neumann saddle, its rigid constraints and physical diagnostics."""
        coarse = self.skeleton.mesh
        fine = coarse.submesh(cell, self.refinement)
        mass, div, asym, force, plain, weighted, scale = mixed_elasticity_operators_3d(
            fine,
            self.family,
            lame_lambda=self.lame_lambda,
            lame_mu=self.lame_mu,
            source=self.source,
            compliance=self.compliance,
            quadrature_order=self.order,
        )
        ns, nu = mass.shape[0], div.shape[0]
        selector = boundary_selector(fine, self.family, ns)
        nb = selector.shape[1]
        z = sparse.csc_matrix
        matrix = sparse.bmat(
            [
                [mass, div.T, asym.T, -selector],
                [div, z((nu, nu)), z((nu, nu)), z((nu, nb))],
                [asym, z((nu, nu)), z((nu, nu)), z((nu, nb))],
                [-selector.T, z((nb, nu)), z((nb, nu)), z((nb, nb))],
            ],
            format="csc",
        )
        center = coarse.points[coarse.cells[cell]].mean(axis=0)
        rigid, moments, boundary = rigid_coefficients(fine, self.family, center)
        kernel, constraints = np.zeros((matrix.shape[0], 6)), np.zeros((matrix.shape[0], 6))
        kernel[ns : ns + nu] = rigid
        rotations = kernel[ns + nu : ns + 2 * nu].reshape(
            len(fine.cells), self.family.pressure_size, 3, 6
        )
        rotations[:, 0, :, 3:] = -np.eye(3)
        kernel[-nb:] = boundary
        constraints[ns : ns + nu] = moments
        mapping = np.kron(
            _trace_mapping(self.skeleton.scalar, cell, fine, self.family.normal_degree), np.eye(3)
        )
        coupling = np.zeros((matrix.shape[0], mapping.shape[1]))
        coupling[-nb:] = -mapping
        problem = LocalProblem(
            matrix,
            coupling,
            np.r_[np.zeros(ns), -force, np.zeros(nu + nb)],
            self.skeleton.cell_dofs(cell),
            kernel,
            constraints,
        )
        plain_full, weighted_full = np.zeros(matrix.shape[0]), np.zeros(matrix.shape[0])
        plain_full[:ns], weighted_full[:ns] = plain, weighted
        return LocalAssembly(problem, (fine, ns, nu, plain_full, weighted_full, scale))


def solve_elasticity_mixed_3d(
    mesh: AffineMixedMesh,
    *,
    stress_degree: int = 2,
    trace_degree: int = 1,
    subdivisions: int = 1,
    local_refinement: int = 2,
    lame_lambda: Any = 1.0,
    lame_mu: Any = 1.0,
    compliance: Any = None,
    source: Any = (0.0, 0.0, 0.0),
    dirichlet: Any = (0.0, 0.0, 0.0),
    traction: dict[int, Any] | None = None,
    rigid_moments: Any = (0.0,) * 6,
    mean_pressure: float = 0.0,
    quadrature_order: int = 5,
    solver: str = "scipy",
    local_solver: str = "scipy",
    local_refinement_precision: Literal["double", "extended"] = "double",
    global_refinement_precision: Literal["double", "extended"] = "double",
    backend: Literal["serial", "thread", "process"] = "serial",
    workers: int | None = None,
) -> MixedElasticity3DSolution:
    """Solve -div(sigma)=f using tetrahedral AFW BDM_k/DG P_(k-1)/DG P_(k-1).

    k>=2 preserves all six exact rigid motions of each local Neumann problem.
    Lambda may include infinity; mu must be positive. The full Cartesian
    compliance option replaces the Lamé law and explicitly extends to skew
    tensors. Prescribed traction is outward sigma n; other faces carry weak
    displacement data. The multiplier is -sigma n in the canonical normal.
    Full displacement data impose the exact integral identity
    integral(tr(A sigma))=integral(g.n). At infinite lambda its compatible
    limit sets mean(-tr(sigma)/3)=mean_pressure. Pure traction uses six physical
    integrated displacement moments about the domain centroid; rotation is
    never separately gauged. Trace subdivisions must align with local faces.
    Classical AFW stability does not remove the MHM trace-rank requirement.
    """
    degree = positive_int(stress_degree, "stress degree", 2)
    refinement = positive_int(local_refinement, "local refinement")
    skeleton = TractionSkeleton3D(mesh, trace_degree, subdivisions)
    if trace_degree > degree or refinement % subdivisions:
        raise ValueError(
            "trace degree must not exceed stress degree and subdivisions must divide refinement"
        )
    family = HDiv3DFamily("tetrahedron", degree - 1, degree)
    order = max(positive_int(quadrature_order, "quadrature order"), degree + 2)
    data = {} if traction is None else traction
    if not np.isfinite(mean_pressure):
        raise ValueError("mean_pressure must be finite")
    if data and mean_pressure != 0:
        raise ValueError("mean_pressure only applies to full displacement boundary data")
    load, fixed, volume_flux, volume_scale = _boundary(skeleton, dirichlet, data, order)
    system = HybridSystem.from_local_factory(
        _Factory(skeleton, family, refinement, lame_lambda, lame_mu, source, compliance, order),
        range(len(mesh.cells)),
        boundary_load=load,
        local_solver=local_solver,
        local_refinement_precision=local_refinement_precision,
        backend=backend,
        workers=workers,
    )
    gauges = []
    metadata = system.local_metadata
    if not data:
        scale = max(info[5] for info in metadata)
        if scale == 0:
            require_compatible_displacement_flux(volume_flux, volume_scale)
            gauges.append(
                system.mean_constraint(
                    [info[3] for info in metadata], -3 * mean_pressure * sum(mesh.volumes)
                )
            )
        else:
            if mean_pressure != 0:
                raise ValueError("mean_pressure is a gauge only in the incompressible limit")
            gauges.append(
                system.mean_constraint([info[4] / scale for info in metadata], volume_flux / scale)
            )
    if set(data) == set(mesh.boundary_faces):
        if np.iscomplexobj(rigid_moments):
            raise ValueError("rigid_moments requires six finite real integrals")
        targets = np.asarray(rigid_moments, dtype=float)
        if targets.shape != (6,) or not np.isfinite(targets).all():
            raise ValueError("rigid_moments requires six finite real integrals")
        center = sum(
            fine.volumes @ fine.points[fine.cells].mean(axis=1) for fine, *_ in metadata
        ) / sum(mesh.volumes)
        weights = []
        for response, (fine, ns, nu, *_) in zip(system.responses, metadata, strict=True):
            moment = np.zeros((len(response.problem.load), 6))
            moment[ns : ns + nu] = rigid_coefficients(fine, family, center)[1]
            weights.append(moment)
        gauges.extend(
            system.mean_constraint([w[:, i] for w in weights], targets[i]) for i in range(6)
        )
    hybrid = system.solve(
        solver=solver,
        fixed=fixed,
        constraints=gauges,
        refinement_precision=global_refinement_precision,
    )
    stresses, displacements, rotations = [], [], []
    for field, (_, ns, nu, *_) in zip(hybrid.fields, metadata, strict=True):
        stresses.append(field[:ns].reshape(-1, 3))
        displacements.append(field[ns : ns + nu].reshape(-1, family.pressure_size, 3))
        rotations.append(field[ns + nu : ns + 2 * nu].reshape(-1, family.pressure_size, 3))
    return MixedElasticity3DSolution(
        skeleton,
        family,
        tuple(info[0] for info in metadata),
        tuple(stresses),
        tuple(displacements),
        tuple(rotations),
        hybrid,
        source,
        order,
    )
