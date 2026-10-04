"""Tetrahedral AFW mixed MHM elasticity with H(div) stress and weak symmetry in 3D."""

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy import sparse

from pymhm._legacy.models.darcy.hdiv_3d import Mixed3DSkeleton, _trace_mapping
from pymhm._legacy.models.elasticity.boundary import require_compatible_displacement_flux
from pymhm._legacy.models.elasticity.stress_forms_3d import (
    boundary_selector,
    mixed_elasticity_operators_3d,
    rigid_coefficients,
)
from pymhm.core.contracts import HybridSolution, LocalAssembly, LocalProblem
from pymhm.core.system import HybridSystem
from pymhm.core.validation import FloatArray, IntArray, positive_int
from pymhm.fem.hdiv.family_3d import (
    HDiv3DFamily,
    cell_quadrature,
    face_polynomials,
    face_quadrature,
    face_shape,
)
from pymhm.materials.evaluation import vector_values_3d
from pymhm.meshes.mixed import AffineMixedMesh, hdiv3d_basis, hdiv3d_dofs


class TractionSkeleton3D:
    """Three interleaved Cartesian moments of -sigma n on each original triangular macroface.

    Every subtriangle carries an independent complete polynomial. Coefficients
    are physical integral moments, not nodal tractions; canonical normals point
    outward from the first neighboring macrocell.
    """

    def __init__(self, mesh: AffineMixedMesh, degree: int = 1, subdivisions: int = 1) -> None:
        """Create the vector trace from the shared oriented scalar moment partition."""
        if not isinstance(mesh, AffineMixedMesh) or mesh.kind != "tetrahedron":
            raise ValueError("mixed elasticity requires affine tetrahedral macrocells")
        self.scalar = Mixed3DSkeleton(mesh, degree, subdivisions)
        self.mesh, self.degree, self.subdivisions = (
            mesh,
            self.scalar.degree,
            self.scalar.subdivisions,
        )
        self.size = 3 * self.scalar.size

    def cell_dofs(self, cell: int) -> IntArray:
        """Return interleaved traction coordinates on one macrocell's original faces."""
        return (3 * self.scalar.cell_dofs(cell)[:, None] + np.arange(3)).ravel()


def _boundary(
    skeleton: TractionSkeleton3D, dirichlet: Any, traction: dict[int, Any], order: int
) -> tuple[FloatArray, dict[int, float], float, float]:
    """Integrate displacement loads and physical outward traction moments, including volume flux."""
    mesh, scalar = skeleton.mesh, skeleton.scalar
    if not set(traction).issubset(set(mesh.boundary_faces)):
        raise ValueError("traction keys must identify exterior macrofaces")
    load, volume_flux, volume_scale = np.zeros(skeleton.size), 0.0, 0.0
    fixed: dict[int, float] = {}
    uv, w = face_quadrature(3, order)
    tests = face_polynomials(uv, 3, skeleton.degree)
    for face in mesh.boundary_faces:
        partition = scalar.partitions[face]
        for piece, ids in enumerate(partition.cells):
            canonical = face_shape(uv, 3) @ partition.nodes[ids]
            physical = face_shape(canonical, 3) @ mesh.points[mesh.faces[face]]
            _, det = partition.coordinates(piece, canonical)
            weights = w * det * partition.measure
            indices = scalar.offsets[face] + piece * partition.width + np.arange(partition.width)
            vector_ids = (3 * indices[:, None] + np.arange(3)).ravel()
            if face in traction:
                values = -tests.T @ (weights[:, None] * vector_values_3d(traction[face], physical))
                fixed.update(zip(vector_ids.tolist(), values.ravel().tolist(), strict=True))
            else:
                values = vector_values_3d(dirichlet, physical)
                moments = partition.evaluate(piece, canonical).T @ (weights[:, None] * values)
                load[vector_ids] = -moments.ravel()
                normal_moments = (tests.T @ weights)[:, None] * mesh.normals[face]
                volume_flux += float(np.sum(moments * normal_moments))
                volume_scale += float(np.sum(abs(moments * normal_moments)))
    return load, fixed, volume_flux, volume_scale


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


@dataclass(frozen=True)
class MixedElasticity3DSolution:
    """Row-wise H(div) stress, DG displacement and axial weak rotation on local tetrahedra.

    Rotation is ((du_y/dz-du_z/dy), (du_z/dx-du_x/dz),
    (du_x/dy-du_y/dx))/2. Symmetry holds in DG P_(k-1) moments, not pointwise.
    Stress arrays have shape (Hdiv DOFs,3), with columns denoting stress rows;
    displacement and rotation have shape (fine cells,scalar modes,3).
    """

    skeleton: TractionSkeleton3D
    family: HDiv3DFamily
    local_meshes: tuple[AffineMixedMesh, ...]
    stress: tuple[FloatArray, ...]
    displacement: tuple[FloatArray, ...]
    rotation: tuple[FloatArray, ...]
    hybrid: HybridSolution
    source: Any
    quadrature_order: int

    def evaluate(
        self, cell: int, points: FloatArray
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        """Evaluate displacement, full stress, axial rotation and divergence at reference points."""
        fine = self.local_meshes[cell]
        vectors, divergence, scalar = hdiv3d_basis(fine, self.family, points)
        stress = self.stress[cell][hdiv3d_dofs(fine, self.family)]
        return (
            np.einsum("qi,tia->tqa", scalar, self.displacement[cell]),
            np.einsum("tqib,tia->tqab", vectors, stress),
            np.einsum("qi,tia->tqa", scalar, self.rotation[cell]),
            np.einsum("tqi,tia->tqa", divergence, stress),
        )

    def errors(
        self, displacement: Any, stress: Any, rotation: Any, *, order: int = 6
    ) -> dict[str, float]:
        """Integrate displacement, Frobenius stress and axial-rotation physical L2 errors."""
        xi, weights = cell_quadrature("tetrahedron", positive_int(order, "error quadrature order"))
        total = np.zeros(3)
        for cell, fine in enumerate(self.local_meshes):
            points = fine.geometry(xi).reshape(-1, 3)
            values = self.evaluate(cell, xi)
            target_stress = stress(points) if callable(stress) else stress
            if np.iscomplexobj(target_stress) or not np.isfinite(target_stress).all():
                raise ValueError("exact stress must be real and finite")
            targets = (
                vector_values_3d(displacement, points).reshape(values[0].shape),
                np.broadcast_to(target_stress, (len(points), 3, 3)).reshape(values[1].shape),
                vector_values_3d(rotation, points).reshape(values[2].shape),
            )
            for i, target in enumerate(targets):
                delta = (values[i] - target).reshape(len(fine.cells), len(xi), -1)
                total[i] += float(fine.determinants @ (np.sum(delta**2, axis=2) @ weights))
        return dict(
            zip(
                ("displacement_l2", "stress_l2", "rotation_l2"),
                np.sqrt(total).tolist(),
                strict=True,
            )
        )

    def fine_force_residuals(self) -> tuple[FloatArray, ...]:
        """Return every DG displacement moment of div(sigma)+f on every fine tetrahedron."""
        xi, weights = cell_quadrature("tetrahedron", self.quadrature_order)
        scalar = self.family.tabulate(xi)[2]
        result = []
        for cell, fine in enumerate(self.local_meshes):
            divergence = self.evaluate(cell, xi)[3]
            force = vector_values_3d(self.source, fine.geometry(xi).reshape(-1, 3)).reshape(
                divergence.shape
            )
            result.append(
                np.einsum("t,q,qi,tqa->tia", fine.determinants, weights, scalar, divergence + force)
            )
        return tuple(result)

    def weak_symmetry_residuals(self) -> tuple[FloatArray, ...]:
        """Return the three independent skew-stress DG moments without symmetrizing the field."""
        xi, weights = cell_quadrature("tetrahedron", self.family.pressure_degree + 3)
        scalar = self.family.tabulate(xi)[2]
        result = []
        for cell, fine in enumerate(self.local_meshes):
            stress = self.evaluate(cell, xi)[1]
            asym = np.stack(
                [stress[..., i, j] - stress[..., j, i] for i, j in ((1, 2), (2, 0), (0, 1))],
                axis=-1,
            )
            result.append(np.einsum("t,q,qi,tqa->tia", fine.determinants, weights, scalar, asym))
        return tuple(result)

    def normal_traction_residuals(self) -> tuple[FloatArray, ...]:
        """Return canonical normal moments of sigma n plus the physical skeletal traction."""
        result = []
        for cell, fine in enumerate(self.local_meshes):
            selector = boundary_selector(fine, self.family, self.stress[cell].size)
            mapping = np.kron(
                _trace_mapping(self.skeleton.scalar, cell, fine, self.family.normal_degree),
                np.eye(3),
            )
            expected = mapping @ self.hybrid.trace[self.skeleton.cell_dofs(cell)]
            result.append(selector.T @ self.stress[cell].ravel() + expected)
        return tuple(result)


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
