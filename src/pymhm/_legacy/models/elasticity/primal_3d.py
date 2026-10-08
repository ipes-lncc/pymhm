"""Primal tetrahedral MHM elasticity with general stiffness and six rigid modes.

Kelvin ordering is (xx, yy, zz, sqrt(2) yz, sqrt(2) xz, sqrt(2) xy).
This displacement formulation is not claimed to be uniformly locking-free.
"""

from dataclasses import dataclass
from functools import partial
from typing import Any, Literal

import numpy as np
from scipy import sparse

from pymhm.core.contracts import HybridSolution, LocalAssembly, LocalProblem
from pymhm.core.system import HybridSystem
from pymhm.core.validation import FloatArray, positive_int
from pymhm.fem.scalar.operators import triangle_quadrature
from pymhm.fem.scalar.tetrahedron import tetra_nodal_space, tetra_tabulate, tetrahedron_quadrature
from pymhm.fem.traces.triangle_3d import TriangularSkeleton, _boundary, tetra_trace_coupling
from pymhm.fem.vector.operators import _assemble_blocks
from pymhm.materials.evaluation import scalar_values_3d, vector_values_3d
from pymhm.meshes.tetrahedron import TetraMesh, _dyadic

_KELVIN3 = np.zeros((6, 3, 3))
_KELVIN3[:3] = np.einsum("ai,aj->aij", np.eye(3), np.eye(3))
for _mode, (_a, _b) in enumerate(((1, 2), (0, 2), (0, 1)), start=3):
    _KELVIN3[_mode, _a, _b] = _KELVIN3[_mode, _b, _a] = 1 / np.sqrt(2)


def constitutive_values_3d(
    material: Any, points: FloatArray, *, lame_lambda: Any = 1.0, lame_mu: Any = 1.0
) -> FloatArray:
    """Validate general SPD Kelvin/Cartesian stiffness, or finite isotropic Lamé fields."""
    if material is None:
        lam, mu = scalar_values_3d(lame_lambda, points), scalar_values_3d(lame_mu, points)
        if np.any(lam < 0) or np.any(mu <= 0):
            raise ValueError("Lame lambda must be nonnegative and mu positive")
        identity = np.r_[np.ones(3), np.zeros(3)]
        return 2 * mu[:, None, None] * np.eye(6) + lam[:, None, None] * np.outer(identity, identity)
    raw = material(points) if callable(material) else material
    if np.iscomplexobj(raw) or not np.isfinite(raw).all():
        raise ValueError("constitutive tensor must be finite and real")
    values = np.asarray(raw, dtype=float)
    if values.shape[-4:] == (3, 3, 3, 3):
        values = np.broadcast_to(values, (len(points), 3, 3, 3, 3))
        scale = np.max(np.abs(values), axis=(1, 2, 3, 4))
        for permutation in (
            values.swapaxes(1, 2),
            values.swapaxes(3, 4),
            values.transpose(0, 3, 4, 1, 2),
        ):
            if np.any(
                np.max(np.abs(values - permutation), axis=(1, 2, 3, 4))
                > 64 * np.finfo(float).eps * scale
            ):
                raise ValueError("constitutive tensor requires major and minor symmetries")
        values = np.einsum("aij,nijkl,bkl->nab", _KELVIN3, values, _KELVIN3)
    else:
        if values.shape[-2:] != (6, 6):
            raise ValueError(
                "constitutive tensor requires Kelvin (6,6) or Cartesian (3,3,3,3) shape"
            )
        values = np.broadcast_to(values, (len(points), 6, 6))
    if np.any(
        np.max(np.abs(values - values.swapaxes(1, 2)), axis=(1, 2))
        > 64 * np.finfo(float).eps * np.max(np.abs(values), axis=(1, 2))
    ):
        raise ValueError("constitutive Kelvin matrix must be symmetric")
    if np.any(np.linalg.eigvalsh(values)[:, 0] <= 0):
        raise ValueError("constitutive tensor must be positive definite on symmetric strains")
    return values


def rigid_modes_3d(points: FloatArray, center: FloatArray) -> FloatArray:
    """Return three translations followed by e_x/e_y/e_z cross (x-center)."""
    points = np.asarray(points) - np.asarray(center)
    result = np.zeros((*points.shape[:-1], 3, 6))
    result[..., :3] = np.eye(3)
    for axis in range(3):
        result[..., 3 + axis] = np.cross(np.eye(3)[axis], points)
    return result


def _boundary_vector(
    skeleton: TriangularSkeleton, dirichlet: Any, traction: dict[int, Any], order: int
) -> tuple[FloatArray, dict[int, float]]:
    """Reuse signed scalar face quadrature independently for each physical vector component."""
    load = np.zeros((skeleton.size, 3))
    fixed = {}
    for component in range(3):
        prescribed = {
            face: partial(_component, datum=value, component=component)
            for face, value in traction.items()
        }
        load[:, component], scalar_fixed = _boundary(
            skeleton, partial(_component, datum=dirichlet, component=component), prescribed, order
        )
        fixed.update({3 * index + component: -value for index, value in scalar_fixed.items()})
    return load.ravel(), fixed


def _component(points: FloatArray, *, datum: Any, component: int) -> FloatArray:
    """Evaluate one validated physical-vector component for a scalar trace operation."""
    return vector_values_3d(datum, points)[:, component]


def _local(
    cell: int,
    *,
    mesh: TetraMesh,
    skeleton: TriangularSkeleton,
    degree: int,
    refinement: int,
    constitutive: Any,
    lame_lambda: Any,
    lame_mu: Any,
    source: Any,
    order: int,
) -> LocalAssembly:
    """Assemble Kelvin energy and force with six physical L2 rigid-motion constraints."""
    fine = mesh.submesh(cell, refinement)
    bary, weights = tetrahedron_quadrature(order)
    dofs, nodes, basis, gradient = tetra_tabulate(fine, degree, bary)
    udofs = (3 * dofs[:, :, None] + np.arange(3)).reshape(len(fine.cells), -1)
    physical = np.einsum("qi,tij->tqj", bary, fine.points[fine.cells])
    stiffness = constitutive_values_3d(
        constitutive, physical.reshape(-1, 3), lame_lambda=lame_lambda, lame_mu=lame_mu
    ).reshape(*physical.shape[:2], 6, 6)
    strain = np.einsum("aij,tqnj->tqani", _KELVIN3, gradient).reshape(*gradient.shape[:2], 6, -1)
    blocks = np.einsum(
        "t,q,tqai,tqab,tqbj->tij",
        fine.volumes,
        weights,
        strain,
        stiffness,
        strain,
        optimize=True,
    )
    size = 3 * len(nodes)
    matrix = _assemble_blocks(blocks, udofs, size)
    mass = _assemble_blocks(
        np.einsum("t,q,qi,qj->tij", fine.volumes, weights, basis, basis), dofs, len(nodes)
    )
    force = vector_values_3d(source, physical.reshape(-1, 3)).reshape(physical.shape)
    load = np.bincount(
        udofs.ravel(),
        weights=np.einsum("t,q,qi,tqa->tia", fine.volumes, weights, basis, force).ravel(),
        minlength=size,
    )
    coupling = np.kron(tetra_trace_coupling(mesh, cell, fine, skeleton, degree), np.eye(3))
    center = mesh.points[mesh.cells[cell]].mean(axis=0)
    rigid = rigid_modes_3d(nodes, center).reshape(size, 6)
    moments = sparse.kron(mass, sparse.eye(3)) @ rigid
    global_center = mesh.volumes @ mesh.points[mesh.cells].mean(axis=1) / mesh.volumes.sum()
    global_moments = sparse.kron(mass, sparse.eye(3)) @ rigid_modes_3d(
        nodes, global_center
    ).reshape(size, 6)
    indices = (3 * skeleton.cell_dofs(cell)[:, None] + np.arange(3)).ravel()
    return LocalAssembly(
        LocalProblem(matrix, coupling, load, indices, rigid, moments), (fine, global_moments)
    )


@dataclass(frozen=True)
class Elasticity3DSolution:
    """Broken tetrahedral displacement and raw symmetric three-dimensional Cauchy stress.

    The scalar triangular skeleton defines the geometry and P1 face modes;
    ``hybrid.trace`` interleaves three negative-traction components per scalar
    skeleton coefficient. Raw stress is not automatically H(div)-conforming.
    """

    skeleton: TriangularSkeleton
    local_meshes: tuple[TetraMesh, ...]
    values: tuple[FloatArray, ...]
    hybrid: HybridSolution
    degree: int
    constitutive: Any
    lame_lambda: Any
    lame_mu: Any
    source: Any
    quadrature_order: int

    def evaluate(self, cell: int, bary: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Return displacement, full gradient and symmetric stress at barycentric points."""
        fine = self.local_meshes[cell]
        dofs, _, basis, gradient = tetra_tabulate(fine, self.degree, bary)
        values = self.values[cell][dofs]
        displacement = np.einsum("qi,tia->tqa", basis, values)
        grad = np.einsum("tqib,tia->tqab", gradient, values)
        physical = np.einsum("qi,tij->tqj", bary, fine.points[fine.cells])
        stiffness = constitutive_values_3d(
            self.constitutive,
            physical.reshape(-1, 3),
            lame_lambda=self.lame_lambda,
            lame_mu=self.lame_mu,
        )
        strain = np.einsum("aij,tqij->tqa", _KELVIN3, grad).reshape(-1, 6)
        sigma = np.einsum("nab,nb,aij->nij", stiffness, strain, _KELVIN3).reshape(
            *physical.shape[:2], 3, 3
        )
        return displacement, grad, sigma

    def errors(self, displacement: Any, stress: Any, order: int = 7) -> dict[str, float]:
        """Integrate displacement L2 and raw-stress Frobenius errors over the physical volume."""
        bary, weights = tetrahedron_quadrature(order)
        totals = np.zeros(2)
        for cell, fine in enumerate(self.local_meshes):
            physical = np.einsum("qi,tij->tqj", bary, fine.points[fine.cells]).reshape(-1, 3)
            u, _, sigma = self.evaluate(cell, bary)
            exact_u = vector_values_3d(displacement, physical).reshape(u.shape)
            raw = stress(physical) if callable(stress) else stress
            if np.iscomplexobj(raw) or not np.isfinite(raw).all():
                raise ValueError("exact stress must be finite and real")
            exact_sigma = np.broadcast_to(raw, (len(physical), 3, 3)).reshape(sigma.shape)
            totals[0] += fine.volumes @ (np.sum((u - exact_u) ** 2, axis=-1) @ weights)
            totals[1] += fine.volumes @ (
                np.sum((sigma - exact_sigma) ** 2, axis=(-1, -2)) @ weights
            )
        return dict(zip(("displacement_l2", "stress_l2"), np.sqrt(totals), strict=True))

    def equilibrium_residuals(self) -> FloatArray:
        """Return three force and three moment defects from body force and skeletal traction."""
        mesh = self.skeleton.mesh
        bary, weights = tetrahedron_quadrature(self.quadrature_order)
        fbary, fweights = triangle_quadrature(self.quadrature_order)
        result = np.zeros((len(mesh.cells), 6))
        for cell, fine in enumerate(self.local_meshes):
            center = mesh.points[mesh.cells[cell]].mean(axis=0)
            physical = np.einsum("qi,tij->tqj", bary, fine.points[fine.cells])
            force = vector_values_3d(self.source, physical.reshape(-1, 3)).reshape(physical.shape)
            result[cell] -= np.einsum(
                "t,q,tqa,tqak->k", fine.volumes, weights, force, rigid_modes_3d(physical, center)
            )
            for side, face in enumerate(mesh.cell_faces[cell]):
                partitions = self.skeleton.face_partition(int(face))
                subvertices = partitions @ mesh.points[mesh.faces[face]]
                points = np.einsum("qi,sij->sqj", fbary, subvertices)
                coefficients = np.array(
                    [
                        self.hybrid.trace.reshape(-1, 3)[
                            self.skeleton.subtriangle_dofs(int(face), segment)
                        ]
                        for segment in range(len(partitions))
                    ]
                )
                traction = np.einsum(
                    "qi,sia->sqa", self.skeleton.basis(int(face), fbary), coefficients
                )
                result[cell] += (
                    mesh.signs[cell, side]
                    * mesh.areas[face]
                    * np.einsum(
                        "s,q,sqa,sqak->k",
                        self.skeleton.face_weights(int(face)),
                        fweights,
                        traction,
                        rigid_modes_3d(points, center),
                    )
                )
        return result


def solve_elasticity_3d(
    mesh: TetraMesh,
    *,
    constitutive: Any = None,
    lame_lambda: Any = 1.0,
    lame_mu: Any = 1.0,
    source: Any = (0.0, 0.0, 0.0),
    dirichlet: Any = (0.0, 0.0, 0.0),
    traction: dict[int, Any] | None = None,
    skeleton: TriangularSkeleton | None = None,
    degree: int = 2,
    local_refinement: int = 2,
    quadrature_order: int = 6,
    rigid_moments: Any = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    solver: str = "scipy",
    local_solver: str = "scipy",
    local_refinement_precision: Literal["double", "extended"] = "double",
    backend: Literal["serial", "thread", "process"] = "serial",
    workers: int | None = None,
) -> Elasticity3DSolution:
    """Solve primal 3D elasticity with P1–P4 locals, P1 face traces and six rigid modes.

    Tensor inputs use Kelvin (6,6) or Cartesian (3,3,3,3) coordinates with
    major/minor symmetry and sampled positive definiteness. Pure traction fixes
    six integrated displacement moments: translations then rotations e_i cross
    (x-centroid). Face data are physical outward Cauchy traction. Dyadic face
    subdivisions must align with local tetrahedral refinements. The P1 trace
    contract contains rigid-motion traces, but arbitrary local choices still
    face a numerical rank check. Material interfaces require adequate fitted
    discretization/integration; no cut-tetrahedron rule is implied. This primal
    method has no uniform incompressible-limit guarantee and requires finite lambda.
    ``local_refinement_precision="extended"`` explicitly accumulates local defect
    corrections in extended precision while retaining the original residual criterion.
    """
    refinement = _dyadic(local_refinement, "local_refinement")
    tetra_nodal_space(mesh, degree)
    order = positive_int(quadrature_order, "quadrature_order", degree + 2)
    skeleton = TriangularSkeleton(mesh, degree=1) if skeleton is None else skeleton
    if skeleton.mesh is not mesh:
        raise ValueError("skeleton must belong to the tetrahedral mesh")
    if np.any(skeleton.degrees != 1):
        raise ValueError("3D elasticity requires P1 traces containing all rigid-motion traces")
    if np.any(skeleton.subdivisions > refinement):
        raise ValueError("local refinement must resolve every face subdivision")
    data = {} if traction is None else traction
    pure_traction = set(data) == set(mesh.boundary_faces)
    targets = np.asarray(rigid_moments)
    if pure_traction and (
        targets.shape != (6,) or np.iscomplexobj(targets) or not np.isfinite(targets).all()
    ):
        raise ValueError("rigid_moments must contain six finite real integrated moments")
    boundary, fixed = _boundary_vector(skeleton, dirichlet, data, order)
    factory = partial(
        _local,
        mesh=mesh,
        skeleton=skeleton,
        degree=degree,
        refinement=refinement,
        constitutive=constitutive,
        lame_lambda=lame_lambda,
        lame_mu=lame_mu,
        source=source,
        order=order,
    )
    system = HybridSystem.from_local_factory(
        factory,
        range(len(mesh.cells)),
        boundary_load=boundary,
        local_solver=local_solver,
        local_refinement_precision=local_refinement_precision,
        backend=backend,
        workers=workers,
    )
    gauges = []
    if pure_traction:
        gauges = [
            system.mean_constraint([m[1][:, i] for m in system.local_metadata], targets[i])
            for i in range(6)
        ]
    hybrid = system.solve(fixed=fixed, constraints=gauges, solver=solver)
    return Elasticity3DSolution(
        skeleton,
        tuple(m[0] for m in system.local_metadata),
        tuple(field.reshape(-1, 3) for field in hybrid.fields),
        hybrid,
        degree,
        constitutive,
        lame_lambda,
        lame_mu,
        source,
        order,
    )
