"""Native tetrahedral MHM Stokes, Brinkman and Oseen velocity-pressure problems."""

from dataclasses import dataclass
from functools import partial
from typing import Any, Literal

import numpy as np

from pymhm._legacy.models.flow.forms_3d import flow_contract_3d, tetra_flow_operators
from pymhm._legacy.models.transport.rad_3d import _boundary_tangent_3d
from pymhm.core.contracts import HybridSolution, LocalAssembly, LocalProblem
from pymhm.core.system import HybridSystem
from pymhm.core.validation import FloatArray, positive_int
from pymhm.core.validation import real_array as _real
from pymhm.fem.scalar.operators import triangle_quadrature
from pymhm.fem.scalar.tetrahedron import tetra_tabulate, tetrahedron_quadrature
from pymhm.fem.traces.triangle_3d import TriangularSkeleton, _boundary, tetra_trace_coupling
from pymhm.materials.evaluation import scalar_values_3d, vector_values_3d
from pymhm.meshes.tetrahedron import TetraMesh, _dyadic


def _component(points: FloatArray, *, datum: Any, component: int) -> FloatArray:
    """Evaluate one vector component through the shared three-dimensional validator."""
    return vector_values_3d(datum, points)[:, component]


def _boundary_flow(
    skeleton: TriangularSkeleton,
    dirichlet: Any,
    traction: dict[int, Any],
    components: dict[int, dict[int, Any]],
    order: int,
) -> tuple[FloatArray, dict[int, float]]:
    """Project physical outward traction and assemble all remaining velocity moments."""
    exterior = set(skeleton.mesh.boundary_faces)
    if not set(traction).issubset(exterior) or not set(components).issubset(exterior):
        raise ValueError("traction keys must identify exterior faces")
    if set(traction) & set(components):
        raise ValueError("full and componentwise traction must not overlap")
    for values in components.values():
        if (
            not isinstance(values, dict)
            or not values
            or any(
                isinstance(component, (bool, np.bool_)) or component not in (0, 1, 2)
                for component in values
            )
        ):
            raise ValueError(
                "traction_components needs nonempty dictionaries with components 0, 1 or 2"
            )
    load = np.zeros((skeleton.size, 3))
    fixed = {}
    for component in range(3):
        prescribed = {
            face: partial(_component, datum=value, component=component)
            for face, value in traction.items()
        }
        prescribed.update(
            {face: values[component] for face, values in components.items() if component in values}
        )
        load[:, component], scalar_fixed = _boundary(
            skeleton, partial(_component, datum=dirichlet, component=component), prescribed, order
        )
        fixed.update({3 * int(index) + component: -value for index, value in scalar_fixed.items()})
    return load.ravel(), fixed


def _local_flow(
    cell: int,
    *,
    mesh: TetraMesh,
    skeleton: TriangularSkeleton,
    refinement: int,
    degree: int,
    options: dict[str, Any],
) -> LocalAssembly:
    """Assemble a complete local mixed operator with three retained physical translations."""
    fine = mesh.submesh(cell, refinement)
    forms = tetra_flow_operators(fine, degree=degree, **options)
    nv, size = len(forms.velocity_nodes), len(forms.load)
    coupling = np.zeros((size, 3 * len(skeleton.cell_dofs(cell))))
    coupling[: 3 * nv] = np.kron(
        tetra_trace_coupling(mesh, cell, fine, skeleton, degree), np.eye(3)
    )
    translation = np.zeros((size, 3))
    translation[: 3 * nv] = np.tile(np.eye(3), (nv, 1))
    moments = np.zeros_like(translation)
    moments[: 3 * nv] = (forms.velocity_moments[:, None, None] * np.eye(3)).reshape(3 * nv, 3)
    beta = options["advection"]
    pure = np.all(forms.zero_columns) and not callable(beta) and not np.any(beta)
    retained = {"kernel": translation} if pure else {"coarse_basis": translation}
    indices = (3 * skeleton.cell_dofs(cell)[:, None] + np.arange(3)).ravel()
    problem = LocalProblem(
        forms.matrix, coupling, forms.load, indices, constraints=moments, **retained
    )
    pressure_moment = np.r_[np.zeros(3 * nv), forms.pressure_moments]
    return LocalAssembly(problem, (fine, forms, moments, pressure_moment))


@dataclass(frozen=True)
class Flow3DSolution:
    """Broken velocity, pressure and grad-grad pseudostress on tetrahedral local meshes.

    The scalar ``skeleton`` supplies face geometry and scalar modes. Its hybrid
    trace interleaves three canonical negative-traction coefficients per mode.
    Velocity and pressure are continuous only inside each macrocell. The raw
    pseudostress is not claimed to be globally H(div)-conforming.
    """

    skeleton: TriangularSkeleton
    local_meshes: tuple[TetraMesh, ...]
    velocity: tuple[FloatArray, ...]
    pressure: tuple[FloatArray, ...]
    hybrid: HybridSolution
    degree: int
    pressure_degree: int
    viscosity: float
    advection: Any
    formulation: str

    def _index(self, cell: int) -> int:
        """Validate a macrocell index before array access."""
        cell = positive_int(cell, "cell", 0)
        if cell >= len(self.local_meshes):
            raise ValueError("cell outside local meshes")
        return cell

    def evaluate(self, cell: int, bary: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Evaluate one-sided velocity and pressure on all fine tetrahedra of a macrocell."""
        cell = self._index(cell)
        fine = self.local_meshes[cell]
        udofs, _, ubasis, _ = tetra_tabulate(fine, self.degree, bary)
        pdofs, _, pbasis, _ = tetra_tabulate(fine, self.pressure_degree, bary)
        return np.einsum("qi,tia->tqa", ubasis, self.velocity[cell][udofs]), self.pressure[cell][
            pdofs
        ] @ pbasis.T

    def gradient(self, cell: int, bary: FloatArray) -> FloatArray:
        """Evaluate grad(u)[component, derivative] without averaging macro interfaces."""
        cell = self._index(cell)
        dofs, _, _, gradient = tetra_tabulate(self.local_meshes[cell], self.degree, bary)
        return np.einsum("tia,tqib->tqab", self.velocity[cell][dofs], gradient)

    def pseudostress(self, cell: int, bary: FloatArray) -> FloatArray:
        """Return -nu*grad(u)+p*I+u tensor beta/2, whose normal is the MHM multiplier."""
        velocity, pressure = self.evaluate(cell, bary)
        fine = self.local_meshes[cell]
        points = np.einsum("qi,tia->tqa", bary, fine.points[fine.cells])
        beta = vector_values_3d(self.advection, points.reshape(-1, 3)).reshape(velocity.shape)
        return (
            -self.viscosity * self.gradient(cell, bary)
            + pressure[..., None, None] * np.eye(3)
            + np.einsum("tqa,tqb->tqab", velocity, beta) / 2
        )

    def _error(self, exact: Any, kind: str, order: int) -> float:
        """Integrate one selected physical squared error using positive volume quadrature."""
        bary, weights = tetrahedron_quadrature(order)
        total = 0.0
        for cell, fine in enumerate(self.local_meshes):
            points = np.einsum("qi,tia->tqa", bary, fine.points[fine.cells]).reshape(-1, 3)
            if kind == "gradient":
                values = self.gradient(cell, bary)
                expected = _real(exact(points) if callable(exact) else exact, "exact gradient")
                expected = np.broadcast_to(expected, (len(points), 3, 3)).reshape(values.shape)
                difference = np.sum((values - expected) ** 2, axis=(-1, -2))
            elif kind == "velocity":
                values = self.evaluate(cell, bary)[0]
                expected = vector_values_3d(exact, points).reshape(values.shape)
                difference = np.sum((values - expected) ** 2, axis=-1)
            else:
                values = self.evaluate(cell, bary)[1]
                expected = scalar_values_3d(exact, points).reshape(values.shape)
                difference = (values - expected) ** 2
            total += float(fine.volumes @ (difference @ weights))
        return float(np.sqrt(total))

    def l2_error(self, exact: Any, order: int = 7) -> float:
        """Return the complete broken velocity L2 error."""
        return self._error(exact, "velocity", order)

    def pressure_l2_error(self, exact: Any, order: int = 7) -> float:
        """Return pressure L2 error with the explicitly imposed physical gauge."""
        return self._error(exact, "pressure", order)

    def h1_seminorm_error(self, exact_gradient: Any, order: int = 7) -> float:
        """Return the complete broken velocity-gradient L2 error."""
        return self._error(exact_gradient, "gradient", order)

    def divergence_l2(self, order: int = 7) -> float:
        """Integrate the raw pointwise velocity divergence; no H(div) recovery is implied."""
        bary, weights = tetrahedron_quadrature(order)
        return float(
            np.sqrt(
                sum(
                    float(
                        fine.volumes
                        @ (np.trace(self.gradient(cell, bary), axis1=-2, axis2=-1) ** 2 @ weights)
                    )
                    for cell, fine in enumerate(self.local_meshes)
                )
            )
        )


def _translation_trace_compatible(skeleton: TriangularSkeleton, beta: Any, order: int) -> bool:
    """Check that beta.n times a constant translation belongs to the declared face space.

    External tangency alone does not preserve a discrete translation kernel:
    its internal Robin traction must also be representable. The L2 projection
    is checked at the declared face integration points, with only floating-point
    cancellation allowed. This is not a certified bound for arbitrary callbacks
    between quadrature points.
    The weighted least-squares projection avoids squaring the basis condition
    number; a numerically unresolved polynomial basis is rejected.
    """
    bary, weights = triangle_quadrature(max(8, order, int(skeleton.degrees.max()) + 1))
    for face, ids in enumerate(skeleton.mesh.faces):
        vertices = skeleton.mesh.points[ids]
        partitions = skeleton.face_partition(int(face))
        triangles = vertices[0] + partitions[:, :, 1:] @ (vertices[1:] - vertices[0])
        points = triangles[:, :1] + np.einsum(
            "qi,sia->sqa", bary[:, 1:], triangles[:, 1:] - triangles[:, :1]
        )
        vectors = vector_values_3d(beta, points.reshape(-1, 3)).reshape(points.shape)
        normal = skeleton.mesh.normals[face]
        values = vectors @ normal
        if skeleton.continuous[face]:
            original_bary = np.einsum("qi,sij->sqj", bary, partitions)
            basis = skeleton.evaluate(int(face), original_bary.reshape(-1, 3))
            area_weights = (skeleton.face_weights(int(face))[:, None] * weights).ravel()
            weighted = np.sqrt(area_weights)
            coefficients, _, rank, _ = np.linalg.lstsq(
                weighted[:, None] * basis, weighted * values.ravel(), rcond=None
            )
            if rank != basis.shape[1]:
                return False
            difference = values - (basis @ coefficients).reshape(values.shape)
        else:
            basis = skeleton.basis(int(face), bary)
            weighted = np.sqrt(weights)
            projection, _, rank, _ = np.linalg.lstsq(
                weighted[:, None] * basis, weighted[:, None] * values.T, rcond=None
            )
            if rank != basis.shape[1]:
                return False
            coefficients = projection.T
            difference = values - coefficients @ basis.T
        scale = np.max(np.abs(vectors) @ np.abs(normal), axis=1)
        if np.any(np.max(np.abs(difference), axis=1) > 256 * np.finfo(float).eps * scale):
            return False
    return True


def _translation_constraints(
    system: HybridSystem,
    skeleton: TriangularSkeleton,
    traction: dict[int, Any],
    components: dict[int, dict[int, Any]],
    beta: Any,
    mean: Any,
    declared: Any,
    order: int,
) -> list[Any]:
    """Identify physically free unresisted translations, without absolute resistance cutoffs."""
    free = np.array(
        [
            all(
                face in traction or component in components.get(int(face), {})
                for face in skeleton.mesh.boundary_faces
            )
            for component in range(3)
        ]
    )
    eligible = np.any(free) and _boundary_tangent_3d(skeleton, beta, order)
    if declared is not None and not eligible:
        raise ValueError(
            "translation_kernel requires free traction directions and tangential advection"
        )
    value = vector_values_3d(mean, np.zeros((1, 3)))[0]
    if not eligible:
        if np.any(value):
            raise ValueError("mean_velocity is only available for free unresisted translations")
        return []
    if declared is None:
        structural = np.all([data[1].zero_columns for data in system.local_metadata], axis=0) & free
        nullspace = np.eye(3)[:, structural]
    else:
        nullspace = _real(declared, "translation_kernel")
        if (
            nullspace.ndim != 2
            or nullspace.shape[0] != 3
            or not 1 <= nullspace.shape[1] <= 3
            or np.linalg.matrix_rank(nullspace) != nullspace.shape[1]
        ):
            raise ValueError("translation_kernel must be an independent 3-by-r basis")
        if np.any(nullspace[~free] != 0):
            raise ValueError("translation_kernel changes a Dirichlet component")
        nullspace = np.linalg.qr(nullspace)[0]
        material = sum(data[1].resistance_moment for data in system.local_metadata)
        scale = sum(data[1].absolute_resistance_moment for data in system.local_metadata) @ np.abs(
            nullspace
        )
        if np.any(np.abs(material @ nullspace) > 128 * np.finfo(float).eps * scale):
            raise ValueError("translation_kernel is resisted by the material")
    if np.linalg.norm(value - nullspace @ (nullspace.T @ value)) > 1e-12 * max(
        1.0, np.linalg.norm(value)
    ):
        raise ValueError("mean_velocity may prescribe only unresisted translation directions")
    if nullspace.shape[1] and not _translation_trace_compatible(skeleton, beta, order):
        raise ValueError("translation gauge requires representable normal advection traces")
    volume = skeleton.mesh.volumes.sum()
    return [
        system.mean_constraint(
            [data[2] @ direction for data in system.local_metadata],
            float(direction @ value) * volume,
        )
        for direction in nullspace.T
    ]


def solve_flow_3d(
    mesh: TetraMesh,
    *,
    viscosity: float = 1.0,
    drag: Any = 0.0,
    advection: Any = (0.0, 0.0, 0.0),
    advection_divergence: Any = None,
    advection_bound: float | None = None,
    source: Any = (0.0, 0.0, 0.0),
    dirichlet: Any = (0.0, 0.0, 0.0),
    traction: dict[int, Any] | None = None,
    traction_components: dict[int, dict[int, Any]] | None = None,
    skeleton: TriangularSkeleton | None = None,
    degree: int | None = None,
    local_refinement: int | None = None,
    quadrature_order: int = 5,
    formulation: str = "taylor-hood",
    stabilization: Literal["tensor-2025", "minimum-2017", "pointwise-2017"] = "tensor-2025",
    gamma_min: float | None = None,
    mean_pressure: float = 0.0,
    mean_velocity: Any = (0.0, 0.0, 0.0),
    translation_kernel: Any = None,
    solver: str = "scipy",
    local_solver: str = "scipy",
    backend: Literal["serial", "thread", "process"] = "serial",
    workers: int | None = None,
) -> Flow3DSolution:
    """Solve native 3D Stokes/Brinkman/Oseen with continuous tetrahedral local spaces.

    Taylor--Hood uses velocity Pk/pressure P(k-1), k=2--4. USFEM and stabilized
    Oseen use Pk/Pk, k=1--4. Three constant velocity modes and their physical
    moments are retained even with arbitrarily small positive resistance.
    Local/skeletal dyadic partitions must align. Bernstein Pk triangular face modes
    are interleaved into three negative-pseudotraction components. The default
    P1 face trace uses local refinement four for P1 and two for higher degree.
    Inadequate discrete trace/local pairs are rejected by the numerical rank
    check; a successful residual is not a mathematical inf-sup certificate.

    Every unspecified exterior component has weak Dirichlet velocity.
    ``traction[face]`` prescribes (nu*grad(u)-p*I)n-beta.n*u/2; componentwise
    dictionaries prescribe only those Cartesian components. This is grad-grad
    pseudotraction, not symmetric Cauchy traction. A pressure mean is imposed
    exactly when no prescribed component has a nonzero outward normal component.
    Fully free unresisted translation directions use ``mean_velocity``; rotated
    resistance kernels can be supplied explicitly and are checked physically.
    Advection must be tangential on the exterior and its normal trace must be
    representable in every skeletal face space for translation gauges.

    USFEM stabilization follows the declared coefficient conventions of
    ``tetra_flow_operators``: tensor-2025 uses the sampled elementwise maximum
    eigenvalue, minimum-2017 an explicit global lower bound, and pointwise-2017
    the pointwise smallest eigenvalue. A global minimum alone does not certify
    coercivity in heterogeneous materials. Stabilized Oseen uses constant scalar
    resistance and an explicitly bounded advection field; its strict published
    coercivity hypothesis is drag-div(beta)/2>0. Viscosity is constant positive.
    Material discontinuities require fitted local tetrahedra; callbacks do not
    trigger cut integration. Assembly and condensation run in the chosen backend.
    """
    nu, beta, divergence, bound, minimum, k = flow_contract_3d(
        viscosity,
        drag,
        advection,
        advection_divergence,
        advection_bound,
        formulation,
        stabilization,
        gamma_min,
        degree,
    )
    refinement = _dyadic(
        (4 if k == 1 else 2) if local_refinement is None else local_refinement, "local_refinement"
    )
    order = max(positive_int(quadrature_order, "quadrature_order"), k + 2)
    skeleton = TriangularSkeleton(mesh, degree=1) if skeleton is None else skeleton
    if skeleton.mesh is not mesh:
        raise ValueError("skeleton must belong to the supplied tetrahedral mesh")
    if np.any(skeleton.subdivisions > refinement):
        raise ValueError("local refinement must resolve every skeleton subdivision")
    pressure_mean = _real(mean_pressure, "mean_pressure")
    if pressure_mean.ndim != 0:
        raise ValueError("mean_pressure must be a scalar")
    traction = {} if traction is None else traction
    components = {} if traction_components is None else traction_components
    boundary, fixed = _boundary_flow(skeleton, dirichlet, traction, components, order)
    normal_traction = bool(traction) or any(
        mesh.normals[face, component] != 0
        for face, values in components.items()
        for component in values
    )
    if normal_traction and pressure_mean != 0:
        raise ValueError("mean_pressure is only a gauge when no normal traction is prescribed")
    options = dict(
        viscosity=nu,
        drag=drag,
        advection=beta,
        advection_divergence=divergence,
        advection_bound=bound,
        source=source,
        formulation=formulation,
        stabilization=stabilization,
        gamma_min=minimum,
        order=order,
    )
    system = HybridSystem.from_local_factory(
        partial(
            _local_flow,
            mesh=mesh,
            skeleton=skeleton,
            refinement=refinement,
            degree=k,
            options=options,
        ),
        range(len(mesh.cells)),
        boundary_load=boundary,
        local_solver=local_solver,
        backend=backend,
        workers=workers,
    )
    gauges = []
    if not normal_traction:
        gauges.append(
            system.mean_constraint(
                [data[3] for data in system.local_metadata],
                float(pressure_mean) * mesh.volumes.sum(),
            )
        )
    gauges.extend(
        _translation_constraints(
            system, skeleton, traction, components, beta, mean_velocity, translation_kernel, order
        )
    )
    result = system.solve(solver=solver, fixed=fixed, constraints=gauges)
    counts = [len(data[1].velocity_nodes) for data in system.local_metadata]
    return Flow3DSolution(
        skeleton,
        tuple(data[0] for data in system.local_metadata),
        tuple(
            field[: 3 * nv].reshape(-1, 3) for field, nv in zip(result.fields, counts, strict=True)
        ),
        tuple(field[3 * nv :] for field, nv in zip(result.fields, counts, strict=True)),
        result,
        k,
        k - 1 if formulation == "taylor-hood" else k,
        nu,
        beta,
        formulation,
    )
