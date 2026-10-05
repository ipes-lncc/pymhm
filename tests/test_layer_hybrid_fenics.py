"""Independent native facet pairings and uncondensed layer-method equations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import linalg
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits

from pymhm.core.equations import Equation, LocalEquations
from pymhm.core.multiscale import MultiscaleProblem, assemble
from pymhm.fem.scalar.operators import boundary_data
from pymhm.fem.scalar.triangle import nodal_space, trace_coupling
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace
from pymhm.meshes.triangle import TriangleMesh

pytestmark = pytest.mark.fem


@dataclass(frozen=True)
class NativeBlocks:
    """Store independent native operators, physical moments and expected coordinates."""

    equations: LocalEquations
    matrix: np.ndarray
    load: np.ndarray
    coupling: np.ndarray
    mean: np.ndarray
    exact: np.ndarray | None
    velocity_dofs: np.ndarray | None = None


def _native_stack() -> tuple[Any, Any, Any, Any]:
    """Import the optional native stack only after checking its availability."""
    dolfinx = pytest.importorskip("dolfinx")
    import basix.ufl
    import ufl
    from mpi4py import MPI

    return dolfinx, basix.ufl, ufl, MPI


def _native_mesh(fine: TriangleMesh, stack: tuple[Any, Any, Any, Any]) -> Any:
    """Create a serial affine native mesh with the identical physical triangles."""
    dolfinx, basix_ufl, ufl, mpi = stack
    coordinate = ufl.Mesh(basix_ufl.element("Lagrange", "triangle", 1, shape=(2,)))
    return dolfinx.mesh.create_mesh(
        mpi.COMM_SELF, fine.cells.copy(), fine.points.copy(), coordinate
    )


def _native_matrix(form: Any, fem: Any) -> np.ndarray:
    """Assemble native test-row/trial-column coordinates without PyMHM compilation."""
    assembled = fem.assemble_matrix(fem.form(form))
    assembled.scatter_reverse()
    return assembled.to_scipy().toarray().copy()


def _native_vector(form: Any, fem: Any) -> np.ndarray:
    """Copy an independently assembled serial native load or physical moment."""
    return fem.assemble_vector(fem.form(form)).array.copy()


def _canonical_order(
    space: Any,
    fine: TriangleMesh,
    degree: int,
    *,
    components: int = 1,
    parent: Any = None,
) -> np.ndarray:
    """Map each canonical nodal component to its actual native coordinate."""
    _, nodes = nodal_space(fine, degree)
    native = space.tabulate_dof_coordinates()[:, :2]
    distance, indices = cKDTree(native).query(nodes)
    assert distance.max() < 1e-12
    assert len(np.unique(indices)) == len(nodes)
    order = (components * indices[:, None] + np.arange(components)).ravel()
    return order if parent is None else np.asarray(parent, dtype=np.int64)[order]


def _native_face_forms(
    domain: Any,
    macro: TriangleMesh,
    cell: int,
    test: Any,
    degree: int,
    components: int,
    stack: tuple[Any, Any, Any, Any],
) -> tuple[tuple[Any, ...], Any]:
    """Integrate traces using native outward normals and physical Legendre coordinates.

    The native normal dotted with the declared global normal supplies each
    incident orientation independently of the package's signed incidence map.
    P0 and P1 face bases are written as 1 and 2t-1, respectively. Vector modes
    are interleaved in Cartesian component order.
    """
    dolfinx, _, ufl, _ = stack
    entities, tags = [], []
    for side, face in enumerate(macro.cell_faces[cell]):
        start, end = macro.points[macro.faces[face]]
        tangent = end - start

        def on_face(
            coordinates: np.ndarray, origin: np.ndarray = start, edge: np.ndarray = tangent
        ) -> np.ndarray:
            """Identify this physical macro edge through its line equation."""
            return np.isclose(
                (coordinates[0] - origin[0]) * edge[1] - (coordinates[1] - origin[1]) * edge[0],
                0,
                atol=1e-13,
                rtol=0,
            )

        facets = dolfinx.mesh.locate_entities_boundary(domain, 1, on_face)
        entities.extend(facets)
        tags.extend([side + 1] * len(facets))
    assert len(set(entities)) == len(entities)
    order = np.argsort(entities)
    markers = dolfinx.mesh.meshtags(
        domain,
        1,
        np.asarray(entities, dtype=np.int32)[order],
        np.asarray(tags, dtype=np.int32)[order],
    )
    measure = ufl.Measure(
        "ds", domain=domain, subdomain_data=markers, metadata={"quadrature_degree": 10}
    )
    x, outward = ufl.SpatialCoordinate(domain), ufl.FacetNormal(domain)
    forms = []
    for side, face in enumerate(macro.cell_faces[cell]):
        start, end = macro.points[macro.faces[face]]
        tangent = end - start
        parameter = sum((x[axis] - start[axis]) * tangent[axis] for axis in range(2)) / (
            tangent @ tangent
        )
        orientation = ufl.dot(outward, ufl.as_vector(macro.normals[face]))
        basis = (1,) if degree == 0 else (1, 2 * parameter - 1)
        for mode in basis:
            for component in range(components):
                value = test if components == 1 else test[component]
                forms.append(orientation * mode * value * measure(side + 1))
    return tuple(forms), measure


def _scalar_blocks(
    macro: TriangleMesh,
    skeleton: SkeletonSpace,
    cell: int,
    *,
    epsilon: float,
    stabilized: bool,
    homogeneous: bool,
    stack: tuple[Any, Any, Any, Any],
) -> NativeBlocks:
    """Declare P1 reaction-diffusion forms and check their closed element moments."""
    dolfinx, _, ufl, _ = stack
    fine = macro.submesh(cell, 2)
    domain = _native_mesh(fine, stack)
    space = dolfinx.fem.functionspace(domain, ("Lagrange", 1))
    trial, test = ufl.TrialFunction(space), ufl.TestFunction(space)
    x = ufl.SpatialCoordinate(domain)
    exact = 1 + 0.7 * x[0] - 0.2 * x[1]
    force = 1 if homogeneous else exact
    dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 10})
    h = ufl.CellDiameter(domain)
    tau = (h**2 / 3) / (ufl.max_value(h**2 / 3, 2 * epsilon) + 2 * epsilon)
    a = (epsilon * ufl.inner(ufl.grad(trial), ufl.grad(test)) + trial * test) * dx
    load = force * test * dx
    if stabilized:
        residual = -epsilon * ufl.div(ufl.grad(trial)) + trial
        test_residual = -epsilon * ufl.div(ufl.grad(test)) + test
        a -= tau * residual * test_residual * dx
        load -= tau * force * test_residual * dx
    native_a = _native_matrix(a, dolfinx.fem)
    native_f = _native_vector(load, dolfinx.fem)
    canonical = _canonical_order(space, fine, 1)
    closed_a, closed_f = np.zeros_like(native_a), np.zeros_like(native_f)
    for nodes in fine.cells:
        vertices = fine.points[nodes]
        affine = np.column_stack((np.ones(3), vertices))
        gradient = np.linalg.inv(affine)[1:].T
        determinant = np.linalg.det((vertices[1:] - vertices[0]).T)
        area = abs(determinant) / 2
        mass = area / 12 * (np.ones((3, 3)) + np.eye(3))
        stiffness = area * gradient @ gradient.T
        diameter = np.linalg.norm(vertices[:, None] - vertices[None], axis=-1).max()
        scalar_tau = diameter**2 / 3 / (max(diameter**2 / 3, 2 * epsilon) + 2 * epsilon)
        effective = 1 - scalar_tau if stabilized else 1.0
        closed_a[np.ix_(nodes, nodes)] += epsilon * stiffness + effective * mass
        nodal_force = np.ones(3) if homogeneous else 1 + vertices @ np.array([0.7, -0.2])
        closed_f[nodes] += effective * mass @ nodal_force
    assert_allclose(native_a[np.ix_(canonical, canonical)], closed_a, atol=2e-15, rtol=1e-13)
    assert_allclose(native_f[canonical], closed_f, atol=2e-15, rtol=1e-13)
    forms, _ = _native_face_forms(domain, macro, cell, test, 0, 1, stack)
    native_b = np.column_stack([_native_vector(form, dolfinx.fem) for form in forms])
    coupling = np.zeros_like(native_b)
    coupling[canonical] = trace_coupling(macro, cell, fine, skeleton, 1)
    assert_allclose(coupling, native_b, atol=2e-14, rtol=1e-13)
    exact_values = None
    if not homogeneous:
        coordinates = space.tabulate_dof_coordinates()[:, :2]
        exact_values = 1 + coordinates @ np.array([0.7, -0.2])
    return NativeBlocks(
        LocalEquations(a=a, L=load, b=coupling, dofs=skeleton.cell_dofs(cell), c=-coupling.T),
        native_a,
        native_f,
        native_b,
        np.zeros(len(native_f)),
        exact_values,
    )


def _mixed_blocks(
    macro: TriangleMesh,
    skeleton: SkeletonSpace,
    cell: int,
    *,
    stabilized: bool,
    quadratic: bool,
    stack: tuple[Any, Any, Any, Any],
) -> NativeBlocks:
    """Declare vector-Laplacian TH/USFEM forms with independently derived forcing."""
    from basix import LagrangeVariant

    dolfinx, basix_ufl, ufl, _ = stack
    fine = macro.submesh(cell, 2)
    domain = _native_mesh(fine, stack)
    degree = 3 if quadratic else 2
    pressure_degree = degree if stabilized else degree - 1
    element = basix_ufl.mixed_element(
        [
            basix_ufl.element(
                "Lagrange",
                "triangle",
                degree,
                shape=(2,),
                lagrange_variant=LagrangeVariant.equispaced,
            ),
            basix_ufl.element(
                "Lagrange",
                "triangle",
                pressure_degree,
                lagrange_variant=LagrangeVariant.equispaced,
            ),
        ]
    )
    space = dolfinx.fem.functionspace(domain, element)
    velocity, velocity_map = space.sub(0).collapse()
    pressure, pressure_map = space.sub(1).collapse()
    u, p = ufl.TrialFunctions(space)
    v, q = ufl.TestFunctions(space)
    x = ufl.SpatialCoordinate(domain)
    nu, gamma, mean = 0.07, 1.6, 2.3
    exact_u = (
        ufl.as_vector((x[1] ** 2, -(x[0] ** 2))) if quadratic else (ufl.as_vector((x[1], -x[0])))
    )
    force = gamma * exact_u
    if quadratic:
        # Delta(y^2,-x^2)=(2,-2), grad(x-y+mean)=(1,-1).
        force += ufl.as_vector((1 - 2 * nu, -1 + 2 * nu))
    dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 12})
    a = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v))
        + gamma * ufl.inner(u, v)
        - p * ufl.div(v)
        - q * ufl.div(u)
    ) * dx
    load = ufl.inner(force, v) * dx
    if stabilized:
        # A declared conservative inverse bound is checked through native
        # polynomial Gram matrices below, rather than inferred from the patch.
        inverse_m = 1 / 1000 if quadratic else 1 / 100
        probe = TriangleMesh(fine.points[fine.cells[0]], np.array([[0, 1, 2]]))
        probe_domain = _native_mesh(probe, stack)
        scalar = dolfinx.fem.functionspace(probe_domain, ("Lagrange", degree))
        probe_dx = ufl.Measure("dx", domain=probe_domain, metadata={"quadrature_degree": 12})
        scalar_u, scalar_v = ufl.TrialFunction(scalar), ufl.TestFunction(scalar)
        energy = _native_matrix(
            ufl.inner(ufl.grad(scalar_u), ufl.grad(scalar_v)) * probe_dx, dolfinx.fem
        )
        strong = _native_matrix(
            ufl.div(ufl.grad(scalar_u)) * ufl.div(ufl.grad(scalar_v)) * probe_dx, dolfinx.fem
        )
        maximum = linalg.eigvalsh(strong[1:, 1:], energy[1:, 1:])[-1]
        diameter = fine.lengths[fine.cell_faces].max()
        assert inverse_m * diameter**2 * maximum < 1
        h = ufl.CellDiameter(domain)
        tau = h**2 / (ufl.max_value(gamma * h**2, 4 * nu / inverse_m) + 4 * nu / inverse_m)
        residual = -nu * ufl.div(ufl.grad(u)) + gamma * u + ufl.grad(p)
        test_residual = -nu * ufl.div(ufl.grad(v)) + gamma * v + ufl.grad(q)
        a -= tau * ufl.inner(residual, test_residual) * dx
        load -= tau * ufl.inner(force, test_residual) * dx
    native_a = _native_matrix(a, dolfinx.fem)
    native_f = _native_vector(load, dolfinx.fem)
    mean_weights = _native_vector(q * dx, dolfinx.fem)
    velocity_order = _canonical_order(velocity, fine, degree, components=2, parent=velocity_map)
    pressure_order = _canonical_order(pressure, fine, pressure_degree, parent=pressure_map)
    assert len(np.unique(np.r_[velocity_order, pressure_order])) == len(native_f)
    forms, _ = _native_face_forms(domain, macro, cell, v, int(quadratic), 2, stack)
    native_b = np.column_stack([_native_vector(form, dolfinx.fem) for form in forms])
    coupling = np.zeros_like(native_b)
    coupling[velocity_order] = np.kron(
        trace_coupling(macro, cell, fine, skeleton, degree), np.eye(2)
    )
    assert_allclose(coupling, native_b, atol=3e-14, rtol=2e-13)
    exact = np.zeros_like(native_f)
    velocity_coordinates = velocity.tabulate_dof_coordinates()[:, :2]
    power = 2 if quadratic else 1
    exact[np.asarray(velocity_map)] = np.column_stack(
        (velocity_coordinates[:, 1] ** power, -(velocity_coordinates[:, 0] ** power))
    ).ravel()
    pressure_coordinates = pressure.tabulate_dof_coordinates()[:, :2]
    exact[np.asarray(pressure_map)] = (
        mean + pressure_coordinates[:, 0] - pressure_coordinates[:, 1] if quadratic else mean
    )
    return NativeBlocks(
        LocalEquations(a=a, L=load, b=coupling, dofs=skeleton.cell_dofs(cell), c=-coupling.T),
        native_a,
        native_f,
        native_b,
        mean_weights,
        exact,
        np.asarray(velocity_map, dtype=np.int64),
    )


def _independent_kkt(
    blocks: Sequence[NativeBlocks],
    skeleton: SkeletonSpace,
    boundary: np.ndarray,
    mean: float | None,
) -> tuple[tuple[np.ndarray, ...], np.ndarray, float]:
    """Solve all native local fields, multiplier and pressure gauge without condensation."""
    offsets = np.r_[0, np.cumsum([len(block.load) for block in blocks])]
    a = linalg.block_diag(*(block.matrix for block in blocks))
    b = np.zeros((len(a), skeleton.size))
    for cell, block in enumerate(blocks):
        local = np.arange(offsets[cell], offsets[cell + 1])
        b[np.ix_(local, skeleton.cell_dofs(cell))] += block.coupling
    zeros = np.zeros((skeleton.size, skeleton.size))
    matrix = np.block([[a, b], [b.T, zeros]])
    rhs = np.r_[np.concatenate([block.load for block in blocks]), boundary]
    if mean is not None:
        moment = np.r_[np.concatenate([block.mean for block in blocks]), np.zeros(skeleton.size)]
        matrix = np.block([[matrix, moment[:, None]], [moment[None], np.zeros((1, 1))]])
        rhs = np.r_[rhs, mean]
    solution = linalg.solve(matrix, rhs, assume_a="sym")
    assert_allclose(matrix @ solution, rhs, atol=3e-13, rtol=1e-12)
    fields = tuple(solution[offsets[cell] : offsets[cell + 1]] for cell in range(len(blocks)))
    trace = solution[offsets[-1] : offsets[-1] + skeleton.size]
    return fields, trace, float(solution[-1]) if mean is not None else 0.0


def _compare_hybrid(
    blocks: Sequence[NativeBlocks],
    skeleton: SkeletonSpace,
    boundary: np.ndarray,
    mean: float | None,
) -> Any:
    """Compare complete physical rows and native KKT fields with the general problem API."""

    def provider(cell: int) -> LocalEquations:
        """Return the user-declared UFL local equations in deterministic cell order."""
        return blocks[cell].equations

    problem = MultiscaleProblem(
        Equation(0, -boundary),
        provider,
        range(len(blocks)),
        skeleton.size,
        (0,) * len(blocks),
    )
    system = assemble(problem)
    constraints = (
        [] if mean is None else [system.mean_constraint([block.mean for block in blocks], mean)]
    )
    solution = system.solve(constraints=constraints)
    native_fields, native_trace, rho = _independent_kkt(blocks, skeleton, boundary, mean)
    assert_allclose(solution.trace, native_trace, atol=2e-11, rtol=1e-11)
    trace_balance = np.zeros(skeleton.size)
    for cell, (block, field, expected) in enumerate(
        zip(blocks, solution.fields, native_fields, strict=True)
    ):
        assert_allclose(field, expected, atol=2e-11, rtol=1e-11)
        local_trace = solution.trace[skeleton.cell_dofs(cell)]
        residual = block.matrix @ field + block.coupling @ local_trace - block.load
        if block.velocity_dofs is None:
            assert_allclose(residual, 0, atol=2e-12, rtol=0)
        else:
            assert_allclose(residual[block.velocity_dofs], 0, atol=2e-12, rtol=0)
            pressure_dofs = np.setdiff1d(np.arange(len(field)), block.velocity_dofs)
            assert_allclose(residual[pressure_dofs], 0, atol=2e-12, rtol=0)
        np.add.at(trace_balance, skeleton.cell_dofs(cell), block.coupling.T @ field)
        if block.exact is not None:
            assert_allclose(field, block.exact, atol=2e-11, rtol=0)
    assert_allclose(trace_balance, boundary, atol=2e-12, rtol=0)
    if mean is not None:
        actual_mean = sum(
            block.mean @ field for block, field in zip(blocks, solution.fields, strict=True)
        )
        assert_allclose(actual_mean, mean, atol=2e-12, rtol=0)
        assert abs(rho) < 2e-12
        assert_allclose(solution.gauge_multipliers, 0, atol=2e-12, rtol=0)
    return system


@pytest.mark.parametrize("epsilon", [1e-3, 1e-5])
@pytest.mark.parametrize("stabilized", [False, True])
@pytest.mark.parametrize("homogeneous", [False, True])
def test_scalar_layer_native_closed_p1_and_full_hybrid(
    epsilon: float, stabilized: bool, homogeneous: bool
) -> None:
    """Verify published P1/P0 scalar spaces, full source and all exterior normal directions."""
    stack = _native_stack()
    with threadpool_limits(1):
        macro = TriangleMesh.unit_square()
        skeleton = SkeletonSpace(macro)
        assert set(map(tuple, macro.normals[macro.boundary_faces])) == {
            (0.0, -1.0),
            (0.0, 1.0),
            (-1.0, 0.0),
            (1.0, 0.0),
        }
        blocks = [
            _scalar_blocks(
                macro,
                skeleton,
                cell,
                epsilon=epsilon,
                stabilized=stabilized,
                homogeneous=homogeneous,
                stack=stack,
            )
            for cell in range(len(macro.cells))
        ]
        prescribed = 0.0 if homogeneous else lambda points: 1 + points @ np.array([0.7, -0.2])
        boundary, fixed = boundary_data(skeleton, prescribed, order=8)
        assert not fixed
        independent = np.zeros(skeleton.size)
        for face in macro.boundary_faces:
            vertices = macro.points[macro.faces[face]]
            independent[skeleton.dofs(int(face))] = (
                0
                if homogeneous
                else (macro.lengths[face] * (1 + vertices.mean(axis=0) @ np.array([0.7, -0.2])))
            )
        assert_allclose(boundary, independent, atol=2e-15, rtol=0)
        _compare_hybrid(blocks, skeleton, independent, None)


@pytest.mark.parametrize("stabilized", [False, True])
@pytest.mark.parametrize("quadratic", [False, True])
def test_brinkman_native_full_residual_pressure_and_uncondensed_gauge(
    stabilized: bool, quadratic: bool
) -> None:
    """Check pseudotractions, Laplacians, pressure gradients and a nonzero physical gauge."""
    stack = _native_stack()
    with threadpool_limits(1):
        macro = TriangleMesh.unit_square()
        skeleton = SkeletonSpace(
            macro, tuple(FaceSpace.uniform(int(quadratic)) for _ in macro.faces), components=2
        )
        blocks = [
            _mixed_blocks(
                macro, skeleton, cell, stabilized=stabilized, quadratic=quadratic, stack=stack
            )
            for cell in range(len(macro.cells))
        ]

        def prescribed(points: np.ndarray) -> np.ndarray:
            """Return the independently manufactured incompressible boundary velocity."""
            power = 2 if quadratic else 1
            return np.column_stack((points[:, 1] ** power, -(points[:, 0] ** power)))

        boundary, fixed = boundary_data(skeleton, prescribed, order=8)
        assert not fixed
        system = _compare_hybrid(blocks, skeleton, boundary, 2.3)
        pressure_offset_trace = np.zeros(skeleton.size)
        for face, space in enumerate(skeleton.faces):
            pressure_offset_trace[skeleton.dofs(face)] = np.outer(
                space.constant_coefficients(), macro.normals[face]
            ).ravel()
        for cell, (response, block) in enumerate(zip(system.responses, blocks, strict=True)):
            assert block.velocity_dofs is not None
            constant_pressure = np.ones(len(block.load))
            constant_pressure[block.velocity_dofs] = 0
            null_rows = (
                block.matrix @ constant_pressure
                + block.coupling @ (pressure_offset_trace[skeleton.cell_dofs(cell)])
            )
            assert_allclose(null_rows, 0, atol=2e-13, rtol=0)
            assert_allclose(response.problem.coupling, block.coupling, atol=3e-14, rtol=2e-13)
        assert_allclose(system.matrix @ pressure_offset_trace, 0, atol=2e-12, rtol=0)
