"""Slip walls and Cartesian component traction conditions with physical gauges."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from pymhm import FaceSpace, SkeletonSpace, TriangleMesh, solve_brinkman
from pymhm.elements import boundary_data


def channel_mesh():
    """Return a rectangular channel with vertical slip walls and height three."""
    reference = TriangleMesh.unit_square(2)
    return TriangleMesh(reference.points * [2.0, 3.0], reference.cells)


def side_conditions(mesh):
    """Select actual vertical and top boundary faces by their exact normals."""
    side = {int(f): {1: 0.0} for f in mesh.boundary_faces if mesh.normals[f, 0] != 0}
    top = {int(f): (0.0, 0.0) for f in mesh.boundary_faces if mesh.normals[f, 1] > 0}
    return side, top


@pytest.mark.parametrize(
    "method,degree,refinement",
    [
        ("usfem", 1, 4),
        ("usfem", 2, 2),
        ("usfem", 3, 1),
        ("taylor-hood", 2, 2),
        ("taylor-hood", 3, 1),
    ],
)
@pytest.mark.parametrize("top_traction", [False, True])
def test_constant_channel_flow_with_slip_and_physical_pressure(
    method, degree, refinement, top_traction
):
    """A pressure gradient drives exact vertical flow with free tangential side traction."""
    mesh = channel_mesh()
    slip, top = side_conditions(mesh)
    gamma = 2.4
    mean = 0.0 if top_traction else 2.3
    solution = solve_brinkman(
        mesh,
        viscosity=0.3,
        drag=gamma,
        source=0.0,
        dirichlet=(0.0, 1.0),
        traction=top if top_traction else None,
        traction_components=slip,
        skeleton=SkeletonSpace(mesh, tuple(FaceSpace(degrees=(1,)) for _ in mesh.faces), 2),
        formulation=method,
        degree=degree,
        local_refinement=refinement,
        mean_pressure=mean,
    )

    def pressure(points):
        """Return the pressure fixed either by top traction or by its physical mean."""
        return gamma * (3.0 - points[:, 1]) if top_traction else gamma * (1.5 - points[:, 1]) + mean

    assert solution.l2_error((0.0, 1.0)) < 3e-12
    assert solution.pressure_l2_error(pressure) < 4e-11
    assert solution.divergence_l2() < 2e-11
    for face in slip:
        dofs = solution.skeleton.dofs(face).reshape(-1, 2)
        assert_allclose(solution.hybrid.trace[dofs[:, 1]], 0, atol=1e-14)


@pytest.mark.parametrize(
    "method,degree", [("usfem", 1), ("usfem", 3), ("taylor-hood", 2), ("taylor-hood", 3)]
)
def test_nonzero_tangential_traction_uses_outward_physical_sign(method, degree):
    """Opposite side tractions support an exactly represented vertical shear."""
    mesh = channel_mesh()
    gamma, viscosity = 1.7, 0.3
    slip, top = side_conditions(mesh)
    traction = {face: {1: viscosity * mesh.normals[face, 0]} for face in slip}

    def velocity(points):
        """Return an incompressible affine shear with zero normal side velocity."""
        return np.column_stack((np.zeros(len(points)), points[:, 0]))

    def force(points):
        """Return the exact Brinkman force, since the affine viscous residual vanishes."""
        return gamma * velocity(points)

    result = solve_brinkman(
        mesh,
        viscosity=viscosity,
        drag=gamma,
        source=force,
        dirichlet=velocity,
        traction=top,
        traction_components=traction,
        formulation=method,
        degree=degree,
        local_refinement=4,
    )
    assert result.l2_error(velocity) < 3e-12
    assert result.pressure_l2_error(0.0) < 4e-11
    for face, components in traction.items():
        fixed = result.hybrid.trace[result.skeleton.dofs(face).reshape(-1, 2)[:, 1]]
        points = np.linspace(0, 1, 19)
        assert_allclose(
            result.skeleton.faces[face].evaluate(points) @ fixed, -components[1], atol=1e-14
        )


def test_component_neumann_projection_preserves_other_dirichlet_moments():
    """Project one quadratic Cartesian component independently on a nonuniform hp face."""
    mesh = TriangleMesh.unit_square()
    spaces = tuple(FaceSpace((0.0, 0.3, 1.0), (2, 3)) for _ in mesh.faces)
    skeleton = SkeletonSpace(mesh, spaces, 2)
    face = int(mesh.boundary_faces[0])
    start, end = mesh.points[mesh.faces[face]]
    tangent = end - start

    def prescribed(points):
        """Evaluate a quadratic trace in the face's canonical parameter."""
        t = (points - start) @ tangent / (tangent @ tangent)
        return 1 + t + t * t

    load, fixed = boundary_data(
        skeleton, (2.0, -3.0), neumann_components={face: {1: prescribed}}, order=8
    )
    dofs = skeleton.dofs(face).reshape(-1, 2)
    assert set(fixed) == set(dofs[:, 1])
    parameter, weights = spaces[face].quadrature(8)
    basis = spaces[face].evaluate(parameter)
    assert_allclose(load[dofs[:, 0]], 2 * mesh.lengths[face] * (weights @ basis), atol=1e-15)
    assert_array_equal(load[dofs[:, 1]], 0)
    coefficients = np.array([fixed[int(d)] for d in dofs[:, 1]])
    t = np.linspace(0, 1, 51)
    assert_allclose(spaces[face].evaluate(t) @ coefficients, 1 + t + t * t, atol=3e-15)


@pytest.mark.parametrize("components", [1, 2])
def test_component_maps_reproduce_full_neumann_without_evaluating_unused_dirichlet(components):
    """Treat complete component maps exactly like full prescribed flux fields."""
    mesh = TriangleMesh.unit_square()
    skeleton = SkeletonSpace(mesh, components=components)
    value = np.arange(1, components + 1)

    def unused(points):
        """Fail if a fully Neumann face incorrectly requests Dirichlet data."""
        raise AssertionError("unused Dirichlet data evaluated")

    component_data = {int(f): {i: value[i] for i in range(components)} for f in mesh.boundary_faces}
    full = {int(f): value[0] if components == 1 else value for f in mesh.boundary_faces}
    load, fixed = boundary_data(skeleton, unused, neumann_components=component_data)
    reference_load, reference_fixed = boundary_data(skeleton, unused, full)
    assert_array_equal(load, reference_load)
    assert fixed == reference_fixed


@pytest.mark.parametrize("component", [0, 1])
def test_prescribed_normal_component_removes_pressure_gauge(component):
    """A Cartesian traction component fixes pressure whenever its normal entry is nonzero."""
    mesh = TriangleMesh.unit_square()
    face = next(int(f) for f in mesh.boundary_faces if mesh.normals[f, component] != 0)
    with pytest.raises(ValueError, match="normal traction"):
        solve_brinkman(mesh, traction_components={face: {component: 0}}, mean_pressure=1.0)


def test_oblique_component_traction_fixes_pressure_even_if_normal_entry_is_small():
    """Do not turn a genuinely nonzero normal component into an artificial gauge."""
    mesh = TriangleMesh([[0.0, 0.0], [1.0, 1e-14], [0.0, 1.0]], [[0, 1, 2]])
    face = next(int(f) for f in mesh.boundary_faces if 0 < abs(mesh.normals[f, 0]) < 1e-12)
    with pytest.raises(ValueError, match="normal traction"):
        solve_brinkman(mesh, traction_components={face: {0: 0}}, mean_pressure=1.0)


@pytest.mark.parametrize("split", [False, True])
def test_free_translation_gauge_with_componentwise_or_full_traction(split):
    """Retain only the unresisted translation not fixed by any Dirichlet component."""
    mesh = TriangleMesh.unit_square()
    fields = {int(f): {1: 0.0} if split else {0: 0.0, 1: 0.0} for f in mesh.boundary_faces}
    result = solve_brinkman(
        mesh, drag=np.diag([1.0, 0.0]), traction_components=fields, mean_velocity=(0.0, 1.0)
    )
    assert result.l2_error((0.0, 1.0)) < 2e-12
    assert result.pressure_l2_error(0.0) < 2e-11
    if split:
        with pytest.raises(ValueError, match="Dirichlet boundary component"):
            solve_brinkman(mesh, traction_components=fields, translation_kernel=[[1.0], [0.0]])
        declared = solve_brinkman(
            mesh,
            drag=np.diag([1.0, 0.0]),
            traction_components=fields,
            mean_velocity=(0.0, 1.0),
            translation_kernel=[[0.0], [1.0]],
        )
        assert declared.l2_error((0.0, 1.0)) < 2e-12


@pytest.mark.parametrize("backend", ["thread", "process"])
def test_slip_channel_parallel_factories_match_serial(backend):
    """Keep component boundary projection identical across worker assembly backends."""
    mesh = channel_mesh()
    slip, top = side_conditions(mesh)
    data = dict(
        viscosity=0.3, drag=2.4, dirichlet=(0.0, 1.0), traction=top, traction_components=slip
    )
    serial = solve_brinkman(mesh, **data)
    parallel = solve_brinkman(mesh, backend=backend, workers=2, **data)
    for actual, expected in zip(parallel.values, serial.values, strict=True):
        assert_allclose(actual, expected, atol=2e-12)
    for actual, expected in zip(parallel.pressure, serial.pressure, strict=True):
        assert_allclose(actual, expected, atol=2e-11)


@pytest.mark.parametrize(
    "which,field",
    [
        ("traction", []),
        ("traction_components", []),
        ("traction_components", {0: []}),
        ("traction_components", {0: {}}),
        ("traction_components", {0: {2: 0}}),
        ("traction_components", {0: {-1: 0}}),
        ("traction_components", {0: {True: 0}}),
        ("traction_components", {0: {1.0: 0}}),
        ("traction_components", {0: {"x": 0}}),
        ("traction_components", {0: {1: np.nan}}),
        ("traction_components", {0: {1: 1j}}),
        ("traction_components", {0: {1: (1.0, 2.0)}}),
        ("traction", {True: (0.0, 0.0)}),
        ("traction_components", {1.5: {1: 0}}),
    ],
)
def test_invalid_component_boundary_contracts(which, field):
    """Reject ambiguous maps, component IDs and invalid physical scalar fields."""
    mesh = TriangleMesh.unit_square()
    if isinstance(field, dict) and 0 in field:
        field = {int(mesh.boundary_faces[0]): field[0]}
    with pytest.raises(ValueError):
        solve_brinkman(mesh, **{which: field})


def test_component_boundary_rejects_interior_faces_and_full_overlap():
    """Never prescribe conflicting traces or treat an interior interface as exterior."""
    mesh = TriangleMesh.unit_square()
    interior = int(np.flatnonzero(mesh.face_cells[:, 1] >= 0)[0])
    with pytest.raises(ValueError, match="boundary faces"):
        solve_brinkman(mesh, traction_components={interior: {1: 0}})
    face = int(mesh.boundary_faces[0])
    with pytest.raises(ValueError, match="overlap"):
        solve_brinkman(mesh, traction={face: (0, 0)}, traction_components={face: {1: 0}})
