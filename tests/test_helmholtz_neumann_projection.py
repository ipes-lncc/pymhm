"""Preserve constant quadrature data in exact trace coordinates before restriction."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from examples.helmholtz_compact_family import CompactFamily
from examples.helmholtz_response_store import ResponseStore
from examples.helmholtz_trace_family import restrict_helmholtz_trace
from pymhm.helmholtz import _HelmholtzFactory, solve_helmholtz
from pymhm.helmholtz_spaces import helmholtz_skeleton
from pymhm.loads import split_point_sources
from pymhm.quadrilateral import CartesianMacroMesh


def constant_callback(points, normals):
    """Return exactly constant complex quadrature data through the callback API."""
    return np.full(len(points), 0.13 + 0.27j)


@pytest.mark.parametrize("prescribed", [0j, 0.13 + 0.27j, constant_callback])
def test_constant_neumann_coordinates_survive_every_nested_degree(prescribed, tmp_path):
    """Compare each reduced solve with direct assembly, memory and disk responses."""
    mesh = CartesianMacroMesh(2, 1)
    exterior = mesh.boundary_faces
    options = dict(
        omega=1.2,
        degree=3,
        local_refinement=2,
        quadrature_order=9,
        density=1.0,
        bulk_modulus=3.0,
        source=1.0 + 0.3j,
        dirichlet=0.2 + 0.4j,
        absorbing={int(exterior[0]): -0.2j},
        neumann={int(exterior[-1]): prescribed},
    )
    high = helmholtz_skeleton(mesh, 1.2, degree=4)
    original = solve_helmholtz(mesh, skeleton=high, **options)
    factory = _HelmholtzFactory(
        mesh,
        high,
        1.2,
        3,
        2,
        9,
        1.0,
        3.0,
        1.0 + 0.3j,
        split_point_sources(mesh, ()),
        0.2 + 0.4j,
        options["absorbing"],
        options["neumann"],
        None,
    )
    memory = CompactFamily.prepare(factory)
    store = ResponseStore.prepare(
        factory,
        tmp_path,
        sources={"owner": "constant-projection-test"},
        configuration={"omega": 1.2, "boundary": "constant Neumann with complex absorption"},
        batch_size=1,
    )
    disk = CompactFamily.from_locals(high, store)
    fixed = {k: v for item in memory.local for k, v in item.fixed.items()}
    values = np.array([fixed[dof] for dof in high.dofs(int(exterior[-1]))]).reshape(-1, 2)
    expected = 0j if prescribed == 0j else 0.13 + 0.27j
    assert_array_equal(values[0], [expected.real, expected.imag])
    assert_array_equal(values[1:], 0.0)
    for degree in range(5):
        reduced = restrict_helmholtz_trace(original, degree)
        direct = solve_helmholtz(mesh, skeleton=reduced.skeleton, **options)
        assert_allclose(reduced.trace, direct.trace, rtol=3e-11, atol=3e-12)
        first, second = memory.solve(degree), disk.solve(degree)
        assert_array_equal(first.trace, second.trace)
        assert_array_equal(first.pressure, second.pressure)
        for a, b, c in zip(reduced.solution.pressure, direct.pressure, first.pressure, strict=True):
            assert_allclose(a, b, rtol=3e-11, atol=3e-12)
            assert_allclose(a, c, rtol=3e-11, atol=3e-12)


@pytest.mark.parametrize("scale", [1.0, 1e-25])
def test_nonconstant_neumann_is_never_removed_by_a_coefficient_tolerance(scale):
    """A nonzero omitted polynomial mode remains inadmissible at either amplitude."""
    mesh = CartesianMacroMesh(1, 1)
    result = solve_helmholtz(
        mesh,
        omega=1.2,
        degree=3,
        local_refinement=2,
        absorbing=None,
        skeleton=helmholtz_skeleton(mesh, 1.2, degree=4),
        neumann={int(mesh.boundary_faces[0]): lambda x, n: scale * (x[:, 0] + x[:, 1])},
    )
    with pytest.raises(ValueError, match="Neumann trace"):
        restrict_helmholtz_trace(result, 0)


def test_constant_projection_uses_executed_oscillatory_coordinates():
    """Preserve the declared constant representation also for the oscillatory basis."""
    mesh = CartesianMacroMesh(1, 1)
    skeleton = helmholtz_skeleton(mesh, 1.2, degree=4, oscillatory=True)
    face = int(mesh.boundary_faces[0])
    factory = _HelmholtzFactory(
        mesh,
        skeleton,
        1.2,
        3,
        2,
        9,
        1.0,
        3.0,
        0j,
        split_point_sources(mesh, ()),
        0j,
        {},
        {face: constant_callback},
        None,
    )
    response = factory(0)
    prescribed = response.metadata[3]
    side = list(mesh.cell_faces[0]).index(face)
    offset = sum(skeleton.faces[int(f)].size for f in mesh.cell_faces[0][:side])
    actual = np.array([prescribed[offset + j] for j in range(skeleton.faces[face].size)])
    expected = (0.13 + 0.27j) * skeleton.faces[face].constant_coefficients()
    assert_array_equal(actual, expected)
    assert_allclose(
        skeleton.faces[face].evaluate(np.linspace(0, 1, 21)) @ actual,
        0.13 + 0.27j,
        rtol=2e-14,
        atol=2e-14,
    )


@pytest.mark.parametrize(
    "coefficients", [(0.13 + 0.27j,), (0.23 + 0.27j, 0.1), (0.19 + 0.27j, 0.1, 1 / 30)]
)
def test_declared_degree_and_oriented_segment_pullback(coefficients):
    """Match prescribed scalar, affine and quadratic traces with direct low-degree solves."""
    from pymhm import PolynomialNeumannTrace
    from pymhm.mesh import FaceSpace, SkeletonSpace

    data = PolynomialNeumannTrace(coefficients)
    mesh = CartesianMacroMesh(2, 1, (-0.7, 1.2, 0.3, 1.9))
    high_face = FaceSpace((0.0, 0.3, 1.0), (4, 4))
    high = SkeletonSpace(mesh, tuple(high_face for _ in mesh.faces), components=2)
    degree = len(coefficients) - 1
    for face in (int(mesh.boundary_faces[0]), int(mesh.boundary_faces[-1])):
        start, end = mesh.points[mesh.faces[face]]
        tangent = end - start

        def physical_callback(points, normals, start=start, tangent=tangent):
            """Independently recover the macroface coordinate from physical points."""
            return data.evaluate((points - start) @ tangent / (tangent @ tangent))

        options = dict(
            omega=1.2,
            degree=3,
            local_refinement=4,
            quadrature_order=10,
            source=0.2 + 0.7j,
            dirichlet=0.1 - 0.2j,
            absorbing=None,
        )
        prepared = solve_helmholtz(mesh, skeleton=high, neumann={face: data}, **options)
        sampled = solve_helmholtz(mesh, skeleton=high, neumann={face: physical_callback}, **options)
        reduced = restrict_helmholtz_trace(prepared, degree)
        direct = solve_helmholtz(mesh, skeleton=reduced.skeleton, neumann={face: data}, **options)
        assert_allclose(prepared.trace, sampled.trace, rtol=3e-10, atol=3e-11)
        assert_allclose(reduced.trace, direct.trace, rtol=3e-10, atol=3e-11)
        for a, b in zip(reduced.solution.pressure, direct.pressure, strict=True):
            assert_allclose(a, b, rtol=3e-10, atol=3e-11)
        exact_coordinates = data.coefficients_on(high_face).reshape(2, 5)
        assert_array_equal(exact_coordinates[:, degree + 1 :], 0)
        parameter = np.array([0.0, 0.11, 0.3, 0.67, 1.0])
        assert_allclose(
            high_face.evaluate(parameter) @ exact_coordinates.ravel(),
            data.evaluate(parameter),
            rtol=2e-15,
            atol=2e-15,
        )


def test_declared_tiny_omitted_mode_cannot_be_erased_by_constant_samples():
    """Keep an explicit mode even when adding it to a large datum rounds at every sample."""
    from pymhm import PolynomialNeumannTrace

    mesh = CartesianMacroMesh(1, 1)
    data = PolynomialNeumannTrace((1.0, 0.0, 1e-25))
    assert_array_equal(data.evaluate(np.linspace(0, 1, 13)), 1.0)
    solution = solve_helmholtz(
        mesh,
        omega=1.2,
        degree=3,
        local_refinement=2,
        absorbing=None,
        skeleton=helmholtz_skeleton(mesh, 1.2, degree=4),
        neumann={int(mesh.boundary_faces[0]): data},
    )
    with pytest.raises(ValueError, match="Neumann trace"):
        restrict_helmholtz_trace(solution, 1)
    with pytest.raises(ValueError, match="not represented"):
        solve_helmholtz(
            mesh,
            omega=1.2,
            absorbing=None,
            skeleton=helmholtz_skeleton(mesh, 1.2, degree=1),
            neumann={int(mesh.boundary_faces[0]): data},
        )


def test_declared_trace_pickle_spawn_and_public_exports():
    """Use the same exact polynomial input in independent spawn workers."""
    import pickle

    from pymhm import PolynomialNeumannTrace

    mesh = CartesianMacroMesh(2, 1)
    data = PolynomialNeumannTrace((0.2 + 0.1j, -0.03j))
    assert pickle.loads(pickle.dumps(data)) == data
    options = dict(
        omega=1.2,
        degree=3,
        local_refinement=2,
        absorbing=None,
        neumann={int(mesh.boundary_faces[-1]): data},
        skeleton=helmholtz_skeleton(mesh, 1.2, degree=2),
    )
    first = solve_helmholtz(mesh, **options)
    second = solve_helmholtz(mesh, backend="process", workers=2, **options)
    assert_array_equal(first.trace, second.trace)
    assert_array_equal(first.pressure, second.pressure)


@pytest.mark.parametrize("coefficients", [(), (np.nan,), (np.inf + 1j,), ((1.0,),)])
def test_declared_neumann_validation_rejects_invalid_coefficients(coefficients):
    """Reject nonfinite, empty or multidimensional coefficient data."""
    from pymhm import PolynomialNeumannTrace

    with pytest.raises(ValueError, match="finite nonempty"):
        PolynomialNeumannTrace(coefficients)


@pytest.mark.parametrize("points", [[-0.1], [1.1], [np.nan], [[0.2]], [0.2j]])
def test_declared_neumann_validation_rejects_invalid_parameter(points):
    """Reject evaluations outside the declared oriented macroface interval."""
    from pymhm import PolynomialNeumannTrace

    with pytest.raises(ValueError, match="face coordinates"):
        PolynomialNeumannTrace((1.0,)).evaluate(points)


def test_declared_trace_continuous_nodes_and_constant_oscillatory_basis():
    """Retain exact span contracts in both ordinary nodal and oscillatory bases."""
    from pymhm import PolynomialNeumannTrace
    from pymhm.helmholtz_spaces import OscillatoryFaceSpace
    from pymhm.mesh import FaceSpace

    data = PolynomialNeumannTrace((0.2 + 0.1j, -0.3, 0j, 0j))
    assert len(data.coefficients) == 2
    space = FaceSpace((0.0, 0.3, 1.0), (2, 3), continuous=True)
    points = np.linspace(0, 1, 21)
    assert_allclose(
        space.evaluate(points) @ data.coefficients_on(space),
        data.evaluate(points),
        rtol=2e-15,
        atol=2e-15,
    )
    wave = OscillatoryFaceSpace((0.0, 1.0), (3,), wave_number_length=1.2)
    with pytest.raises(ValueError, match="not represented"):
        data.coefficients_on(wave)
    constant = PolynomialNeumannTrace((0.2 + 0.1j,))
    assert_array_equal(constant.coefficients_on(wave), (0.2 + 0.1j) * wave.constant_coefficients())
