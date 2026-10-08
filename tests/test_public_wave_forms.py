"""Original coefficient equations and physical patches for public wave recipes."""

import numpy as np
import pytest
from scipy import sparse
from scipy.sparse.linalg import spsolve

from examples.tutorial_elastodynamic_equations import advance, initialize, prepare
from examples.tutorial_helmholtz_equations import (
    acoustic_equations,
    acoustic_prescribed,
    acoustic_problem,
    recover_acoustic,
)
from pymhm.core.equations import Equation, LocalEquations, compile_form
from pymhm.core.multiscale import MultiscaleProblem, assemble
from pymhm.core.original import assemble_original_blocks
from pymhm.fem.assembly import assemble_element_blocks
from pymhm.fem.traces.helmholtz import PolynomialNeumannTrace, helmholtz_skeleton
from pymhm.linalg.complex import complexify_vector, realify_operator, realify_vector
from pymhm.meshes.cartesian import CartesianMacroMesh
from pymhm.meshes.tetrahedron import TetraMesh
from pymhm.meshes.triangle import TriangleMesh


def test_complex_nonsymmetric_four_blocks_match_original_complex_system():
    """Realification and condensation preserve independently signed complex B/C/D."""
    a = np.array([[3 + 0.2j, 0.4 + 0.1j], [-0.3j, 2 - 0.1j]])
    b = np.array([[1 + 0.2j], [-0.4 + 0.5j]])
    c = np.array([[0.6 - 0.7j, 0.9 + 0.3j]])
    d = np.array([[2 + 0.8j]])
    f, g = np.array([0.4 + 0.7j, -0.3 + 0.1j]), np.array([1 - 0.6j])

    def local(_):
        return LocalEquations(
            realify_operator(a),
            realify_vector(f),
            realify_operator(b),
            realify_operator(c),
            [0, 1],
            d=realify_operator(d),
            g=realify_vector(g),
        )

    result = assemble(MultiscaleProblem(Equation(0, 0), local, [0], 2, (0,))).solve()
    reference = np.linalg.solve(np.block([[a, b], [c, d]]), np.r_[f, g])
    np.testing.assert_allclose(
        complexify_vector(result.fields[0]), reference[:2], atol=1e-12, rtol=1e-10
    )
    np.testing.assert_allclose(
        complexify_vector(result.trace), reference[2:], atol=1e-12, rtol=1e-10
    )
    np.testing.assert_allclose(
        realify_operator(b) @ realify_vector(reference[2:]),
        realify_vector(b @ reference[2:]),
        atol=1e-12,
        rtol=1e-10,
    )
    np.testing.assert_array_equal(
        complexify_vector(realify_vector(np.array([], dtype=complex))), []
    )
    np.testing.assert_array_equal(realify_vector([2, -3]), [2, 0, -3, 0])
    wider = np.array([1], dtype=np.longdouble)
    assert realify_vector(wider).dtype == wider.dtype

    global_a, global_g = np.array([[0.3 + 0.2j]]), np.array([0.6 - 0.5j])
    system = assemble(
        MultiscaleProblem(
            Equation(realify_operator(global_a), realify_vector(global_g)),
            local,
            [0],
            2,
            (0,),
        )
    )
    original = assemble_original_blocks(system)
    expected_a = realify_operator(np.block([[a, b], [c, d + global_a]]))
    np.testing.assert_array_equal(original.matrix.toarray(), expected_a.toarray())
    np.testing.assert_array_equal(original.load, realify_vector(np.r_[f, g + global_g]))
    np.testing.assert_array_equal(original.local_offsets, [0, 4])
    assert original.trace_offset == 4
    assert not original.local_offsets.flags.writeable


def test_original_blocks_rejects_missing_original_layout():
    """Retained modes and recursive coordinates cannot silently flatten to a different model."""
    with pytest.raises(TypeError, match="MultiscaleSystem"):
        assemble_original_blocks(None)

    def retained(_):
        return LocalEquations([[0]], [0], [[1]], [[-1]], [0], kernel=[[1]], moments=[[1]])

    system = assemble(MultiscaleProblem(Equation(0, 0), retained, [0], 1, (1,)))
    with pytest.raises(ValueError, match="without retained"):
        assemble_original_blocks(system)

    def scalar(_):
        return LocalEquations([[1]], [0], [[1]], [[-1]], [0])

    inner = MultiscaleProblem(Equation(0, 0), scalar, [0], 1, (0,))

    def outer(_):
        return LocalEquations(inner, 0, [[1]], [[-1]], [0])

    nested = assemble(MultiscaleProblem(Equation(0, 0), outer, [0], 1, (0,)))
    with pytest.raises(ValueError, match="leaf equations"):
        assemble_original_blocks(nested)


@pytest.mark.parametrize("value", [2, [[1]], [np.nan], ["value"]])
def test_realification_rejects_invalid_vectors(value):
    with pytest.raises(ValueError, match="vector"):
        realify_vector(value)


@pytest.mark.parametrize("value", [1, [[1, 0]], [1], [1j, 0], [np.nan, 0], ["0", "0"]])
def test_complexification_rejects_invalid_coordinates(value):
    with pytest.raises(ValueError, match="vector"):
        complexify_vector(value)


@pytest.mark.parametrize("value", [1, [1, 0], [["1"]], [[np.inf]], sparse.csc_matrix([[np.nan]])])
def test_operator_realification_rejects_invalid_entries(value):
    with pytest.raises(ValueError, match="operator"):
        realify_operator(value)


def test_rectangular_element_scatter_preserves_declared_pairings_and_duplicates():
    """Independent element test/trial maps sum oriented complex contributions."""
    blocks = np.array([[[1j, 2, -1], [3, 1 + 1j, 4]], [[5, 6, 7], [8, 9, 10]]])
    tests, trials = np.array([[0, 2], [2, 2]]), np.array([[1, 0, 1], [2, 1, 0]])
    actual = assemble_element_blocks(blocks, tests, trials, (3, 4)).toarray()
    expected = np.zeros((3, 4), dtype=complex)
    for cell in range(2):
        for row in range(2):
            for column in range(3):
                expected[tests[cell, row], trials[cell, column]] += blocks[cell, row, column]
    np.testing.assert_array_equal(actual, expected)
    empty = assemble_element_blocks(
        np.empty((0, 2, 3)), np.empty((0, 2), int), np.empty((0, 3), int), (0, 0)
    )
    assert empty.shape == (0, 0)


@pytest.mark.parametrize(
    "defect",
    [
        "shape",
        "negative_size",
        "fractional_size",
        "test_rank",
        "trial_rank",
        "test_type",
        "trial_type",
        "cell_count",
        "block_shape",
        "values_type",
        "nonfinite",
        "negative_test",
        "negative_trial",
        "test_range",
        "trial_range",
    ],
)
def test_element_scatter_rejects_incompatible_forms(defect):
    blocks, tests, trials, shape = np.ones((1, 1, 1)), np.array([[0]]), np.array([[0]]), (1, 1)
    if defect == "shape":
        shape = (1,)
    elif defect == "negative_size":
        shape = (-1, 1)
    elif defect == "fractional_size":
        shape = (1.2, 1)
    elif defect == "test_rank":
        tests = np.array([0])
    elif defect == "trial_rank":
        trials = np.array([0])
    elif defect == "test_type":
        tests = tests.astype(float)
    elif defect == "trial_type":
        trials = trials.astype(float)
    elif defect == "cell_count":
        trials = np.zeros((2, 1), int)
    elif defect == "block_shape":
        blocks = np.ones((1, 2, 1))
    elif defect == "values_type":
        blocks = np.array([[["1"]]])
    elif defect == "nonfinite":
        blocks[0, 0, 0] = np.nan
    elif defect == "negative_test":
        tests[0, 0] = -1
    elif defect == "negative_trial":
        trials[0, 0] = -1
    elif defect == "test_range":
        tests[0, 0] = 1
    else:
        trials[0, 0] = 1
    with pytest.raises(ValueError, match="element blocks"):
        assemble_element_blocks(blocks, tests, trials, shape)


@pytest.mark.parametrize(
    "case", ["absorbing", "dirichlet", "neumann", "mixed", "pml", "point", "oscillatory"]
)
def test_acoustic_public_equations_match_uncondensed_original_system(case):
    """The declared physical boundary regimes match a direct original-block solve."""
    mesh = CartesianMacroMesh(2, 1)
    options = dict(omega=1.2, degree=2, local_refinement=2, source=1 + 0.3j, dirichlet=0.4 + 0.2j)
    face = int(mesh.boundary_faces[0])
    if case == "absorbing":
        options.update(absorbing=0.2j)
    elif case == "dirichlet":
        options.update(absorbing=None)
    elif case == "neumann":
        options.update(absorbing=None, neumann={int(f): 0.7 - 0.2j for f in mesh.boundary_faces})
    elif case == "mixed":
        options.update(
            absorbing={face: 0.1j},
            neumann={int(mesh.boundary_faces[-1]): PolynomialNeumannTrace([0.2, 0.1j])},
        )
    elif case == "pml":
        options.update(absorbing=None, pml_stretch=(1 + 0.2j, 1 + 0.3j))
    elif case == "point":
        options.update(point_sources=[[0.25, 0.4, 1]])
    else:
        options.update(skeleton=helmholtz_skeleton(mesh, 1.2, degree=1, oscillatory=True))
    problem = acoustic_problem(mesh, **options)
    system = assemble(problem)
    fixed = acoustic_prescribed(system)
    result = system.solve(fixed=fixed)
    equations = tuple(acoustic_equations(item) for item in problem.items)
    local_sizes = [compile_form(eq.a).shape[0] for eq in equations]
    volume_size = sum(local_sizes)
    original = sparse.lil_matrix((volume_size + problem.trace_size,) * 2)
    forcing = np.zeros(original.shape[0])
    offset = 0
    for eq, size in zip(equations, local_sizes, strict=True):
        ids = np.arange(offset, offset + size)
        trace = volume_size + eq.dofs
        original[np.ix_(ids, ids)] = compile_form(eq.a)
        original[np.ix_(ids, trace)] = compile_form(eq.b)
        original[np.ix_(trace, ids)] = compile_form(eq.c)
        forcing[ids] = compile_form(eq.L)
        np.add.at(forcing, trace, compile_form(eq.g))
        offset += size
    reference = np.zeros(original.shape[0])
    for index, value in fixed.items():
        reference[volume_size + index] = value
    free = np.setdiff1d(np.arange(len(reference)), volume_size + np.array(list(fixed), dtype=int))
    original = original.tocsc()
    reference[free] = spsolve(original[free][:, free], (forcing - original @ reference)[free])
    np.testing.assert_allclose(
        np.r_[*result.fields, result.trace], reference, atol=1e-12, rtol=1e-10
    )
    physical = recover_acoustic(problem, system, result)
    assert max(abs(physical.conservation_residuals())) < 1e-12
    assembled = assemble_original_blocks(system)
    np.testing.assert_allclose(
        assembled.matrix.toarray(), original.toarray(), atol=1e-12, rtol=1e-10
    )
    np.testing.assert_array_equal(assembled.load, forcing)


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("substeps", [1, 3])
def test_newmark_public_forms_reproduce_rigid_acceleration(dimension, substeps):
    """Physical force, traction orientation and local subcycling retain exact rigid motion."""
    mesh = TriangleMesh.unit_square() if dimension == 2 else TetraMesh.unit_cube()
    acceleration = np.arange(1, dimension + 1) * 0.2
    density = 1.3
    traction = {int(face): np.zeros(dimension) for face in mesh.boundary_faces}
    with prepare(
        mesh, time_step=0.02, local_substeps=substeps, density=density, traction=traction
    ) as data:
        state = initialize(data)
        for _ in range(3):
            state = advance(data, state, density * acceleration)
        for u, v in zip(state.displacement, state.velocity, strict=True):
            np.testing.assert_allclose(
                u.reshape(-1, dimension),
                np.broadcast_to(0.5 * state.time**2 * acceleration, u.reshape(-1, dimension).shape),
                atol=1e-12,
                rtol=1e-10,
            )
            np.testing.assert_allclose(
                v.reshape(-1, dimension),
                np.broadcast_to(state.time * acceleration, v.reshape(-1, dimension).shape),
                atol=1e-12,
                rtol=1e-10,
            )
        assert state.constraint_residual < 1e-12
