"""Interface transports preserve variational equations and native geometry owners."""

import pickle
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse

from pymhm import Equation, MultiscaleProblem, compile_local_equations, solve
from pymhm.core.spaces import (
    MeshHierarchy,
    TraceBinding,
    bind_interface,
    bind_local_equations,
    validate_trace_binding,
)
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace, interface_pairing
from pymhm.fem.traces.pressure_3d import PressureTraceSpace3D
from pymhm.fem.traces.triangle_3d import TriangularSkeleton
from pymhm.meshes.cartesian import CartesianMacroMesh
from pymhm.meshes.tetrahedron import TetraMesh
from pymhm.meshes.triangle import TriangleMesh


def _tetra_mesh():
    return TetraMesh(
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1.0]]),
        np.array([[0, 1, 2, 3], [1, 2, 3, 4]]),
    )


def test_nonorthogonal_independent_trace_bases_preserve_original_equations():
    """An independently constructed monolithic system checks fields and coefficients."""
    a = np.array([[4.0, -1, 0], [0, 3, 1], [-1, 0, 2]])
    b = np.array([[1.0, 2], [-1, 1], [2, 0]])
    c = np.array([[1.0, 0, -2], [0, 3, 1]])
    d = np.array([[6.0, 1], [-1, 5]])
    r = np.array([[1.0, 2], [0, -1]])
    s = np.array([[0.0, 2], [-1, 1]])
    binding = TraceBinding([1, 0], r, [0, 1], s, basis_id="declared nonorthogonal faces")
    expected_u = np.array([0.4, -0.8, 1.2])
    expected_trace = np.array([0.3, -0.2])
    trial = r @ expected_trace[binding.dofs]
    load = a @ expected_u + b @ trial
    boundary_load = c @ expected_u + d @ trial
    declaration = bind_local_equations(binding, a=a, L=load, b=b, c=c, d=d, g=boundary_load)
    result = solve(MultiscaleProblem(Equation(0, 0), lambda _: declaration, [0], 2, (0,)))
    assert_allclose(result.trace, expected_trace, atol=1e-11, rtol=1e-10)
    assert_allclose(result.fields[0], expected_u, atol=1e-11, rtol=1e-10)
    assert_allclose(a @ result.fields[0] + b @ (r @ result.trace[[1, 0]]), load, atol=1e-11)
    assert_allclose(c @ result.fields[0] + d @ (r @ result.trace[[1, 0]]), boundary_load)
    # Coordinate assembly is deliberately independent of bind_local_equations.
    original = np.zeros((5, 5))
    original[:3, :3] = a
    original[:3, 3 + binding.dofs] = b @ r
    original[3 + binding.test_dofs, :3] = s.T @ c
    original[np.ix_(3 + binding.test_dofs, 3 + binding.dofs)] = s.T @ d @ r
    rhs = np.r_[load, s.T @ boundary_load]
    assert_allclose(np.r_[result.fields[0], result.trace], np.linalg.solve(original, rhs))


def test_projection_maps_and_independent_support_embed_into_one_global_union():
    binding = TraceBinding(
        [3, 0], [[1.0, 2]], test_dofs=[1, 3, 2], test_map=[[2.0, 0, 1], [0, 3, 0]]
    )
    declaration = bind_local_equations(
        binding,
        a=[[2.0]],
        L=[1.0],
        b=[[4.0]],
        c=[[5.0], [6.0]],
        d=[[7.0], [8.0]],
        g=[9.0, 10.0],
    )
    compiled = compile_local_equations(declaration)
    assert_array_equal(compiled.problem.trace_dofs, [3, 0, 1, 2])
    assert_array_equal(compiled.problem.coupling, [[4, 8, 0, 0]])
    assert_array_equal(compiled.problem.test_coupling, [[-18, 0, -10, -5]])
    assert_array_equal(compiled.load, [30, 0, 18, 9])
    assert_array_equal(
        compiled.matrix, [[24, 48, 0, 0], [0, 0, 0, 0], [14, 28, 0, 0], [7, 14, 0, 0]]
    )


def test_signed_blocks_can_declare_global_coordinates_without_double_transport():
    binding = TraceBinding([0, 1], [[0.0, -2], [3, 1]])
    declaration = bind_local_equations(
        binding,
        a=sparse.eye(2),
        L=[1.0, 2.0],
        b=sparse.eye(2),
        c=sparse.eye(2) * 4,
        d=sparse.eye(2) * 5,
        g=[6.0, 7],
        coordinates="global",
        metadata={"orientation": "already canonical"},
    )
    assert sparse.isspmatrix_csc(declaration.a)
    assert_array_equal(declaration.b, np.eye(2))
    assert_array_equal(declaration.c, 4 * np.eye(2))
    assert_array_equal(declaration.d, 5 * np.eye(2))
    assert_array_equal(declaration.g, [6, 7])
    assert declaration.metadata == {"orientation": "already canonical"}


def test_retained_petrov_modes_and_moments_stay_in_the_declared_volume_basis():
    right, left = np.array([[1.0], [0]]), np.array([[-2.0], [1]])
    moments, test_moments = [[2.0], [1]], [[-1.0], [2]]
    declaration = bind_local_equations(
        TraceBinding([0], [[-1.0]]),
        a=[[0.0, 1], [0, 2]],
        L=[0.0, 0],
        b=[[1.0], [2]],
        c=[[3.0, 4]],
        kernel=right,
        left_kernel=left,
        moments=moments,
        test_moments=test_moments,
    )
    compiled = compile_local_equations(declaration)
    assert_array_equal(compiled.problem.kernel, right)
    assert_array_equal(compiled.problem.left_kernel, left)
    assert_array_equal(compiled.problem.constraints, moments)
    assert_array_equal(compiled.problem.test_constraints, test_moments)
    retained = replace(
        declaration, a=np.eye(2), kernel=None, left_kernel=None, coarse_basis=right, test_basis=left
    )
    assert_array_equal(compile_local_equations(retained).problem.test_basis, left)


def test_face_only_and_empty_interface_bindings_need_no_volume_or_optional_backend():
    binding = TraceBinding([2, 0], [[2.0, 1], [0, -1]])
    declaration = bind_local_equations(
        binding, a=np.zeros((0, 0)), L=[], b=0, c=0, d=np.eye(2), g=[1.0, 2]
    )
    assert_array_equal(declaration.d, [[4, 2], [2, 2]])
    assert_array_equal(declaration.g, [2, -1])
    empty = TraceBinding([])
    assert empty.trial_size == empty.test_size == 0
    assert validate_trace_binding(empty, 0) is empty
    plain = bind_local_equations(empty, a=[[2.0]], L=[4.0], b=0, c=0)
    assert compile_local_equations(plain).problem.coupling.shape == (1, 0)


def test_bindings_own_frozen_maps_and_identity_survives_independent_construction():
    dofs, matrix = np.array([1, 0]), np.array([[2.0, 1], [0, 1]])
    binding = TraceBinding(dofs, matrix, basis_id="executed Legendre basis", require_injective=True)
    identity = TraceBinding([1, 0], [[2.0, 1], [0, 1]], basis_id=binding.basis_id)
    assert binding.basis_digest == identity.basis_digest
    assert binding.test_map is binding.trial_map and binding.test_dofs is binding.dofs
    dofs[:] = 0
    matrix[:] = 3
    assert_array_equal(binding.dofs, [1, 0])
    assert_array_equal(binding.trial_map, [[2, 1], [0, 1]])
    assert (
        TraceBinding([1, 0], [[2.0, 1], [0, 1]], basis_id="another basis").basis_digest
        != binding.basis_digest
    )
    for value in (binding.dofs, binding.test_dofs, binding.trial_map, binding.test_map):
        assert not value.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        binding.trial_map[0, 0] = 2
    separate = TraceBinding([1, 0], matrix, test_dofs=[0, 1])
    assert_array_equal(separate.test_map, np.eye(2))
    restored = pickle.loads(pickle.dumps(binding))
    assert restored.basis_digest == binding.basis_digest
    assert restored.require_injective
    for value in (restored.dofs, restored.test_dofs, restored.trial_map, restored.test_map):
        assert not value.flags.writeable
    assert_array_equal(restored.trial_map, binding.trial_map)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"trial_map": [1.0]},
        {"trial_map": [[1, 2]]},
        {"trial_map": [[np.inf]]},
        {"trial_map": [[1j]]},
        {"test_map": [[1, 2]]},
        {"dofs": [-1]},
        {"dofs": [0, 0]},
        {"trial_map": [[0.0]], "require_injective": True},
        {"dofs": [0, 1], "trial_map": [[1.0, 2]], "require_injective": True},
    ],
)
def test_trace_bindings_reject_invalid_representation_data(kwargs):
    with pytest.raises(ValueError):
        TraceBinding(**({"dofs": [0]} | kwargs))


def test_binding_validation_rejects_stale_identity_and_global_range():
    with pytest.raises(TypeError, match="basis_id"):
        TraceBinding([0], basis_id=3)
    with pytest.raises(TypeError, match="boolean"):
        TraceBinding([0], require_injective=1)
    with pytest.raises(TypeError, match="TraceBinding"):
        validate_trace_binding(None, 1)
    with pytest.raises(ValueError, match="outside"):
        validate_trace_binding(TraceBinding([1]), 1)
    stale = TraceBinding([0])
    stale.trial_map.setflags(write=True)
    stale.trial_map[0, 0] = 2
    with pytest.raises(ValueError, match="stale"):
        validate_trace_binding(stale, 1)
    with pytest.raises(ValueError, match="stale"):
        bind_local_equations(stale, a=[[1]], L=[0], b=0, c=0)
    with pytest.raises(ValueError, match="stale"):
        pickle.dumps(stale)
    with pytest.raises(TypeError, match="TraceBinding"):
        bind_local_equations(None, a=[[1]], L=[0], b=0, c=0)
    with pytest.raises(ValueError, match="coordinates"):
        bind_local_equations(TraceBinding([0]), a=[[1]], L=[0], b=0, c=0, coordinates="native")
    with pytest.raises(ValueError, match="square"):
        bind_local_equations(TraceBinding([0]), a=[1], L=[0], b=0, c=0)


@pytest.mark.parametrize("geometry", ["triangle", "rectangle", "tetrahedron"])
def test_builtin_normal_binding_reuses_mesh_incidence_with_vector_or_partitioned_faces(geometry):
    if geometry == "triangle":
        mesh = TriangleMesh(
            np.array([[0.0, 0], [1, 0], [0, 1], [1, 1]]), np.array([[0, 1, 2], [1, 3, 2]])
        )
        skeleton = SkeletonSpace(
            mesh, tuple(FaceSpace.uniform(2, 2) for _ in mesh.faces), components=2
        )
    elif geometry == "rectangle":
        mesh = CartesianMacroMesh(2)
        skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(1, 2) for _ in mesh.faces))
    else:
        mesh = _tetra_mesh()
        skeleton = TriangularSkeleton(mesh, 2, degree=1)
    normal, value = bind_interface(skeleton, convention="normal"), bind_interface(skeleton)
    assert normal.mesh is mesh and normal.size == skeleton.size
    for cell in range(len(mesh.cells)):
        expected = np.concatenate(
            [
                np.full(len(skeleton.dofs(int(face))), mesh.signs[cell, side])
                for side, face in enumerate(mesh.cell_faces[cell])
            ]
        )
        binding = normal.binding(cell)
        assert_array_equal(binding.dofs, skeleton.cell_dofs(cell))
        assert_array_equal(binding.trial_map, np.diag(expected))
        assert_array_equal(value.binding(cell).trial_map, np.eye(len(binding.dofs)))
    with pytest.raises(ValueError, match="outside"):
        value.binding(len(mesh.cells))


def test_continuous_pressure_trace_adapter_keeps_shared_topological_nodes():
    skeleton = PressureTraceSpace3D(_tetra_mesh())
    adapter = bind_interface(skeleton)
    assert_array_equal(adapter.binding(1).dofs, skeleton.cell_dofs(1))
    with pytest.raises(TypeError, match="normal trace"):
        bind_interface(skeleton, convention="normal")


def test_basis_identity_distinguishes_equal_width_spaces_and_rejects_stale_source():
    macro = CartesianMacroMesh(1)
    segmented = SkeletonSpace(macro, tuple(FaceSpace.uniform(2, 2) for _ in macro.faces))
    unpartitioned = SkeletonSpace(macro, tuple(FaceSpace.uniform(5) for _ in macro.faces))
    first, second = bind_interface(segmented).binding(0), bind_interface(unpartitioned).binding(0)
    assert first.trial_size == second.trial_size
    assert first.basis_digest != second.basis_digest
    source = TriangularSkeleton(_tetra_mesh())
    adapter = bind_interface(source)
    source.degrees[0] = 1
    with pytest.raises(ValueError, match="stale"):
        adapter.binding(0)
    source = TriangularSkeleton(_tetra_mesh())
    adapter = bind_interface(source)
    source.offsets[1] += 1
    with pytest.raises(ValueError, match="stale"):
        adapter.binding(0)


def test_builtin_adapter_rejects_ambiguous_face_support_and_invalid_incidence():
    with pytest.raises(ValueError, match="convention"):
        bind_interface(None, convention="tangential")
    with pytest.raises(TypeError, match="mesh-associated"):
        bind_interface(SimpleNamespace(size=1))
    mesh = SimpleNamespace(cell_faces=[[0, 1]], signs=np.array([[1, -1]]))
    space = SimpleNamespace(mesh=mesh, size=2, cell_dofs=lambda cell: [0, 1], dofs=lambda face: [0])
    with pytest.raises(ValueError, match="independent"):
        bind_interface(space, convention="normal").binding(0)
    space.dofs = lambda face: [0] if face == 0 else [2]
    with pytest.raises(ValueError, match="cell-owned"):
        bind_interface(space, convention="normal").binding(0)
    space.dofs = lambda face: [face]
    mesh.signs[0, 1] = 0
    with pytest.raises(ValueError, match="incidence"):
        bind_interface(space, convention="normal").binding(0)
    mesh.signs[0, 1] = -1
    space.dofs = lambda face: []
    with pytest.raises(ValueError, match="cover"):
        bind_interface(space, convention="normal").binding(0)


def test_mesh_hierarchy_constructs_local_meshes_lazily_and_retains_order():
    macro = CartesianMacroMesh(2, 1)
    calls = []
    hierarchy = MeshHierarchy(
        macro, lambda cell: calls.append(cell) or macro.submesh(cell, 2), items=(1, 0)
    )
    assert calls == [] and hierarchy.items == (1, 0)
    assert hierarchy.local_mesh(1).bounds == macro.submesh(1, 2).bounds
    assert calls == [1]
    supplied = [macro.submesh(cell, 2) for cell in range(2)]
    direct = MeshHierarchy(macro, supplied)
    supplied[0] = None
    assert direct.items == (0, 1) and direct.local_mesh(0) is not None
    with pytest.raises(ValueError, match="selected"):
        hierarchy.local_mesh(2)
    with pytest.raises(TypeError, match="topology"):
        MeshHierarchy(None, [])
    with pytest.raises(ValueError, match="outside"):
        MeshHierarchy(macro, [], items=(2,))
    with pytest.raises(ValueError, match="one local mesh"):
        MeshHierarchy(macro, [])


def test_interface_pairing_preserves_partitioned_shared_and_reordered_coordinates():
    """Boundary mass integrates both polynomial spaces before any coordinate transport."""
    from pymhm.methods.three_field import PressureTraceSpace

    macro = TriangleMesh.unit_square()
    left = SkeletonSpace(macro, tuple(FaceSpace((0, 0.3, 1), (3, 9)) for _ in macro.faces))
    right = PressureTraceSpace(
        macro, tuple(FaceSpace((0, 0.5, 1), (8, 2), True) for _ in macro.faces)
    )
    mass = interface_pairing(left, right, 0, order=1)
    assert_allclose(mass, interface_pairing(left, right, 0, order=24), atol=2e-13, rtol=2e-11)
    assert_allclose(mass.T, interface_pairing(right, left, 0), atol=2e-13, rtol=2e-11)
    constant = np.concatenate(
        [left.faces[face].constant_coefficients() for face in macro.cell_faces[0]]
    )
    perimeter = macro.lengths[macro.cell_faces[0]].sum()
    assert_allclose(constant @ mass @ np.ones(mass.shape[1]), perimeter, atol=2e-13)
    shared_mass = interface_pairing(right, right, 0)
    assert_allclose(
        np.ones(len(shared_mass)) @ shared_mass @ np.ones(len(shared_mass)), perimeter, atol=2e-13
    )
    assert np.linalg.eigvalsh(shared_mass).min() > 0
    reordered = SimpleNamespace(
        mesh=macro,
        faces=right.faces,
        size=right.size,
        face_dofs=right.face_dofs,
        cell_dofs=lambda cell: right.cell_dofs(cell)[::-1],
    )
    assert_allclose(interface_pairing(left, reordered, 0), mass[:, ::-1], atol=2e-13, rtol=2e-11)
    simple = SkeletonSpace(macro)
    vector = SkeletonSpace(macro, components=2)
    assert_allclose(
        interface_pairing(vector, vector, 0),
        np.kron(interface_pairing(simple, simple, 0), np.eye(2)),
    )
    # Normal incidence belongs to the explicit binding, not the mass integration.
    signed = bind_interface(left, convention="normal").binding(0)
    assert_allclose(interface_pairing(left, right, 0), mass, atol=2e-13, rtol=2e-11)
    assert_allclose(
        signed.trial_map @ constant,
        np.repeat(
            macro.signs[0], [face.size for face in (left.faces[f] for f in macro.cell_faces[0])]
        )
        * constant,
    )
    transform = np.eye(mass.shape[1])
    transform[0, 1] = 0.7
    binding = TraceBinding(right.cell_dofs(0), transform, basis_id="nonorthogonal pressure trace")
    declaration = bind_local_equations(binding, a=np.eye(mass.shape[0]), L=0, b=mass, c=mass.T)
    assert_allclose(declaration.b, mass @ transform)
    assert_allclose(declaration.c, transform.T @ mass.T)


def test_interface_pairing_rejects_unsupported_geometry_and_inconsistent_support():
    """Undeclared bases and invalid scatter maps fail before any boundary integration."""
    macro = TriangleMesh.unit_square()
    valid = SkeletonSpace(macro)
    with pytest.raises(TypeError, match="mesh-associated"):
        interface_pairing(object(), valid, 0)
    with pytest.raises(TypeError, match="mesh-associated"):
        interface_pairing(SimpleNamespace(mesh=object()), valid, 0)
    with pytest.raises(ValueError, match="same macro"):
        interface_pairing(valid, SkeletonSpace(TriangleMesh.unit_square()), 0)
    volume = TriangularSkeleton(_tetra_mesh())
    with pytest.raises(TypeError, match="planar"):
        interface_pairing(volume, volume, 0)
    with pytest.raises(ValueError, match="outside"):
        interface_pairing(valid, valid, len(macro.cells))
    with pytest.raises(ValueError, match="quadrature order"):
        interface_pairing(valid, valid, 0, order=0)
    with pytest.raises(ValueError, match="component count"):
        interface_pairing(valid, SkeletonSpace(macro, components=2), 0)
    attributes = dict(
        mesh=macro, faces=valid.faces, size=valid.size, cell_dofs=valid.cell_dofs, dofs=valid.dofs
    )
    for changes, error, message in (
        ({"faces": ()}, TypeError, "FaceSpace"),
        ({"faces": (object(),) * len(macro.faces)}, TypeError, "FaceSpace"),
        ({"cell_dofs": None}, TypeError, "FaceSpace"),
        ({"dofs": None}, TypeError, "FaceSpace"),
        ({"cell_dofs": lambda cell: [0.0]}, ValueError, "integer"),
        ({"cell_dofs": lambda cell: [0, 0]}, ValueError, "integer"),
        ({"cell_dofs": lambda cell: [-1]}, ValueError, "integer"),
        ({"cell_dofs": lambda cell: [[0]]}, ValueError, "integer"),
        ({"cell_dofs": lambda cell: [valid.size]}, ValueError, "integer"),
        ({"dofs": lambda face: [0, 1]}, ValueError, "face basis width"),
        ({"cell_dofs": lambda cell: [0]}, ValueError, "cell support"),
        ({"cell_dofs": lambda cell: np.arange(valid.size)}, ValueError, "cover"),
    ):
        with pytest.raises(error, match=message):
            interface_pairing(SimpleNamespace(**(attributes | changes)), valid, 0)
    lengths = macro.lengths.copy()
    lengths[macro.cell_faces[0, 0]] = np.nan
    invalid_mesh = SimpleNamespace(
        points=macro.points,
        faces=macro.faces,
        cell_faces=macro.cell_faces,
        cells=macro.cells,
        lengths=lengths,
    )
    invalid_space = SimpleNamespace(**(attributes | {"mesh": invalid_mesh}))
    with pytest.raises(ValueError, match="face lengths"):
        interface_pairing(invalid_space, invalid_space, 0)
    del invalid_mesh.lengths
    with pytest.raises(TypeError, match="physical face lengths"):
        interface_pairing(invalid_space, invalid_space, 0)
