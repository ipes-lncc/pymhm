"""Storage-only condensation changes preserve original physical equations."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from examples.helmholtz_compact_family import CompactFamily, CompactLocal
from pymhm._legacy.models.waves.helmholtz import _HelmholtzFactory, solve_helmholtz
from pymhm.fem.loads import split_point_sources
from pymhm.fem.traces.helmholtz import helmholtz_skeleton
from pymhm.materials.cartesian import CartesianCellField
from pymhm.meshes.cartesian import CartesianMacroMesh


@pytest.mark.parametrize("boundary", ["absorbing", "dirichlet", "mixed"])
def test_compact_all_degrees_match_original_equations(boundary):
    mesh = CartesianMacroMesh(2, 1)
    skeleton = helmholtz_skeleton(mesh, 1.2, degree=4)
    exterior = list(mesh.boundary_faces)
    absorbing = (
        dict.fromkeys(exterior, 0.2 + 0.1j)
        if boundary == "absorbing"
        else {int(exterior[0]): -0.2j}
        if boundary == "mixed"
        else {}
    )
    neumann = {int(exterior[-1]): 0.0} if boundary == "mixed" else {}
    material = CartesianCellField(np.array([[1.0], [2.0], [4.0]]), (1 / 3, 1))
    points = ((0.5, 0.5, 0.1),)
    options = dict(
        omega=1.2,
        density=material,
        bulk_modulus=3.0,
        source=1 + 0.3j,
        dirichlet=0.2 + 0.4j,
        absorbing=absorbing,
        neumann=neumann,
        point_sources=points,
        degree=3,
        local_refinement=2,
        quadrature_order=9,
    )
    original = solve_helmholtz(mesh, skeleton=skeleton, **options)
    factory = _HelmholtzFactory(
        mesh,
        skeleton,
        1.2,
        3,
        2,
        9,
        material,
        3.0,
        1 + 0.3j,
        split_point_sources(mesh, points),
        0.2 + 0.4j,
        absorbing,
        neumann,
        None,
    )
    compact = CompactFamily.prepare(factory)
    assert_array_equal(compact.matrix.toarray(), original.system.matrix.toarray())
    assert_array_equal(compact.rhs, original.system.rhs)
    for item, response in zip(compact.local, original.system.responses, strict=True):
        assert_array_equal(item.coupling.toarray(), response.problem.coupling)
        assert_array_equal(item.source, response.source)
        assert_array_equal(item.lifts, response.lifts)
        assert item.storage_bytes > 0
    for degree in range(5):
        value = compact.solve(degree)
        trace = compact.solve_trace(degree)
        streamed = list(compact.reconstruct(trace.prepared_coordinates))
        assert_array_equal(trace.trace, value.trace)
        assert_array_equal([item[0] for item in streamed], value.pressure)
        assert_array_equal([item[1] for item in streamed], value.balance)
        assert max(item[2] for item in streamed) == value.local_residual_max
        assert trace.residual == value.residual
        direct = solve_helmholtz(
            mesh, skeleton=helmholtz_skeleton(mesh, 1.2, degree=degree), **options
        )
        assert_allclose(value.trace, direct.trace, rtol=2e-10, atol=2e-11)
        assert_allclose(value.pressure, direct.pressure, rtol=2e-10, atol=2e-11)
        assert value.residual < 1e-12
        assert value.local_residual_max < 1e-12
        assert value.original_trace_residual < 1e-12
        assert_allclose(value.balance, 0, atol=2e-12)


def test_compact_rejects_retained_modes_and_corrupted_responses():
    from pymhm.core.contracts import LocalProblem

    problem = LocalProblem(
        np.eye(2), np.eye(2), np.ones(2), np.arange(2), coarse_basis=np.ones((2, 1))
    )
    with pytest.raises(ValueError, match="retained"):
        CompactLocal.from_response(problem.condense(), (None, None, np.zeros(2), {}))
    problem = LocalProblem(np.eye(2), np.eye(2), np.ones(2), np.arange(2))
    item = CompactLocal.from_response(problem.condense(), (None, None, np.zeros(2), {}))
    bad = replace(item, source=item.source + 0.1)
    with pytest.raises(ValueError, match="original Helmholtz local"):
        bad.reconstruct(np.zeros(2))
    bad = replace(item, source=np.full_like(item.source, np.nan))
    with pytest.raises(ValueError, match="finite real interleaved"):
        bad.reconstruct(np.zeros(2))


def test_compact_spawn_preserves_arrays_and_fields():
    """Compare independent serial/spawn calculations with scaled roundoff bounds."""
    mesh = CartesianMacroMesh(2, 1)
    skeleton = helmholtz_skeleton(mesh, 1.2, degree=2)
    factory = _HelmholtzFactory(
        mesh,
        skeleton,
        1.2,
        3,
        2,
        8,
        1.0,
        2.0,
        1.0 + 0.4j,
        split_point_sources(mesh, ((0.5, 0.5, 0.1),)),
        0j,
        dict.fromkeys(mesh.boundary_faces, 0.0),
        {},
        None,
    )
    serial = CompactFamily.prepare(factory)
    parallel = CompactFamily.prepare(factory, backend="process", workers=2)
    first, second = serial.solve(1), parallel.solve(1)
    # Both execution modes limit native threads to one, but independently loaded
    # BLAS libraries can round differently. Local condition numbers are about
    # 660; the field bounds match the independent direct-solve checks above.
    # A unit scale is appropriate for this dimensionless, unit-amplitude source:
    # symmetry makes the RHS and trace nearly zero, so relative error alone is
    # meaningless. Exact response-copy/storage checks remain separate.
    for original, spawned, rtol, atol in (
        (serial.matrix.toarray(), parallel.matrix.toarray(), 2e-12, 2e-13),
        (serial.rhs, parallel.rhs, 2e-12, 2e-13),
        (first.trace, second.trace, 2e-10, 2e-11),
        (np.asarray(first.pressure), np.asarray(second.pressure), 2e-10, 2e-11),
    ):
        scale = max(1.0, float(np.max(np.abs(original))))
        assert_allclose(original, spawned, rtol=rtol, atol=atol * scale)
    for value in (first, second):
        assert value.residual < 1e-12
        assert value.local_residual_max < 1e-12
        assert value.original_trace_residual < 1e-12


def test_trace_only_solve_defers_reconstruction_and_checks_coordinate_contract(monkeypatch):
    """The global pass never loads fields, and the field pass consumes one at a time."""
    mesh = CartesianMacroMesh(2, 1)
    skeleton = helmholtz_skeleton(mesh, 1.2, degree=2)
    factory = _HelmholtzFactory(
        mesh,
        skeleton,
        1.2,
        3,
        2,
        8,
        1.0,
        2.0,
        1.0 + 0.4j,
        split_point_sources(mesh, ()),
        0j,
        dict.fromkeys(mesh.boundary_faces, 0.0),
        {},
        None,
    )
    compact = CompactFamily.prepare(factory)
    visited = []
    original = CompactLocal.reconstruct

    def observed(self, trace):
        visited.append(self)
        return original(self, trace)

    monkeypatch.setattr(CompactLocal, "reconstruct", observed)
    trace = compact.solve_trace(1)
    assert visited == []
    fields = compact.reconstruct(trace.prepared_coordinates)
    next(fields)
    assert len(visited) == 1
    next(fields)
    assert len(visited) == 2
    with pytest.raises(StopIteration):
        next(fields)
    for invalid in (trace.trace, np.zeros(skeleton.size - 1), np.full(skeleton.size, np.nan)):
        with pytest.raises(ValueError, match="finite real interleaved"):
            list(compact.reconstruct(invalid))


@pytest.mark.skipif(
    np.finfo(np.longdouble).eps >= np.finfo(float).eps,
    reason="executed correction digits require a wider native mantissa",
)
def test_global_assembly_keeps_executed_schur_and_load_correction_digits():
    """A representable correction below double epsilon survives the sparse assembly."""
    from pymhm.core.contracts import LocalProblem

    response = LocalProblem(np.eye(2), np.eye(2), np.ones(2), np.arange(2)).condense()
    original = CompactLocal.from_response(response, (None, None, np.zeros(2), {}))
    digits = np.array([1, 2], dtype=np.longdouble) + np.longdouble(2) ** -60
    item = replace(
        original, schur=np.diag(digits), rhs=digits, boundary=np.zeros(2, dtype=np.longdouble)
    )
    schur_bytes, lift_bytes = item.schur.tobytes(), item.lifts.tobytes()
    family = CompactFamily.from_locals(SimpleNamespace(size=2), (item,))
    assert family.matrix.dtype == np.dtype(np.longdouble)
    assert family.rhs.dtype == np.dtype(np.longdouble)
    assert_array_equal(family.matrix.diagonal(), digits, strict=True)
    assert_array_equal(family.rhs, digits, strict=True)
    assert np.all(family.matrix.diagonal() != digits.astype(float).astype(np.longdouble))
    assert item.schur.tobytes() == schur_bytes
    assert item.lifts.tobytes() == lift_bytes


def test_original_field_gate_detects_wrong_boundary_functional_after_trace_solve():
    """An unchanged Schur residual cannot certify a changed physical continuity load."""
    mesh = CartesianMacroMesh(2, 1)
    skeleton = helmholtz_skeleton(mesh, 1.2, degree=0)
    factory = _HelmholtzFactory(
        mesh,
        skeleton,
        1.2,
        3,
        2,
        8,
        1.0,
        1.0,
        1 + 0.2j,
        split_point_sources(mesh, ()),
        0.3j,
        {},
        {},
        None,
    )
    family = CompactFamily.prepare(factory)
    trace = family.solve_trace(0)
    fields = [item[0] for item in family.reconstruct(trace.prepared_coordinates)]
    assert family.verify_fields(trace, fields) < 1e-12
    changed = tuple(replace(item, boundary=item.boundary + 0.1) for item in family.local)
    stale = replace(family, local=changed)
    assert stale.solve_trace(0).residual == trace.residual
    with pytest.raises(ValueError, match="original Helmholtz trace"):
        stale.verify_fields(trace, fields)
    with pytest.raises(ValueError, match="finite"):
        family.verify_fields(trace, [np.full_like(field, np.nan) for field in fields])
