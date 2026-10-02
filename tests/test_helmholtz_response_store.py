"""Persistent responses preserve operators, physical fields and resume contracts."""

import json
from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import examples.helmholtz_response_store as owner
from examples.helmholtz_compact_family import CompactFactory, CompactFamily
from examples.helmholtz_response_store import ResponseStore
from pymhm.helmholtz import _HelmholtzFactory
from pymhm.helmholtz_spaces import helmholtz_skeleton
from pymhm.loads import split_point_sources
from pymhm.quadrilateral import CartesianMacroMesh
from pymhm.reservoir import CartesianCellField


@pytest.fixture
def factory():
    mesh = CartesianMacroMesh(2, 1)
    skeleton = helmholtz_skeleton(mesh, 1.2, degree=4)
    material = CartesianCellField(np.array([[1.0], [2.0], [4.0]]), (1 / 3, 1))
    exterior = mesh.boundary_faces
    return _HelmholtzFactory(
        mesh,
        skeleton,
        1.2,
        3,
        2,
        9,
        material,
        3.0,
        1 + 0.3j,
        split_point_sources(mesh, ((0.5, 0.5, 0.1),)),
        0.2 + 0.4j,
        {int(exterior[0]): -0.2j},
        {int(exterior[-1]): 0.0},
        None,
    )


def prepare(factory, path, **kwargs):
    return ResponseStore.prepare(
        factory,
        path,
        sources={"operator": "checked-by-test"},
        configuration={"space": "Q3/P4", "quadrature": 9},
        batch_size=1,
        **kwargs,
    )


@pytest.mark.parametrize("homogeneous", [False, True])
@pytest.mark.parametrize(
    "precision",
    [
        "double",
        pytest.param(
            "extended",
            marks=pytest.mark.skipif(
                np.finfo(np.longdouble).eps >= np.finfo(float).eps,
                reason="extended local correction requires a wider native NumPy type",
            ),
        ),
    ],
)
def test_streamed_responses_match_every_buffer_and_all_five_fields(
    tmp_path, factory, homogeneous, precision
):
    if homogeneous:
        factory = replace(factory, dirichlet=0j, absorbing=dict.fromkeys(factory.absorbing, 0j))
    expected = CompactFamily.prepare(factory, local_refinement_precision=precision)
    store = prepare(factory, tmp_path, local_refinement_precision=precision)
    assert store.storage_bytes == sum(item.storage_bytes for item in expected.local)
    assert store.record["complete"]
    for actual, original in zip(store, expected.local, strict=True):
        for name in owner._ARRAYS:
            assert_array_equal(getattr(actual, name), getattr(original, name), strict=True)
        assert actual.fixed == original.fixed
        if precision == "extended":
            assert actual.source.dtype == np.dtype(np.longdouble)
            assert actual.lifts.dtype == np.dtype(np.longdouble)
        for name in ("matrix", "coupling"):
            first, second = getattr(actual, name), getattr(original, name)
            assert first.format == second.format
            for key in ("data", "indices", "indptr"):
                assert_array_equal(getattr(first, key), getattr(second, key), strict=True)
    streamed = CompactFamily.from_locals(factory.skeleton, store)
    for key in ("data", "indices", "indptr"):
        assert_array_equal(
            getattr(streamed.matrix, key), getattr(expected.matrix, key), strict=True
        )
    assert_array_equal(streamed.rhs, expected.rhs, strict=True)
    for degree in range(5):
        first, second = streamed.solve(degree), expected.solve(degree)
        assert_array_equal(first.trace, second.trace, strict=True)
        assert_array_equal(first.pressure, second.pressure, strict=True)
        assert_array_equal(first.balance, second.balance, strict=True)
        assert first.residual == second.residual
        assert first.local_residual_max == second.local_residual_max


def test_spawn_store_preserves_serial_responses(tmp_path, factory):
    serial = prepare(factory, tmp_path / "serial")
    spawned = prepare(factory, tmp_path / "spawn", backend="process", workers=2)
    for first, second in zip(serial, spawned, strict=True):
        assert_array_equal(first.schur, second.schur, strict=True)
        assert_array_equal(first.lifts, second.lifts, strict=True)
        assert_array_equal(first.source, second.source, strict=True)


def test_resume_only_reacquires_uncommitted_batch(tmp_path, factory, monkeypatch):
    original = owner._write_batch
    acquired = []

    def interrupted(path, rows):
        acquired.append(path.name)
        original(path, rows)
        if len(acquired) == 2:
            raise RuntimeError("interrupt after archive and before manifest commit")

    monkeypatch.setattr(owner, "_write_batch", interrupted)
    with pytest.raises(RuntimeError, match="interrupt"):
        prepare(factory, tmp_path)
    manifest = json.loads((tmp_path / "responses.json").read_text())
    assert len(manifest["batches"]) == 1
    assert not manifest["complete"]
    committed = (tmp_path / "responses-0000000.npz").read_bytes()
    calls = []
    numerical_owner = CompactFactory.__call__

    def tracked(self, cell):
        calls.append(cell)
        return numerical_owner(self, cell)

    monkeypatch.setattr(owner, "_write_batch", original)
    monkeypatch.setattr(CompactFactory, "__call__", tracked)
    resumed = prepare(factory, tmp_path)
    assert calls == [1]
    assert (tmp_path / "responses-0000000.npz").read_bytes() == committed
    assert resumed.record["complete"]
    calls.clear()
    prepare(factory, tmp_path)
    assert calls == []


@pytest.mark.parametrize("when", ["resume", "read"])
def test_changed_batch_is_rejected(tmp_path, factory, when):
    store = prepare(factory, tmp_path)
    path = tmp_path / store.record["batches"][0]["archive"]
    with path.open("ab") as stream:
        stream.write(b"modified")
    with pytest.raises(ValueError, match="digest"):
        prepare(factory, tmp_path) if when == "resume" else list(store)


@pytest.mark.parametrize("change", ["sources", "configuration", "ordering", "completeness"])
def test_resume_rejects_changed_contract(tmp_path, factory, change):
    prepare(factory, tmp_path)
    path = tmp_path / "responses.json"
    record = json.loads(path.read_text())
    if change == "sources":
        record["sources"]["operator"] = "different"
    elif change == "configuration":
        record["identity"]["configuration"]["quadrature"] = 8
    elif change == "ordering":
        record["batches"][0]["archive"] = "../outside.npz"
    else:
        record["batches"].pop()
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        prepare(factory, tmp_path)


@pytest.mark.parametrize("batch_size", [0, -1, 1.5, True])
def test_batch_size_must_be_positive_integer(tmp_path, factory, batch_size):
    with pytest.raises(ValueError):
        ResponseStore.prepare(
            factory, tmp_path, sources={"s": "h"}, configuration={"a": 1}, batch_size=batch_size
        )


def test_input_contracts_and_repeatable_collection_are_required(tmp_path, factory):
    with pytest.raises(ValueError, match="configuration"):
        ResponseStore.prepare(factory, tmp_path, sources={"s": "h"}, configuration={})
    with pytest.raises(ValueError, match="sources"):
        ResponseStore.prepare(factory, tmp_path, sources={}, configuration={"a": 1})
    with pytest.raises(ValueError, match="repeatable"):
        CompactFamily.from_locals(factory.skeleton, iter(()))


def test_resume_normalizes_complete_batch_set_and_rejects_extra_empty_batch(tmp_path, factory):
    prepare(factory, tmp_path)
    path = tmp_path / "responses.json"
    record = json.loads(path.read_text())
    record["complete"] = False
    path.write_text(json.dumps(record))
    assert prepare(factory, tmp_path).record["complete"]
    record = json.loads(path.read_text())
    record["batches"].append({"start": 2, "stop": 2, "archive": "responses-0000002.npz"})
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="ordering"):
        prepare(factory, tmp_path)


def test_directory_flush_is_portable_without_directory_descriptors(tmp_path, monkeypatch):
    monkeypatch.delattr(owner.os, "O_DIRECTORY", raising=False)
    owner._sync_directory(tmp_path)


@pytest.mark.skipif(
    np.finfo(np.longdouble).eps >= np.finfo(float).eps,
    reason="extended local correction requires a wider native NumPy type",
)
def test_resume_binds_local_precision_and_keeps_completed_buffers(tmp_path, factory):
    """A restart retains correction digits and rejects an unrequested policy change."""
    store = prepare(factory, tmp_path, local_refinement_precision="extended")
    before = {
        row["archive"]: (tmp_path / row["archive"]).read_bytes() for row in store.record["batches"]
    }
    resumed = prepare(factory, tmp_path, local_refinement_precision="extended")
    for original, actual in zip(store, resumed, strict=True):
        assert_array_equal(original.source, actual.source, strict=True)
        assert_array_equal(original.lifts, actual.lifts, strict=True)
    assert all((tmp_path / name).read_bytes() == data for name, data in before.items())
    with pytest.raises(ValueError, match="identity"):
        prepare(factory, tmp_path)
    with pytest.raises(ValueError, match="identity"):
        prepare(factory, tmp_path, local_solver="pypardiso", local_refinement_precision="extended")


def test_invalid_local_precision_is_rejected_by_shared_condensation(tmp_path, factory):
    """The adapter forwards unsupported precision to the shared numerical owner."""
    with pytest.raises(ValueError, match="double or extended"):
        prepare(factory, tmp_path, local_refinement_precision="quadruple")
