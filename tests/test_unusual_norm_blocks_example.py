"""Ordered cell-block integration preserves the original physical norm reductions."""

import importlib
import json
import runpy
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pymhm.meshes.triangle import TriangleMesh


def fields():
    """Return nonzero P1/P2 fields on unequal, material-aligned triangulations."""
    module = importlib.import_module("examples.compare_unusual_spe10")
    baseline = importlib.import_module("examples.solve_unusual_spe10_reference")
    x = np.array([0.0, 0.3, 1.0, 2.0])
    y = np.array([0.0, 0.75, 1.5, 3.0])

    def nodes(axis):
        """Interleave vertices and physical midpoints in the native CG2 archive order."""
        result = np.empty(2 * len(axis) - 1)
        result[::2], result[1::2] = axis, (axis[1:] + axis[:-1]) / 2
        return result

    xx, yy = np.meshgrid(nodes(x), nodes(y))
    reference = baseline.CG2Field(
        2 + xx**2 + yy**2, np.array([[1.0, 4.0], [2.0, 8.0]]), (0, 2, 0, 3), x, y
    )
    macro = TriangleMesh(TriangleMesh.unit_square().points * [2, 3], [[0, 1, 3], [0, 3, 2]])
    meshes = tuple(macro.submesh(i, 3) for i in range(2))

    def evaluate_local(cell, points, owners):
        """Retain an explicit broken macro shift and the physical coefficient in flux."""
        gradient = np.tile([1.0, 2.0], (len(points), 1))
        pressure = 1 + points[:, 0] + 2 * points[:, 1] + cell
        return pressure, gradient, -reference.material(points)[:, None] * gradient

    candidate = SimpleNamespace(
        vertices=tuple(mesh.points[mesh.cells] for mesh in meshes), evaluate_local=evaluate_local
    )
    return module, candidate, reference


@pytest.mark.parametrize("batch_size", [1, 4, 32])
def test_cell_blocks_keep_each_original_longdouble_addition(monkeypatch, batch_size):
    """Changing task boundaries never changes the physical contributions or ordered sum."""
    module, candidate, reference = fields()
    monkeypatch.setattr(module, "_FIELDS", (candidate, reference))
    counts = np.array([len(vertices) for vertices in candidate.vertices])
    for order in (3, 4):
        expected = np.asarray([module.integrate((cell, order)) for cell in range(len(counts))])
        blocks = map(module.integrate_block, module.cell_tasks(counts, order, batch_size))
        actual = module.accumulate_blocks(blocks, counts)
        assert_array_equal(actual, expected)
        assert_array_equal(
            np.sum(actual, axis=0, dtype=np.longdouble),
            np.sum(expected, axis=0, dtype=np.longdouble),
        )
        assert np.all(actual > 0)


def test_block_contracts_reject_missing_reordered_or_malformed_contributions(monkeypatch):
    """A missing/repeated cell cannot silently alter a physical norm or denominator."""
    module, _, _ = fields()
    with pytest.raises(ValueError, match="positive"):
        list(module.cell_tasks([1], 3, 0))
    assert list(module.cell_tasks([0, 2], 3, 4)) == [(1, 3, 0, 2)]
    for block in (
        (-1, 0, np.ones((1, 6))),
        (1, 0, np.ones((1, 6))),
        (0, 1, np.ones((1, 6))),
        (0, 0, np.ones(6)),
        (0, 0, np.ones((1, 5))),
        (0, 0, np.ones((0, 6))),
        (0, 0, np.ones((3, 6))),
    ):
        with pytest.raises(ValueError, match="original cell order"):
            module.accumulate_blocks([block], np.array([2]))
    with pytest.raises(ValueError, match="every original"):
        module.accumulate_blocks([(0, 0, np.ones((1, 6)))], np.array([2]))
    assert_array_equal(module.accumulate_blocks([], np.array([0])), np.zeros((1, 6)))
    monkeypatch.setattr(module, "_FIELDS", None)
    for function, task in ((module.integrate, (0, 3)), (module.integrate_block, (0, 3, 0, 1))):
        with pytest.raises(RuntimeError, match="initialize"):
            function(task)
    with pytest.raises(ValueError, match="workers"):
        module.acquire(None, None, None, workers=0)
    with pytest.raises(ValueError, match="nonnegative"):
        module.acquire(None, None, None, cell_batch_size=-1)


def test_acquisition_records_identical_norms_for_macro_and_cell_scheduling(monkeypatch, tmp_path):
    """The acquisition preserves the norm denominator while recording its task granularity."""
    module, candidate, reference = fields()
    monkeypatch.setattr(module, "_FIELDS", None)
    monkeypatch.setattr(module, "_LIMIT", None)
    monkeypatch.setattr(module, "UnusualSPE10Field", lambda path: candidate)
    monkeypatch.setattr(module, "CG2Field", SimpleNamespace(load=lambda path: reference))

    class OrderedExecutor:
        """Exercise the real worker entrypoints without spawning an optional data fixture."""

        def __init__(self, *, initializer, initargs, **kwargs):
            """Retain the same once-per-worker initialization used by native spawn."""
            initializer(*initargs)

        def __enter__(self):
            """Expose the ordered map contract."""
            return self

        def __exit__(self, *args):
            """Restore native thread counts after this in-process worker."""
            module._LIMIT.restore_original_limits()

        @staticmethod
        def map(function, tasks):
            """Keep input order exactly as ProcessPoolExecutor.map does."""
            return map(function, tasks)

    monkeypatch.setattr(module, "ProcessPoolExecutor", OrderedExecutor)
    archive, baseline = tmp_path / "candidate.npz", tmp_path / "reference.npz"
    counts = [len(vertices) for vertices in candidate.vertices]
    np.savez(archive, macro_cells=np.zeros((2, 3)), cell_offsets=np.r_[0, np.cumsum(counts)])
    baseline.write_bytes(b"field fixture supplied by the independent polynomial constructor")
    expected = module.acquire(archive, baseline, tmp_path / "macro.json", cell_batch_size=0)
    actual = module.acquire(archive, baseline, tmp_path / "blocks.json", cell_batch_size=4)
    for first, second in zip(expected["rows"], actual["rows"], strict=True):
        assert {k: v for k, v in first.items() if k != "seconds"} == {
            k: v for k, v in second.items() if k != "seconds"
        }
    assert json.loads((tmp_path / "blocks.json").read_text())["cell_batch_size"] == 4
    assert not actual["source_changed_during_run"]

    original_hash, calls = module.hashlib.sha256, []

    def changed_source_hash(content):
        """Model an edited source after six fingerprints and the two field digests."""
        calls.append(None)
        return original_hash(content + (b"changed" if len(calls) > 8 else b""))

    monkeypatch.setattr(module, "hashlib", SimpleNamespace(sha256=changed_source_hash))
    with pytest.raises(RuntimeError, match="norm source changed"):
        module.acquire(archive, baseline, tmp_path / "unaccepted.json", cell_batch_size=4)
    assert not (tmp_path / "unaccepted.json").exists()


def test_overlay_partition_and_cli_reject_invalid_norm_acquisitions(monkeypatch):
    """Incomplete intersections and invalid execution settings cannot produce a report."""
    module, candidate, reference = fields()
    monkeypatch.setattr(module, "_clip_affine_polygon", lambda *args: np.empty((0, 2)))
    with pytest.raises(ValueError, match="partition"):
        module.overlay_quadrature(candidate.vertices[0][0], reference, 3)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare",
            "candidate.npz",
            "--reference",
            "reference.npz",
            "--output",
            "result.json",
            "--workers",
            "0",
        ],
    )
    with pytest.raises(ValueError, match="workers must be positive"):
        runpy.run_path(module.__file__, run_name="__main__")
