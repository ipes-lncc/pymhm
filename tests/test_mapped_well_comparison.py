"""Joint norms retain pairwise quadrature, Piola maps and reduction conventions."""

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from pymhm.meshes.hexahedron import HexMesh, _geometry


def _example(name: str) -> ModuleType:
    """Load original examples without installing their namespace as package runtime."""
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).resolve().parents[1] / "examples" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


fields = _example("mapped_well_fields")
MappedWellField, difference = fields.MappedWellField, fields.difference
differences = _example("mapped_well_comparison").differences


def _field(shape: tuple[int, int, int], seed: int) -> MappedWellField:
    """Build a nonaffine common base with independent pressure and flux coefficients."""
    parent = HexMesh.unit_cube()
    corners = parent.points[parent.cells[0]].copy()
    corners[:, 0] += 0.2 * corners[:, 1] * corners[:, 2]
    corners[:, 2] += 0.15 * corners[:, 0] * corners[:, 1]
    base = HexMesh(corners, np.arange(8).reshape(1, 8))
    coordinates = np.array(list(np.ndindex(tuple(n + 1 for n in shape)))) / shape
    points = _geometry(base.points[base.cells], coordinates)[0][0]
    corners = np.array(list(np.ndindex((2, 2, 2))))
    cells = np.array(
        [
            np.ravel_multi_index((index + corners).T, tuple(n + 1 for n in shape))
            for index in np.ndindex(shape)
        ]
    )
    fine = HexMesh(points, cells)
    rng = np.random.default_rng(seed)
    return MappedWellField(
        fine.points[fine.cells],
        rng.normal(size=(len(fine.cells), 8)),
        rng.normal(size=(len(fine.cells), 36)),
        shape[0],
        shape[2],
    )


@pytest.mark.parametrize("order", [(6, 6, 3), (8, 8, 3)])
@pytest.mark.parametrize("backend,workers", [("thread", 1), ("thread", 4), ("process", 2)])
def test_joint_matches_pairwise_physical_norms(
    order: tuple[int, int, int], workers: int, backend: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Spawned BLAS evaluations preserve norms within the declared numerical precision."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    fine = _field((4, 4, 1), 31)
    candidates = [_field((2, 2, 2), seed) for seed in (4, 7, 12)]
    # An admissible roundoff-sized geometry difference must not be reused as exact equality.
    vertices = candidates[2].vertices.copy()
    vertices[..., 0] += 1e-14
    candidates[2] = replace(candidates[2], vertices=vertices)
    progress = []
    with threadpool_limits(1):
        expected = [difference(fine, field, order, pressure_offset=25.0) for field in candidates]
        actual = differences(
            fine,
            candidates,
            order,
            pressure_offset=25.0,
            workers=workers,
            backend=backend,
            progress=lambda i, n: progress.append((i, n)),
        )
    for result, reference in zip(actual, expected, strict=True):
        assert result.keys() == reference.keys()
        for name, value in reference.items():
            if isinstance(value, float):
                assert result[name] == pytest.approx(value, rel=1e-10, abs=1e-12)
            else:
                assert result[name] == value
    assert progress[-1] == (8, 8)


def test_geometry_reuse_requires_identical_coordinates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Numerical norm tolerances never permit reuse of a distinct geometry array."""
    module = _example("mapped_well_comparison")
    constructor = module._GroupIntegrator
    owners = []

    def record(*args: object) -> object:
        integrator = constructor(*args)
        owners.append(integrator.geometry_owner)
        return integrator

    monkeypatch.setattr(module, "_GroupIntegrator", record)
    fine = _field((2, 2, 1), 9)
    first, second = (_field((1, 1, 1), seed) for seed in (3, 4))
    vertices = second.vertices.copy()
    vertices[..., 0] += 1e-14
    third = replace(second, vertices=vertices)
    differences(fine, [first, second, third], (3, 3, 3))
    assert owners == [[0, 0, 2]]


def test_different_shapes_keep_each_pair_partition_and_zero_norms() -> None:
    """Shape groups preserve input order and the original zero-denominator convention."""
    fine = _field((4, 4, 1), 5)
    fine = replace(fine, pressure=np.zeros_like(fine.pressure), flux=np.zeros_like(fine.flux))
    candidates = [_field(shape, seed) for shape, seed in [((2, 2, 2), 3), ((1, 1, 1), 4)]]
    assert differences(fine, candidates, (3, 3, 3)) == [
        difference(fine, field, (3, 3, 3)) for field in candidates
    ]
    assert differences(fine, [fine], (3, 3, 3))[0]["flux_l2"] == 0


def test_geometry_and_partition_sentinels_remain_active() -> None:
    """Reuse does not accept incompatible grids, base meshes or physical coordinates."""
    fine = _field((2, 2, 1), 11)
    coarse = _field((1, 1, 1), 12)
    with pytest.raises(ValueError, match="backend"):
        differences(fine, [coarse], backend="unknown")
    with pytest.raises(ValueError, match="workers"):
        differences(fine, [coarse], workers=0)
    with pytest.raises(ValueError, match="candidate"):
        differences(fine, [])
    with pytest.raises(ValueError, match="nested"):
        differences(fine, [_field((3, 3, 1), 7)])
    with pytest.raises(ValueError, match="base mesh"):
        differences(fine, [replace(coarse, vertices=np.tile(coarse.vertices, (2, 1, 1)))])
    with pytest.raises(ValueError, match="inconsistent physical geometry"):
        differences(fine, [replace(coarse, vertices=coarse.vertices + [1e-6, 0, 0])])
