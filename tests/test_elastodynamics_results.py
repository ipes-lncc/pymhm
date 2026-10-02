"""Independent physical integrals for persisted elastic-wave comparisons."""

import importlib
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from pymhm.tetrahedral import TetraMesh, tetra_nodal_space


@pytest.fixture
def difference(monkeypatch: pytest.MonkeyPatch) -> Callable[[Path, Path], dict[str, float]]:
    """Import the original example under the same package namespace used by its runner."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    return importlib.import_module("examples.elastodynamics_results").difference


def _archive(path: Path, axis: int, *, nonzero: bool, time: float = 0.5) -> None:
    """Store quadratic displacement and linear velocity on independent local meshes."""
    mesh = TetraMesh.unit_cube()
    displacement, velocity = [], []
    for macro in range(len(mesh.cells)):
        fine = mesh.submesh(macro, 2)
        _, points = tetra_nodal_space(fine, 3)
        u, v = np.zeros_like(points), np.zeros_like(points)
        if nonzero:
            u[:, axis] = points[:, axis] ** 2
            v[:, (axis + 1) % 3] = 2 * points[:, (axis + 1) % 3]
        displacement.append(u.ravel())
        velocity.append(v.ravel())
    np.savez(
        path,
        macro_points=mesh.points,
        macro_cells=mesh.cells,
        displacement=np.array(displacement),
        velocity=np.array(velocity),
        local_degree=3,
        local_refinement=2,
        time=time,
    )


@pytest.mark.parametrize("axis", (0, 1, 2))
def test_temporal_difference_has_exact_physical_norms(
    tmp_path: Path, axis: int, difference: Callable[[Path, Path], dict[str, float]]
) -> None:
    """Check all component directions, stress divergence and full H1/H(div) norms."""
    first, second = tmp_path / "first.npz", tmp_path / "second.npz"
    _archive(first, axis, nonzero=True)
    _archive(second, axis, nonzero=False)
    # On the unit cube, u=x_i^2 e_i and v=2 x_j e_j, with lambda=mu=2/5.
    # sigma has diagonal entries 12 x_i/5, 4 x_i/5, 4 x_i/5;
    # div(sigma)=12 e_i/5. Integrate these monomials analytically.
    squared = {
        "displacement_l2": 1 / 5,
        "velocity_l2": 4 / 3,
        "displacement_h1": 23 / 15,
        "velocity_h1": 16 / 3,
        "stress_l2": 176 / 75,
        "stress_broken_hdiv": 608 / 75,
    }
    measured = difference(first, second)
    for key, expected in squared.items():
        assert measured[key] ** 2 == pytest.approx(expected, rel=3e-13)
    assert difference(second, first) == measured
    assert all(value == 0 for value in difference(first, first).values())


def test_temporal_difference_rejects_different_physical_times(
    tmp_path: Path, difference: Callable[[Path, Path], dict[str, float]]
) -> None:
    """A spatially compatible archive at another time is not a time-refinement pair."""
    first, second = tmp_path / "first.npz", tmp_path / "second.npz"
    _archive(first, 0, nonzero=True)
    _archive(second, 0, nonzero=True, time=0.6)
    with pytest.raises(ValueError, match="matching time"):
        difference(first, second)
