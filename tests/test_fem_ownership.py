"""FEM owners remain generic and current solution records retain scientific fields."""

from __future__ import annotations

import importlib
import pickle
import subprocess
import sys

import numpy as np
import pytest
from numpy.testing import assert_array_equal


@pytest.mark.parametrize(
    "module_name",
    ["pymhm.fem.vector.operators", "pymhm.fem.hdiv.tensor_rt"],
)
def test_generic_kernel_import_does_not_load_physical_models(module_name: str) -> None:
    """Loading generic kernels leaves PDE drivers and optional native runtimes unopened."""
    script = f"""
import importlib
import sys
module = importlib.import_module({module_name!r})
assert not [name for name in sys.modules if name.startswith('pymhm._legacy.models')]
native = ('dolfinx', 'ufl', 'mpi4py', 'petsc4py')
assert not [name for name in sys.modules if name.split('.')[0] in native]
"""
    run = subprocess.run([sys.executable, "-I", "-c", script], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr


@pytest.mark.parametrize(
    "module_name,owner,symbols",
    [
        (
            "pymhm.fem.vector.operators",
            "pymhm._legacy.models.vector",
            ("VectorSolution", "solve_brinkman", "solve_elasticity"),
        ),
        (
            "pymhm.fem.hdiv.tensor_rt",
            "pymhm._legacy.models.darcy.tensor",
            ("TensorRTDarcySolution", "solve_darcy_tensor_rt"),
        ),
    ],
)
def test_physical_symbols_require_their_explicit_owner(
    module_name: str, owner: str, symbols: tuple[str, ...]
) -> None:
    """A FEM module has no lazy redirect for solution records or PDE solve functions."""
    generic = importlib.import_module(module_name)
    physical = importlib.import_module(owner)
    for name in symbols:
        assert getattr(physical, name).__module__ == owner
        with pytest.raises(AttributeError, match=f"has no attribute '{name}'"):
            getattr(generic, name)
    assert "__getattr__" not in vars(generic)
    with pytest.raises(AttributeError, match="has no attribute 'unrelated_solver'"):
        _ = generic.unrelated_solver


@pytest.mark.parametrize(
    "module_name,symbol",
    [
        ("pymhm.fem.vector.operators", "VectorSolution"),
        ("pymhm.fem.hdiv.tensor_rt", "TensorRTDarcySolution"),
    ],
)
def test_removed_fem_pickle_globals_are_rejected(module_name: str, symbol: str) -> None:
    """Generic owners cannot deserialize records formerly supplied by lazy PDE aliases."""
    serialized = f"c{module_name}\n{symbol}\n.".encode("ascii")
    with pytest.raises(AttributeError, match=f"{symbol!r}"):
        pickle.loads(serialized)


def test_current_vector_pickle_preserves_fields_and_physical_evaluation() -> None:
    """The current record preserves independent velocity/pressure fields and norms."""
    from pymhm._legacy.models.vector import VectorSolution
    from pymhm.core.contracts import HybridSolution
    from pymhm.fem.traces.interval import SkeletonSpace
    from pymhm.meshes.triangle import TriangleMesh

    mesh = TriangleMesh.unit_square()
    skeleton = SkeletonSpace(mesh, components=2)
    values = np.column_stack((mesh.points[:, 1], -mesh.points[:, 0]))
    pressure = mesh.points[:, 0] + 2 * mesh.points[:, 1]
    hybrid = HybridSolution(
        np.zeros(skeleton.size), (np.ones(1),), (values.ravel(),), 0.0, np.zeros(1)
    )
    solution = VectorSolution(skeleton, (mesh,), (values,), (pressure,), hybrid, 1)
    archived = pickle.dumps(solution)
    assert b"pymhm._legacy.models.vector" in archived
    replay = pickle.loads(archived)
    assert type(replay) is VectorSolution
    assert replay.degree == 1 and replay.pressure_degree == 1
    assert_array_equal(replay.values[0], values)
    assert_array_equal(replay.pressure[0], pressure)
    assert_array_equal(replay.hybrid.trace, hybrid.trace)
    assert_array_equal(replay.hybrid.coarse[0], hybrid.coarse[0])
    assert_array_equal(replay.hybrid.fields[0], hybrid.fields[0])

    def exact(x):
        return np.column_stack((x[:, 1], -x[:, 0]))

    assert replay.l2_error(exact) == solution.l2_error(exact)
    assert replay.pressure_l2_error(lambda x: x[:, 0] + 2 * x[:, 1]) == (
        solution.pressure_l2_error(lambda x: x[:, 0] + 2 * x[:, 1])
    )
    assert replay.divergence_l2() == solution.divergence_l2()


def test_current_tensor_pickle_preserves_oriented_flux_and_pressure_gauge() -> None:
    """The current H(div) record retains fields, oriented traces and the physical gauge."""
    from pymhm._legacy.models.darcy.tensor import TensorRTDarcySolution, solve_darcy_tensor_rt
    from pymhm.meshes.cartesian import CartesianMacroMesh

    mesh = CartesianMacroMesh(2, 1)
    neumann = {int(face): float(mesh.normals[face] @ [-1.0, -2.0]) for face in mesh.boundary_faces}
    solution = solve_darcy_tensor_rt(
        mesh, degree=1, enrichment=1, neumann=neumann, mean_pressure=3.5, local_refinement=1
    )
    archived = pickle.dumps(solution)
    assert b"pymhm._legacy.models.darcy.tensor" in archived
    replay = pickle.loads(archived)
    assert type(replay) is TensorRTDarcySolution
    assert replay.degree == solution.degree and replay.enrichment == solution.enrichment
    for field in ("pressure", "flux"):
        for actual, expected in zip(getattr(replay, field), getattr(solution, field), strict=True):
            assert_array_equal(actual, expected)
    assert_array_equal(replay.hybrid.trace, solution.hybrid.trace)
    assert_array_equal(replay.hybrid.gauge_multipliers, solution.hybrid.gauge_multipliers)
    points = np.array([[0.13, 0.21], [0.59, 0.83]])
    for cell in range(len(mesh.cells)):
        for actual, expected in zip(
            replay.evaluate(cell, points), solution.evaluate(cell, points), strict=True
        ):
            assert_array_equal(actual, expected)
    for name in ("equilibrium_residuals", "normal_flux_residuals"):
        for actual, expected in zip(
            getattr(replay, name)(), getattr(solution, name)(), strict=True
        ):
            assert_array_equal(actual, expected)
    assert replay.errors(0.0, [0.0, 0.0], 0.0) == solution.errors(0.0, [0.0, 0.0], 0.0)
