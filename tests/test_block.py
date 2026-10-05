"""Saddle block algebra, real MHM systems and optional native GPU AMG."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from pymhm.core.contracts import LocalProblem
from pymhm.core.system import HybridSystem
from pymhm.fem.scalar.operators import face_integration, p1_operators
from pymhm.fem.traces.interval import SkeletonSpace
from pymhm.linalg import block
from pymhm.linalg.block import SaddleBlockSolver, _positive
from pymhm.linalg.linear import LinearSolveError
from pymhm.meshes.triangle import TriangleMesh


def mhm_system(n=2):
    mesh = TriangleMesh.unit_square(n)
    skeleton = SkeletonSpace(mesh)
    problems = []
    for cell in range(len(mesh.cells)):
        fine = mesh.submesh(cell, 4)
        A, M, f = p1_operators(fine, 1.0, 1.0)
        B = face_integration(mesh, cell, fine, skeleton)[0]
        Z = np.ones((A.shape[0], 1))
        problems.append(LocalProblem(A, B, f, skeleton.cell_dofs(cell), Z, M @ Z))
    return HybridSystem(problems)


def test_actual_mhm_saddle_cpu_amg_and_factor_reuse():
    pytest.importorskip("pyamg")
    system = mhm_system()
    expected = system.solve()
    with SaddleBlockSolver(system.matrix, system.trace_size) as prepared:
        factor = prepared.as_factorization()
        actual = system.solve(factorization=factor)
        np.testing.assert_allclose(actual.fields, expected.fields, atol=2e-12)
        np.testing.assert_allclose(actual.trace, expected.trace, atol=2e-12)
        rhs = np.column_stack((system.rhs, -2 * system.rhs, np.zeros_like(system.rhs)))
        answers = prepared.solve(rhs)
        np.testing.assert_allclose(system.matrix @ answers, rhs, atol=2e-12)
        assert prepared.iterations[0] > 0 and prepared.iterations[-1] == 0
    prepared.close()
    for method in (prepared.__enter__, prepared.as_factorization):
        with pytest.raises(RuntimeError, match="closed"):
            method()
    with pytest.raises(RuntimeError, match="closed"):
        prepared.solve(system.rhs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"split": 0},
        {"split": 3},
        {"backend": "unknown"},
        {"augmentation": -1},
        {"augmentation": np.nan},
        {"rtol": -1},
        {"maxiter": 0},
        {"restart": 0},
    ],
)
def test_block_invalid_options(kwargs):
    A = np.array([[2.0, -1, 1], [-1, 2, 1], [1, 1, 0]])
    with pytest.raises(ValueError):
        SaddleBlockSolver(A, **({"split": 2} | kwargs))


def test_invalid_block_structures_and_positive_guard():
    with pytest.raises(ValueError, match="real data"):
        SaddleBlockSolver(np.array([[1.0, 1.0], [1.0, 0.0]]) + 0j, 1)
    with pytest.raises(ValueError, match="coupling"):
        SaddleBlockSolver(np.diag([1.0, -1.0]), 1)
    with pytest.raises(ValueError, match="positive definite"):
        SaddleBlockSolver([[-2.0, 1.0], [1.0, 0.0]], 1, augmentation=1)
    with pytest.raises(ValueError, match="positive definite"):
        SaddleBlockSolver([[1.0, 1.0], [1.0, 2.0]], 1)
    with pytest.raises(ValueError, match="positive definite"):
        _positive(sparse.csr_matrix((2, 2)), "zero")
    _positive(sparse.eye(257, format="csr"), "identity")
    with pytest.raises(ValueError, match="positive definite"):
        SaddleBlockSolver([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], 1)


def test_outer_failure_complex_rhs_and_resource_cleanup(monkeypatch):
    pytest.importorskip("pyamg")
    system = mhm_system()
    with SaddleBlockSolver(system.matrix, system.trace_size, maxiter=1, restart=1) as prepared:
        with pytest.raises(ValueError, match="real rhs"):
            prepared.solve(system.rhs + 1j)
        with pytest.raises(LinearSolveError, match="converge"):
            prepared.solve(system.rhs)
    closed = []
    original = block.factorize

    def counted(matrix):
        factor = original(matrix)
        close = factor._release
        factor._release = lambda: (closed.append(True), close())
        return factor

    monkeypatch.setattr(block, "factorize", counted)
    monkeypatch.setattr(
        block, "_optional", lambda *args: (_ for _ in ()).throw(ImportError("unavailable"))
    )
    with pytest.raises(ImportError):
        SaddleBlockSolver(system.matrix, system.trace_size)
    assert closed == [True]


@pytest.fixture
def simulated_amgx(monkeypatch):
    """Exercise GPU resource contracts with explicitly simulated exact CPU subsolves."""
    events = []

    class Handle:
        corrupt = False

        def create_from_dict(self, config):
            return self

        def create_simple(self, config):
            return self

        def create(self, *args, **kwargs):
            return self

        def destroy(self):
            events.append("destroy")

        def upload_CSR(self, matrix):
            self.matrix = matrix

        def setup(self, operator):
            self.operator = operator

        def upload(self, values):
            self.values = values.copy()

        def solve(self, rhs, result, **kwargs):
            result.values = np.linalg.solve(self.operator.matrix.toarray(), rhs.values)
            if self.corrupt:
                result.values[:] = np.nan

        def download(self):
            return self.values

    amgx = SimpleNamespace(
        initialize=lambda: events.append("initialize"),
        finalize=lambda: events.append("finalize"),
        Config=Handle,
        Resources=Handle,
        Matrix=Handle,
        Solver=Handle,
        Vector=Handle,
    )
    monkeypatch.setattr(block, "_optional", lambda *args: amgx)
    return events, Handle


def test_simulated_gpu_amg_reuse_and_invalid_vector(simulated_amgx):
    events, handle = simulated_amgx
    system = mhm_system()
    with SaddleBlockSolver(system.matrix, system.trace_size, backend="amgx") as prepared:
        x = prepared.solve(system.rhs)
        np.testing.assert_allclose(system.matrix @ x, system.rhs, atol=2e-12)
        np.testing.assert_allclose(prepared._preconditioner @ np.zeros(len(system.rhs)), 0)
        handle.corrupt = True
        with pytest.raises(LinearSolveError, match="invalid vector"):
            prepared.solve(system.rhs)
    assert events.count("initialize") == 1 and events.count("finalize") == 1
    assert events.count("destroy") == 6


@pytest.mark.serial
@pytest.mark.gpu
def test_native_gpu_amg_saddle_fields():
    pytest.importorskip("pyamgx")
    system = mhm_system(4)
    exact = system.solve()
    with SaddleBlockSolver(system.matrix, system.trace_size, backend="amgx") as prepared:
        actual = system.solve(factorization=prepared.as_factorization())
        np.testing.assert_allclose(actual.fields, exact.fields, atol=3e-11)
        np.testing.assert_allclose(actual.trace, exact.trace, atol=3e-11)
        assert prepared.iterations[0] > 0
