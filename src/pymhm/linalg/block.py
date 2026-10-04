"""AMG block preconditioning of real symmetric mixed saddle systems."""

from __future__ import annotations

from contextlib import ExitStack
from types import TracebackType
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse import linalg

from pymhm.core.validation import positive_int
from pymhm.linalg.linear import (
    _AMGX_LOCK,
    LinearFactorization,
    LinearSolveError,
    _checked,
    _hermitian,
    _matrix,
    _optional,
    _rhs,
    _tolerances,
    factorize,
)


def _positive(matrix: Any, name: str) -> None:
    """Check positive definiteness of a symmetric preconditioner block at physical scale."""
    if not matrix.nnz:
        raise ValueError(f"{name} is not numerically positive definite")
    if matrix.shape[0] <= 256:
        minimum = float(np.linalg.eigvalsh(matrix.toarray())[0])
    else:
        minimum = float(
            linalg.eigsh(matrix, k=1, which="SA", return_eigenvectors=False, tol=1e-8)[0]
        )
    if minimum <= np.finfo(float).eps * matrix.shape[0] * np.max(np.abs(matrix.data)):
        raise ValueError(f"{name} is not numerically positive definite")


def _amgx_cycle(matrix: Any, resources: ExitStack) -> Any:
    """Prepare one reusable GPU aggregation V-cycle, retaining all native resources."""
    amgx = _optional("pyamgx", "Build pyamgx against NVIDIA AmgX and a compatible CUDA runtime.")
    _AMGX_LOCK.acquire()
    resources.callback(_AMGX_LOCK.release)
    amgx.initialize()
    resources.callback(amgx.finalize)
    configuration = amgx.Config().create_from_dict(
        {
            "config_version": 2,
            "determinism_flag": 1,
            "exception_handling": 1,
            "solver": {
                "solver": "AMG",
                "algorithm": "AGGREGATION",
                "selector": "SIZE_2",
                "smoother": "BLOCK_JACOBI",
                "presweeps": 1,
                "postsweeps": 1,
                "max_iters": 1,
                "max_levels": 30,
                "cycle": "V",
                "coarse_solver": "DENSE_LU_SOLVER",
                "monitor_residual": 0,
                "print_solve_stats": 0,
            },
        }
    )
    resources.callback(configuration.destroy)
    owner = amgx.Resources().create_simple(configuration)
    resources.callback(owner.destroy)
    operator = amgx.Matrix().create(owner, mode="dDDI")
    resources.callback(operator.destroy)
    operator.upload_CSR(matrix)
    engine = amgx.Solver().create(owner, configuration, mode="dDDI")
    resources.callback(engine.destroy)
    engine.setup(operator)
    forcing = amgx.Vector().create(owner, mode="dDDI")
    resources.callback(forcing.destroy)
    result = amgx.Vector().create(owner, mode="dDDI")
    resources.callback(result.destroy)

    def apply(rhs: np.ndarray) -> np.ndarray:
        """Apply one device V-cycle; global Krylov convergence is checked independently."""
        if not np.any(rhs):
            return np.zeros_like(rhs)
        forcing.upload(np.ascontiguousarray(rhs, dtype=float))
        result.upload(np.zeros(len(rhs)))
        engine.solve(forcing, result, zero_initial_guess=True)
        answer = np.asarray(result.download())
        if answer.shape != rhs.shape or not np.isfinite(answer).all():
            raise LinearSolveError("AmgX block cycle returned an invalid vector")
        return answer

    return apply


class SaddleBlockSolver:
    """Reusable block-diagonal preconditioned GMRES for ``[A B; B.T -C]``.

    The original matrix is never shifted. AMG acts on the positive augmented
    preconditioner block ``P=A+alpha*B@B.T``; the second block is
    ``S=C+B.T@diag(P)^-1@B`` and uses a sparse direct factorization. Both blocks
    must be positive definite. This supports a semidefinite A when the coupling
    controls its kernel. It is not an arbitrary indefinite-matrix AMG adapter.

    ``backend='pyamg'`` uses a retained CPU smoothed-aggregation V-cycle;
    ``backend='amgx'`` retains a GPU aggregation hierarchy, transferring each
    block vector. The outer Krylov matrix and coarse Schur block remain on CPU.
    An AmgX context exclusively owns its process-global native lifecycle.
    """

    def __init__(
        self,
        matrix: Any,
        split: int,
        *,
        backend: str = "pyamg",
        augmentation: float | None = None,
        rtol: float = 1e-10,
        atol: float = 0.0,
        maxiter: int = 200,
        restart: int = 50,
    ) -> None:
        """Prepare positive block hierarchies for an explicit contiguous primal/coarse split."""
        self.matrix = _matrix(matrix)
        _hermitian(self.matrix, "saddle block preconditioner")
        if np.iscomplexobj(self.matrix.data):
            raise ValueError("saddle block solver requires real data")
        self.split = positive_int(split, "split")
        if self.split >= self.matrix.shape[0]:
            raise ValueError("split must leave a nonempty coarse block")
        if backend not in ("pyamg", "amgx"):
            raise ValueError("backend must be pyamg or amgx")
        _tolerances(rtol, atol)
        self.rtol, self.atol = rtol, atol
        self.maxiter, self.restart = (
            positive_int(maxiter, "maxiter"),
            positive_int(restart, "restart"),
        )
        self.backend = backend
        self.iterations: list[int] = []
        self._closed = False
        A = self.matrix[:split, :split].tocsr()
        B = self.matrix[:split, split:].tocsr()
        C = -self.matrix[split:, split:].tocsr()
        if B.nnz == 0:
            raise ValueError("saddle coupling must be nonzero")
        if augmentation is None:
            augmentation = max(
                float(np.max(np.abs(A.data))) if A.nnz else 0.0, np.finfo(float).tiny
            ) / float(np.max(np.asarray(B.power(2).sum(axis=1))))
        if not np.isfinite(augmentation) or augmentation <= 0:
            raise ValueError("augmentation must be positive and finite")
        self.augmentation = augmentation
        primal = (A + augmentation * (B @ B.T)).tocsr()
        _positive(primal, "augmented primal block")
        schur = (C + B.T @ sparse.diags(1 / primal.diagonal()) @ B).tocsr()
        _positive(schur, "Schur preconditioner block")
        self._resources = ExitStack()
        try:
            coarse = self._resources.enter_context(factorize(schur))
            if backend == "pyamg":
                pyamg = _optional("pyamg", "Install pymhm[amg] to use the CPU block hierarchy.")
                hierarchy = pyamg.smoothed_aggregation_solver(primal, symmetry="symmetric")
                apply_primal = hierarchy.aspreconditioner(cycle="V").matvec
            else:
                apply_primal = _amgx_cycle(primal, self._resources)

            def apply(vector: np.ndarray) -> np.ndarray:
                """Apply independent positive primal and Schur preconditioner blocks."""
                return np.r_[apply_primal(vector[:split]), coarse.solve(vector[split:])]

            self._preconditioner = linalg.LinearOperator(
                self.matrix.shape, matvec=apply, dtype=float
            )
        except Exception:
            self._resources.close()
            raise

    def solve(self, rhs: Any) -> np.ndarray:
        """Solve every RHS against the original saddle operator with strict residual checks."""
        if self._closed:
            raise RuntimeError("saddle block solver is closed")
        values = _rhs(rhs, self.matrix.shape[0])
        if np.iscomplexobj(values):
            raise ValueError("saddle block solver requires real rhs")
        columns = values[:, None] if values.ndim == 1 else values
        answers = []
        for column in columns.T:
            history: list[float] = []
            answer, status = linalg.gmres(
                self.matrix,
                column,
                M=self._preconditioner,
                rtol=self.rtol * 0.1,
                atol=self.atol * 0.1,
                restart=self.restart,
                maxiter=self.maxiter,
                callback=history.append,
                callback_type="pr_norm",
            )
            self.iterations.append(len(history))
            if status != 0:
                raise LinearSolveError(f"block GMRES failed to converge: info={status}")
            answers.append(_checked(self.matrix, column, answer, self.rtol, self.atol))
        result = np.column_stack(answers)
        return result[:, 0] if values.ndim == 1 else result

    def as_factorization(self) -> LinearFactorization:
        """Expose the prepared solve through HybridSystem's exact-matrix reuse contract."""
        if self._closed:
            raise RuntimeError("saddle block solver is closed")
        return LinearFactorization(
            self.matrix, self.solve, self.close, f"block-{self.backend}", self.rtol, self.atol
        )

    def close(self) -> None:
        """Release retained coarse/GPU resources exactly once."""
        self._closed = True
        self._resources.close()

    def __enter__(self) -> SaddleBlockSolver:
        """Enter an open block solver context."""
        if self._closed:
            raise RuntimeError("saddle block solver is closed")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release all owned resources on normal or exceptional exit."""
        self.close()
