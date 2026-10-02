"""Explicit serial NumPy simulation of the PETSc API used by distributed assembly.

This fixture tests resource and assembly contracts. It is not MPI evidence;
``mpi_cases.py`` separately executes the real libraries on multiple ranks.
"""

from types import SimpleNamespace
from typing import Any

import numpy as np


class Vec:
    """Store one serial vector and mimic additive PETSc insertion semantics."""

    def __init__(self, size: int = 0) -> None:
        """Allocate test-owned host values."""
        self.values = np.zeros(size)

    def createSeq(self, size: int, **kwargs: Any) -> "Vec":
        """Allocate sequential output storage for index scattering."""
        self.values = np.zeros(size)
        return self

    def duplicate(self) -> "Vec":
        """Allocate a zero vector of the same size."""
        return Vec(len(self.values))

    def copy(self) -> "Vec":
        """Copy the unmodified physical right-hand side."""
        result = self.duplicate()
        result.values[:] = self.values
        return result

    def setValues(self, indices: Any, values: Any, addv: Any = None) -> None:
        """Insert or add values, including repeated contribution indices."""
        if addv == 1:
            np.add.at(self.values, indices, values)
        else:
            self.values[indices] = values

    def setValue(self, index: int, value: float, addv: Any = None) -> None:
        """Insert one gauge right-hand-side contribution."""
        self.setValues([index], [value], addv)

    def assemble(self) -> None:
        """Complete serial insertion, which requires no communication."""

    def destroy(self) -> None:
        """Expose explicit native-resource release in contract tests."""

    def getOwnershipRange(self) -> tuple[int, int]:
        """Return the entire serial vector's owned interval."""
        return 0, len(self.values)

    def getArray(self, readonly: bool = False) -> np.ndarray:
        """Return the test-owned array for residual checks."""
        return self.values

    def norm(self) -> float:
        """Compute the genuine Euclidean norm used by the contract."""
        return float(np.linalg.norm(self.values))

    def axpy(self, alpha: float, vector: "Vec") -> None:
        """Compute a linear combination in original physical units."""
        self.values += alpha * vector.values


class Mat:
    """Assemble actual dense numerical blocks while simulating PETSc allocation."""

    Option = SimpleNamespace(NEW_NONZERO_ALLOCATION_ERR=0)

    def createAIJ(self, size: tuple[int, int], **kwargs: Any) -> "Mat":
        """Allocate a serial matrix with the requested global shape."""
        self.values = np.zeros(size)
        return self

    def setOption(self, *args: Any) -> None:
        """Accept the explicit dynamic-preallocation option."""

    def createVecLeft(self) -> Vec:
        """Allocate a compatible residual/right-hand-side vector."""
        return Vec(len(self.values))

    def setValues(self, rows: Any, cols: Any, values: Any, addv: Any = None) -> None:
        """Add a dense contribution at its global row/column indices."""
        self.values[np.ix_(rows, cols)] += values

    def assemble(self) -> None:
        """Complete matrix insertion without emulating communication."""

    def destroy(self) -> None:
        """Expose deterministic matrix-resource cleanup."""

    def copy(self) -> "Mat":
        """Preserve the original physical equations before elimination."""
        result = Mat().createAIJ(self.values.shape)
        result.values[:] = self.values
        return result

    def getValuesCSR(self) -> tuple[Any, Any, Any]:
        """Expose the owned-row CSR values of the serial contract operator."""
        from scipy import sparse

        matrix = sparse.csr_matrix(self.values)
        return matrix.indptr, matrix.indices, matrix.data

    def setValuesCSR(self, indptr: Any, indices: Any, values: Any) -> None:
        """Replace existing owned-row entries using CSR insertion semantics."""
        from scipy import sparse

        self.values[:] = sparse.csr_matrix(
            (values, indices, indptr), shape=self.values.shape
        ).toarray()

    def zeroRowsColumns(self, rows: Any, diag: float, x: Vec, b: Vec) -> None:
        """Apply exact inhomogeneous Dirichlet lifting and row/column elimination."""
        b.values -= self.values @ x.values
        self.values[rows, :] = 0
        self.values[:, rows] = 0
        self.values[rows, rows] = diag
        b.values[rows] = diag * x.values[rows]

    def mult(self, x: Vec, y: Vec) -> None:
        """Multiply the stored numerical operator by an independent vector."""
        y.values[:] = self.values @ x.values


class IS:
    """Hold test-owned scatter indices."""

    def createGeneral(self, indices: Any, **kwargs: Any) -> "IS":
        """Store an arbitrary ordered subset of global indices."""
        self.indices = np.asarray(indices, dtype=int)
        return self

    def createStride(self, size: int, step: int = 0, **kwargs: Any) -> "IS":
        """Preserve PETSc's explicit stride semantics, including its zero default."""
        self.indices = np.arange(size) * step
        return self

    def destroy(self) -> None:
        """Expose index-set cleanup."""


class Scatter:
    """Apply index extraction with actual values under serial API simulation."""

    def create(self, vector: Vec, source: IS, target: Vec, destination: IS) -> "Scatter":
        """Retain the source and destination indexing contract."""
        self.source, self.destination = source, destination
        return self

    def scatter(self, vector: Vec, target: Vec, **kwargs: Any) -> None:
        """Extract only the requested coefficients in their specified order."""
        target.values[self.destination.indices] = vector.values[self.source.indices]

    def destroy(self) -> None:
        """Expose scatter-resource cleanup."""


class KSP:
    """Use NumPy's direct solver solely for the portable native-API contract."""

    corrupt = False

    def create(self, **kwargs: Any) -> "KSP":
        """Return the newly allocated test solver."""
        return self

    def setOperators(self, matrix: Mat) -> None:
        """Keep the constrained numerical matrix."""
        self.matrix = matrix

    def setType(self, name: str) -> None:
        """Accept explicit direct-solver/preconditioner selection."""

    def getPC(self) -> "KSP":
        """Represent the solver's LU preconditioner configuration."""
        return self

    def setFactorSolverType(self, name: str) -> None:
        """Accept the mandatory MUMPS selection in the simulated interface."""

    def setErrorIfNotConverged(self, value: bool) -> None:
        """Accept propagation of a native solve failure."""

    def solve(self, rhs: Vec, solution: Vec) -> None:
        """Solve the assembled system or expose an intentionally incorrect result."""
        solution.values[:] = np.linalg.solve(self.matrix.values, rhs.values)
        if self.corrupt:
            solution.values[:] += 1.0

    def destroy(self) -> None:
        """Expose direct-solver resource cleanup."""


PETSC = SimpleNamespace(
    Mat=Mat,
    Vec=Vec,
    KSP=KSP,
    IS=IS,
    Scatter=Scatter,
    Sys=SimpleNamespace(hasExternalPackage=lambda name: True),
    IntType=np.int32,
    COMM_SELF=None,
    InsertMode=SimpleNamespace(ADD_VALUES=1, INSERT_VALUES=0),
    Error=np.linalg.LinAlgError,
)
