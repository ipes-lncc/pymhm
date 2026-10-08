"""Broken scalar/vector polynomial fields in explicitly ordered modal coordinates."""

from dataclasses import dataclass
from hashlib import sha256
from typing import Any

import numpy as np

from pymhm.core.validation import positive_int, real_array
from pymhm.fem.geometry import pullback_points
from pymhm.fem.reference import monomial_coefficients, tabulate_archived_basis
from pymhm.postprocessing.fields import FieldDefinition


@dataclass(frozen=True)
class _ModalEvaluator:
    """Literal interval polynomial matrix and explicit multiindices on each fine cell."""

    degree: int
    powers: tuple[tuple[int, ...], ...]
    basis_matrix: Any
    components: int
    derivative: bool = False

    def __post_init__(self) -> None:
        """Own the executed polynomial matrix and retain it on pickle replay."""
        matrix = np.array(self.basis_matrix, copy=True)
        matrix.setflags(write=False)
        object.__setattr__(self, "basis_matrix", matrix)

    def __reduce__(self) -> Any:
        """Replay the literal matrix instead of deriving a new modal coordinate map."""
        return type(self), (
            self.degree,
            self.powers,
            self.basis_matrix,
            self.components,
            self.derivative,
        )

    def __call__(self, mesh: Any, coefficients: Any, points: Any, *, cells: Any = None) -> Any:
        """Evaluate cellwise polynomial values or their physical Cartesian gradient."""
        owners, coordinates, jacobian, _ = pullback_points(mesh, points, cells=cells)
        values = real_array(coefficients, "modal coefficients")
        width, dimension = len(self.powers), coordinates.shape[1]
        if values.shape != (len(mesh.cells) * width * self.components,):
            raise ValueError("modal coefficients must match the declared cell/polynomial map")
        powers = np.array(self.powers)
        factors = [
            tabulate_archived_basis(
                "interval",
                self.degree,
                self.basis_matrix,
                coordinates[:, a : a + 1],
                nderiv=int(self.derivative),
            )
            for a in range(dimension)
        ]
        local = values.reshape(len(mesh.cells), width, self.components)[owners]
        basis = np.ones((len(points), width))
        for a in range(dimension):
            basis *= (factors[a][0] if self.derivative else factors[a])[:, powers[:, a]]
        if self.derivative:
            reference_gradient = np.ones((len(points), width, dimension))
            for axis in range(dimension):
                for a in range(dimension):
                    reference_gradient[..., axis] *= factors[a][int(a == axis)][:, powers[:, a]]
            gradient = np.einsum("qid,qda->qia", reference_gradient, np.linalg.inv(jacobian))
            output = np.einsum("qia,qic->qca", gradient, local)
        else:
            output = np.einsum("qi,qic->qc", basis, local)
        return output[:, 0] if self.components == 1 else output


def modal_field(
    name: str,
    mesh: Any,
    powers: tuple[tuple[int, ...], ...],
    *,
    convention: str = "monomial",
    components: int = 1,
    reconstruction: Any = None,
) -> FieldDefinition:
    """Declare cellwise polynomial coefficients with an explicit exponent order.

    ``monomial`` uses reference coordinate powers. ``legendre`` uses conventional
    L_j(2*x-1), normalized to one at x=1, independently in each direction. This
    covers complete simplex, tensor Qk and restricted complete tensor Pk spaces
    through the supplied multiindices. Coefficients interleave physical components
    within a polynomial, then concatenate cells. The actual interval coordinate
    matrix is archived and reused for evaluation and physical gradients.
    """
    components = positive_int(components, "components")
    dimension = mesh.points.shape[1]
    if not powers or any(
        len(e) != dimension or any(not isinstance(a, int) or a < 0 for a in e) for e in powers
    ):
        raise ValueError("modal powers must be nonnegative multiindices in the physical dimension")
    degree = max(max(e) for e in powers)
    if convention == "monomial":
        matrix = monomial_coefficients(degree).T
    elif convention == "legendre":
        matrix = np.diag(1 / np.sqrt(2 * np.arange(degree + 1) + 1))
    else:
        raise ValueError("modal convention must be monomial or legendre")
    evaluator = _ModalEvaluator(degree, powers, matrix, components)
    gradient = _ModalEvaluator(degree, powers, matrix, components, True)
    digest = sha256(repr((powers, convention, components)).encode())
    digest.update(evaluator.basis_matrix.tobytes())
    return FieldDefinition(
        name,
        mesh=mesh,
        evaluator=evaluator,
        gradient_evaluator=gradient,
        reconstruction=reconstruction,
        basis_id=digest.hexdigest(),
    )
