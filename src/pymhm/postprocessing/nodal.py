"""Portable named nodal fields using the recorded polynomial basis and topology.

Evaluation uses archived Basix coefficient matrices directly. Mesh interfaces
keep independent one-sided values through an optional explicit cell owner.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any

import numpy as np

from pymhm.core.validation import FloatArray, positive_int, real_array
from pymhm.fem.geometry import pullback_points
from pymhm.fem.reference import (
    ReferenceElementSpec,
    create_reference_element,
    simplex_lagrange_basis,
    tabulate_archived_nodal_basis,
)
from pymhm.fem.scalar.quadrilateral import qk_space
from pymhm.fem.scalar.tetrahedron import tetra_nodal_space
from pymhm.fem.scalar.tetrahedron_topology import tetra_indices
from pymhm.fem.scalar.triangle import multiindices, nodal_space
from pymhm.meshes.cartesian import CartesianMacroMesh
from pymhm.meshes.tetrahedron import TetraMesh
from pymhm.meshes.triangle import TriangleMesh
from pymhm.postprocessing.fields import FieldDefinition


@dataclass(frozen=True)
class _NodalEvaluator:
    """Executed nodal map and polynomial matrix, without native mesh resources."""

    cell_type: str
    degree: int
    components: int
    dofs: Any
    size: int
    basis_matrix: Any
    derivative: bool = False

    def __post_init__(self) -> None:
        """Own immutable executed basis and topology through construction and pickle replay."""
        dofs = np.array(self.dofs, dtype=np.int64, copy=True)
        matrix = np.array(self.basis_matrix, dtype=float, copy=True)
        dofs.setflags(write=False)
        matrix.setflags(write=False)
        object.__setattr__(self, "dofs", dofs)
        object.__setattr__(self, "basis_matrix", matrix)

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        """Replay the executed matrix rather than regenerate native basis coefficients."""
        return type(self), (
            self.cell_type,
            self.degree,
            self.components,
            self.dofs,
            self.size,
            self.basis_matrix,
            self.derivative,
        )

    def __call__(
        self, mesh: Any, coefficients: Any, points: Any, *, cells: Any = None
    ) -> FloatArray:
        """Evaluate the recorded basis after affine geometry pullback."""
        dimension = mesh.points.shape[1]
        points = real_array(points, "field points")
        if points.ndim != 2 or points.shape[1] != dimension:
            raise ValueError("field points must have shape (point, physical_dimension)")
        values = real_array(coefficients, "field coefficients")
        if values.shape != (self.size * self.components,):
            raise ValueError("field coefficients must match the recorded nodal map")
        owners, reference, jacobians, _ = pullback_points(mesh, points, cells=cells)
        if self.derivative:
            if isinstance(mesh, (TriangleMesh, TetraMesh, CartesianMacroMesh)):
                # Affine Jacobians depend on the cell, not the sampling point.
                # Keep the original point order, including incident duplicates.
                _, first, ordering = np.unique(owners, return_index=True, return_inverse=True)
                inverse = np.linalg.inv(jacobians[first])[ordering]
            else:
                inverse = np.linalg.inv(jacobians)
        if self.cell_type == "quadrilateral":
            factors = [
                tabulate_archived_nodal_basis(
                    "interval",
                    self.degree,
                    self.basis_matrix,
                    reference[:, axis : axis + 1],
                    nderiv=1 if self.derivative else 0,
                )
                for axis in range(dimension)
            ]
            if self.derivative:
                x, y = factors
                tables = np.einsum("qi,qj->qij", y[0], x[0]).reshape(len(points), -1)
                derivative = np.stack(
                    (
                        np.einsum("qi,qj->qij", y[0], x[1]),
                        np.einsum("qi,qj->qij", y[1], x[0]),
                    ),
                    axis=-1,
                ).reshape(len(points), -1, dimension)
                gradient = np.einsum("qid,qda->qia", derivative, inverse)
            else:
                tables = np.einsum("qi,qj->qij", factors[1], factors[0]).reshape(len(points), -1)
        else:
            tables = tabulate_archived_nodal_basis(
                self.cell_type,
                self.degree,
                self.basis_matrix,
                reference,
                nderiv=1 if self.derivative else 0,
            )
            if self.derivative:
                gradient = np.einsum("dqi,qda->qia", tables[1 : dimension + 1], inverse)
        local = values.reshape(self.size, self.components)[self.dofs[owners]]
        if self.derivative:
            output = np.einsum("qia,qic->qca", gradient, local)
        else:
            output = np.einsum("qi,qic->qc", tables, local)
        return output[:, 0] if self.components == 1 else output


def nodal_field(
    name: str,
    mesh: TriangleMesh | TetraMesh | CartesianMacroMesh,
    degree: int,
    *,
    components: int = 1,
    discontinuous: bool = False,
    reconstruction: Any = None,
    trace_reconstruction: Any = None,
    trace_dofs: Any = None,
    offset: Any = None,
) -> FieldDefinition:
    """Declare a scalar/vector Pk/Qk field with portable evaluation and gradients.

    Components are interleaved at each node. Continuous simplex coordinates use
    the package's topological nodal order; DG coordinates concatenate that local
    order by cell, with degree zero representing one cell value. Cartesian Qk
    coordinates use x-fastest nodes and the actual archived interval factors
    used by Qk assembly; no independent tensor basis is regenerated.
    ``reconstruction`` maps the user's complete
    local unknown vector into this field's coordinates, including block selection
    and scaling. The executed basis matrix and DOF map are retained by the
    picklable evaluators and identified by ``basis_id``.
    """
    degree = positive_int(degree, "degree", 0)
    components = positive_int(components, "components")
    if not isinstance(discontinuous, bool):
        raise TypeError("discontinuous must be boolean")
    if degree == 0 and not discontinuous:
        raise ValueError("degree-zero nodal fields must be discontinuous")
    if isinstance(mesh, CartesianMacroMesh):
        cell_type = "quadrilateral"
        if degree:
            coordinate = np.linspace(0, 1, degree + 1)
            basis = simplex_lagrange_basis(
                "interval", degree, nodes=np.column_stack((1 - coordinate, coordinate))
            )
            dofs, physical = qk_space(mesh, degree)
    elif isinstance(mesh, (TriangleMesh, TetraMesh)):
        cell_type = "triangle" if isinstance(mesh, TriangleMesh) else "tetrahedron"
        if degree:
            nodes = multiindices(degree) if cell_type == "triangle" else tetra_indices(degree)
            basis = simplex_lagrange_basis(cell_type, degree, nodes=nodes / degree)
            dofs, physical = (
                nodal_space(mesh, degree)
                if isinstance(mesh, TriangleMesh)
                else tetra_nodal_space(mesh, degree)
            )
    else:
        raise TypeError("nodal fields require a triangular, tetrahedral or Cartesian mesh")
    if degree:
        matrix = basis.basis_matrix.copy()
        if discontinuous:
            width = (degree + 1) ** 2 if cell_type == "quadrilateral" else len(matrix)
            dofs = np.arange(len(mesh.cells) * width).reshape(len(mesh.cells), width)
            size = dofs.size
        else:
            size = len(physical)
    else:
        matrix = create_reference_element(
            ReferenceElementSpec(
                "P",
                "interval" if cell_type == "quadrilateral" else cell_type,
                0,
                discontinuous=True,
            )
        ).basis_matrix.copy()
        dofs = np.arange(len(mesh.cells))[:, None]
        size = len(mesh.cells)
    matrix.setflags(write=False)
    dofs.setflags(write=False)
    identity = sha256(matrix.tobytes() + dofs.tobytes()).hexdigest()
    evaluator = _NodalEvaluator(cell_type, degree, components, dofs, size, matrix)
    gradient = _NodalEvaluator(cell_type, degree, components, dofs, size, matrix, True)
    return FieldDefinition(
        name,
        mesh=mesh,
        evaluator=evaluator,
        reconstruction=reconstruction,
        basis_id=f"{cell_type}:nodal:{degree}:{components}:{identity}",
        gradient_evaluator=gradient,
        trace_reconstruction=trace_reconstruction,
        trace_dofs=trace_dofs,
        offset=offset,
    )
