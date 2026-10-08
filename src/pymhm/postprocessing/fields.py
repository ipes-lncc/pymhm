"""Named field views carrying the executed coefficient and evaluation contract."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

import numpy as np
from scipy import sparse

from pymhm.core.spaces import validate_trace_binding
from pymhm.core.validation import FloatArray, positive_int, real_array


def _immutable_matrix(value: Any, name: str) -> Any:
    """Own finite real dense/sparse reconstruction coordinates in immutable storage."""
    if sparse.issparse(value):
        matrix = sparse.csr_matrix(value, copy=True)
        matrix.data = real_array(matrix.data, name)
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        matrix.sort_indices()
        for array in (matrix.data, matrix.indices, matrix.indptr):
            array.setflags(write=False)
    else:
        matrix = real_array(value, name)
        matrix.setflags(write=False)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a matrix")
    return matrix


@dataclass(frozen=True)
class FieldDefinition:
    """Portable description of one physical field reconstructed from a local vector.

    A native nodal descriptor stores its executed basis/map. A custom evaluator
    has signature evaluator(mesh, coefficients, points, *, cells=None); its
    declared basis_id identifies the coordinate convention. reconstruction maps
    local coefficients to the evaluator's coefficients. Without an evaluator or
    descriptor, the field supports coefficient access but no spatial evaluation.
    The reconstructed coefficients are R*local + T*global_trace[trace_dofs] +
    offset. Every trace orientation and basis change is explicit in T; no
    incident normal sign or local binding is inferred. All maps and the affine
    offset form part of the archived coefficient and basis-digest contract.
    """

    name: str
    mesh: Any = None
    descriptor: Any = None
    evaluator: Callable[..., Any] | None = None
    reconstruction: Any = None
    basis_id: str = ""
    gradient_evaluator: Callable[..., Any] | None = None
    trace_reconstruction: Any = None
    trace_dofs: Any = None
    offset: Any = None

    def __post_init__(self) -> None:
        """Own a reconstruction map and reject ambiguous evaluation conventions."""
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("field name must be a nonempty string")
        if self.evaluator is not None and not callable(self.evaluator):
            raise TypeError("field evaluator must be callable")
        if self.gradient_evaluator is not None and not callable(self.gradient_evaluator):
            raise TypeError("field gradient_evaluator must be callable")
        if self.descriptor is not None and (
            self.evaluator is not None or self.gradient_evaluator is not None
        ):
            raise ValueError("declare one field evaluation convention")
        if not isinstance(self.basis_id, str):
            raise TypeError("field basis_id must be a string")
        if (
            self.evaluator is not None or self.gradient_evaluator is not None
        ) and not self.basis_id:
            raise ValueError("custom field evaluation requires a declared basis_id")
        if self.reconstruction is not None:
            matrix = _immutable_matrix(self.reconstruction, "field reconstruction")
            if self.descriptor is not None and matrix.shape[0] != self.descriptor.size:
                raise ValueError("field reconstruction must produce the descriptor's coordinates")
            object.__setattr__(self, "reconstruction", matrix)
        if (self.trace_reconstruction is None) != (self.trace_dofs is None):
            raise ValueError(
                "trace reconstruction and explicit global trace coordinates are paired"
            )
        if self.trace_reconstruction is not None:
            matrix = _immutable_matrix(self.trace_reconstruction, "field trace reconstruction")
            indices = np.asarray(self.trace_dofs)
            if indices.size == 0 and indices.ndim == 1:
                indices = indices.astype(np.int64)
            if (
                indices.ndim != 1
                or indices.dtype.kind not in "iu"
                or np.any(indices < 0)
                or len(np.unique(indices)) != len(indices)
                or np.any(indices > np.iinfo(np.int64).max)
            ):
                raise ValueError("field trace coordinates must be distinct nonnegative integers")
            indices = np.array(indices, dtype=np.int64, copy=True)
            indices.setflags(write=False)
            if matrix.shape[1] != len(indices):
                raise ValueError(
                    "field trace reconstruction must match its global trace coordinates"
                )
            width = (
                self.reconstruction.shape[0]
                if self.reconstruction is not None
                else self.descriptor.size
                if self.descriptor is not None
                else None
            )
            if width is not None and matrix.shape[0] != width:
                raise ValueError(
                    "field local and trace reconstructions must produce the same coordinates"
                )
            object.__setattr__(self, "trace_reconstruction", matrix)
            object.__setattr__(self, "trace_dofs", indices)
        if self.offset is not None:
            offset = real_array(self.offset, "field affine offset")
            if offset.ndim != 1:
                raise ValueError("field affine offset must be a vector")
            width = (
                self.reconstruction.shape[0]
                if self.reconstruction is not None
                else self.trace_reconstruction.shape[0]
                if self.trace_reconstruction is not None
                else self.descriptor.size
                if self.descriptor is not None
                else None
            )
            if width is not None and len(offset) != width:
                raise ValueError("field affine offset must match its reconstructed coordinates")
            offset.setflags(write=False)
            object.__setattr__(self, "offset", offset)

    @property
    def basis_matrix(self) -> FloatArray:
        """Return an owned read-only copy of a declared single polynomial matrix.

        Nodal/native and modal evaluators declare this coordinate matrix. A
        custom or composite Piola basis may require several matrices and then
        raises TypeError. Archive this entire definition with its digest to
        retain cell maps, orientation and reconstruction as well as the matrix.
        """
        owner = self.descriptor if self.descriptor is not None else self.evaluator
        matrix = getattr(owner, "basis_matrix", None)
        if matrix is None:
            raise TypeError("this field has no declared single basis matrix")
        return _immutable_matrix(matrix, "field basis matrix")

    @property
    def basis_digest(self) -> str:
        """Identify the executed descriptor and any additional reconstruction map."""
        digest = sha256(self.basis_id.encode())
        digest.update(getattr(self.descriptor, "basis_digest", "").encode())
        for name, value in (
            ("", self.reconstruction),
            ("trace_reconstruction", self.trace_reconstruction),
            ("trace_dofs", self.trace_dofs),
            ("offset", self.offset),
        ):
            if value is None:
                continue
            digest.update(name.encode())
            digest.update(str(value.shape).encode())
            arrays = (
                (value.data, value.indices, value.indptr) if sparse.issparse(value) else (value,)
            )
            for array in arrays:
                digest.update(array.dtype.str.encode())
                digest.update(array.tobytes())
        return digest.hexdigest()

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        """Restore an owned immutable reconstruction through the validating constructor."""
        return type(self), (
            self.name,
            self.mesh,
            self.descriptor,
            self.evaluator,
            self.reconstruction,
            self.basis_id,
            self.gradient_evaluator,
            self.trace_reconstruction,
            self.trace_dofs,
            self.offset,
        )


@dataclass(frozen=True)
class DiscreteField:
    """One reconstructed field associated with its mesh and executed basis.

    Coefficients are independently owned and read-only. Separate macrocell fields
    preserve one-sided interface values; evaluation does not average or smooth.
    """

    definition: FieldDefinition
    coefficients: Any

    def __post_init__(self) -> None:
        """Freeze the reconstructed field's finite real coefficients."""
        values = real_array(self.coefficients, "field coefficients")
        if values.ndim != 1:
            raise ValueError("field coefficients must be a vector")
        values.setflags(write=False)
        object.__setattr__(self, "coefficients", values)

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        """Restore immutable coefficients through the same checked field constructor."""
        return type(self), (self.definition, self.coefficients)

    @property
    def mesh(self) -> Any:
        """Return the declared physical mesh, without constructing native resources."""
        return (
            self.definition.mesh
            if self.definition.descriptor is None
            else self.definition.descriptor.mesh
        )

    @property
    def basis_digest(self) -> str:
        """Return the field's executed basis/reconstruction identity."""
        return self.definition.basis_digest

    @property
    def portable_coefficients(self) -> FloatArray:
        """Delegate coefficient access in the descriptor's recorded portable order."""
        return portable_field_coefficients(self)

    def evaluate(self, points: Any, *, cells: Any = None) -> FloatArray:
        """Delegate spatial evaluation with optional explicit one-sided cell owners."""
        return evaluate_field(self, points, cells=cells)

    def gradient(self, points: Any, *, cells: Any = None) -> FloatArray:
        """Delegate physical gradients, retaining explicit one-sided cell owners."""
        return evaluate_field_gradient(self, points, cells=cells)

    def values_and_gradient(
        self, points: Any, *, cells: Any = None
    ) -> tuple[FloatArray, FloatArray]:
        """Delegate values and physical gradients with a shared native sampling pass."""
        return evaluate_field_and_gradient(self, points, cells=cells)


def evaluate_field(field: DiscreteField, points: Any, *, cells: Any = None) -> FloatArray:
    """Evaluate a declared native or custom field; reject absent capabilities."""
    definition = field.definition
    if definition.descriptor is not None:
        from pymhm.backends.spaces import evaluate_descriptor

        return evaluate_descriptor(definition.descriptor, field.coefficients, points, cells=cells)
    if definition.evaluator is None:
        raise TypeError("this field has no declared spatial evaluation capability")
    return real_array(
        definition.evaluator(field.mesh, field.coefficients, points, cells=cells), "field values"
    )


def evaluate_field_gradient(field: DiscreteField, points: Any, *, cells: Any = None) -> FloatArray:
    """Evaluate physical derivatives in the executed native or custom field basis.

    Scalar gradients have axes (point, coordinate); vector gradients use
    (point, component, coordinate). These are raw derivatives, not a conservative
    H(div) flux reconstruction. Custom fields declare gradient_evaluator with
    the same arguments as their value evaluator; absent capabilities fail.
    """
    definition = field.definition
    if definition.descriptor is not None:
        from pymhm.backends.spaces import evaluate_descriptor_gradient

        return evaluate_descriptor_gradient(
            definition.descriptor, field.coefficients, points, cells=cells
        )
    if definition.gradient_evaluator is None:
        raise TypeError("this field has no declared gradient evaluation capability")
    return real_array(
        definition.gradient_evaluator(field.mesh, field.coefficients, points, cells=cells),
        "field gradients",
    )


def portable_field_coefficients(field: DiscreteField) -> FloatArray:
    """Return coefficients through the executed descriptor's portable coordinate map.

    The result is independently owned and read-only, with mixed components
    selected by their stored map. This only changes ordering; it does not
    regenerate the basis or discard its digest. These are nodal values when
    the executed basis is the corresponding nodal basis. A custom field without
    this declared map has no automatic portable coefficient conversion.
    """
    descriptor = field.definition.descriptor
    if descriptor is None:
        raise TypeError("this field has no declared portable coefficient map")
    result = field.coefficients[descriptor.mapping]
    result.setflags(write=False)
    return result


def evaluate_field_and_gradient(
    field: DiscreteField, points: Any, *, cells: Any = None
) -> tuple[FloatArray, FloatArray]:
    """Sample values and raw physical derivatives without duplicate native geometry.

    Native descriptors share cell location, geometry pullback and executed basis
    tabulation. Custom fields delegate to their independent value and gradient
    callbacks. Output axes and one-sided conventions match evaluate_field and
    evaluate_field_gradient; this does not construct a conservative flux.
    """
    if field.definition.descriptor is not None:
        from pymhm.backends.spaces import evaluate_descriptor_data

        return evaluate_descriptor_data(
            field.definition.descriptor, field.coefficients, points, cells=cells
        )
    return evaluate_field(field, points, cells=cells), evaluate_field_gradient(
        field, points, cells=cells
    )


def solution_field(
    solution: Any, name: str, *, recursive: bool = False
) -> tuple[DiscreteField, ...]:
    """Recover named fields in deterministic macrocell order.

    With recursive=True, a cell without its own definition delegates to its
    recovered child solution. Cells declaring the field are visited once; their
    children are not also returned. Each reconstruction uses its own level's
    trace vector, retaining one-sided values and executed orientations.
    """
    if not isinstance(recursive, bool):
        raise TypeError("recursive must be boolean")
    result: list[DiscreteField] = []
    for index, (definitions, values) in enumerate(
        zip(solution.field_data, solution.fields, strict=True)
    ):
        matches = [definition for definition in definitions if definition.name == name]
        if recursive and not matches:
            children = getattr(solution, "children", ())
            child = children[index] if index < len(children) else None
            if child is not None:
                result.extend(solution_field(child, name, recursive=True))
                continue
        if len(matches) != 1:
            raise KeyError(f"field {name!r} must be declared once in each local context")
        definition = matches[0]
        coefficients = values
        if definition.reconstruction is not None:
            if definition.reconstruction.shape[1] != len(values):
                raise ValueError("field reconstruction does not match the executed local vector")
            coefficients = definition.reconstruction @ values
        if definition.trace_reconstruction is not None:
            if np.any(definition.trace_dofs >= len(solution.trace)):
                raise ValueError("field trace coordinates must belong to the executed global trace")
            coefficients = (
                coefficients
                + definition.trace_reconstruction @ solution.trace[definition.trace_dofs]
            )
        if definition.offset is not None:
            if definition.offset.shape != coefficients.shape:
                raise ValueError(
                    "field affine offset must match the executed reconstructed coordinates"
                )
            coefficients = coefficients + definition.offset
        result.append(DiscreteField(definition, coefficients))
    if not result:
        raise KeyError(f"field {name!r} is not declared")
    return tuple(result)


def local_trace(solution: Any, index: int, *, test: bool = False) -> FloatArray:
    """Recover selected global coefficients in their executed local trace basis.

    index is a position in deterministic assembly item order, like solution.fields.
    The trial map is used by default; test=True uses the independent test map.
    Both maps already declare geometric orientation and basis changes. This
    operation does not infer a physical boundary sign or average adjacent fields.
    Explicit equations without a stored binding have no local-trace capability.
    """
    index = positive_int(index, "local trace index", 0)
    if not isinstance(test, bool):
        raise TypeError("test must be boolean")
    if index >= len(solution.trace_bindings):
        raise ValueError("local trace index outside the executed binding layout")
    binding = solution.trace_bindings[index]
    if binding is None:
        raise TypeError("this solution has no declared local trace binding")
    binding = validate_trace_binding(binding, len(solution.trace))
    dofs, transform = (
        (binding.test_dofs, binding.test_map) if test else (binding.dofs, binding.trial_map)
    )
    return transform @ solution.trace[dofs]
