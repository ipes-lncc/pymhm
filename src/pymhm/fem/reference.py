"""Basix reference-element tabulation with explicit executed basis data.

Basix supplies polynomial bases, derivatives and entity transformations. Its
native family/degree convention is retained: for example Basix RT degree one
is the lowest-order RT element. Reference capabilities do not establish the
compatibility, physical conformity or error estimates of a hybrid formulation.
The native library is loaded when tabulation is requested; algebraic providers
do not require a native element handle.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import import_module
from math import factorial
from threading import RLock
from types import ModuleType
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from pymhm.core.validation import FloatArray, IntArray

_ELEMENT_LOCK = RLock()


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    """Validate integer orders without accepting booleans or truncating floats."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _readonly(values: Any) -> FloatArray:
    """Copy native binary64 data so clients cannot mutate a cached basis."""
    result = np.array(values, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _digest(matrix: FloatArray) -> str:
    """Identify literal C-order binary64 basis coefficients."""
    return hashlib.sha256(matrix.tobytes(order="C")).hexdigest()


@dataclass(frozen=True)
class ReferenceElementSpec:
    """Declare an element using native Basix family, cell and degree names.

    Basix checks supported family/cell/order/variant combinations; this record
    introduces no degree shifts or new admissibility hypotheses. Family aliases
    are resolved by Basix's own ``string_to_family``. ``dof_ordering`` follows
    Basix's native old-to-new permutation convention: index ``j`` contains the
    executed slot of canonical DOF ``j``. An empty tuple requests Basix's
    default order, as does ``None``. Supported permutation/family choices
    are checked by Basix. Elements use binary64. Descriptors are picklable;
    create the native element separately inside each worker.
    """

    family: str
    cell: str
    degree: int
    lagrange_variant: str = "unset"
    dpc_variant: str = "unset"
    discontinuous: bool = False
    dof_ordering: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        """Freeze the declared permutation and reject malformed specifications."""
        for name in ("family", "cell", "lagrange_variant", "dpc_variant"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError(f"{name} must be a nonempty native name without outer whitespace")
        object.__setattr__(self, "degree", _integer(self.degree, "degree"))
        if not isinstance(self.discontinuous, bool):
            raise ValueError("discontinuous must be a boolean")
        if self.dof_ordering is not None:
            order = tuple(_integer(index, "dof_ordering") for index in self.dof_ordering)
            if sorted(order) != list(range(len(order))):
                raise ValueError("dof_ordering must be a permutation of consecutive indices")
            object.__setattr__(self, "dof_ordering", order)


@dataclass(frozen=True)
class ReferenceElement:
    """A process-local native element and read-only executed basis metadata.

    ``basis_matrix`` contains ``FiniteElement.coefficient_matrix`` after native
    dualization, with rows reordered into the actual executed DOF order, rather
    than ``wcoeffs`` defining only a polynomial span. Its digest identifies the
    literal binary64 matrix. Replaying coefficients also
    requires the declared cell, family, variants, order, polyset and backend
    version. Native handles remain inside their worker; persist metadata and
    actual basis arrays rather than serializing this object across processes.
    """

    spec: ReferenceElementSpec
    backend_version: str
    cell_dimension: int
    dimension: int
    value_shape: tuple[int, ...]
    map_type: str
    sobolev_space: str
    polyset_type: str
    basis_matrix: FloatArray = field(repr=False, compare=False)
    basis_sha256: str
    _native: Any = field(repr=False, compare=False)


def _require_basix() -> ModuleType:
    """Load the reference-element dependency only when needed."""
    try:
        return import_module("basix")
    except ImportError as exc:
        raise ImportError("Basix tabulation requires the fenics-basix runtime dependency") from exc


@lru_cache(maxsize=32)
def _cached_reference_element(spec: ReferenceElementSpec) -> ReferenceElement:
    """Create one native basis per descriptor under the cache owner's lock."""
    basix = _require_basix()
    try:
        cell = basix.CellType[spec.cell]
        family = (
            basix.ElementFamily[spec.family]
            if spec.family in basix.ElementFamily.__members__
            else basix.finite_element.string_to_family(spec.family, spec.cell)
        )
        lagrange = basix.LagrangeVariant[spec.lagrange_variant]
        dpc = basix.DPCVariant[spec.dpc_variant]
    except (KeyError, ValueError) as exc:
        raise ValueError(f"unknown Basix element name in {spec}") from exc
    native = basix.create_element(
        family,
        cell,
        spec.degree,
        lagrange_variant=lagrange,
        dpc_variant=dpc,
        discontinuous=spec.discontinuous,
        dof_ordering=None if spec.dof_ordering is None else list(spec.dof_ordering),
        dtype=np.float64,
    )
    coefficients = native.coefficient_matrix
    if spec.dof_ordering:
        coefficients = coefficients[np.argsort(spec.dof_ordering)]
    coefficients = _readonly(coefficients)
    return ReferenceElement(
        spec=spec,
        backend_version=str(basix.__version__),
        cell_dimension=int(basix.geometry(cell).shape[1]),
        dimension=int(native.dim),
        value_shape=tuple(native.value_shape),
        map_type=native.map_type.name,
        sobolev_space=native.sobolev_space.name,
        polyset_type=native.polyset_type.name,
        basis_matrix=coefficients,
        basis_sha256=_digest(coefficients),
        _native=native,
    )


def create_reference_element(spec: ReferenceElementSpec) -> ReferenceElement:
    """Create or reuse a read-only Basix element with its native conventions.

    Supported shapes, families, variants and orders are inherited from the
    installed Basix version. Unsupported combinations raise its creation error.
    The cache is process-local and synchronized during creation; tabulation
    does not modify the native element or cached basis metadata.
    """
    if not isinstance(spec, ReferenceElementSpec):
        raise TypeError("spec must be a ReferenceElementSpec")
    with _ELEMENT_LOCK:
        return _cached_reference_element(spec)


def _points(points: ArrayLike, dimension: int, name: str = "points") -> FloatArray:
    """Validate finite real reference coordinates, including exterior points."""
    raw = np.asarray(points)
    if raw.dtype.kind not in "fiu" or not np.isfinite(raw).all():
        raise ValueError(f"{name} must contain finite real coordinates")
    if raw.ndim != 2 or raw.shape[1] != dimension:
        raise ValueError(f"{name} must have shape (n, {dimension})")
    return np.ascontiguousarray(raw, dtype=np.float64)


def tabulate_reference(element: ReferenceElement, points: ArrayLike, nderiv: int = 0) -> FloatArray:
    """Evaluate native basis values and Cartesian derivatives through ``nderiv``.

    The binary64 output has axes ``(derivative, point, basis, value_component)``.
    Scalar elements retain a final axis of size one; vector/tensor values use
    Basix's flattened component order. Derivatives follow ``basix.index``; any
    nonnegative derivative order supported by Basix is allowed. Coordinates
    have shape ``(points, cell_dimension)`` and may lie outside the cell. No
    physical Piola map, cell orientation or global DOF identification is applied.
    """
    if not isinstance(element, ReferenceElement):
        raise TypeError("element must be a ReferenceElement")
    order = _integer(nderiv, "nderiv")
    coordinates = _points(points, element.cell_dimension)
    return np.asarray(element._native.tabulate(order, coordinates), dtype=np.float64)


def tabulate_archived_basis(
    cell: str,
    degree: int,
    basis_matrix: ArrayLike,
    points: ArrayLike,
    *,
    components: int = 1,
    nderiv: int = 0,
) -> FloatArray:
    """Evaluate an executed scalar/vector polynomial basis from its archived matrix.

    ``basis_matrix`` contains rows of the actual Basix ``coefficient_matrix``
    in its native orthonormal Legendre polynomial coordinates. Its rows may
    include a declared change of basis. The returned array has axes
    ``(points, executed_basis)`` for a scalar and
    ``(points, executed_basis, components)`` for a vector for ``nderiv=0``.
    Positive derivative orders
    add a leading derivative axis in Basix's ``index`` convention, including
    values at index zero. Evaluation uses the supplied matrix directly, without
    dualizing or rebuilding finite-element basis coefficients. Physical maps,
    continuity, vector blocking and global DOF order belong to the caller.
    """
    basix = _require_basix()
    order = _integer(degree, "degree")
    derivatives = _integer(nderiv, "nderiv")
    try:
        native_cell = basix.CellType[cell]
    except KeyError as error:
        raise ValueError("unknown archived basis reference cell") from error
    dimension = basix.geometry(native_cell).shape[1]
    coordinates = _points(points, dimension, "points")
    matrix = np.asarray(basis_matrix)
    if matrix.dtype.kind not in "fiu" or matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("archived basis_matrix must be a finite real matrix")
    width = basix.polynomials.dim(basix.PolynomialType.legendre, native_cell, order)
    components = _integer(components, "components", minimum=1)
    if matrix.shape[1] != components * width:
        raise ValueError("archived basis columns must match the reference polynomial dimension")
    if derivatives == 0:
        polynomials = basix.polynomials.tabulate_polynomials(
            basix.PolynomialType.legendre, native_cell, order, coordinates
        )
        if components == 1:
            return np.asarray(polynomials.T @ matrix.T, dtype=np.float64)
        return np.einsum("pq,iap->qia", polynomials, matrix.reshape(-1, components, width))
    tables = basix.polynomials.tabulate_polynomial_set(
        native_cell, basix.PolysetType.standard, order, derivatives, coordinates
    )
    if components == 1:
        return np.stack([table.T @ matrix.T for table in tables])
    return np.einsum("dpq,iap->dqia", tables, matrix.reshape(-1, components, width))


def tabulate_archived_nodal_basis(
    cell: str, degree: int, basis_matrix: ArrayLike, points: ArrayLike, *, nderiv: int = 0
) -> FloatArray:
    """Evaluate the literal executed scalar basis in archived orthonormal coordinates.

    Return (point,basis), or (derivative,point,basis) when nderiv is positive.
    No finite element or independent coefficient matrix is regenerated.
    """
    return tabulate_archived_basis(cell, degree, basis_matrix, points, nderiv=nderiv)


def reference_interpolation_points(element: ReferenceElement) -> FloatArray:
    """Return read-only native interpolation points in reference coordinates."""
    if not isinstance(element, ReferenceElement):
        raise TypeError("element must be a ReferenceElement")
    return _readonly(element._native.points)


def interpolate_reference(element: ReferenceElement, values: ArrayLike) -> FloatArray:
    """Apply native value interpolation to several candidate functions.

    Input axes are (interpolation_point, candidate, flattened_value_component).
    The output has axes (executed_dof, candidate), including a declared DOF
    permutation. Only value-based functionals are accepted; derivative-based
    elements require derivative data and are outside this helper's contract.
    No physical Piola map or entity orientation is inferred.
    """
    if not isinstance(element, ReferenceElement):
        raise TypeError("element must be a ReferenceElement")
    if element._native.interpolation_nderivs:
        raise ValueError("value interpolation does not accept derivative-based functionals")
    raw = np.asarray(values)
    if (
        raw.dtype.kind not in "fiu"
        or not np.isfinite(raw).all()
        or raw.ndim != 3
        or raw.shape[0] != len(element._native.points)
        or raw.shape[2] != int(np.prod(element.value_shape))
    ):
        raise ValueError(
            "values must have finite real shape (interpolation_point, candidate, value)"
        )
    data = (
        np.asarray(raw, dtype=np.float64)
        .transpose(2, 0, 1)
        .reshape(raw.shape[0] * raw.shape[2], raw.shape[1])
    )
    return np.asarray(element._native.interpolation_matrix @ data, dtype=np.float64)


def reference_entity_transformations(element: ReferenceElement) -> dict[str, FloatArray]:
    """Return read-only native entity maps, including H(div)/H(curl) signs.

    Matrices act in the intrinsic DOF order of each entity, even when a global
    ``dof_ordering`` is declared. :func:`reference_entity_dofs` gives those
    local DOFs' executed indices; embed these maps using those indices instead
    of applying a global permutation to a small entity matrix. Entity names
    retain Basix's reference orientation convention. They do not infer PyMHM
    macroface normals, mesh incidence or a physical trace map.
    """
    return {
        name: _readonly(matrix) for name, matrix in element._native.entity_transformations().items()
    }


def reference_entity_dofs(element: ReferenceElement) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Return each entity's intrinsic local order as executed DOF indices.

    Axes are ``(entity_dimension, entity_number, entity_local_dof)``. Basix
    already applies the declared old-to-new global permutation to these
    indices. Pair them with its unchanged entity-local transformations.
    """
    return tuple(
        tuple(tuple(int(dof) for dof in entity) for entity in dimension)
        for dimension in element._native.entity_dofs
    )


def reference_base_transformations(element: ReferenceElement) -> FloatArray:
    """Return full DOF maps in the actual executed basis order.

    Axes are ``(transformation, output_dof, input_dof)``. Basix supplies edge
    reversals and face rotations/reflections in canonical order. A declared
    ``dof_ordering`` conjugates those maps into the tabulation/basis order.
    No physical or global trace orientation is selected by this function.
    """
    transformations = element._native.base_transformations()
    if element.spec.dof_ordering:
        inverse = np.argsort(element.spec.dof_ordering)
        transformations = transformations[:, inverse][:, :, inverse]
    return _readonly(transformations)


@dataclass(frozen=True)
class NodalReferenceBasis:
    """An equispaced scalar basis in a caller's literal node order.

    ``permutation[j]`` identifies the native basis function for caller node j.
    ``basis_matrix`` and its digest capture the actually reordered coefficient
    rows. The native reference basis and declared permutation must be preserved
    together when persisting coefficients. Arrays are read-only cache data.
    """

    element: ReferenceElement
    nodes: FloatArray = field(repr=False, compare=False)
    permutation: IntArray = field(repr=False, compare=False)
    basis_matrix: FloatArray = field(repr=False, compare=False)
    basis_sha256: str


def _simplex_dimension(cell: str) -> int:
    """Return the dimension of supported affine simplex reference cells."""
    dimensions = {"interval": 1, "triangle": 2, "tetrahedron": 3}
    if cell not in dimensions:
        raise ValueError("simplex cell must be interval, triangle or tetrahedron")
    return dimensions[cell]


def _barycentric(points: ArrayLike, dimension: int, name: str) -> FloatArray:
    """Require coordinates on the unit-sum hyperplane, without an interior restriction."""
    result = _points(points, dimension + 1, name)
    scale = np.maximum(1, np.abs(result).sum(axis=1))
    if np.any(np.abs(result.sum(axis=1) - 1) > 32 * np.finfo(float).eps * scale):
        raise ValueError(f"{name} barycentric coordinates must sum to one")
    return result


@lru_cache(maxsize=32)
def _cached_nodal_basis(cell: str, degree: int, literal_nodes: bytes) -> NodalReferenceBasis:
    """Match all declared nodes uniquely to native equispaced interpolation nodes."""
    dimension = _simplex_dimension(cell)
    nodes = np.frombuffer(literal_nodes, dtype=np.float64).reshape(-1, dimension + 1)
    return _ordered_nodal_basis(cell, degree, nodes, nodes[:, 1:])


def _ordered_nodal_basis(
    cell: str, degree: int, nodes: FloatArray, locations: FloatArray
) -> NodalReferenceBasis:
    """Apply a bijective node-numbering map to one native equispaced element."""
    element = create_reference_element(
        ReferenceElementSpec("P", cell, degree, lagrange_variant="equispaced")
    )
    native_nodes = reference_interpolation_points(element)
    if len(nodes) != element.dimension or native_nodes.shape != locations.shape:
        raise ValueError("nodes must contain the complete equispaced nodal set")
    distances = np.max(np.abs(locations[:, None] - native_nodes[None]), axis=2)
    matches = distances <= 32 * np.finfo(float).eps
    if not (np.all(matches.sum(axis=1) == 1) and np.all(matches.sum(axis=0) == 1)):
        raise ValueError("nodes must uniquely match the native equispaced nodal set")
    permutation = np.argmax(matches, axis=1).astype(np.int64)
    permutation.setflags(write=False)
    coefficients = _readonly(element.basis_matrix[permutation])
    return NodalReferenceBasis(element, nodes, permutation, coefficients, _digest(coefficients))


def simplex_lagrange_basis(cell: str, degree: int, *, nodes: ArrayLike) -> NodalReferenceBasis:
    """Declare a native equispaced Pk basis using literal barycentric node order.

    Positive degree is unrestricted subject to Basix's implementation. Node
    coordinates must contain its complete equispaced lattice exactly once;
    this changes only numbering, never solves for a new polynomial basis or
    merges global nodes by geometric proximity. High-order equispaced nodal
    interpolation retains its inherent conditioning limitations.
    """
    dimension = _simplex_dimension(cell)
    order = _integer(degree, "degree", minimum=1)
    coordinates = _barycentric(nodes, dimension, "nodes")
    with _ELEMENT_LOCK:
        return _cached_nodal_basis(cell, order, coordinates.tobytes(order="C"))


@lru_cache(maxsize=32)
def _cached_tensor_basis(cell: str, degree: int, literal_nodes: bytes) -> NodalReferenceBasis:
    """Match a declared Cartesian tensor lattice to the native nodal order."""
    dimension = {"interval": 1, "quadrilateral": 2, "hexahedron": 3}[cell]
    nodes = np.frombuffer(literal_nodes, dtype=np.float64).reshape(-1, dimension)
    return _ordered_nodal_basis(cell, degree, nodes, nodes)


def tensor_lagrange_basis(cell: str, degree: int, *, nodes: ArrayLike) -> NodalReferenceBasis:
    """Declare a native equispaced Qk basis in explicit Cartesian node order.

    Cells are interval, quadrilateral or hexahedron; degree is positive.
    Nodes are a complete lattice of shape (basis, reference_dimension).
    The returned coefficient matrix follows that order, for archive/replay.
    """
    dimensions = {"interval": 1, "quadrilateral": 2, "hexahedron": 3}
    if cell not in dimensions:
        raise ValueError("tensor cell must be interval, quadrilateral or hexahedron")
    order = _integer(degree, "degree", minimum=1)
    coordinates = _points(nodes, dimensions[cell], "nodes")
    with _ELEMENT_LOCK:
        return _cached_tensor_basis(cell, order, coordinates.tobytes(order="C"))


def tensor_lagrange_tabulation(
    cell: str,
    degree: int,
    points: ArrayLike,
    *,
    nodes: ArrayLike,
    nderiv: int = 1,
) -> tuple[FloatArray, FloatArray]:
    """Return ordered native Qk values and Cartesian reference gradients.

    Axes are (point, basis) and (point, basis, reference_dimension). Coordinates
    may lie outside the cell. Order zero returns an empty gradient coordinate
    axis; order one returns all Cartesian first derivatives. Mapping into
    physical coordinates is the caller's explicit geometric operation. Values
    at literally declared interpolation nodes are the exact Kronecker rows;
    derivatives retain the executed native tabulation.
    """
    order = _integer(nderiv, "nderiv")
    if order > 1:
        raise ValueError("tensor tabulation nderiv must be 0 or 1")
    basis = tensor_lagrange_basis(cell, degree, nodes=nodes)
    coordinates = _points(points, basis.element.cell_dimension)
    table = tabulate_reference(basis.element, coordinates, order)[:, :, basis.permutation, 0]
    _honor_nodal_values(table[0], coordinates, basis.nodes)
    gradients = table[1:].transpose(1, 2, 0) if order else np.empty((*table[0].shape, 0))
    return table[0], gradients


def _honor_nodal_values(values: FloatArray, points: FloatArray, nodes: FloatArray) -> None:
    """Apply the interpolation functional only at literally equal node coordinates."""
    rows, columns = np.nonzero(np.all(points[:, None, :] == nodes[None, :, :], axis=-1))
    values[rows] = 0.0
    values[rows, columns] = 1.0


def nodal_base_transformations(basis: NodalReferenceBasis) -> FloatArray:
    """Conjugate native DOF orientation maps into the declared nodal order."""
    maps = reference_base_transformations(basis.element)
    return _readonly(maps[:, basis.permutation][:, :, basis.permutation])


def simplex_lagrange_tabulation(
    cell: str,
    degree: int,
    bary: ArrayLike,
    *,
    nodes: ArrayLike,
    nderiv: int = 2,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Tabulate ordered simplex Pk values, Cartesian gradients and Hessians.

    Reference coordinates are ``bary[:, 1:]`` with ``bary[:, 0] = 1-sum(x)``.
    Return shapes are ``(points, basis)``, ``(points, basis, dimension)`` and
    ``(points, basis, dimension, dimension)``. Only derivative orders 0/1/2
    are returned here; unrequested gradient/Hessian coordinate axes are empty.
    These are derivatives on the unit-sum simplex, not derivatives in all
    independent barycentric variables.
    Physical derivatives require the caller's affine Jacobian, explicitly.
    Values at literally declared interpolation nodes are exact Kronecker rows;
    no proximity threshold or derivative correction is applied.
    """
    order = _integer(nderiv, "nderiv")
    if order > 2:
        raise ValueError("simplex tabulation nderiv must be 0, 1 or 2")
    basis = simplex_lagrange_basis(cell, degree, nodes=nodes)
    dimension = basis.element.cell_dimension
    coordinates = _barycentric(bary, dimension, "bary")
    native = tabulate_reference(basis.element, coordinates[:, 1:], order)
    table = native[:, :, basis.permutation, 0]
    values = table[0]
    _honor_nodal_values(values, coordinates, basis.nodes)
    gradient = np.empty((*values.shape, dimension if order else 0))
    hessian = np.empty(
        (*values.shape, dimension if order == 2 else 0, dimension if order == 2 else 0)
    )
    if order:
        basix = _require_basix()
        for a in range(dimension):
            derivative = np.eye(dimension, dtype=int)[a]
            gradient[..., a] = table[basix.index(*derivative)]
            if order == 2:
                for b in range(dimension):
                    mixed = derivative + np.eye(dimension, dtype=int)[b]
                    hessian[..., a, b] = table[basix.index(*mixed)]
    return values, gradient, hessian


def barycentric_simplex_tabulation(
    cell: str,
    degree: int,
    bary: ArrayLike,
    *,
    nodes: ArrayLike,
    nderiv: int = 2,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Express native simplex derivatives in the canonical barycentric extension.

    On the unit-sum hyperplane, the basis is the ordered Basix Pk basis.
    Its extension is p(lambda_1,...,lambda_d), independent of lambda_0.
    Returned gradients and Hessians have d+1 coordinate axes; entries involving
    lambda_0 are zero. Contracting with physical barycentric gradients gives
    the usual affine derivatives. The extension itself has no physical role.
    """
    values, first, second = simplex_lagrange_tabulation(
        cell, degree, bary, nodes=nodes, nderiv=nderiv
    )
    dimension = _simplex_dimension(cell)
    gradients = np.zeros((*values.shape, dimension + 1))
    hessians = np.zeros((*values.shape, dimension + 1, dimension + 1))
    if nderiv:
        gradients[..., 1:] = first
    if nderiv == 2:
        hessians[..., 1:, 1:] = second
    return values, gradients, hessians


def orthogonal_polynomial_tabulation(
    cell: str, degree: int, points: ArrayLike, nderiv: int = 0
) -> FloatArray:
    """Tabulate Basix's orthonormal polynomial set and Cartesian derivatives.

    Values use native reference-cell measure and ordering, with axes
    (derivative, point, polynomial). Derivative indices follow basix.index.
    This supplies polynomial spans, rather than dualized finite elements.
    """
    order = _integer(degree, "degree")
    derivatives = _integer(nderiv, "nderiv")
    basix = _require_basix()
    try:
        native_cell = basix.CellType[cell]
    except KeyError as exc:
        raise ValueError(f"unknown Basix cell {cell!r}") from exc
    dimension = basix.geometry(native_cell).shape[1]
    coordinates = _points(points, dimension)
    table = basix.polynomials.tabulate_polynomial_set(
        native_cell, basix.PolysetType.standard, order, derivatives, coordinates
    )
    return np.asarray(table, dtype=np.float64).swapaxes(1, 2)


def legendre_tabulation(points: ArrayLike, degree: int, nderiv: int = 0) -> FloatArray:
    """Evaluate conventional Legendre polynomials on [-1,1] through Basix.

    L_j(1)=1; derivatives are with respect to the supplied coordinate, including
    extrapolation points. Axes are (derivative, *points.shape, degree+1).
    The affine coordinate map and sqrt(2j+1) normalization convert Basix's
    orthonormal interval polynomial set into this persisted moment convention.
    """
    order = _integer(degree, "degree")
    derivatives = _integer(nderiv, "nderiv")
    raw = np.asarray(points)
    if raw.dtype.kind not in "fiu" or not np.isfinite(raw).all():
        raise ValueError("Legendre coordinates must contain finite real values")
    coordinates = np.asarray(raw, dtype=np.float64)
    table = orthogonal_polynomial_tabulation(
        "interval", order, ((coordinates.ravel() + 1) / 2)[:, None], derivatives
    )
    table /= np.sqrt(2 * np.arange(order + 1) + 1)
    table /= 2.0 ** np.arange(derivatives + 1)[:, None, None]
    return table.reshape(derivatives + 1, *coordinates.shape, order + 1)


def legendre_values(points: ArrayLike, degree: int) -> FloatArray:
    """Return conventional L_0,...,L_degree at finite real coordinates."""
    return legendre_tabulation(points, degree)[0]


@lru_cache(maxsize=32)
def monomial_coefficients(degree: int) -> FloatArray:
    """Map orthonormal interval coefficients to the declared power convention.

    The coefficients are exact integral moments of x^k against the shifted
    Legendre polynomials. This changes polynomial coordinates only; Basix owns
    every repeated value and derivative evaluation.
    """
    matrix = np.zeros((degree + 1, degree + 1))
    for power in range(degree + 1):
        for mode in range(power + 1):
            matrix[mode, power] = np.sqrt(2 * mode + 1) * (
                factorial(power) ** 2 / (factorial(power - mode) * factorial(power + mode + 1))
            )
    matrix.setflags(write=False)
    return matrix


def monomial_tabulation(
    points: ArrayLike, powers: tuple[tuple[int, ...], ...], *, nderiv: int = 1
) -> FloatArray:
    """Tabulate declared component powers using Basix interval polynomial factors.

    Preserve the supplied exponent order, including incomplete polynomial
    subspaces used for MHM moment definitions and archived coefficient vectors.
    Axes are (derivative, point, monomial); order one contains the value followed
    by each Cartesian derivative. Any finite real coordinates are allowed.
    The product construction works in arbitrary coordinate dimension, without
    imposing a simplex or tensor-cell approximation space on the caller.
    """
    raw = np.asarray(points)
    if raw.ndim != 2 or raw.shape[1] == 0:
        raise ValueError("monomial points must have shape (n, positive_dimension)")
    dimension = raw.shape[1]
    coordinates = _points(points, dimension)
    derivatives = _integer(nderiv, "nderiv")
    if derivatives > 1:
        raise ValueError("monomial nderiv must be 0 or 1")
    exponents = tuple(tuple(_integer(p, "power") for p in exponent) for exponent in powers)
    if any(len(exponent) != dimension for exponent in exponents):
        raise ValueError("each monomial exponent must match the coordinate dimension")
    result = np.ones((1 + derivatives * dimension, len(coordinates), len(exponents)))
    if not exponents:
        return result
    indices = np.asarray(exponents, dtype=np.int64)
    degree = int(indices.max())
    factors = [
        orthogonal_polynomial_tabulation(
            "interval", degree, coordinates[:, axis, None], derivatives
        )
        @ monomial_coefficients(degree)
        for axis in range(dimension)
    ]
    for derivative in range(len(result)):
        for axis in range(dimension):
            result[derivative] *= factors[axis][derivatives * (derivative == axis + 1)][
                :, indices[:, axis]
            ]
    return result


@lru_cache(maxsize=32)
def _bernstein_order(
    dimension: int, degree: int
) -> tuple[ReferenceElement, tuple[tuple[int, ...], ...]]:
    """Identify native Bernstein exponents from exact integral first moments."""
    cell = {1: "interval", 2: "triangle", 3: "tetrahedron"}[dimension]
    element = create_reference_element(
        ReferenceElementSpec(
            "P",
            cell,
            degree,
            lagrange_variant="bernstein" if degree else "equispaced",
            discontinuous=degree == 0,
        )
    )
    if dimension == 1:
        from numpy.polynomial.legendre import leggauss

        x, weights = leggauss(degree + 2)
        points = (x[:, None] + 1) / 2
    elif dimension == 2:
        from pymhm.fem.scalar.operators import triangle_quadrature

        bary, weights = triangle_quadrature(degree + 2)
        points = bary[:, 1:]
    else:
        from pymhm.fem.scalar.tetrahedron import tetrahedron_quadrature

        bary, weights = tetrahedron_quadrature(degree + 2)
        points = bary[:, 1:]
    values = tabulate_reference(element, points)[0, :, :, 0]
    bary = np.column_stack((1 - points.sum(axis=1), points))
    integrals = weights @ values
    centroids = np.einsum("q,qi,qa->ia", weights, values, bary) / integrals[:, None]
    # For B_alpha on a d-simplex, its normalized first moment is
    # (alpha_a+1)/(degree+d+1). Integer exponents fix numbering without fitting.
    exponents = np.rint((degree + dimension + 1) * centroids - 1).astype(int)
    if np.any(exponents < 0) or np.any(exponents.sum(axis=1) != degree):
        raise ArithmeticError("native Bernstein moments do not identify the declared degree")
    return element, tuple(tuple(int(a) for a in row) for row in exponents)


def bernstein_basis_matrix(
    dimension: int, degree: int, exponents: tuple[tuple[int, ...], ...]
) -> FloatArray:
    """Return the literal executed Bernstein matrix in a declared barycentric order.

    Rows are only permuted from the native matrix used by bernstein_tabulation;
    columns retain Basix's orthonormal reference polynomial coordinates. No
    interpolation fit or new basis is generated to describe these coefficients.
    """
    element, native_exponents = _bernstein_order(dimension, degree)
    lookup = {exponent: index for index, exponent in enumerate(native_exponents)}
    try:
        order = [lookup[exponent] for exponent in exponents]
    except KeyError as error:
        raise ValueError("Bernstein exponents must belong to the declared degree") from error
    return _readonly(element.basis_matrix[order])


def bernstein_tabulation(
    points: FloatArray, exponents: tuple[tuple[int, ...], ...]
) -> tuple[FloatArray, FloatArray]:
    """Tabulate simplex Bernstein polynomials in an explicit barycentric order.

    ``exponents`` lists all barycentric multiindices of one degree. The native
    basis is only permuted; no polynomial fit or numerical basis rotation is
    performed. Cartesian gradients have axes (point, polynomial, coordinate).
    """
    dimension = points.shape[1]
    element, native_exponents = _bernstein_order(dimension, sum(exponents[0]))
    lookup = {exponent: index for index, exponent in enumerate(native_exponents)}
    order = [lookup[exponent] for exponent in exponents]
    table = tabulate_reference(element, points, 1)
    return table[0, :, order, 0].T, np.stack(
        [table[axis + 1, :, order, 0].T for axis in range(dimension)], axis=-1
    )


def physical_simplex_tabulation(
    cell: str,
    degree: int,
    bary: ArrayLike,
    *,
    nodes: ArrayLike,
    reference_gradients: ArrayLike,
    nderiv: int = 2,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Map ordered native Pk derivatives using explicitly supplied affine geometry.

    ``reference_gradients`` has shape ``(cells, dimension, physical_dimension)``
    and contains the physical gradients of lambda_1 through lambda_dimension.
    It is the caller's affine Jacobian data; no geometry or orientation is
    inferred. ``bary`` is either shared ``(points, dimension+1)`` or per-cell
    ``(cells, points, dimension+1)``. Values retain that shared/per-cell layout;
    derivatives always have leading cell and point axes. Unrequested derivative
    coordinate axes are empty. Hessians use the affine chain rule; curved cells
    require additional geometric derivatives and are outside this adapter.
    """
    dimension = _simplex_dimension(cell)
    order = _integer(nderiv, "nderiv")
    geometry = np.asarray(reference_gradients)
    if (
        geometry.dtype.kind not in "fiu"
        or not np.isfinite(geometry).all()
        or geometry.ndim != 3
        or geometry.shape[1] != dimension
        or geometry.shape[2] < dimension
    ):
        raise ValueError(
            "reference_gradients must be finite real affine geometry (cells, dim, physical_dim)"
        )
    geometry = np.asarray(geometry, dtype=np.float64)
    points = np.asarray(bary)
    if points.ndim not in (2, 3) or points.shape[-1] != dimension + 1:
        raise ValueError(
            "bary must have shared (points, dim+1) or per-cell (cells, points, dim+1) shape"
        )
    if points.ndim == 3 and points.shape[0] != len(geometry):
        raise ValueError("per-cell bary must have one entry per geometric cell")
    values, first, second = simplex_lagrange_tabulation(
        cell, degree, points.reshape(-1, dimension + 1), nodes=nodes, nderiv=order
    )
    if points.ndim == 3:
        values = values.reshape(*points.shape[:2], values.shape[-1])
        first = first.reshape(*values.shape, first.shape[-1])
        second = second.reshape(*values.shape, *second.shape[-2:])
    else:
        first = np.broadcast_to(first, (len(geometry), *first.shape))
        second = np.broadcast_to(second, (len(geometry), *second.shape))
    gradient = (
        np.einsum("tqna,tap->tqnp", first, geometry) if order else np.empty((*first.shape[:3], 0))
    )
    hessian = (
        np.einsum("tqnab,tap,tbr->tqnpr", second, geometry, geometry)
        if order == 2
        else np.empty((*second.shape[:3], 0, 0))
    )
    return values, gradient, hessian


_monomial_legendre_map = monomial_coefficients
