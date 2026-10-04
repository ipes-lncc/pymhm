"""Executed analytic P3 basis recipes and literal one-sided three-layer tables.

The cardinal basis is a product of ordered barycentric Polynomial factors;
it has no arbitrary nullspace or fitted cardinal matrix. Its actual factor
coefficients and derivatives are observed through the shared owner's unchanged
code object. Local mass tables remain the arrays held by the actual producer.
New field quadrature tables are separate, explicitly executed observations.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from types import FunctionType
from typing import Any

import numpy as np
from numpy.polynomial import Polynomial

from pymhm import lagrange
from pymhm.elasticity_primal import _KELVIN, constitutive_values
from pymhm.elastodynamics import ElastodynamicLocal
from pymhm.elements import p1_geometry
from pymhm.lagrange import element_tabulate, multiindices, reference_basis
from pymhm.maxwell_dg import physical_points, quadrature

SCHEMA = "pymhm-three-layer-product-basis-v1"


def array_digest(array: np.ndarray) -> str:
    """Bind complete dtype, shape and ordered bytes without narrowing coefficients."""
    value = np.asarray(array)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode())
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def product_runtime() -> dict[str, Any]:
    """Identify the actual Polynomial evaluator and effective binary64 significand."""
    import numpy.polynomial.polynomial as owner

    return {
        "numpy": np.__version__,
        "polynomial_source_sha256": _file_digest(Path(owner.__file__)),
        "basis_dtype": np.dtype(float).str,
        "basis_precision_bits": np.finfo(float).nmant + 1,
    }


def _observed_recipe(bary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Observe real factor coefficients through the unchanged shared code objects.

    A separate function-globals dictionary owns the delegating Polynomial
    subclass. Neither the shared module nor Polynomial class is mutated, so
    threads and spawned processes cannot see an observer left behind.
    """
    observed: list[tuple[int, np.ndarray]] = []

    class ObservedPolynomial(lagrange.Polynomial):
        def deriv(self, m: int = 1) -> Polynomial:
            result = super().deriv(m)
            observed.append((m, np.array(result.coef, copy=True)))
            return result

    original_table = lagrange._reference_table
    table_globals = original_table.__globals__ | {"Polynomial": ObservedPolynomial}
    table = FunctionType(original_table.__code__, table_globals)
    original_basis = lagrange.reference_basis
    basis = FunctionType(
        original_basis.__code__, original_basis.__globals__ | {"_reference_table": table}
    )
    actual = basis(3, bary)
    expected = reference_basis(3, bary)
    if any(not np.array_equal(a, b) for a, b in zip(actual, expected, strict=True)):
        raise ArithmeticError("Observed shared P3 values or derivatives are not bitwise identical")
    if len(observed) != 36:
        raise ArithmeticError("Shared ordered P3 factor construction changed")
    coefficients = np.zeros((3, 4, 3, 4))
    sizes = np.zeros((3, 4, 3), dtype=np.int64)
    for coordinate in range(3):
        for exponent in range(4):
            for derivative in range(3):
                index = (coordinate * 4 + exponent) * 3 + derivative
                order, values = observed[index]
                if order != derivative:
                    raise ArithmeticError("Shared P3 derivative evaluation order changed")
                # These are the derivative coefficients actually consumed by
                # the delegated Polynomial, rather than a fitted basis matrix.
                coefficients[coordinate, exponent, derivative, : len(values)] = values
                sizes[coordinate, exponent, derivative] = len(values)
    return coefficients, sizes


def tabulate_product(arrays: Any, bary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate archived factor recipes in their declared multiply order.

    No basis matrix, new interpolation problem or nullspace is constructed.
    ``bary`` contains finite binary64 triples, including extrapolation points.
    Returned derivatives are with respect to all three barycentric coordinates.
    """
    bary = np.asarray(bary)
    if bary.dtype != np.float64 or bary.ndim != 2 or bary.shape[1] != 3:
        raise ValueError("Finite binary64 barycentric triples required")
    if not np.isfinite(bary).all():
        raise ValueError("Finite binary64 barycentric triples required")
    coefficients = arrays["product_factor_coefficients"]
    sizes = arrays["product_factor_sizes"]
    indices = arrays["product_multiindices"]
    table = np.empty((3, 4, 2, len(bary)))
    for coordinate in range(3):
        for exponent in range(4):
            for derivative in range(2):
                length = sizes[coordinate, exponent, derivative]
                table[coordinate, exponent, derivative] = Polynomial(
                    coefficients[coordinate, exponent, derivative, :length]
                )(bary[:, coordinate])
    values = np.ones((len(bary), len(indices)))
    first = np.ones((*values.shape, 3))
    for node, exponents in enumerate(indices):
        for coordinate in range(3):
            values[:, node] *= table[coordinate, exponents[coordinate], 0]
            for derivative in range(3):
                first[:, node, derivative] *= table[
                    coordinate, exponents[coordinate], int(coordinate == derivative)
                ]
    return values, first


def _physical_gradients(first: np.ndarray, arrays: Any) -> np.ndarray:
    """Preserve the producer's arithmetic order under a coherent column permutation."""
    indices = arrays["product_multiindices"]
    positions = {tuple(row): index for index, row in enumerate(indices)}
    order = np.array([positions[tuple(row)] for row in multiindices(3)])
    canonical = np.ascontiguousarray(first[..., order, :])
    # p1_geometry returns a slice of a transposed 3x3 inverse. NPZ retains
    # its values but cannot retain this non-contiguous slice's strides.
    # Restore that declared storage layout before the original contraction.
    geometry = np.empty((len(first), 3, 3)).swapaxes(1, 2)[..., :2]
    geometry[...] = arrays["product_physical_barycentric_gradients"]
    result = np.einsum("tqib,tba->tqia", canonical, geometry)
    return result[..., np.argsort(order), :]


@lru_cache(maxsize=4)
def _recipe(owner_sha256: str, runtime_json: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Cache a fixed observed recipe without exposing mutable backing arrays."""
    nodes = multiindices(3) / 3
    factors, sizes = _observed_recipe(nodes)
    arrays = {
        "product_factor_coefficients": factors,
        "product_factor_sizes": sizes,
        "product_multiindices": multiindices(3),
        "product_reference_nodes": nodes,
    }
    for key, value in arrays.items():
        arrays[key] = np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)
    values, first = tabulate_product(arrays, nodes)
    actual = reference_basis(3, nodes)
    if not np.array_equal(values, actual[0]) or not np.array_equal(first, actual[1]):
        raise ArithmeticError("Archived recipe differs from executed shared P3 basis")
    record = {
        "schema": SCHEMA,
        "degree": 3,
        "dimension": 2,
        "scalar_dimension": 10,
        "basis_representation": "ordered analytic barycentric Polynomial product recipe",
        "multiply_order": [0, 1, 2],
        "field_arithmetic_column_order": (
            "shared canonical multiindices(3), before coordinate permutation"
        ),
        "component_convention": "global coefficient2*node+Cartesian component",
        "shared_lagrange_source_sha256": owner_sha256,
        "recipe_runtime": json.loads(runtime_json),
        "observation_bitwise_identical_to_shared_owner": True,
        "arbitrary_or_fitted_basis_matrix_used": False,
        "recipe_arrays_sha256": {key: array_digest(value) for key, value in arrays.items()},
    }
    return record, arrays


def product_recipe() -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Capture the producer's actual deterministic P3 factors, retaining their order."""
    record, arrays = _recipe(
        _file_digest(Path(lagrange.__file__)), json.dumps(product_runtime(), sort_keys=True)
    )
    return json.loads(json.dumps(record)), dict(arrays)


def actual_local_tables(local: ElastodynamicLocal) -> dict[str, np.ndarray]:
    """Retain literal arrays held by the actual local model, without new tabulation."""
    if local.degree != 3 or local.nodes.shape[1] != 2:
        raise ValueError("The selected planar P3 local model is required")
    return {
        "actual_mass_basis_values": local.basis,
        "actual_mass_quadrature_points": local.points,
        "actual_mass_quadrature_weights": local.weights,
        "actual_mass_density_values": local.density_values,
    }


def capture_field_basis(
    local: ElastodynamicLocal, *, orders: tuple[int, ...] = (5, 7)
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Save an executed recipe and independent one-sided field quadrature observations.

    ``actual_mass_*`` are arrays already held by the original local model.
    ``field_q*`` are new producer evaluations through shared tabulation and
    material owners; they neither replace any operator nor the solution basis.
    Their physical gradients use the same affine map as the field evaluator.
    """
    if (
        not orders
        or len(set(orders)) != len(orders)
        or any(type(q) is not int or q < 5 for q in orders)
    ):
        raise ValueError("Distinct integer field quadrature orders at least5 required")
    record, arrays = product_recipe()
    arrays.update(actual_local_tables(local))
    arrays.update(
        product_local_points=local.mesh.points,
        product_local_cells=local.mesh.cells,
        product_local_nodes=local.nodes,
        product_local_dofs=local.dofs,
    )
    arrays["product_physical_barycentric_gradients"] = p1_geometry(local.mesh)[0]
    geometry = arrays["product_physical_barycentric_gradients"]
    if geometry.strides != (72, 8, 24):
        raise ValueError("The declared shared binary64 affine-gradient storage layout changed")
    arrays["product_kelvin_map"] = _KELVIN
    for order in orders:
        bary, weights, material = quadrature(local.mesh, local.constitutive, order)
        dofs, nodes, values, gradient, _ = element_tabulate(local.mesh, local.degree, bary)
        if not np.array_equal(dofs, local.dofs) or not np.array_equal(nodes, local.nodes):
            raise ValueError("Executed field tabulation changed local coefficient coordinates")
        recipe_values, recipe_first = tabulate_product(arrays, bary.reshape(-1, 3))
        shape = (*bary.shape[:2], 10)
        recipe_gradient = _physical_gradients(recipe_first.reshape(*shape, 3), arrays)
        if not np.array_equal(values, recipe_values.reshape(shape)) or not np.array_equal(
            gradient, recipe_gradient
        ):
            raise ArithmeticError("Literal field values/gradients differ from archived recipe")
        points = physical_points(local.mesh, bary)
        constitutive = constitutive_values(
            material,
            points.reshape(-1, 2),
            lame_lambda=local.lame_lambda,
            lame_mu=local.lame_mu,
        ).reshape(*weights.shape, 3, 3)
        arrays.update(
            {
                f"field_q{order}_bary": bary,
                f"field_q{order}_points": points,
                f"field_q{order}_weights": weights,
                f"field_q{order}_values": values,
                f"field_q{order}_gradients": gradient,
                f"field_q{order}_constitutive": constitutive,
            }
        )
    record.update(
        field_quadrature_orders=list(orders),
        local_scalar_nodes=len(local.nodes),
        local_fine_cells=len(local.mesh.cells),
        physical_gradient_storage_strides_bytes=list(geometry.strides),
        coefficient_dtype=np.dtype(float).str,
        coefficient_precision_bits=np.finfo(float).nmant + 1,
        source_sha256={
            path.name: _file_digest(path)
            for path in (
                Path(__file__),
                Path(lagrange.__file__),
                Path(__import__("pymhm.elements", fromlist=["__file__"]).__file__),
                Path(__import__("pymhm.maxwell_dg", fromlist=["__file__"]).__file__),
                Path(__import__("pymhm.elasticity_primal", fromlist=["__file__"]).__file__),
            )
        },
        arrays_sha256={key: array_digest(value) for key, value in arrays.items()},
        capture_scope=(
            "Actual held local mass tables plus newly executed producer field tables; "
            "same analytic P3 coefficients, no operator replacement or fitted basis"
        ),
    )
    return record, arrays


def validate_field_basis(
    record: dict[str, Any], arrays: Any, *, replay_recipe: bool = True
) -> None:
    """Validate producer tables, optionally replaying them in the original runtime.

    ``replay_recipe=False`` supports literal-table consumers in another NumPy
    runtime. That consumer must verify the producer's source snapshot and bound
    archive digest; it evaluates the archived literal tables, not a newly
    tabulated recipe. Bitwise recipe replay requires the original evaluator.
    """
    if (
        record.get("schema") != SCHEMA
        or (replay_recipe and record.get("recipe_runtime") != product_runtime())
        or record.get("multiply_order") != [0, 1, 2]
        or record.get("field_arithmetic_column_order")
        != "shared canonical multiindices(3), before coordinate permutation"
        or record.get("coefficient_dtype") != np.dtype(float).str
        or record.get("coefficient_precision_bits") != 53
        or record.get("physical_gradient_storage_strides_bytes") != [72, 8, 24]
        or record.get("degree") != 3
        or record.get("component_convention") != "global coefficient2*node+Cartesian component"
    ):
        raise ValueError("Executed product basis, runtime or coefficient convention mismatch")
    runtime = record.get("recipe_runtime", {})
    if (
        runtime.get("basis_dtype") != np.dtype(float).str
        or runtime.get("basis_precision_bits") != 53
        or not isinstance(runtime.get("numpy"), str)
        or len(runtime.get("polynomial_source_sha256", "")) != 64
    ):
        raise ValueError("Actual producer Polynomial runtime and precision required")
    hashes = record.get("arrays_sha256", {})
    if not hashes or any(
        key not in arrays
        or not np.isfinite(arrays[key]).all()
        or array_digest(arrays[key]) != value
        for key, value in hashes.items()
    ):
        raise ValueError("Executed product basis arrays changed or are incomplete")
    orders = record.get("field_quadrature_orders")
    if (
        not isinstance(orders, list)
        or not orders
        or len(set(orders)) != len(orders)
        or any(type(order) is not int or order < 5 for order in orders)
    ):
        raise ValueError("Declared executed field quadrature orders required")
    required = (
        set(product_recipe()[1])
        | {
            "actual_mass_basis_values",
            "actual_mass_quadrature_points",
            "actual_mass_quadrature_weights",
            "actual_mass_density_values",
            "product_physical_barycentric_gradients",
            "product_kelvin_map",
            "product_local_points",
            "product_local_cells",
            "product_local_nodes",
            "product_local_dofs",
        }
        | {
            f"field_q{order}_{suffix}"
            for order in orders
            for suffix in ("bary", "points", "weights", "values", "gradients", "constitutive")
        }
    )
    if set(hashes) != required:
        raise ValueError("Complete declared executed basis payload required")
    recipe_hashes = record.get("recipe_arrays_sha256", {})
    if set(recipe_hashes) != set(product_recipe()[1]) or any(
        hashes.get(key) != value for key, value in recipe_hashes.items()
    ):
        raise ValueError("Executed recipe and field payload digests differ")
    if any(
        arrays[key].dtype != np.float64
        for key in required
        if key
        not in (
            "product_factor_sizes",
            "product_multiindices",
            "product_local_cells",
            "product_local_dofs",
        )
    ):
        raise ValueError("Actual field table, affine map and factor precision differ")
    indices = arrays["product_multiindices"]
    if indices.shape != (10, 3) or indices.dtype.kind not in "iu":
        raise ValueError("Complete ordered P3 multiindices required")
    expected = {tuple(row) for row in multiindices(3)}
    if {tuple(row) for row in indices} != expected or not np.array_equal(
        arrays["product_reference_nodes"], indices / 3
    ):
        raise ValueError("Executed product nodes and coefficient order differ")
    if (
        arrays["product_factor_coefficients"].shape != (3, 4, 3, 4)
        or arrays["product_factor_sizes"].shape != (3, 4, 3)
        or arrays["product_factor_sizes"].dtype.kind not in "iu"
        or np.any(arrays["product_factor_sizes"] < 1)
        or np.any(arrays["product_factor_sizes"] > 4)
    ):
        raise ValueError("Complete finite executed product factors required")
    canonical = product_recipe()[1]
    if any(
        not np.array_equal(arrays[key], canonical[key])
        for key in ("product_factor_coefficients", "product_factor_sizes")
    ):
        raise ValueError("Executed analytic product factors or derivative coefficients differ")
    cells, points = arrays["product_local_cells"], arrays["product_local_points"]
    nodes, dofs = arrays["product_local_nodes"], arrays["product_local_dofs"]
    if (
        points.ndim != 2
        or points.shape[1] != 2
        or points.dtype != np.float64
        or cells.ndim != 2
        or cells.shape[1] != 3
        or cells.dtype.kind not in "iu"
        or np.any(cells < 0)
        or np.any(cells >= len(points))
        or nodes.shape != (record.get("local_scalar_nodes"), 2)
        or nodes.dtype != np.float64
        or dofs.shape != (record.get("local_fine_cells"), 10)
        or dofs.dtype.kind not in "iu"
        or len(cells) != len(dofs)
        or np.any(dofs < 0)
        or np.any(dofs >= len(nodes))
        or arrays["product_physical_barycentric_gradients"].shape != (len(cells), 3, 2)
        or not np.array_equal(arrays["product_kelvin_map"], _KELVIN)
    ):
        raise ValueError("Executed affine geometry, Cartesian map or nodal injection differ")
    vertices = points[cells]
    geometry = arrays["product_physical_barycentric_gradients"]
    affine_action = np.einsum("tia,tib->tab", vertices - vertices[:, :1], geometry)
    affine_scale = np.einsum("tia,tib->tab", abs(vertices - vertices[:, :1]), abs(geometry))
    if np.any(
        abs(affine_action - np.eye(2)) > 256 * np.finfo(float).eps * np.maximum(1, affine_scale)
    ) or np.any(
        abs(geometry.sum(axis=1))
        > 256 * np.finfo(float).eps * np.maximum(1, abs(geometry).sum(axis=1))
    ):
        raise ValueError("Saved barycentric physical derivatives and affine geometry differ")
    mass_values = arrays["actual_mass_basis_values"]
    mass_weights = arrays["actual_mass_quadrature_weights"]
    if (
        mass_values.ndim != 3
        or mass_values.shape[0] != len(cells)
        or mass_values.shape[2] != 10
        or mass_weights.shape != mass_values.shape[:2]
        or arrays["actual_mass_quadrature_points"].shape != (*mass_weights.shape, 2)
        or arrays["actual_mass_density_values"].shape != mass_weights.shape
        or np.any(mass_weights < 0)
        or np.any(arrays["actual_mass_density_values"] <= 0)
    ):
        raise ValueError("Actual held local mass basis, quadrature or density differ")
    coordinate_scale = max(1.0, float(np.max(abs(vertices))))
    if not np.allclose(
        nodes[dofs],
        np.einsum("qi,tia->tqa", indices / 3, vertices),
        rtol=0,
        atol=256 * np.finfo(float).eps * coordinate_scale,
    ):
        raise ValueError("Executed coefficient nodes and one-sided basis order differ")
    values, _ = tabulate_product(arrays, arrays["product_reference_nodes"])
    if np.max(abs(values - np.eye(10))) > 128 * np.finfo(float).eps:
        raise ValueError("Archived product is not the declared cardinal P3 basis")
    for order in orders:
        bary = arrays[f"field_q{order}_bary"]
        weights = arrays[f"field_q{order}_weights"]
        if (
            bary.ndim != 3
            or bary.shape[0] != len(cells)
            or bary.shape[2] != 3
            or bary.dtype != np.float64
            or weights.shape != bary.shape[:2]
            or np.any(weights < 0)
            or arrays[f"field_q{order}_values"].shape != (*weights.shape, 10)
            or arrays[f"field_q{order}_gradients"].shape != (*weights.shape, 10, 2)
            or arrays[f"field_q{order}_constitutive"].shape != (*weights.shape, 3, 3)
            or not np.array_equal(
                arrays[f"field_q{order}_points"], np.einsum("tqi,tia->tqa", bary, vertices)
            )
        ):
            raise ValueError("Literal executed physical quadrature tables or geometry differ")
        values, first = tabulate_product(arrays, bary.reshape(-1, 3))
        shape = (*bary.shape[:2], 10)
        gradient = _physical_gradients(first.reshape(*shape, 3), arrays)
        if replay_recipe:
            if not np.array_equal(
                values.reshape(shape), arrays[f"field_q{order}_values"]
            ) or not np.array_equal(gradient, arrays[f"field_q{order}_gradients"]):
                raise ValueError("Executed product recipe and literal field tables differ")
        else:
            # A semantic audit in another Polynomial runtime uses a Horner
            # absolute-sum bound. It never replaces the producer's tables or
            # claims a bitwise replay in that runtime. The P3 factor/product
            # and three-term affine contraction need fewer than32 operations.
            absolute = dict(arrays)
            absolute["product_factor_coefficients"] = abs(arrays["product_factor_coefficients"])
            positive_values, positive_first = tabulate_product(absolute, abs(bary.reshape(-1, 3)))
            value_bound = 256 * np.finfo(float).eps * np.maximum(1, positive_values.reshape(shape))
            gradient_bound = (
                256
                * np.finfo(float).eps
                * np.maximum(
                    1, np.einsum("tqib,tba->tqia", positive_first.reshape(*shape, 3), abs(geometry))
                )
            )
            if np.any(
                abs(values.reshape(shape) - arrays[f"field_q{order}_values"]) > value_bound
            ) or np.any(abs(gradient - arrays[f"field_q{order}_gradients"]) > gradient_bound):
                raise ValueError(
                    "Literal producer tables fail the declared product/affine semantic audit"
                )


def evaluate_saved_field(
    arrays: Any, dofs: np.ndarray, coefficients: np.ndarray, *, order: int = 5
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Contract validated literal producer tables with interleaved nodal coefficients.

    Return displacement, raw physical gradient and raw Cauchy stress. Neither
    gradients nor stress are H(div) reconstructions. No basis or operator is
    regenerated. The caller validates metadata/digests before using arrays.
    """
    coefficients = np.asarray(coefficients)
    if coefficients.dtype.kind != "f" or coefficients.ndim != 1 or coefficients.size % 2:
        raise ValueError("Finite interleaved real nodal coefficients required")
    if (
        not np.isfinite(coefficients).all()
        or dofs.ndim != 2
        or dofs.shape[1] != 10
        or dofs.dtype.kind not in "iu"
        or np.any(dofs < 0)
        or np.any(dofs >= coefficients.size // 2)
    ):
        raise ValueError("Finite executed P3 coefficient injection required")
    local = coefficients.reshape(-1, 2)[dofs]
    values = np.einsum("tqj,tja->tqa", arrays[f"field_q{order}_values"], local)
    gradient = np.einsum("tqjb,tja->tqab", arrays[f"field_q{order}_gradients"], local)
    kelvin = arrays["product_kelvin_map"]
    strain = np.einsum("aij,tqij->tqa", kelvin, gradient)
    stress = np.einsum("tqab,tqb,aij->tqij", arrays[f"field_q{order}_constitutive"], strain, kelvin)
    return values, gradient, stress
