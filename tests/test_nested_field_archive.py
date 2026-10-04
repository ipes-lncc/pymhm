"""Original recursive equations, one-sided orientation and executed field replay."""

import hashlib
import json

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from examples import nested_field_archive as archive
from examples import verify_nested as driver
from examples.archive_precision import precision_fields


@pytest.fixture(scope="module")
def acquired(tmp_path_factory):
    """Acquire two small whole domains with both original physical boundary data."""
    base = tmp_path_factory.mktemp("nested")
    result = {}
    with threadpool_limits(1):
        for n in (1, 2):
            for nonhomogeneous in (False, True):
                path = base / f"n{n}-{nonhomogeneous}.npz"
                row, fields = driver.acquire(n, nonhomogeneous=nonhomogeneous, archive=path)
                record, arrays = archive.read_archive(path)
                result[n, nonhomogeneous] = row, record, arrays, fields, path
    return result


@pytest.mark.parametrize("n", (1, 2))
@pytest.mark.parametrize("nonhomogeneous", (False, True))
def test_actual_bases_original_equations_and_quadrature(acquired, n, nonhomogeneous):
    """The entire Q2 leaf discretization retains rank8, source and physical constraints."""
    row, record, values, fields, _ = acquired[n, nonhomogeneous]
    count = (2 * n) ** 2
    for name, shape in {
        "leaf_points": (count, 9, 2),
        "leaf_cells": (count, 4, 4),
        "leaf_nodes": (count, 25, 2),
        "leaf_dofs": (count, 4, 9),
        "leaf_A": (count, 25, 25),
        "leaf_B": (count, 25, 8),
        "leaf_f": (count, 25),
        "leaf_trace_dofs": (count, 8),
        "leaf_pressure_recursive": (count, 25),
        "leaf_pressure_flat": (count, 25),
    }.items():
        assert values[name].shape == shape
    assert not fields
    assert max(archive.original_checks(values).values()) < 1e-12
    assert record["finite_case_acceptance"]
    assert not record["mesh_uniform_inf_sup_estimate_verified"]
    assert row["relative_leaf_difference"] < 1e-11
    assert row["norm_quadrature_relative_sensitivity"] < 1e-9
    assert np.linalg.matrix_rank(values["leaf_B"]).min() == 8
    # The analytical integral of 2*pi²*sin(pi*x)*sin(pi*y) on the square is8.
    assert abs(np.sum(archive.restore(values, "leaf_f")) - 8) < 1e-12
    assert np.max(abs(values["leaf_A"] @ values["leaf_Z"])) < 2e-14
    for prefix in ("leaf", "flat", "parent"):
        np.testing.assert_allclose(
            archive.replay_responses(values, prefix),
            archive.restore(values, f"{prefix}_pressure"),
            atol=1e-13,
            rtol=1e-12,
        )


@pytest.mark.parametrize("n", (1, 2))
def test_affine_lifting_preserves_source_and_oriented_physical_flux(acquired, n):
    """Nonzero pressure lifting adds the exact affine field and its signed normal flux."""
    zero = acquired[n, False][2]
    affine = acquired[n, True][2]
    np.testing.assert_array_equal(zero["leaf_f"], affine["leaf_f"])
    expected = 1 + zero["leaf_nodes"][..., 0] - zero["leaf_nodes"][..., 1]
    for name in ("leaf_pressure_recursive", "leaf_pressure_flat"):
        actual = archive.restore(affine, name) - archive.restore(zero, name)
        np.testing.assert_allclose(actual, expected, atol=4e-13, rtol=0)
    for name in ("trace_recursive", "trace_flat"):
        actual = archive.restore(affine, name) - archive.restore(zero, name)
        expected_flux = -(zero["flat_face_normals"] @ np.array([1.0, -1.0]))
        np.testing.assert_allclose(actual[::2], expected_flux, atol=2e-12, rtol=0)
        np.testing.assert_allclose(actual[1::2], 0, atol=2e-12, rtol=0)


def test_archived_cardinal_tables_and_one_sided_raw_gradient(acquired):
    """Executed monomial coefficients agree with both saved rules and exact Q2 derivatives."""
    from pymhm.fem.reference import simplex_lagrange_basis

    arrays = {k: v.copy() for k, v in acquired[2, True][2].items()}
    x, y = arrays["leaf_nodes"].transpose(2, 0, 1)
    polynomial = 1 + x**2 + 3 * x * y + 2 * y**2
    arrays.update(precision_fields("leaf_pressure_recursive", polynomial))
    nodes = np.linspace(0, 1, 3)
    factor = simplex_lagrange_basis("interval", 2, nodes=np.column_stack((1 - nodes, nodes)))
    modes = np.arange(3)
    normalization = np.sqrt(2 * modes + 1)
    factor_value_bound = np.abs(factor.basis_matrix) @ normalization
    factor_derivative_bound = np.abs(factor.basis_matrix) @ (normalization * modes * (modes + 1))
    for q in (8, 10):
        points = arrays[f"q{q}_reference"]
        powers = arrays["q2_monomial_powers"]
        monomials = np.prod(points[:, None] ** powers[None], axis=-1)
        # Seventeen multiply/add operations in the nine-term power dot;
        # The native degree-two interval value/first-derivative graph has at
        # most 47 scalar operations: 12 for the values, 26 for the derivative
        # recurrences and nine for normalization. Counting the entire graph
        # also counts exact/zero terms and operations the compiler may remove.
        # Five coefficient-dot operations and one tensor multiplication follow.
        epsilon = np.finfo(arrays["q2_cardinal_matrix"].dtype).eps
        operations = 17 + 47 + 5 + 1
        gamma = operations * epsilon / (1 - operations * epsilon)
        basis_bound = gamma * (
            monomials @ np.abs(arrays["q2_cardinal_matrix"]).T
            + np.outer(factor_value_bound, factor_value_bound).ravel()
        )
        represented = monomials @ arrays["q2_cardinal_matrix"].T
        assert np.all(np.abs(represented - arrays[f"q{q}_basis"]) <= basis_bound)
        physical, pressure, raw, _ = archive.evaluate(arrays, "recursive", q)
        x, y = physical.transpose(3, 0, 1, 2)
        np.testing.assert_allclose(pressure, 1 + x * x + 3 * x * y + 2 * y * y, atol=5e-15, rtol=0)
        # Forward arithmetic bound for the executed SDK factors, rather than
        # an absolute threshold tied to the previous polynomial representation.
        # |L_j| <= 1 and |d L_j(2s-1)/ds| <= j(j+1) on [0,1].
        # Conservatively count all 47 native interval operations above, five
        # three-term coefficient-dot operations, a tensor product, seventeen
        # nine-term field-dot operations and one physical-width division.
        gradient_bound = np.stack(
            (
                np.outer(factor_value_bound, factor_derivative_bound).ravel(),
                np.outer(factor_derivative_bound, factor_value_bound).ravel(),
            ),
            axis=-1,
        )
        coefficients = archive.restore(arrays, "leaf_pressure_recursive")
        values = coefficients[np.arange(len(coefficients))[:, None, None], arrays["leaf_dofs"]]
        widths = (arrays["leaf_points"][:, 8] - arrays["leaf_points"][:, 0]) / 2
        operation_count = 47 + 5 + 1 + 17 + 1
        epsilon = np.finfo(factor.basis_matrix.dtype).eps
        gamma = operation_count * epsilon / (1 - operation_count * epsilon)
        bound = gamma * np.einsum("cfj,ja->cfa", np.abs(values), gradient_bound)
        bound = bound[:, :, None] / widths[:, None, None]
        expected = np.stack((2 * x + 3 * y, 3 * x + 4 * y), axis=-1)
        assert np.all(np.abs(raw - expected) <= bound)
    # Change only one independent leaf's constant; its neighbor must remain unchanged.
    polynomial[0] += 2
    arrays.update(precision_fields("leaf_pressure_recursive", polynomial))
    _, pressure, _, _ = archive.evaluate(arrays, "recursive", 8)
    assert pressure[0].min() > 3
    assert pressure[1].max() < 4
    with pytest.raises(ValueError, match="executed norm"):
        archive.evaluate(arrays, "recursive", 6)


def test_coherent_kernel_sign_and_thread_replay(acquired):
    """Executed E and coarse coordinates must transform together across thread counts."""
    original = acquired[1, True][2]
    arrays = {k: v.copy() for k, v in original.items()}
    for prefix in ("leaf", "flat", "parent"):
        before = archive.replay_responses(arrays, prefix)
        for key in ("E", "Z", "C", "W", "test_C", "coarse"):
            name = f"{prefix}_{key}"
            arrays.update(precision_fields(name, -archive.restore(arrays, name)))
        with threadpool_limits(2):
            after = archive.replay_responses(arrays, prefix)
        np.testing.assert_array_equal(after, before)
        arrays.update(
            precision_fields(f"{prefix}_coarse", -archive.restore(arrays, f"{prefix}_coarse"))
        )
        assert np.linalg.norm(archive.replay_responses(arrays, prefix) - before) > 0.1


def test_basis_digest_and_original_boundary_tamper(acquired, tmp_path):
    """A changed executed matrix cannot replay with old digests or satisfy old physical data."""
    row, _, arrays, _, _ = acquired[1, True]
    path = tmp_path / "tamper.npz"
    archive.write_archive(path, arrays, row)
    altered = {k: v.copy() for k, v in arrays.items()}
    altered["leaf_E"][0, 0, 0] += 1
    np.savez(path, **altered)
    with pytest.raises(ValueError, match="field archive digest"):
        archive.read_archive(path)
    metadata = json.loads(path.with_suffix(".json").read_text())
    metadata["archive_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(".json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="array/basis digest"):
        archive.read_archive(path)
    altered = {k: v.copy() for k, v in arrays.items()}
    altered["boundary_load"] += 0.1
    assert archive.original_checks(altered)["leaf_original_constraint_relative_residual"] > 1e-3
    with pytest.raises(ValueError, match="fresh"):
        archive.write_archive(path, arrays, row)
    altered["leaf_E"][0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="finite portable"):
        archive.write_archive(tmp_path / "invalid.npz", altered, row)


@pytest.mark.parametrize("n", (0, 3, True))
def test_excluded_level_rejected_before_assembly(n):
    """The executable publication sequence rejects unsupported dimensions and boolean counts."""
    with pytest.raises(ValueError, match="levels"):
        driver.acquire(n)


def test_face_endpoint_and_normal_contract():
    """Normal reversal and parameter reversal act independently on the P1 moment basis."""
    edge = np.array([[0.0, 0.0], [1.0, 0.0]])
    n = np.array([0.0, 1.0])
    np.testing.assert_array_equal(
        archive._face_transform(edge[::-1], -n, edge, n), np.diag([-1, 1])
    )
    with pytest.raises(ValueError, match="endpoints"):
        archive._face_transform(edge + 1, n, edge, n)
    with pytest.raises(ValueError, match="normal"):
        archive._face_transform(edge, np.array([1.0, 0.0]), edge, n)


def test_incompatible_degrees_dimensions_and_injections(acquired):
    """The adjacent excluded degree and wrong coordinate maps fail before field replay."""
    original = acquired[1, False][2]
    for key, value, message in (
        ("n", np.asarray(True), "supported level"),
        ("leaf_degree", np.asarray(1), "scalar Q2"),
        ("leaf_degree", np.asarray(3), "scalar Q2"),
        ("leaf_degree", np.asarray(2.0), "scalar Q2"),
        ("leaf_dofs", original["leaf_dofs"][:, :, :-1], "dimensions"),
        ("leaf_dofs", original["leaf_dofs"] + 25, "injection"),
        ("leaf_trace_dofs", original["leaf_trace_dofs"].astype(float), "injection"),
    ):
        arrays = dict(original)
        arrays[key] = value
        with pytest.raises(ValueError, match=message):
            archive.validate_layout(arrays)


def test_independently_executed_two_thread_acquisition(acquired, tmp_path):
    """Two native thread counts reproduce the same original fields using their actual bases."""
    with threadpool_limits(2):
        row, _ = driver.acquire(1, nonhomogeneous=True, archive=tmp_path / "threads.npz")
    _, arrays = archive.read_archive(tmp_path / "threads.npz")
    original = acquired[1, True][2]
    assert max(row["original_physical_checks"].values()) < 1e-10
    for name in ("leaf_pressure_recursive", "leaf_pressure_flat", "trace_recursive", "trace_flat"):
        np.testing.assert_allclose(
            archive.restore(arrays, name), archive.restore(original, name), atol=1e-13, rtol=1e-11
        )


def _retag(path, arrays, record):
    """Simulate replacement with self-consistent byte digests to exercise physical validation."""
    np.savez(path, **arrays)
    record = dict(
        record,
        archive=path.name,
        archive_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        arrays_sha256={key: archive.array_digest(value) for key, value in arrays.items()},
    )
    path.with_suffix(".json").write_text(json.dumps(record))


@pytest.mark.parametrize(
    "name",
    (
        "q2_cardinal_matrix",
        "q8_gradient",
        "leaf_A",
        "leaf_B",
        "leaf_C",
        "leaf_E",
        "leaf_source",
        "inner_A",
        "boundary_load",
        "trace_q8_basis",
        "display_gradient",
        "parent_physical_moment_offset",
    ),
)
def test_rehashed_numerical_contract_change_is_rejected(acquired, tmp_path, name):
    """Consistent replacement hashes do not legitimize changed physical operators or bases."""
    _, record, original, _, _ = acquired[1, True]
    arrays = {key: value.copy() for key, value in original.items()}
    arrays[name].flat[0] += 0.1
    path = tmp_path / "changed.npz"
    _retag(path, arrays, record)
    with pytest.raises(ValueError, match="executed"):
        archive.read_archive(path)


@pytest.mark.parametrize(
    "kind",
    (
        "uuid",
        "schema",
        "precision",
        "nonfinite",
        "complex",
        "nodal-map",
        "norm-record",
        "original-record",
    ),
)
def test_rehashed_identity_realness_and_geometry_change_is_rejected(acquired, tmp_path, kind):
    """The JSON identity and finite-real physical coefficient map remain part of replay."""
    _, saved_record, original, _, _ = acquired[1, True]
    record = dict(saved_record)
    arrays = {key: value.copy() for key, value in original.items()}
    if kind == "uuid":
        record["acquisition_uuid"] = "00000000-0000-4000-8000-000000000000"
    elif kind == "schema":
        record["field_archive_schema"] = "other"
    elif kind == "precision":
        arrays["coefficient_precision_bits"] = np.asarray(52)
    elif kind == "nonfinite":
        arrays["leaf_A"].flat[0] = np.nan
    elif kind == "complex":
        arrays["leaf_A"] = arrays["leaf_A"].astype(complex)
    elif kind == "norm-record":
        record = json.loads(json.dumps(record))
        record["norms"]["quadrature_8"]["pressure_l2"] += 0.1
    elif kind == "original-record":
        record = json.loads(json.dumps(record))
        record["original_physical_checks"].pop("leaf_original_source_relative_residual")
    else:
        arrays["leaf_dofs"][0, 0, :2] = arrays["leaf_dofs"][0, 0, 1::-1]
    path = tmp_path / "changed.npz"
    _retag(path, arrays, record)
    with pytest.raises(ValueError, match="identity|finite portable|nodal injection|record"):
        archive.read_archive(path)


def test_executed_display_replay_needs_no_current_tabulator(acquired, monkeypatch):
    """Publication samples replay their saved one-sided derivative and injection tables."""
    arrays = acquired[1, True][2]
    before = archive.display_fields(arrays)

    def forbidden(*args, **kwargs):
        raise AssertionError("fresh basis evaluation must not be used for saved field replay")

    monkeypatch.setattr(archive, "qk_basis", forbidden)
    with threadpool_limits(2):
        after = archive.display_fields(arrays)
    for left, right in zip(before, after, strict=True):
        np.testing.assert_array_equal(left, right)
    with pytest.raises(ValueError, match="displayed field"):
        archive.display_fields(arrays, "other")
