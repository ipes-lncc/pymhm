"""Optional element contracts, literal basis data and inherited native capabilities."""

from __future__ import annotations

import hashlib
import pickle
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from threadpoolctl import threadpool_limits

from pymhm import element_backends as providers
from pymhm.element_backends import (
    ReferenceElementSpec,
    create_reference_element,
    nodal_base_transformations,
    physical_simplex_tabulation,
    reference_base_transformations,
    reference_entity_dofs,
    reference_entity_transformations,
    simplex_lagrange_basis,
    simplex_lagrange_tabulation,
    tabulate_reference,
)
from pymhm.lagrange import multiindices
from pymhm.tetra_lagrange import tetra_indices


def test_import_leaves_optional_libraries_unloaded() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import pymhm.element_backends; "
            "assert not any(name.split('.')[0] in {'basix', 'dolfinx', 'ufl', 'FIAT', 'finat'} "
            "for name in sys.modules)",
        ],
        check=True,
    )


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"family": ""}, "family"),
        ({"cell": " triangle"}, "cell"),
        ({"lagrange_variant": None}, "lagrange_variant"),
        ({"dpc_variant": "unset "}, "dpc_variant"),
        ({"degree": True}, "degree"),
        ({"degree": 1.5}, "degree"),
        ({"degree": -1}, "degree"),
        ({"discontinuous": 1}, "discontinuous"),
        ({"dof_ordering": (0, 0)}, "dof_ordering"),
        ({"dof_ordering": (1, 2)}, "dof_ordering"),
        ({"dof_ordering": (-1, 0)}, "dof_ordering"),
        ({"dof_ordering": (False, 1)}, "dof_ordering"),
    ],
)
def test_descriptor_rejects_malformed_data(kwargs: dict[str, Any], match: str) -> None:
    values: dict[str, Any] = {"family": "P", "cell": "triangle", "degree": 2}
    values.update(kwargs)
    with pytest.raises(ValueError, match=match):
        ReferenceElementSpec(**values)


def test_descriptor_is_frozen_portable_and_uses_literal_native_degree() -> None:
    degree: Any = np.int64(1)
    spec = ReferenceElementSpec("RT", "triangle", degree, dof_ordering=(2, 0, 1))
    assert spec.degree == 1 and spec.dof_ordering == (2, 0, 1)
    assert pickle.loads(pickle.dumps(spec)) == spec
    with pytest.raises(FrozenInstanceError):
        spec.degree = 3  # type: ignore[misc]
    with pytest.raises(TypeError, match="spec"):
        create_reference_element(None)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="element"):
        tabulate_reference(None, [[0, 0]])  # type: ignore[arg-type]


def test_absent_optional_backend_is_reported_at_creation(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(_name: str) -> Any:
        raise ImportError("native library absent")

    monkeypatch.setattr(providers, "import_module", missing)
    with pytest.raises(ImportError, match="fenics-basix"):
        create_reference_element(ReferenceElementSpec("P", "triangle", 53))


@pytest.mark.parametrize(
    "spec",
    [
        ReferenceElementSpec("P", "unknown", 1),
        ReferenceElementSpec("unknown", "triangle", 1),
        ReferenceElementSpec("P", "triangle", 1, lagrange_variant="unknown"),
        ReferenceElementSpec("P", "triangle", 1, dpc_variant="unknown"),
    ],
)
def test_native_names_and_unsupported_families_are_not_silently_substituted(
    spec: ReferenceElementSpec,
) -> None:
    with pytest.raises(ValueError, match="unknown Basix"):
        create_reference_element(spec)
    with pytest.raises(RuntimeError):
        create_reference_element(ReferenceElementSpec("RT", "triangle", 0))


def test_cache_and_archived_basis_survive_threaded_tabulation() -> None:
    spec = ReferenceElementSpec("P", "triangle", 4, lagrange_variant="equispaced")
    with ThreadPoolExecutor(max_workers=3) as executor:
        elements = list(executor.map(create_reference_element, [spec] * 6))
    element = elements[0]
    assert all(other is element for other in elements)
    digest = hashlib.sha256(element.basis_matrix.tobytes()).hexdigest()
    assert element.basis_sha256 == digest
    assert not element.basis_matrix.flags.writeable
    with pytest.raises(ValueError):
        element.basis_matrix[0, 0] = 0
    expected = tabulate_reference(element, [[0.2, 0.3]], 3)
    with ThreadPoolExecutor(max_workers=3) as executor:
        tables = list(
            executor.map(lambda _: tabulate_reference(element, [[0.2, 0.3]], 3), range(6))
        )
    for table in tables:
        assert_array_equal(table, expected)
    assert element.basis_sha256 == hashlib.sha256(element.basis_matrix.tobytes()).hexdigest()


def test_scalar_third_derivatives_reproduce_an_independent_polynomial() -> None:
    import basix

    element = create_reference_element(
        ReferenceElementSpec("P", "triangle", 3, lagrange_variant="equispaced")
    )
    nodes = element._native.points
    coefficients = nodes[:, 0] ** 2 * nodes[:, 1]
    table = tabulate_reference(element, [[0.2, 0.3], [1.2, -0.1]], 3)[..., 0]
    assert_allclose(table[0] @ coefficients, [0.012, -0.144], atol=2e-14)
    assert_allclose(table[basix.index(2, 1)] @ coefficients, 2, atol=8e-13)
    assert_allclose(table[basix.index(0, 3)] @ coefficients, 0, atol=8e-13)


@pytest.mark.parametrize(
    "family, degree, shape", [("RT", 1, (2,)), ("BDM", 2, (2,)), ("Regge", 1, (2, 2))]
)
def test_vector_and_tensor_native_tables_and_orientation_data(
    family: str, degree: int, shape: tuple[int, ...]
) -> None:
    element = create_reference_element(ReferenceElementSpec(family, "triangle", degree))
    table = tabulate_reference(element, [[0.2, 0.3]], 1)
    assert element.value_shape == shape
    assert table.shape == (3, 1, element.dimension, int(np.prod(shape)))
    assert element.map_type == element._native.map_type.name
    assert_array_equal(element.basis_matrix, element._native.coefficient_matrix)
    assert_array_equal(
        reference_base_transformations(element), element._native.base_transformations()
    )
    for name, transform in reference_entity_transformations(element).items():
        assert not transform.flags.writeable
        assert_array_equal(transform, element._native.entity_transformations()[name])


@pytest.mark.parametrize("cell,degree", [("triangle", 3), ("tetrahedron", 4)])
def test_permuted_executed_basis_replays_fields_and_orientation_maps(
    cell: str, degree: int, tmp_path: Path
) -> None:
    """Non-involutive global ordering keeps archived fields and local maps coherent."""
    import basix

    canonical = create_reference_element(
        ReferenceElementSpec("P", cell, degree, lagrange_variant="equispaced")
    )
    # This cycle moves vertex, edge and interior DOFs between global slots.
    ordering = tuple(int(i) for i in np.roll(np.arange(canonical.dimension), 3))
    inverse = np.argsort(ordering)
    assert not np.array_equal(inverse, ordering)
    executed = create_reference_element(
        ReferenceElementSpec(
            "P", cell, degree, lagrange_variant="equispaced", dof_ordering=ordering
        )
    )
    bary = np.random.default_rng(29).dirichlet(np.ones(executed.cell_dimension + 1), size=9)
    points = np.ascontiguousarray(bary[:, 1:])
    coefficients = np.random.default_rng(43).normal(size=executed.dimension)
    table = tabulate_reference(executed, points, 2)
    canonical_table = tabulate_reference(canonical, points, 2)
    assert_array_equal(table, canonical_table[:, :, inverse])
    assert_array_equal(executed.basis_matrix, canonical.basis_matrix[inverse])
    assert executed.basis_sha256 != canonical.basis_sha256

    maps = reference_base_transformations(executed)
    intrinsic = reference_entity_transformations(executed)
    indices = reference_entity_dofs(executed)
    canonical_indices = reference_entity_dofs(canonical)
    for dimension, entities in enumerate(indices):
        for entity, dofs in enumerate(entities):
            assert dofs == tuple(ordering[dof] for dof in canonical_indices[dimension][entity])
    transformation = 0
    for dimension in range(1, executed.cell_dimension):
        local_maps = intrinsic["interval" if dimension == 1 else "triangle"]
        for dofs in indices[dimension]:
            for local_map in local_maps:
                embedded = np.eye(executed.dimension)
                embedded[np.ix_(dofs, dofs)] = local_map
                assert_array_equal(maps[transformation], embedded)
                transformation += 1
    assert transformation == len(maps)
    for name, local_maps in intrinsic.items():
        assert_array_equal(local_maps, reference_entity_transformations(canonical)[name])
        assert not local_maps.flags.writeable
    assert not maps.flags.writeable

    # Replay uses only the archived executed matrix and its native polyset,
    # without recreating or reordering a newly computed finite-element basis.
    archive = tmp_path / "executed-basis.npz"
    np.savez(archive, basis=executed.basis_matrix, coefficients=coefficients, maps=maps)
    polynomial_set = basix.polynomials.tabulate_polynomial_set(
        basix.CellType[cell],
        executed._native.polyset_type,
        executed._native.embedded_superdegree,
        2,
        points,
    )
    with np.load(archive) as saved:
        assert hashlib.sha256(saved["basis"].tobytes()).hexdigest() == executed.basis_sha256
        replay_table = np.einsum("bk,dkq->dqb", saved["basis"], polynomial_set)[..., None]
        assert_allclose(replay_table, table, atol=2e-13, rtol=2e-13)
        for map_index in range(len(maps)):
            oriented = saved["maps"][map_index] @ saved["coefficients"]
            replay = np.einsum("b,dqbv->dqv", oriented, replay_table)
            original = np.einsum("b,dqbv->dqv", oriented, table)
            canonical_coefficients = coefficients[np.asarray(ordering)]
            canonical_oriented = (
                reference_base_transformations(canonical)[map_index] @ canonical_coefficients
            )
            canonical_field = np.einsum("b,dqbv->dqv", canonical_oriented, canonical_table)
            assert_allclose(replay, original, atol=3e-13, rtol=3e-13)
            assert_allclose(replay, canonical_field, atol=3e-13, rtol=3e-13)


def test_empty_ordering_and_zero_entity_dofs_retain_native_defaults() -> None:
    """Empty ordering is not an empty basis; entity-local empty maps stay valid."""
    for degree, discontinuous in ((3, False), (0, True)):
        default = create_reference_element(
            ReferenceElementSpec(
                "P", "triangle", degree, lagrange_variant="equispaced", discontinuous=discontinuous
            )
        )
        empty = create_reference_element(
            ReferenceElementSpec(
                "P",
                "triangle",
                degree,
                lagrange_variant="equispaced",
                discontinuous=discontinuous,
                dof_ordering=(),
            )
        )
        assert_array_equal(empty.basis_matrix, default.basis_matrix)
        assert_array_equal(
            tabulate_reference(empty, [[0.2, 0.3]], 2), tabulate_reference(default, [[0.2, 0.3]], 2)
        )
        assert_array_equal(
            reference_base_transformations(empty), reference_base_transformations(default)
        )
        assert reference_entity_dofs(empty) == reference_entity_dofs(default)
        assert not empty.basis_matrix.flags.writeable
        assert not reference_base_transformations(empty).flags.writeable
        for maps in reference_entity_transformations(empty).values():
            assert not maps.flags.writeable
            if degree == 0:
                assert maps.shape[-2:] == (0, 0)


def test_upstream_rejects_global_dof_reordering_for_nonnodal_rt() -> None:
    """A descriptor cannot silently make a family support an upstream-only P feature."""
    with pytest.raises(RuntimeError, match="DOF ordering only supported for Lagrange"):
        create_reference_element(ReferenceElementSpec("RT", "triangle", 1, dof_ordering=(2, 0, 1)))


@pytest.mark.parametrize("points", [[[1j, 0]], [[np.inf, 0]], [0, 1], [[0, 1, 2]], [["0", "1"]]])
def test_reference_coordinates_are_finite_real_and_dimensioned(points: Any) -> None:
    element = create_reference_element(
        ReferenceElementSpec("P", "triangle", 1, lagrange_variant="equispaced")
    )
    with pytest.raises(ValueError, match="points"):
        tabulate_reference(element, points)
    with pytest.raises(ValueError, match="nderiv"):
        tabulate_reference(element, [[0, 0]], -1)


def test_literal_nodal_order_and_conjugated_maps_are_preserved() -> None:
    nodes = (multiindices(4) / 4)[::-1]
    basis = simplex_lagrange_basis("triangle", 4, nodes=nodes)
    values, first, second = simplex_lagrange_tabulation("triangle", 4, nodes, nodes=nodes)
    assert_allclose(values, np.eye(len(nodes)), atol=4e-14)
    assert first.shape == (len(nodes), len(nodes), 2)
    assert second.shape == (len(nodes), len(nodes), 2, 2)
    assert not basis.nodes.flags.writeable and not basis.permutation.flags.writeable
    assert basis.basis_sha256 == hashlib.sha256(basis.basis_matrix.tobytes()).hexdigest()
    maps = nodal_base_transformations(basis)
    assert_array_equal(
        maps,
        reference_base_transformations(basis.element)[:, basis.permutation][
            :, :, basis.permutation
        ],
    )
    for matrix in maps:
        assert_allclose(matrix @ matrix, np.eye(len(nodes)), rtol=0, atol=16 * np.finfo(float).eps)


def test_fresh_executed_basis_and_replay_survive_native_thread_counts() -> None:
    nodes = tetra_indices(6) / 6
    points = np.random.default_rng(37).dirichlet(np.ones(4), size=11)

    def fresh_basis() -> Any:
        with providers._ELEMENT_LOCK:
            providers._cached_reference_element.cache_clear()
            providers._cached_nodal_basis.cache_clear()
        return simplex_lagrange_basis("tetrahedron", 6, nodes=nodes)

    with threadpool_limits(limits=1):
        original = fresh_basis()
        table = tabulate_reference(original.element, points[:, 1:], 2)
        maps = nodal_base_transformations(original)
    with threadpool_limits(limits=2):
        recomputed = fresh_basis()
        assert_array_equal(recomputed.permutation, original.permutation)
        assert_array_equal(recomputed.basis_matrix, original.basis_matrix)
        assert recomputed.basis_sha256 == original.basis_sha256
        assert_array_equal(tabulate_reference(original.element, points[:, 1:], 2), table)
        assert_array_equal(nodal_base_transformations(original), maps)


@pytest.mark.parametrize("nderiv", [0, 1, 2])
def test_unrequested_derivatives_and_physical_per_cell_layouts(nderiv: int) -> None:
    nodes = multiindices(2) / 2
    bary = np.array([[0.5, 0.2, 0.3], [0.2, 0.4, 0.4]])
    geometry = np.array([[[1, 0], [0, 1]], [[2, 0.1], [0.2, 3]]])
    shared = physical_simplex_tabulation(
        "triangle", 2, bary, nodes=nodes, reference_gradients=geometry, nderiv=nderiv
    )
    cells = physical_simplex_tabulation(
        "triangle",
        2,
        np.broadcast_to(bary, (2, *bary.shape)),
        nodes=nodes,
        reference_gradients=geometry,
        nderiv=nderiv,
    )
    assert_array_equal(cells[0], np.broadcast_to(shared[0], cells[0].shape))
    for a, b in zip(shared[1:], cells[1:], strict=True):
        assert_array_equal(a, b)
    assert shared[1].shape[-1] == (2 if nderiv else 0)
    assert shared[2].shape[-2:] == ((2, 2) if nderiv == 2 else (0, 0))


@pytest.mark.parametrize(
    "cell, degree, nodes, match",
    [
        ("quadrilateral", 2, [[1, 0]], "cell"),
        ("triangle", 0, [[1, 0, 0]], "degree"),
        ("triangle", 1, [[0, 0, 0]], "sum to one"),
        ("triangle", 2, np.eye(3), "complete"),
        ("triangle", 1, [[1, 0, 0], [1, 0, 0], [0, 0, 1]], "uniquely"),
    ],
)
def test_nodal_adapter_rejects_a_changed_or_incomplete_lattice(
    cell: str, degree: int, nodes: Any, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        simplex_lagrange_basis(cell, degree, nodes=nodes)


def test_physical_adapter_rejects_missing_affine_geometry_and_invalid_layouts() -> None:
    nodes = np.eye(3)
    for bary, geometry, match in [
        (nodes, np.eye(2), "reference_gradients"),
        (nodes, np.full((1, 2, 2), np.nan), "reference_gradients"),
        (nodes, np.zeros((1, 2, 1)), "reference_gradients"),
        (np.ones(3), np.eye(2)[None], "bary"),
        (nodes[:, None], np.eye(2)[None], "per-cell"),
    ]:
        with pytest.raises(ValueError, match=match):
            physical_simplex_tabulation(
                "triangle", 1, bary, nodes=nodes, reference_gradients=geometry
            )
    with pytest.raises(ValueError, match="nderiv"):
        simplex_lagrange_tabulation("triangle", 1, nodes, nodes=nodes, nderiv=3)
    with pytest.raises(ValueError, match="sum to one"):
        simplex_lagrange_tabulation("triangle", 1, [[0.2, 0.2, 0.2]], nodes=nodes)
