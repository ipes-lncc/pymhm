"""Verify exact trace restriction, retained constants and original scalar equations."""

from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose

from examples.unfitted_trace_family import ScalarTraceFamily, _Cell, nested_trace_injection
from pymhm._legacy.models.darcy.primal import solve_darcy
from pymhm.fem.traces.interval import FaceSpace
from pymhm.linalg.linear import LinearSolveError
from pymhm.materials.cartesian import CartesianCellField
from pymhm.meshes.triangle import TriangleMesh


def _spawn_source(points):
    """Provide a module-level nonconstant source that can be transferred with spawn."""
    return 1 + points[:, 0] - 0.3 * points[:, 1]


def _spawn_boundary(points):
    """Supply a nonhomogeneous polynomial Dirichlet functional."""
    return 0.2 + points[:, 0] ** 2 - 0.1 * points[:, 1]


@pytest.mark.parametrize(("resolution", "workers"), [(1, 3), (2, 2)])
def test_spawn_collection_preserves_executed_assembly_order(resolution, workers, monkeypatch):
    """Keep executed cell order exact and verify independently computed physical fields."""
    from numpy.testing import assert_array_equal
    from threadpoolctl import threadpool_limits

    from examples import unfitted_trace_family as owner

    original = owner._prepared_cells
    captured = {}

    def observe(factory, count, trace_size, workers, local_solver):
        """Retain actual worker outputs in the order consumed by the coordinator."""
        records = captured[workers] = []
        for record in original(factory, count, trace_size, workers, local_solver):
            records.append(record)
            yield record

    mesh = TriangleMesh.unit_square(resolution)
    options = dict(
        trace_degree=2,
        segments=2,
        local_degree=3,
        local_refinement=4,
        permeability=CartesianCellField(np.array([[2.0, 1.0], [3.0, 1.5]]), (0.5, 0.5)),
        source=_spawn_source,
        dirichlet=_spawn_boundary,
        quadrature_order=6,
    )
    with threadpool_limits(1):
        with monkeypatch.context() as observation:
            observation.setattr(owner, "_prepared_cells", observe)
            serial = ScalarTraceFamily.prepare(mesh, **options)
            parallel = ScalarTraceFamily.prepare(mesh, **options, workers=workers)
        # Identical executed contributions must be reduced in macrocell order
        # with exactly the same coefficients. Reuse these records for replay;
        # independent native initializations need not compute identical bits.
        for family, count in ((serial, 1), (parallel, workers)):
            records = captured[count]
            assert len(records) == len(mesh.cells)
            for cell, record in enumerate(records):
                assert record[3] is family.cells[cell]
                assert_array_equal(
                    record[0], np.r_[family.cells[cell].trace_dofs, family.skeleton.size + cell]
                )
            with monkeypatch.context() as replay:
                replay.setattr(
                    owner, "_prepared_cells", lambda *args, records=records: iter(records)
                )
                repeated = ScalarTraceFamily.prepare(mesh, **options, workers=count)
            assert_array_equal(family.matrix.toarray(), repeated.matrix.toarray())
            assert_array_equal(family.load, repeated.load)
        assert serial.degree == parallel.degree == 3
        assert serial.quadrature_order == parallel.quadrature_order == 6
        for name in ("points", "cells", "faces", "normals"):
            assert_array_equal(
                getattr(serial.skeleton.mesh, name), getattr(parallel.skeleton.mesh, name)
            )
        assert_allclose(serial.matrix.toarray(), parallel.matrix.toarray(), rtol=1e-11, atol=1e-12)
        assert_allclose(serial.load, parallel.load, rtol=1e-11, atol=1e-12)
        for a, b in zip(serial.cells, parallel.cells, strict=True):
            for name in ("points", "cells", "faces", "normals"):
                assert_array_equal(getattr(a.mesh, name), getattr(b.mesh, name))
            assert_array_equal(a.trace_dofs, b.trace_dofs)
            for name in ("matrix", "coupling"):
                assert_allclose(
                    getattr(a, name).toarray(), getattr(b, name).toarray(), rtol=1e-11, atol=1e-12
                )
            for name in ("load", "source", "lifts", "kernel"):
                assert_allclose(getattr(a, name), getattr(b, name), rtol=1e-11, atol=1e-12)
        solutions = [family.solve(1, 1) for family in (serial, parallel)]
        direct = solve_darcy(
            mesh,
            skeleton=solutions[0][0].skeleton,
            degree=3,
            local_refinement=4,
            permeability=options["permeability"],
            source=_spawn_source,
            dirichlet=_spawn_boundary,
            quadrature_order=6,
        )
    for result, diagnostics in solutions:
        assert_allclose(result.hybrid.trace, direct.hybrid.trace, rtol=1e-11, atol=1e-12)
        assert_allclose(result.hybrid.coarse, direct.hybrid.coarse, rtol=1e-11, atol=1e-12)
        assert_allclose(result.pressure, direct.pressure, rtol=1e-11, atol=1e-12)
        assert_allclose(result.flux, direct.flux, rtol=1e-11, atol=1e-12)
        assert_allclose(result.conservation_residuals(), 0, atol=2e-13)
        assert diagnostics["original_trace_residual"] < 1e-12
        assert diagnostics["original_local_residual_max"] < 1e-12
        assert diagnostics["local_componentwise_backward_error_max"] < 1e-12


def test_spawn_worker_count_is_validated_before_assembly():
    """Reject invalid worker counts without starting a native process or local solve."""
    with pytest.raises(ValueError, match="workers"):
        ScalarTraceFamily.prepare(
            TriangleMesh.unit_square(),
            trace_degree=0,
            segments=1,
            local_degree=1,
            local_refinement=1,
            workers=0,
        )


def test_selected_gauss_count_integrates_independent_high_degree_load_moments():
    """A declared q13 reaches both volume and boundary data, beyond the P2 floor."""
    mesh = TriangleMesh.unit_square(1)

    def high_degree(points):
        """Use x²⁰, whose unit-square and horizontal-edge integrals are 1/21."""
        return points[:, 0] ** 20

    options = dict(trace_degree=2, segments=1, local_degree=2, local_refinement=1)
    exact = ScalarTraceFamily.prepare(mesh, **options, source=high_degree, quadrature_order=13)
    underintegrated = ScalarTraceFamily.prepare(
        mesh, **options, source=high_degree, quadrature_order=1
    )
    assert exact.quadrature_order == 13
    assert underintegrated.quadrature_order == 4
    assert_allclose(sum(cell.load.sum() for cell in exact.cells), 1 / 21, rtol=2e-14)
    assert abs(sum(cell.load.sum() for cell in underintegrated.cells) - 1 / 21) > 1e-6

    boundary = ScalarTraceFamily.prepare(
        mesh, **options, dirichlet=high_degree, quadrature_order=13
    )
    low_boundary = ScalarTraceFamily.prepare(
        mesh, **options, dirichlet=high_degree, quadrature_order=1
    )
    horizontal_moments = []
    for face in mesh.boundary_faces:
        vertices = mesh.points[mesh.faces[face]]
        if vertices[0, 1] == vertices[1, 1]:
            dof = boundary.skeleton.offsets[face]
            assert_allclose(-boundary.load[dof], 1 / 21, rtol=2e-14)
            horizontal_moments.append(-low_boundary.load[dof])
    assert len(horizontal_moments) == 2
    assert np.all(np.abs(np.asarray(horizontal_moments) - 1 / 21) > 1e-6)


@pytest.mark.parametrize("quadrature_order", [11, 13])
@pytest.mark.parametrize("threads", [1, 2])
def test_exact_cyclic_trace_kernel_is_excluded_and_refinement_restores_injectivity(
    quadrature_order, threads
):
    """Keep an exact P8/P3 trace null mode distinct from volume integration accuracy.

    The integer pattern contains shifted-Legendre coefficients on two equal
    segments per face. Scaling by inverse physical face length gives the same
    cyclic endpoint moments on each edge. Its global orientation is the executed
    FaceSpace orientation; the coupling tests actual P8 nodal pressure traces.
    Independent rational integration gives zero against every nodal trace.
    """
    from threadpoolctl import threadpool_limits

    mesh = TriangleMesh.unit_square(1)
    options = dict(
        trace_degree=3,
        segments=2,
        local_degree=8,
        source=1.0,
        quadrature_order=quadrature_order,
    )
    with threadpool_limits(threads):
        excluded = ScalarTraceFamily.prepare(mesh, **options, local_refinement=1)
        pattern = np.array([27.0, 129.0, 125.0, 231.0, -27.0, 129.0, -125.0, 231.0])
        multiplier = np.concatenate([pattern / length for length in mesh.lengths])
        multiplier /= np.linalg.norm(multiplier)
        for cell in excluded.cells:
            assert_allclose(cell.coupling @ multiplier[cell.trace_dofs], 0, atol=2e-14)
            assert np.linalg.matrix_rank(cell.coupling.toarray()) == len(cell.trace_dofs) - 1
        with pytest.raises(LinearSolveError, match="insufficient numerical rank"):
            excluded.solve(3, 2)

        resolved = ScalarTraceFamily.prepare(mesh, **options, local_refinement=2)
        for cell in resolved.cells:
            assert np.linalg.matrix_rank(cell.coupling.toarray()) == len(cell.trace_dofs)
            assert np.linalg.norm(cell.coupling @ multiplier[cell.trace_dofs]) > 1e-3
        result, diagnostics = resolved.solve(3, 2)
        assert result.quadrature_order == quadrature_order
        assert diagnostics["original_trace_residual"] < 1e-10
        assert diagnostics["original_local_residual_max"] < 1e-10


def test_piecewise_legendre_injection_preserves_every_resolved_jump():
    """Compare piecewise polynomials away from breaks with nonuniform segment lengths."""
    fine = FaceSpace((0, 0.1, 0.3, 0.6, 1), (3, 3, 4, 3))
    coarse = FaceSpace((0, 0.3, 1), (2, 3))
    injection = nested_trace_injection(fine, coarse)
    values = np.arange(coarse.size) + 0.3
    points = np.array([0.03, 0.19, 0.32, 0.71, 0.99])
    assert_allclose(
        fine.evaluate(points) @ injection @ values,
        coarse.evaluate(points) @ values,
        rtol=2e-14,
        atol=2e-14,
    )
    with pytest.raises(ValueError, match="discontinuous"):
        nested_trace_injection(FaceSpace.uniform(2, continuous=True), coarse)
    with pytest.raises(ValueError, match="breakpoint"):
        nested_trace_injection(FaceSpace.uniform(3, 2), coarse)
    with pytest.raises(ValueError, match="degree"):
        nested_trace_injection(FaceSpace.uniform(1), FaceSpace.uniform(2))


@pytest.mark.parametrize("heterogeneous", [False, True])
@pytest.mark.parametrize("boundary_scale", [0.0, 1.0])
@pytest.mark.parametrize("quadrature_order", [1, 6, 13])
def test_nested_schur_fields_and_constants_equal_independent_direct_solves(
    heterogeneous, boundary_scale, quadrature_order
):
    """Keep local matrices, sources and modes while varying both skeletal resolutions."""
    mesh = TriangleMesh.unit_square(1)
    material = (
        CartesianCellField(np.array([[2.0, 1.0], [3.0, 1.5]]), (0.5, 0.5)) if heterogeneous else 1.0
    )

    def source(x):
        """Provide a nonconstant affine forcing with a nonzero total load."""
        return 1 + x[:, 0] - 0.3 * x[:, 1]

    def dirichlet(x):
        """Exercise a nonzero quadratic boundary functional with exact Gauss integration."""
        return boundary_scale * (0.2 + x[:, 0] ** 2 - 0.1 * x[:, 1])

    family = ScalarTraceFamily.prepare(
        mesh,
        trace_degree=2,
        segments=2,
        local_degree=3,
        local_refinement=4,
        permeability=material,
        source=source,
        dirichlet=dirichlet,
        quadrature_order=quadrature_order,
    )
    assert family.quadrature_order == max(quadrature_order, 5)
    owners = [id(cell.matrix) for cell in family.cells]
    for degree in range(3):
        for segments in (1, 2):
            result, diagnostics = family.solve(degree, segments)
            direct = solve_darcy(
                mesh,
                skeleton=result.skeleton,
                degree=3,
                local_refinement=4,
                permeability=material,
                source=source,
                dirichlet=dirichlet,
                quadrature_order=quadrature_order,
            )
            assert result.quadrature_order == direct.quadrature_order == family.quadrature_order
            assert [id(cell.matrix) for cell in family.cells] == owners
            assert_allclose(result.hybrid.trace, direct.hybrid.trace, rtol=1e-11, atol=1e-12)
            for field, expected in zip(result.pressure, direct.pressure, strict=True):
                assert_allclose(field, expected, rtol=1e-11, atol=1e-12)
            assert_allclose(result.hybrid.coarse, direct.hybrid.coarse, rtol=1e-11, atol=1e-12)
            assert_allclose(result.flux, direct.flux, rtol=1e-11, atol=1e-12)
            assert_allclose(result.conservation_residuals(), 0, atol=2e-13)
            assert diagnostics["original_trace_residual"] < 1e-12
            assert diagnostics["original_local_residual_max"] < 1e-12
            assert diagnostics["local_componentwise_backward_error_max"] < 1e-12


def test_homogeneous_source_and_nonzero_exact_polynomial():
    """Test zero denominators and a representable quartic with nonzero volume forcing."""
    mesh = TriangleMesh.unit_square(1)
    zero = ScalarTraceFamily.prepare(
        mesh, trace_degree=0, segments=1, local_degree=2, local_refinement=2
    )
    result, diagnostics = zero.solve(0, 1)
    assert not any(diagnostics.values())
    assert not np.any(np.concatenate(result.pressure))

    def exact(x):
        """Return the quartic bubble with homogeneous exterior values."""
        return x[:, 0] * (1 - x[:, 0]) * x[:, 1] * (1 - x[:, 1])

    def source(x):
        """Return minus the analytically differentiated Laplacian."""
        return 2 * (x[:, 0] * (1 - x[:, 0]) + x[:, 1] * (1 - x[:, 1]))

    family = ScalarTraceFamily.prepare(
        mesh, trace_degree=3, segments=1, local_degree=4, local_refinement=2, source=source
    )
    result, _ = family.solve(3, 1)
    assert result.l2_error(exact, order=7) < 2e-14


@pytest.mark.parametrize("datum", [1e-8, 2.0, 1e10])
def test_constant_dirichlet_roundoff_scales_with_the_physical_field(datum):
    """Preserve exact kernel solutions without an absolute residual floor."""
    mesh = TriangleMesh.unit_square(1)
    family = ScalarTraceFamily.prepare(
        mesh,
        trace_degree=1,
        segments=1,
        local_degree=2,
        local_refinement=2,
        dirichlet=datum,
    )
    result, diagnostics = family.solve(0, 1)
    assert result.l2_error(datum) / abs(datum) < 5e-14
    assert diagnostics["local_componentwise_backward_error_max"] < 1e-13
    assert diagnostics["local_accepted_with_roundoff_envelope"]
    assert (
        diagnostics["original_local_defect_l2_max"] <= diagnostics["local_roundoff_envelope_l2_max"]
    )


def test_original_equation_checks_detect_corrupted_reconstruction(monkeypatch):
    """A solved reduced Schur system cannot authorize a different physical local field."""
    mesh = TriangleMesh.unit_square(1)
    family = ScalarTraceFamily.prepare(
        mesh, trace_degree=0, segments=1, local_degree=2, local_refinement=2, source=1.0
    )
    bad_cell = replace(
        family.cells[0], source=family.cells[0].source + np.arange(len(family.cells[0].source))
    )
    with pytest.raises(ValueError, match="original local"):
        replace(family, cells=(bad_cell, *family.cells[1:])).solve(0, 1)
    import examples.unfitted_trace_family as owner

    original = owner.solve_linear
    monkeypatch.setattr(owner, "solve_linear", lambda a, b: original(a, b) + 0.1)
    with pytest.raises(ValueError, match="restricted original"):
        family.solve(0, 1)


def test_retained_kernel_contract_is_explicit():
    """Reject generic retained vectors that require a different reconstruction contract."""
    from types import SimpleNamespace

    with pytest.raises(ValueError, match="constant kernel"):
        _Cell.from_response(
            SimpleNamespace(problem=SimpleNamespace(kernel=np.zeros((2, 0)))),
            TriangleMesh.unit_square(1),
        )
    from pymhm.core.contracts import LocalProblem

    matrix = 1e6 * (4 * np.eye(4) - np.ones((4, 4))) + 1e-5 * np.eye(4)
    problem = LocalProblem(matrix, np.eye(4), np.zeros(4), np.arange(4), kernel=np.ones((4, 1)))
    response = problem.condense()
    assert response.coarse_vectors is not None
    cell = _Cell.from_response(response, TriangleMesh.unit_square(1))
    assert_allclose(cell.kernel, response.retained_basis, rtol=0, atol=0)
