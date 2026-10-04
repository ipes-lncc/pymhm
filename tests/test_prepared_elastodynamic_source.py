"""Original load/basis invariants for the explicit separable planar source contract."""

from dataclasses import replace

import numpy as np
import pytest

from pymhm import ElastodynamicStepper, PolylineLayerField, RadialDiskLoad, TetraMesh, TriangleMesh
from pymhm.elastodynamics import PreparedElastodynamicSource
from pymhm.elements import triangle_quadrature
from pymhm.lagrange import multiindices, reference_basis, reference_values
from pymhm.mesh import FaceSpace, SkeletonSpace
from pymhm.triangle_fields import SeparableTriangleField


@pytest.mark.parametrize("degree", [1, 2, 3, 4, 5, 6])
def test_values_and_derivatives_reproduce_reference_polynomials(degree):
    """Independent polynomial derivatives hold inside and outside the reference triangle."""
    bary, _ = triangle_quadrature(5)
    bary = np.vstack([bary, [-0.1, 0.2, 0.9]])
    values = reference_values(degree, bary)
    tabulated_values, first_derivative, second_derivative = reference_basis(degree, bary)
    np.testing.assert_array_equal(values, tabulated_values)
    tangent = np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]])
    gradient = np.einsum("qia,ad->qid", first_derivative, tangent)
    hessian = np.einsum("qiab,ad,be->qide", second_derivative, tangent, tangent)
    nodes = multiindices(degree) / degree
    for first in range(degree + 1):
        for second in range(degree + 1 - first):
            expected = bary[:, 1] ** first * bary[:, 2] ** second
            actual = values @ (nodes[:, 1] ** first * nodes[:, 2] ** second)
            action = abs(values) @ (abs(nodes[:, 1] ** first * nodes[:, 2] ** second))
            assert np.all(
                abs(actual - expected) <= 128 * np.finfo(float).eps * (action + abs(expected) + 1)
            )
            coefficient = nodes[:, 1] ** first * nodes[:, 2] ** second
            for axis in range(2):
                powers = [first, second]
                factor = powers[axis]
                powers[axis] = max(powers[axis] - 1, 0)
                expected_first = factor * bary[:, 1] ** powers[0] * bary[:, 2] ** powers[1]
                actual_first = gradient[:, :, axis] @ coefficient
                scale = abs(gradient[:, :, axis]) @ abs(coefficient)
                assert np.all(
                    abs(actual_first - expected_first)
                    <= 1024 * np.finfo(float).eps * (scale + abs(expected_first) + 1)
                )
                for other in range(2):
                    second_factor = factor * powers[other]
                    derivative_powers = powers.copy()
                    derivative_powers[other] = max(derivative_powers[other] - 1, 0)
                    expected_second = (
                        second_factor
                        * bary[:, 1] ** derivative_powers[0]
                        * bary[:, 2] ** derivative_powers[1]
                    )
                    actual_second = hessian[:, :, axis, other] @ coefficient
                    scale = abs(hessian[:, :, axis, other]) @ abs(coefficient)
                    assert np.all(
                        abs(actual_second - expected_second)
                        <= 1024 * np.finfo(float).eps * (scale + abs(expected_second) + 1)
                    )
    assert reference_values(degree, np.empty((0, 3))).shape == (0, len(nodes))


@pytest.mark.parametrize("bary", [[[1, 0]], [1, 0, 0], [[np.nan, 0, 0]], [[1j, 0, 0]]])
def test_values_reject_invalid_coordinates(bary):
    """Malformed or nonreal coordinates cannot silently enter an original load vector."""
    with pytest.raises(ValueError, match="coordinates"):
        reference_values(3, np.asarray(bary))


@pytest.mark.parametrize("degree", [0, -1, True, 1.5])
def test_values_reject_invalid_degree(degree):
    """Values-only evaluation preserves the discrete polynomial degree contract."""
    with pytest.raises(ValueError, match="integer"):
        reference_values(degree, np.array([[1.0, 0, 0]]))


def test_load_tabulation_uses_values_without_discarded_derivatives(monkeypatch):
    """A true radial load still has analytical zero resultant and first moment."""
    mesh = TriangleMesh([[0.0, 0], [1.0, 0], [0.0, 1]], [[0, 1, 2]])
    with ElastodynamicStepper(
        mesh, time_step=0.01, degree=3, local_refinement=1, quadrature_order=5
    ) as stepper:
        local = stepper.locals[0]
        monkeypatch.setattr(
            "pymhm.elastodynamics.physical_basis",
            lambda *args: pytest.fail("derivative tabulation in load"),
        )
        source = RadialDiskLoad([0.25, 0.25], 0.1, 3.0, time_function=lambda t: 2 * t)
        load = local.load_at_time(source, 0.7).reshape(-1, 2)
        np.testing.assert_allclose(load.sum(axis=0), 0, atol=3e-15)
        np.testing.assert_allclose(
            (local.nodes - source.center).T @ load, np.eye(2) * 4.2 * np.pi * 0.1**3 / 3, atol=3e-16
        )


@pytest.mark.parametrize("substeps", [1, 2])
@pytest.mark.parametrize("nonhomogeneous", [False, True])
def test_prepared_disk_matches_at_time_and_original_nonhomogeneous_boundary_fields(
    substeps, nonhomogeneous
):
    """Preparation changes neither forced physical fields nor exterior traction conventions."""
    mesh = TriangleMesh.unit_square()
    skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(1) for _ in mesh.faces), 2)
    traction = np.array([0.1, -0.03]) if nonhomogeneous else np.zeros(2)
    options = dict(
        time_step=0.01,
        degree=2,
        local_refinement=2,
        local_substeps=substeps,
        skeleton=skeleton,
        traction={int(f): traction for f in mesh.boundary_faces},
        quadrature_order=4,
    )
    source = RadialDiskLoad([0.4, 0.4], 0.12, 2.3, time_function=lambda t: 1 + np.sin(2 * t))
    assert isinstance(source, SeparableTriangleField)
    with (
        ElastodynamicStepper(mesh, **options) as raw,
        ElastodynamicStepper(mesh, **options) as cached,
    ):
        prepared = cached.prepare_source(source)
        for local, load in zip(cached.locals, prepared.loads, strict=True):
            assert not load.flags.writeable
            for time in (0, 0.003, 0.011):
                original = local.load_at_time(source, time)
                actual = local.load_at_time(prepared, time)
                np.testing.assert_allclose(actual, original, rtol=3e-13, atol=3e-17)

        def initial(p):
            """Specify the same nonzero affine initial displacement on both solvers."""
            return np.column_stack((0.01 + 0.02 * p[:, 0], 0.03 * p[:, 1]))

        raw.initialize(initial, [-0.03, 0.01])
        cached.initialize(initial, [-0.03, 0.01])
        for _ in range(3):
            expected = raw.advance(source)
            actual = cached.advance(prepared)
            np.testing.assert_allclose(actual.trace, expected.trace, rtol=3e-12, atol=2e-15)
            for field in ("displacement", "velocity"):
                np.testing.assert_allclose(
                    getattr(actual, field), getattr(expected, field), rtol=3e-12, atol=3e-16
                )
            assert actual.constraint_residual < 1e-12


def test_generic_separable_source_snapshot_reuses_every_original_spatial_vector(monkeypatch):
    """The contract applies to an arbitrary quadrature provider, without case or disk knowledge."""
    mesh = TriangleMesh.unit_square()
    field = PolylineLayerField([0.0, 1.0], [[0.4, 0.6]], [[1.0, -0.2], [0.7, 0.3]])

    class Separated:
        def spatial_field(self):
            return field

        def time_scale(self, time):
            return 1 + 0.2 * time

    with ElastodynamicStepper(mesh, time_step=0.01, degree=2, local_refinement=2) as stepper:
        prepared = stepper.prepare_source(Separated())
        owned = [load.copy() for load in prepared.loads]
        for i, local in enumerate(stepper.locals):
            np.testing.assert_array_equal(owned[i], local.load(field))
        stepper.initialize()
        monkeypatch.setattr(
            "pymhm.elastodynamics.ElastodynamicLocal.load",
            lambda *args, **kw: pytest.fail("spatial rule reintegrated"),
        )
        for _ in range(3):
            stepper.advance(prepared)
        for i, load in enumerate(prepared.loads):
            np.testing.assert_array_equal(load, owned[i])


@pytest.mark.parametrize(
    "defect", ["count", "duplicate", "callable", "shape", "dtype", "complex", "nan", "dimension"]
)
def test_prepared_snapshot_rejects_incompatible_vectors(defect):
    """Basis ownership, precision and dimensions precede any coefficient replay."""
    with ElastodynamicStepper(TriangleMesh.unit_square(), time_step=0.01, degree=2) as stepper:
        locals = stepper.locals
        loads = tuple(np.zeros(local.mass.shape[0]) for local in locals)

        def temporal(t):
            """Use a valid scalar source scale unless the requested defect changes it."""
            return 1.0

        if defect == "count":
            loads = loads[:-1]
        elif defect == "duplicate":
            locals = (locals[0], locals[0])
        elif defect == "callable":
            temporal = None
        elif defect == "shape":
            loads = (loads[0][:-1], loads[1])
        elif defect == "dtype":
            loads = (loads[0].astype(np.float32), loads[1])
        elif defect == "complex":
            loads = (loads[0].astype(complex), loads[1])
        elif defect == "nan":
            loads[0][0] = np.nan
        else:
            locals = (replace(locals[0], nodes=np.zeros((len(locals[0].nodes), 3))), locals[1])
        with pytest.raises(ValueError):
            PreparedElastodynamicSource(locals, loads, temporal)


@pytest.mark.parametrize("bad", [np.nan, np.inf, 1j, [1.0, 2.0]])
def test_prepared_temporal_data_and_foreign_basis_rejected(bad):
    """Temporal validation and actual local ownership are independent of vector dimensions."""
    with ElastodynamicStepper(TriangleMesh.unit_square(), time_step=0.01, degree=2) as stepper:
        source = RadialDiskLoad([2.0, 2.0], 0.1)
        prepared = stepper.prepare_source(source)
        with pytest.raises(ValueError, match="time"):
            prepared.load_at_time(stepper.locals[0], bad)
        bad_scale = replace(prepared, time_function=lambda t: bad)
        with pytest.raises(ValueError, match="scale"):
            bad_scale.load_at_time(stepper.locals[0], 0.1)
        with pytest.raises(ValueError, match="different"):
            prepared.load_at_time(replace(stepper.locals[0], degree=1), 0.1)
        with pytest.raises(ValueError, match="time"):
            source.time_scale(bad)
        with pytest.raises(ValueError, match="scale"):
            replace(source, time_function=lambda t: bad).at_time(0.1)


def test_prepare_rejects_absent_contract_wrong_provider_dimension_and_closed_owner():
    """Unsupported source/dimension contracts fail explicitly on an open numerical owner."""
    mesh = TriangleMesh.unit_square()

    class Wrong:
        def spatial_field(self):
            return [1.0, 0.0]

        def time_scale(self, t):
            return 1.0

    source = RadialDiskLoad([0.4, 0.4], 0.1)
    with ElastodynamicStepper(mesh, time_step=0.01, degree=2) as stepper:
        with pytest.raises(TypeError, match="explicit"):
            stepper.prepare_source(lambda t, p: p)
        with pytest.raises(TypeError, match="quadrature"):
            stepper.prepare_source(Wrong())
    with pytest.raises(RuntimeError, match="open"):
        stepper.prepare_source(source)
    with (
        ElastodynamicStepper(TetraMesh.unit_cube(), time_step=0.01, degree=2) as space,
        pytest.raises(ValueError, match="two dimensions"),
    ):
        space.prepare_source(source)


def test_prepared_source_rejects_finite_factors_that_overflow_binary64_loads():
    """Finite inputs cannot bypass the original finite integrated-force contract."""
    with ElastodynamicStepper(TriangleMesh.unit_square(), time_step=0.01, degree=2) as stepper:
        loads = tuple(np.full(local.mass.shape[0], 1e308) for local in stepper.locals)
        prepared = PreparedElastodynamicSource(stepper.locals, loads, lambda t: 2.0)
        with pytest.raises(ValueError, match="scaled load"):
            prepared.load_at_time(stepper.locals[0], 0.1)
        np.testing.assert_array_equal(prepared.loads[0], loads[0])
        huge = np.longdouble(np.finfo(float).max) * np.longdouble(2)
        if np.isfinite(huge):
            source = replace(prepared, time_function=lambda t: huge)
            with pytest.raises(ValueError, match="binary64"):
                source.load_at_time(stepper.locals[0], 0.1)
            disk = RadialDiskLoad([0.4, 0.4], 0.1, time_function=lambda t: huge)
            with pytest.raises(ValueError, match="binary64"):
                disk.time_scale(0.1)
