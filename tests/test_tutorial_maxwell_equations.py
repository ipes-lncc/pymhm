"""User-defined local/global leapfrog equations preserve the explicit compatibility controls."""

from __future__ import annotations

import pickle
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from threadpoolctl import threadpool_limits

from examples import tutorial_maxwell_equations as tutorial
from examples.tutorial_vector_variants import MaxwellBoundary, _tetra_patch
from pymhm._legacy.models.waves.maxwell import MaxwellStepper
from pymhm.core.equations import Equation
from pymhm.execution.cpu import ExecutionConfig
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace
from pymhm.fem.vector.curl import CurlOperators, TangentialTraceSpace
from pymhm.meshes.cartesian import CartesianMacroMesh
from pymhm.meshes.triangle import TriangleMesh


@pytest.mark.parametrize("original", [False, True])
def test_electric_kick_reports_its_numeric_original_equation_defect(original):
    """The observer returns the integrated local-row residual in both solve routes."""
    rng = np.random.default_rng(19)
    data = tutorial.prepare(TriangleMesh.unit_square(), degree=2, local_refinement=1)
    electric = tuple(rng.normal(size=local.electric_mass.shape[0]) for local in data.locals)
    magnetic = tuple(rng.normal(size=local.magnetic_mass.shape[0]) for local in data.locals)
    forcing = tuple(rng.normal(size=local.electric_mass.shape[0]) for local in data.locals)
    duration = 0.003
    _, trace, means, _, reported = tutorial.electric_kick(
        data,
        electric,
        magnetic,
        forcing,
        duration,
        np.zeros(data.skeleton.size),
        original=original,
    )
    expected = 0.0
    for local, old, h, force, mean in zip(
        data.locals, electric, magnetic, forcing, means, strict=True
    ):
        coupling = duration / 2 * (local.coupling @ trace[local.trace_dofs])
        load = local.electric_mass @ old + duration / 2 * (force - local.curl.T @ h)
        defect = local.electric_mass @ mean + coupling - load
        scale = (
            np.linalg.norm(abs(local.electric_mass) @ abs(mean))
            + np.linalg.norm(abs(coupling))
            + np.linalg.norm(abs(load))
        )
        expected = max(expected, float(np.linalg.norm(defect) / scale))
    assert expected > 0
    assert not isinstance(reported, (bool, np.bool_))
    assert reported == pytest.approx(expected, rel=1e-10, abs=0)


@pytest.mark.parametrize("homogeneous", [False, True])
def test_user_defined_stationary_vector_trajectory_matches_every_original_step(homogeneous):
    electric = np.zeros(3) if homogeneous else np.array([1.2, -0.3, 0.7])
    magnetic = np.zeros(3) if homogeneous else np.array([0.4, 0.8, -0.2])
    mesh = _tetra_patch()
    boundary = MaxwellBoundary(electric, magnetic)
    with threadpool_limits(1):
        data = tutorial.prepare(mesh, boundary_data=boundary)
        state = tutorial.initialize(data, electric, magnetic)
        with MaxwellStepper(
            mesh,
            time_step=0.001,
            degree=2,
            local_refinement=2,
            absorbing=1,
            boundary_data=boundary,
            quadrature_order=5,
        ) as comparator:
            reference = comparator.initialize(electric, magnetic)
            for step in range(5):
                if step:
                    state = tutorial.advance(data, state)
                    reference = comparator.advance()
                difference = tutorial.compare_state(state, reference)
                assert difference["electric_coefficient_linf"] < 2e-12
                assert difference["magnetic_coefficient_linf"] < 2e-12
                assert difference["trace_coefficient_linf"] < 2e-11
                assert difference["energy_difference"] < 2e-14
                assert difference["electric_time_difference"] == 0
                assert difference["magnetic_time_difference"] == 0
                assert max(tutorial.l2_errors(data, state, electric, magnetic)) < 1e-12
                assert state.constraint_moment_norm < 1e-12
                assert state.original_electric_residual < 1e-10
                assert abs(state.energy_balance_residual) < 2e-14
                assert abs(reference.energy_balance_residual) < 2e-14


@pytest.mark.parametrize("geometry", ["triangle", "rectangle", "tetrahedron"])
def test_independent_providers_handle_affine_fields_sources_and_time_data(geometry):
    mesh = (
        _tetra_patch()
        if geometry == "tetrahedron"
        else (TriangleMesh.unit_square() if geometry == "triangle" else CartesianMacroMesh(2, 1))
    )
    dimension = mesh.points.shape[1]
    components = 1 if dimension == 2 else 3
    magnetic = np.arange(1, dimension + 1) / 5

    def electric(points):
        return 1 + points[:, :components] / 10

    def boundary(time, points, normals):
        cross = (
            magnetic[0] * normals[:, 1] - magnetic[1] * normals[:, 0]
            if dimension == 2
            else np.cross(magnetic, normals)
        )
        return electric(points)[:, 0] - cross if dimension == 2 else electric(points) - cross

    def forcing(time, points):
        return np.full((len(points), components), time / 10)

    calls = {"electric": 0, "magnetic": 0, "global": 0}

    def electric_provider(item):
        calls["electric"] += 1
        return tutorial.electric_equations(item)

    def magnetic_provider(item):
        calls["magnetic"] += 1
        return tutorial.magnetic_equations(item)

    def global_provider(data, load):
        calls["global"] += 1
        return Equation(data.impedance, -load)

    with threadpool_limits(1):
        data = tutorial.prepare(mesh, boundary_data=boundary)
        state = tutorial.initialize(
            data,
            electric,
            magnetic,
            source=forcing,
            electric_provider=electric_provider,
            global_provider=global_provider,
        )
        state = tutorial.advance(
            data,
            state,
            source=forcing,
            electric_provider=electric_provider,
            magnetic_provider=magnetic_provider,
            global_provider=global_provider,
        )
        with MaxwellStepper(
            mesh,
            time_step=0.001,
            degree=2,
            local_refinement=2,
            absorbing=1,
            boundary_data=boundary,
            quadrature_order=5,
        ) as comparator:
            comparator.initialize(electric, magnetic, source=forcing)
            reference = comparator.advance(forcing)
    assert calls == {"electric": 2 * len(mesh.cells), "magnetic": len(mesh.cells), "global": 2}
    assert max(tutorial.compare_state(state, reference).values()) < 2e-11
    assert_allclose(
        tutorial.l2_errors(data, state, electric, magnetic),
        reference.l2_errors(electric, magnetic, order=5),
        atol=2e-15,
    )


def test_zero_electric_pec_constraint_uses_uncancelled_lift_scale():
    with threadpool_limits(1):
        data = tutorial.prepare(TriangleMesh.unit_square(), absorbing=None)
        state = tutorial.initialize(data, 0.0, [0.2, 0.4])
        state = tutorial.advance(data, state)
    assert max(tutorial.l2_errors(data, state, 0.0, [0.2, 0.4])) < 2e-13
    assert abs(state.energy_balance_residual) < 2e-14


def test_one_macrocell_without_initial_pec_projection_and_constant_boundary():
    with threadpool_limits(1):
        data = tutorial.prepare(CartesianMacroMesh(), boundary_data=2.0)
        state = tutorial.initialize(data, 2.0, [0.0, 0.0])
        state = tutorial.advance(data, state)
    assert max(tutorial.l2_errors(data, state, 2.0, [0.0, 0.0])) < 2e-13


@pytest.mark.parametrize("time_step", [0.0, -0.1, np.inf, np.nan, 1j, 10.0])
def test_time_step_validation_and_conservative_cfl(time_step):
    with pytest.raises(ValueError, match="time_step"):
        tutorial.prepare(TriangleMesh.unit_square(), time_step=time_step)


def test_boundary_indices_and_original_physical_moment_sign_are_checked():
    mesh = TriangleMesh.unit_square()
    interior = next(face for face in range(len(mesh.faces)) if face not in mesh.boundary_faces)
    with pytest.raises(ValueError, match="exterior"):
        tutorial.prepare(mesh, absorbing={interior: 1.0})
    mesh = CartesianMacroMesh()
    data = tutorial.prepare(mesh, boundary_data=2.0)

    def wrong_pairing(item):
        equations = tutorial.electric_equations(item)
        return replace(equations, c=-equations.c)

    with pytest.raises(RuntimeError, match="original physical balance"):
        tutorial.initialize(data, 2.0, [0.0, 0.0], electric_provider=wrong_pairing)


def test_generic_curl_import_and_current_record_pickle_roundtrip():
    script = (
        "import pymhm.fem.vector.curl\n"
        "import sys\n"
        "assert not [name for name in sys.modules "
        "if name.startswith('pymhm._legacy.models')]\n"
    )
    run = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
        ],
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stderr
    data = tutorial.prepare(_tetra_patch())
    for cls, instance in (
        (TangentialTraceSpace, data.skeleton),
        (CurlOperators, data.locals[0]),
    ):
        archived = pickle.dumps(instance)
        replay = pickle.loads(archived)
        assert type(replay) is cls
        if isinstance(replay, TangentialTraceSpace):
            assert_array_equal(replay.cell_dofs(0), instance.cell_dofs(0))
            assert_array_equal(replay.frames, instance.frames)
        else:
            assert_array_equal(replay.curl.toarray(), instance.curl.toarray())
            assert replay.frequency_bound() == instance.frequency_bound()


def test_compatibility_boundary_rules_retain_shared_tangent_frames():
    mesh = _tetra_patch()
    with MaxwellStepper(mesh, time_step=0.001, degree=2, absorbing=1.0) as comparator:
        rules = list(comparator._boundary_rules())
    assert len(rules) == len(mesh.boundary_faces)
    for face, points, weights, basis, frame, ids in rules:
        assert_array_equal(ids, comparator.skeleton.dofs(face))
        assert points.shape == (len(weights), 3)
        assert basis.shape[0] == len(weights)
        assert np.all(weights > 0)
        assert_allclose(frame.T @ frame, np.eye(2), atol=1e-15)


def test_importable_providers_preserve_spawn_and_serial_coefficients():
    with threadpool_limits(1):
        data = tutorial.prepare(_tetra_patch(), local_refinement=1)
        initial = tutorial.initialize(data, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        serial = tutorial.advance(data, initial)
        spawned = tutorial.advance(
            data, initial, execution=ExecutionConfig(backend="process", workers=2)
        )
    for field in ("electric", "magnetic"):
        for actual, expected in zip(getattr(spawned, field), getattr(serial, field), strict=True):
            assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
    assert_allclose(spawned.trace, serial.trace, rtol=1e-10, atol=1e-12)


def test_explicit_segmented_trace_and_device_callbacks_share_the_declared_equations():
    from examples.maxwell_nanoguide import NanoWaveguide

    mesh = CartesianMacroMesh(2, 2, (0, 10, 0, 10))
    trace = TangentialTraceSpace(
        SkeletonSpace(mesh, tuple(FaceSpace.uniform(1, 2) for _ in mesh.faces))
    )
    model = NanoWaveguide()
    options = dict(
        time_step=0.005,
        degree=2,
        local_refinement=2,
        skeleton=trace,
        permittivity=model.permittivity,
        absorbing=1.0,
        boundary_data=model.boundary,
        quadrature_order=6,
    )
    with threadpool_limits(1):
        data = tutorial.prepare(mesh, **options)
        own = tutorial.initialize(data, 0.0, 0.0)
        with MaxwellStepper(mesh, **options) as comparator:
            reference = comparator.initialize()
            for _ in range(2):
                own = tutorial.advance(data, own)
                reference = comparator.advance()
                assert max(tutorial.compare_state(own, reference).values()) < 2e-11
                assert own.original_electric_residual < 1e-10
    assert data.skeleton is trace
    with pytest.raises(ValueError, match="supplied macro mesh"):
        tutorial.prepare(CartesianMacroMesh(), skeleton=trace)
