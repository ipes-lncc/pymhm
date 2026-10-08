"""Executed layered-source and fine-trace cases remain reproducible through declared equations."""

from types import SimpleNamespace

import numpy as np

from examples.three_layer_2017 import load_case
from examples.three_layer_2017_acquire import original_step_checks
from examples.tutorial_elastodynamic_equations import (
    advance,
    initialize,
    prepare,
    prepare_source,
    spatial_forms,
)
from pymhm._legacy.models.waves.elastodynamics import ElastodynamicStepper, _make_local
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace
from pymhm.meshes.triangle import TriangleMesh


def test_original_three_layer_operator_source_and_one_sided_material_contract() -> None:
    """Match the actual layered material/source data with its P3/r8-P2/s8 local discretization."""
    case = load_case()
    density, stiffness = case.materials()
    skeleton = case.skeleton()
    expected = _make_local(7, case.mesh, skeleton, 3, 8, 5, density, stiffness, None, None)
    actual = spatial_forms(case.mesh, skeleton, 7, 3, 8, 5, density, None, None, stiffness)
    for name in ("mass", "stiffness"):
        np.testing.assert_allclose(
            getattr(actual, name).toarray(),
            getattr(expected, name).toarray(),
            atol=1e-12,
            rtol=1e-10,
        )
    np.testing.assert_allclose(actual.coupling, expected.coupling, atol=1e-12, rtol=1e-10)
    np.testing.assert_array_equal(actual.trace_dofs, expected.trace_dofs)
    source = case.source()
    snapshot = prepare_source(SimpleNamespace(locals=(actual,)), source)
    time = case.source_center_time_s / case.scales[3]
    np.testing.assert_allclose(
        snapshot.load_at_time(actual, time),
        expected.load_at_time(source, time),
        atol=1e-12,
        rtol=1e-10,
    )


def test_fine_P2_trace_newmark_original_momentum_and_legacy_field_agreement() -> None:
    """Use the executed P3/r8 and vector-P2/s8 spaces with physical free traction."""
    mesh = TriangleMesh.unit_square()
    skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(2, 8) for _ in mesh.faces), components=2)
    options = dict(
        time_step=0.001,
        degree=3,
        local_refinement=8,
        skeleton=skeleton,
        quadrature_order=5,
        traction=dict.fromkeys(map(int, mesh.boundary_faces), [0, 0]),
    )
    with prepare(mesh, **options) as operators, ElastodynamicStepper(mesh, **options) as legacy:
        before = initialize(operators)
        after = advance(operators, before, [1, 0.3])
        expected_before = legacy.initialize()
        expected = legacy.advance([1, 0.3])
        checks = original_step_checks(operators, before, after, [1, 0.3])
        assert checks["physical_work_energy_relative_residual"] < 1e-10
        for name in ("displacement", "velocity"):
            for actual, reference in zip(
                getattr(after, name), getattr(expected, name), strict=True
            ):
                np.testing.assert_allclose(actual, reference, atol=1e-12, rtol=1e-10)
        assert before.time == expected_before.time
