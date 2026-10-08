"""Verify editable physical data and spawn-safe scaling declarations."""

import pickle

import numpy as np
import pytest

pytest.importorskip("ufl")

from examples.introduction.scaling_forms import (  # noqa: E402
    LocalProvider,
    PeriodicDarcyData,
    face_length_snapshot,
    refined_interface_template,
    verify_source,
)
from pymhm import MeshHierarchy, bind_interface, bind_problem  # noqa: E402
from pymhm.core.equations import Equation  # noqa: E402
from pymhm.fem.scalar.quadrilateral import quadrilateral_operators  # noqa: E402
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace  # noqa: E402
from pymhm.meshes.cartesian import CartesianMacroMesh  # noqa: E402


@pytest.mark.parametrize("period", [0.1, 0.137, 0.2])
def test_physical_data_reaches_declared_local_operator(period: float) -> None:
    """Compare the executed operator/source with independently supplied same data."""
    data = PeriodicDarcyData(period)
    verify_source(data)
    macro = CartesianMacroMesh(2, 2)
    face = FaceSpace.uniform(1, 2, continuous=True)
    skeleton = SkeletonSpace(macro, tuple(face for _ in macro.faces))
    template, ends = refined_interface_template(macro.spacing, 4, face)
    provider = LocalProvider(
        macro,
        skeleton,
        template,
        ends,
        face_length_snapshot(macro),
        face,
        refinement=4,
        data=data,
    )
    restored = pickle.loads(pickle.dumps(provider))
    assert type(restored) is LocalProvider
    assert restored.data == data
    problem = bind_problem(
        MeshHierarchy(macro, restored.local_mesh),
        bind_interface(skeleton, convention="normal"),
        restored,
        global_equation=Equation(0, 0),
        retained=1,
    )
    local = problem.local_provider(1)
    matrix, mass, load = quadrilateral_operators(
        macro.submesh(1, 4),
        1,
        permeability=data.permeability,
        source=data.source,
        order=4,
    )
    np.testing.assert_array_equal(local.a.toarray(), matrix.toarray())
    np.testing.assert_array_equal(local.L, load)
    np.testing.assert_array_equal(local.moments, mass @ np.ones((len(load), 1)))
    if period != 0.1:
        standard, _, standard_load = quadrilateral_operators(
            macro.submesh(1, 4),
            1,
            permeability=PeriodicDarcyData().permeability,
            source=PeriodicDarcyData().source,
            order=4,
        )
        assert np.linalg.norm((matrix - standard).data) > 1e-3
        assert np.linalg.norm(load - standard_load) > 1e-3


@pytest.mark.parametrize("period", [0.0, -0.1, float("inf"), float("nan")])
def test_invalid_material_period_is_explicit(period: float) -> None:
    """A physical material period must be finite and strictly positive."""
    with pytest.raises(ValueError, match="period"):
        PeriodicDarcyData(period)
