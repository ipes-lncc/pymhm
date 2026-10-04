"""Physical Piola section replay and preservation of independent interface owners."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from threadpoolctl import threadpool_limits

from examples import core_extension_data as exact
from examples.hdiv3d_field_archive import field_arrays, observe_system, replay
from examples.hdiv3d_sections import evaluate, replay_section
from pymhm._legacy.models.darcy.hdiv_3d import solve_darcy_hdiv3d
from pymhm.meshes.mixed import AffineMixedMesh


@pytest.fixture(
    scope="module",
    params=[("tetrahedron", 1, 1), ("tetrahedron", 2, 2), ("tetrahedron", 3, 1), ("prism", 2, 2)],
)
def executed(request):
    """Capture actual production bases and maps for each selected normal convention."""
    kind, pressure, normal = request.param
    with threadpool_limits(1), observe_system() as observed:
        solution = solve_darcy_hdiv3d(
            AffineMixedMesh.unit_cube(1, kind),
            pressure_degree=pressure,
            normal_degree=normal,
            trace_degree=normal,
            local_refinement=1,
            source=exact.source3d,
            quadrature_order=12,
        )
        arrays = field_arrays(solution, observed, assembly_order=12, norm_orders=(6, 8))
    return arrays


def test_arbitrary_physical_points_match_executed_norm_tables(executed):
    """Physical section evaluation agrees with the archived independent volume replay."""
    for macro in range(int(executed["local_count"])):
        origin = executed[f"points_{macro}"][executed[f"cells_{macro}"][0, 0]]
        physical = origin + executed["q8_points"] @ executed[f"jacobian_{macro}"][0].T
        direct = evaluate(executed, macro, 0, physical)
        volume = replay(executed, macro, 8)
        for actual, expected in zip(direct, volume, strict=True):
            assert_allclose(actual, expected[0], rtol=2e-11, atol=2e-11)


def test_interface_values_remain_separate_and_plane_has_unit_area(executed):
    """Sampling never averages pressure jumps or shares independent owners' vertices."""
    fields = replay_section(executed, refinement=2)
    points, cells, owners = fields["points"], fields["cells"], fields["owners"]
    assert np.all(owners[cells] == owners[cells[:, :1]])
    assert_allclose(points[:, 2], 0.37, rtol=0, atol=1e-13)
    triangles = points[cells, :2]
    a, b = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    assert abs(np.sum(abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])) / 2 - 1) < 1e-12
    physical_owners = {}
    for point, owner in zip(points, owners, strict=True):
        physical_owners.setdefault(tuple(point.round(12)), set()).add(int(owner[0]))
    assert any(len(value) > 1 for value in physical_owners.values())
    # A prescribed jump here tests display ownership only; it is never saved
    # or presented as an accepted solution of the physical equations.
    modified = dict(executed)
    modified["pressure_0"] = executed["pressure_0"].copy()
    modified["pressure_0"][0, 0] += 1
    changed = replay_section(modified, refinement=2)
    difference = changed["actual"] - fields["actual"]
    assert_allclose(difference[:, 0], owners[:, 0] == 0, rtol=0, atol=1e-12)
    assert np.array_equal(changed["actual"][:, 1:], fields["actual"][:, 1:])


def test_sections_do_not_construct_a_new_basis_or_moment_map(executed, monkeypatch):
    """Fresh basis/nullspace/orientation constructors cannot affect persisted fields."""
    import pymhm.fem.hdiv.family_3d as legacy
    import pymhm.fem.hdiv.moments_3d as general
    import pymhm.meshes.mixed as maps

    def forbidden(*args, **kwargs):
        raise AssertionError("Fresh numerical field coordinates are forbidden")

    monkeypatch.setattr(legacy, "null_space", forbidden)
    monkeypatch.setattr(general, "coefficients", forbidden)
    monkeypatch.setattr(maps, "hdiv3d_transform", forbidden)
    fields = replay_section(executed, refinement=1)
    assert np.isfinite(fields["actual"]).all()


def test_unowned_or_nonphysical_sampling_is_rejected(executed):
    """Owners and real physical coordinates are explicit parts of the replay contract."""
    for macro, cell, points in (
        (-1, 0, np.zeros((1, 3))),
        (0, -1, np.zeros((1, 3))),
        (0, 0, np.zeros((1, 2))),
        (0, 0, np.full((1, 3), np.nan)),
        (0, 0, np.ones((1, 3), dtype=complex)),
    ):
        with pytest.raises(ValueError, match="archived owner"):
            evaluate(executed, macro, cell, points)
    for refinement in (0, True):
        with pytest.raises(ValueError, match="sampling refinement"):
            replay_section(executed, refinement)


def test_consumer_uses_the_terminal_rule_and_archived_geometry(executed):
    """The display consumer cannot substitute an earlier integration rule or cell kind."""
    from examples.sample_core_sections import mixed_fields
    from examples.verify_hdiv3d import physical_errors

    kind = ("tetrahedron", "prism")[int(executed["cell_kind_code"])]
    fields, norms = mixed_fields(executed, kind, 1, 8)
    assert norms == physical_errors(executed, 8)
    assert fields["actual"].shape[1] == 4
    for wrong_kind, wrong_order in ((kind, 6), ("cube", 8)):
        with pytest.raises(ValueError, match="terminal executed norm"):
            mixed_fields(executed, wrong_kind, 1, wrong_order)
