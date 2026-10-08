"""Native triangular trace adapters reject immediately excluded local partitions."""

import numpy as np
import pytest

from pymhm import (
    GlobalContext,
    LocalContext,
    MeshHierarchy,
    TetraMesh,
    TriangularSkeleton,
    bind_interface,
)

pytestmark = pytest.mark.fem


@pytest.mark.parametrize(
    "case,expected",
    [
        ("unresolved", "resolve"),
        ("misaligned", "align"),
        ("incomplete", "cover"),
        ("hole", "cover"),
    ],
)
def test_native_triangular_trace_partition_admissibility(case, expected):
    pytest.importorskip("dolfinx")
    import ufl

    macro = TetraMesh(
        np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]), np.array([[0, 1, 2, 3]])
    )
    if case == "unresolved":
        skeleton = TriangularSkeleton(macro, 2)
        fine = macro
    elif case == "misaligned":
        default = TriangularSkeleton(macro, 2)
        parts = []
        for face in range(len(macro.faces)):
            part = default.face_partition(face).copy()
            # Move every copy of one shared midpoint along its original edge.
            old = np.array([0.5, 0.5, 0])
            mask = np.all(part == old, axis=2)
            part[mask] = [0.7, 0.3, 0]
            parts.append(part)
        skeleton = TriangularSkeleton(macro, 2, face_partitions=tuple(parts))
        fine = macro.submesh(0, 2)
    elif case == "incomplete":
        skeleton = TriangularSkeleton(macro)
        fine = TetraMesh(macro.points * 0.5 + 0.1, macro.cells)
    else:
        skeleton = TriangularSkeleton(macro)
        full = macro.submesh(0, 4)
        fine = TetraMesh(full.points, full.cells[1:])
    context = GlobalContext(MeshHierarchy(macro, [fine]), bind_interface(skeleton), (0,))
    with LocalContext(context, 0) as local:
        native = local.native_space()
        v = ufl.TestFunction(native.space)
        with pytest.raises(ValueError, match=expected):
            local.trace_pairings(lambda phi, ds: phi * v * ds)


def test_triangular_trace_adapter_checks_dimension_without_guessing_a_surface_map():
    pytest.importorskip("dolfinx")
    import ufl

    from pymhm import TriangleMesh

    macro = TetraMesh.unit_cube()
    context = GlobalContext(
        MeshHierarchy(macro, lambda _: TriangleMesh.unit_square()),
        bind_interface(TriangularSkeleton(macro)),
        (0,) * len(macro.cells),
    )
    with LocalContext(context, 0) as local:
        native = local.native_space()
        v = ufl.TestFunction(native.space)
        with pytest.raises(ValueError, match="three-dimensional"):
            local.trace_pairings(lambda phi, ds: phi * v * ds)
