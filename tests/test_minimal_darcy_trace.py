"""The fixed SPE10 trace needs visible local moments in every macro orientation."""

import numpy as np
import pytest

from pymhm.mesh import FaceSpace, SkeletonSpace
from pymhm.quadrilateral import CartesianMacroMesh, quadrilateral_trace_coupling


@pytest.mark.parametrize("cell", [0, 1, 6, 7])
def test_continuous_p1_s2_has_invisible_mode_on_q1_r2(cell: int) -> None:
    """A nonzero physical trace is invisible to all nine r2 local nodal tests."""
    mesh = CartesianMacroMesh(6, 11, (0, 1200, 0, 2200))
    skeleton = SkeletonSpace(
        mesh, tuple(FaceSpace.uniform(1, 2, continuous=True) for _ in mesh.faces)
    )
    coupling = quadrilateral_trace_coupling(mesh, cell, mesh.submesh(cell, 2), skeleton, 1)
    assert coupling.shape == (9, 12)
    assert np.linalg.matrix_rank(coupling) == 8
    _, _, vectors = np.linalg.svd(coupling, full_matrices=True)
    invisible = vectors[-1]
    assert np.linalg.norm(invisible) > 0.99
    assert np.linalg.norm(coupling @ invisible) < 1e-12 * np.linalg.norm(coupling)


@pytest.mark.parametrize("cell", [0, 1, 6, 7])
def test_continuous_p1_s2_coupling_is_injective_on_q1_r4(cell: int) -> None:
    """The smallest selected level has twelve independently visible trace columns."""
    mesh = CartesianMacroMesh(6, 11, (0, 1200, 0, 2200))
    skeleton = SkeletonSpace(
        mesh, tuple(FaceSpace.uniform(1, 2, continuous=True) for _ in mesh.faces)
    )
    coupling = quadrilateral_trace_coupling(mesh, cell, mesh.submesh(cell, 4), skeleton, 1)
    assert coupling.shape == (25, 12)
    assert np.linalg.matrix_rank(coupling) == 12
