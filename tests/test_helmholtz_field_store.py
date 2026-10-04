"""Disk-streamed fields preserve exact coefficients and transactional byte contracts."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from examples.helmholtz_compact_family import CompactFamily
from examples.helmholtz_field_store import write_coefficients
from pymhm.helmholtz import _HelmholtzFactory
from pymhm.helmholtz_spaces import helmholtz_skeleton
from pymhm.loads import split_point_sources
from pymhm.quadrilateral import CartesianMacroMesh


@pytest.mark.parametrize("precision", ["double", "extended"])
def test_streamed_coefficients_match_full_fields_and_keep_independent_cells(tmp_path, precision):
    if precision == "extended" and np.finfo(np.longdouble).eps >= np.finfo(float).eps:
        pytest.skip("extended correction requires a wider native mantissa")
    mesh = CartesianMacroMesh(2, 1)
    skeleton = helmholtz_skeleton(mesh, 2, degree=2)
    factory = _HelmholtzFactory(
        mesh,
        skeleton,
        2,
        3,
        2,
        8,
        1.0,
        1.0,
        1 + 0.2j,
        split_point_sources(mesh, ((0.2, 0.3, 0.1),)),
        0j,
        dict.fromkeys(mesh.boundary_faces, 0j),
        {},
        None,
    )
    family = CompactFamily.prepare(factory, local_refinement_precision=precision)
    complete = family.solve(1)
    trace = family.solve_trace(1)
    path = tmp_path / "coefficients.npy"
    metadata = write_coefficients(family, trace, path, cell_count=2)
    mapped = np.load(path, mmap_mode="r", allow_pickle=False)
    try:
        assert_array_equal(mapped, np.asarray(complete.pressure), strict=True)
        assert not np.array_equal(mapped[0], mapped[1])
        assert metadata["coefficient_dtype"] == mapped.dtype.str
        assert metadata["coefficient_shape"] == [2, mapped.shape[1]]
        assert metadata["macro_balance_max"] == pytest.approx(
            max(abs(complete.balance)), rel=2e-15, abs=0
        )
        assert metadata["local_equation_residual_max"] == complete.local_residual_max
        assert metadata["original_field_trace_residual"] == complete.original_trace_residual
    finally:
        mapped._mmap.close()
    original = path.read_bytes()
    for count in (1, 3):
        with pytest.raises(ValueError, match="declared"):
            write_coefficients(family, trace, path, cell_count=count)
        assert path.read_bytes() == original
        assert not path.with_suffix(".npy.tmp").exists()
