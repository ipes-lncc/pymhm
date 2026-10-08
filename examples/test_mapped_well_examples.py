"""Scientific contracts for native mapped-well data and archived physical norm integration."""

from dataclasses import replace

import numpy as np
from numpy.testing import assert_allclose
from threadpoolctl import threadpool_limits

from examples.formulations.application import mapped_darcy as solve_darcy_mapped_rt
from examples.mapped_well_fields import MappedWellField, difference
from examples.solve_mapped_oscillatory_well import OscillatoryWellData
from examples.solve_mapped_well import WellData
from pymhm.meshes.hexahedron import HexMesh


def archived(solution):
    """Collect the exact public acquisition arrays for a small native solution."""
    return dict(
        macro_cells=solution.skeleton.mesh.cells,
        local_points=np.stack([m.points for m in solution.local_meshes]),
        local_cells=np.stack([m.cells for m in solution.local_meshes]),
        pressure=np.stack(solution.pressure),
        flux=np.stack(solution.flux),
    )


def test_well_production_sign_and_oscillatory_positive_bounds():
    """The producer has lower inner pressure and a uniformly positive physical tensor."""
    well = WellData()
    points = np.array([[well.inner_radius, 0.0, 0.0], [well.outer_radius, 0.0, 0.0]])
    assert well.pressure(points)[0] < well.pressure(points)[1]
    assert np.all(well.flux(points)[:, 0] < 0)
    rng = np.random.default_rng(41)
    xyz = rng.uniform([-50, -50, -5], [50, 50, 5], (500, 3))
    tensor = OscillatoryWellData().tensor(xyz) * well.viscosity
    diagonal = np.diagonal(tensor, axis1=1, axis2=2)
    scale = 1e-11 * np.array([0.1, 0.001, 0.001])
    assert np.all(diagonal >= scale / 19)
    assert np.all(diagonal <= scale * 19)
    assert_allclose(tensor, diagonal[:, :, None] * np.eye(3))
    assert_allclose(
        OscillatoryWellData(tensor_scale=1e-3).tensor(xyz),
        1e8 * OscillatoryWellData().tensor(xyz),
        rtol=5e-16,
    )


def test_nested_archives_physical_norm_and_nonzero_reference_denominator():
    """Nested Piola evaluation preserves fields and integrates known nonzero physical norms."""
    mesh = HexMesh.unit_cube()

    def exact(points):
        """Return affine pressure represented in both physical discrete spaces."""
        return 1 + points @ np.array([1.0, 2.0, 3.0])

    with threadpool_limits(1):
        coarse = solve_darcy_mapped_rt(mesh, local_refinement=1, dirichlet=exact)
        fine = solve_darcy_mapped_rt(mesh, local_refinement=2, subdivisions=2, dirichlet=exact)
    a = MappedWellField.from_arrays(archived(fine), 2, 1)
    b = MappedWellField.from_arrays(archived(coarse), 1, 1)
    zero = difference(a, b, order=4)
    assert zero["pressure_relative"] < 2e-13
    assert zero["flux_relative"] < 2e-13
    measured = difference(
        a, replace(b, pressure=np.zeros_like(b.pressure), flux=np.zeros_like(b.flux)), order=4
    )
    assert_allclose(measured["pressure_l2"], np.sqrt(16 + 14 / 12), rtol=2e-13)
    assert_allclose(measured["flux_l2"], np.sqrt(14), rtol=2e-13)
    assert_allclose([measured["pressure_relative"], measured["flux_relative"]], [1, 1])
    null_reference = replace(a, pressure=np.zeros_like(a.pressure), flux=np.zeros_like(a.flux))
    undefined = difference(null_reference, b, order=4)
    assert undefined["pressure_relative"] is None
    assert undefined["flux_relative"] is None
    assert undefined["pressure_increment_relative"] is None
    assert_allclose(undefined["flux_l2"], np.sqrt(14), rtol=2e-13)


def test_global_tensor_scale_preserves_dirichlet_pressure_and_scales_flux():
    """A uniform mobility factor changes flux, but not pressure, in the zero-source BVP."""
    physical = OscillatoryWellData()
    printed_scale = replace(physical, tensor_scale=1e-3)
    mesh = HexMesh.annular_prism(np.array([0.2, 0.4]), 1.0, 3)
    with threadpool_limits(1):
        first = solve_darcy_mapped_rt(
            mesh, local_refinement=1, permeability=physical.tensor, dirichlet=physical.pressure
        )
        second = solve_darcy_mapped_rt(
            mesh, local_refinement=1, permeability=printed_scale.tensor, dirichlet=physical.pressure
        )
    # Piola's determinant cancels in each divergence moment. With zero forcing,
    # the RT1 divergence times Q1 test is integrated exactly by two-point Gauss.
    low = replace(first, quadrature_order=2).equilibrium_residuals()
    high = replace(first, quadrature_order=8).equilibrium_residuals()
    for a, b in zip(low, high, strict=True):
        assert_allclose(a, b, atol=1e-14, rtol=1e-10)
    points = np.array([[0.2, 0.3, 0.4], [0.7, 0.5, 0.6]])
    for cell in range(len(mesh.cells)):
        p, q, _ = first.evaluate(cell, points)
        other_p, other_q, _ = second.evaluate(cell, points)
        assert_allclose(other_p, p, rtol=2e-12)
        assert_allclose(other_q / 1e8, q, rtol=2e-11, atol=1e-12)


def test_anisotropic_common_grid_norm_preserves_physical_volume():
    """Different vertical resolutions integrate on their exact common tensor partition."""
    coarse_mesh = HexMesh.unit_cube().refined((2, 2, 1))
    fine_mesh = HexMesh.unit_cube().refined((4, 4, 2))

    def pressure(points):
        """Return affine pressure with known nonzero L2 norm and flux."""
        return 1 + points @ np.array([1.0, 2.0, 3.0])

    fields = []
    with threadpool_limits(1):
        for mesh, shape in ((coarse_mesh, (2, 1)), (fine_mesh, (4, 2))):
            solution = solve_darcy_mapped_rt(mesh, local_refinement=1, dirichlet=pressure)
            fields.append(
                MappedWellField(
                    mesh.points[mesh.cells],
                    np.stack(solution.pressure)[:, 0],
                    np.stack(solution.flux),
                    *shape,
                )
            )
    measured = difference(fields[1], fields[0], order=3)
    assert measured["integration_axis_counts"] == [4, 4, 2]
    assert measured["pressure_relative"] < 3e-13
    assert measured["flux_relative"] < 3e-13
    zero = replace(
        fields[0], pressure=np.zeros_like(fields[0].pressure), flux=np.zeros_like(fields[0].flux)
    )
    measured = difference(fields[1], zero, order=3)
    assert_allclose(measured["pressure_l2"], np.sqrt(16 + 14 / 12), rtol=3e-13)
    assert_allclose(measured["flux_l2"], np.sqrt(14), rtol=3e-13)


def test_invariant_classical_subspace_matches_full_three_dimensional_mixed_field():
    """Exact vertical symmetry removes modes without changing the full Piola solution."""
    from examples.solve_mapped_well_invariant import solve

    mesh = HexMesh.annular_prism(np.array([0.2, 0.5, 1.0]), 1.5, 4)
    data = OscillatoryWellData()
    caps = {
        int(face): 0.0
        for face in mesh.boundary_faces
        if np.ptp(mesh.points[mesh.faces[face], 2]) < 1e-12
    }
    with threadpool_limits(1):
        full = solve_darcy_mapped_rt(
            mesh,
            local_refinement=1,
            permeability=data.tensor,
            dirichlet=data.pressure,
            neumann=caps,
            quadrature_order=(7, 7, 5),
        )
        pressure, flux, diagnostic = solve(mesh, data, 7)
    assert diagnostic["archived_divergence_cell_backward_residual_max"] < 5e-15
    assert abs(sum(diagnostic["boundary_rates"].values())) < 1e-15
    points = np.array([[0.2, 0.3, 0.1], [0.7, 0.5, 0.8]])
    field = MappedWellField(mesh.points[mesh.cells], pressure, flux, 1, 1)
    p, q = field.values(np.arange(len(mesh.cells)), points)
    for cell in range(len(mesh.cells)):
        fp, fq, _ = full.evaluate(cell, points)
        assert_allclose(p[cell], fp[0], rtol=5e-13)
        assert_allclose(q[cell], fq[0], rtol=5e-11, atol=1e-14)
    assert np.count_nonzero(flux[:, 16:24]) == 0
    assert np.count_nonzero(pressure[:, 1::2]) == 0


def test_invariant_subspace_rejects_tilted_caps_and_preserves_global_mobility_scale():
    """The reduction requires actual geometric symmetry and respects dimensional scaling."""
    import pytest

    from examples.solve_mapped_well_invariant import assemble, solve

    cube = HexMesh.unit_cube()
    vertices = cube.points.copy()
    vertices[:, 2] += 0.1 * vertices[:, 0]
    with pytest.raises(ValueError, match="horizontal planar caps"):
        assemble(HexMesh(vertices, cube.cells), OscillatoryWellData(), 4)
    mesh = HexMesh.annular_prism(np.array([0.2, 0.6]), 1.0, 3)
    original = OscillatoryWellData()
    with threadpool_limits(1):
        p, q, diagnostics = solve(mesh, original, 5)
        scaled_p, scaled_q, scaled_diagnostics = solve(
            mesh, replace(original, tensor_scale=1e-3), 5
        )
    assert_allclose(scaled_p, p, rtol=5e-13, atol=1e-8)
    assert_allclose(scaled_q / 1e8, q, rtol=5e-12, atol=1e-16)
    assert max(diagnostics["archived_physical_block_backward_residual_max"]) < 1e-14
    assert max(scaled_diagnostics["archived_physical_block_backward_residual_max"]) < 1e-14


def test_archived_field_scoped_tables_preserve_point_values_bitwise():
    """Reusing reference tables preserves all physical values without a global cache."""
    from pymhm.fem.hdiv.mapped import mapped_rt_basis

    mesh = HexMesh.annular_prism(np.array([0.2, 0.5]), 1.0, 4)
    rng = np.random.default_rng(913)
    field = MappedWellField(
        mesh.points[mesh.cells], rng.normal(size=(4, 8)), rng.normal(size=(4, 36)), 1, 1
    )
    points = rng.uniform(0, 1, (17, 3))
    basis, _, modal = mapped_rt_basis(HexMesh.unit_cube(), 1, points)
    fresh = field.values(np.arange(4), points)
    cached = field.values(np.arange(4), points, tables=(basis, modal))
    for first, second in zip(fresh, cached, strict=True):
        np.testing.assert_array_equal(first, second)
