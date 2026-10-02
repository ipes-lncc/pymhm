"""Independent differentiation of the published elastodynamic manufactured load."""

import numpy as np
from numpy.testing import assert_allclose

from examples.elastodynamics_campaign import ElasticWave
from examples.elastodynamics_results import difference
from pymhm.tetrahedral import TetraMesh, tetra_nodal_space


def test_archived_physical_time_difference_on_a_rigid_translation(tmp_path):
    """A known constant difference integrates exactly and has zero stress/divergence."""
    mesh = TetraMesh.unit_cube()
    coefficients = np.array(
        [
            np.tile([1.0, 2.0, 3.0], len(tetra_nodal_space(mesh.submesh(cell, 2), 3)[1]))
            for cell in range(len(mesh.cells))
        ]
    )
    common = dict(
        macro_points=mesh.points,
        macro_cells=mesh.cells,
        local_degree=3,
        local_refinement=2,
        time=0.5,
    )
    first, second = tmp_path / "translation.npz", tmp_path / "zero.npz"
    np.savez(first, **common, displacement=coefficients, velocity=2 * coefficients)
    np.savez(
        second,
        **common,
        displacement=np.zeros_like(coefficients),
        velocity=np.zeros_like(coefficients),
    )
    actual = difference(first, second)
    for key in ("displacement_l2", "displacement_h1"):
        assert_allclose(actual[key], np.sqrt(14), rtol=2e-14)
    for key in ("velocity_l2", "velocity_h1"):
        assert_allclose(actual[key], 2 * np.sqrt(14), rtol=2e-14)
    assert actual["stress_broken_hdiv"] < 1e-11
    assert all(value == 0 for value in difference(first, first).values())


def test_manufactured_body_force_and_owned_shape_cache():
    """Differentiate displacement twice in time and analytical stress once in space."""
    wave = ElasticWave()
    points = np.random.default_rng(943).uniform(0.1, 0.9, (11, 3))
    step = 1e-5

    def displacement(time, coordinates):
        """Evaluate only the prescribed displacement, independently of source Hessians."""
        return wave.amplitudes(time)[0] * wave.spatial(coordinates)[0]

    def stress(time, coordinates):
        """Use the Cauchy constitutive law and the first analytical derivative."""
        gradient = wave.amplitudes(time)[0] * wave.spatial(coordinates)[1]
        return 0.4 * (
            gradient
            + gradient.swapaxes(-1, -2)
            + np.trace(gradient, axis1=-2, axis2=-1)[:, None, None] * np.eye(3)
        )

    for time in (0.17, 0.43):
        acceleration = (
            displacement(time + step, points)
            - 2 * displacement(time, points)
            + displacement(time - step, points)
        ) / step**2
        divergence = np.zeros_like(points)
        for direction in range(3):
            offset = step * np.eye(3)[direction]
            divergence += (
                stress(time, points + offset)[..., direction]
                - stress(time, points - offset)[..., direction]
            ) / (2 * step)
        assert_allclose(wave.source(time, points), acceleration - divergence, rtol=1e-6, atol=3e-6)
    assert len(wave._source_cache) == 1
    assert all(not value.flags.writeable for value in next(iter(wave._source_cache.values())))
