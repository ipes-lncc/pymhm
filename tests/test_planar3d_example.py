"""Independent source/flux derivation and a material-fitted mixed-boundary patch."""

import importlib.util
import sys
from functools import partial
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
from threadpoolctl import threadpool_limits

from pymhm.darcy3d import TriangularSkeleton, solve_darcy_3d
from pymhm.planar_fitting import fit_planar_material, planar_face_partitions
from pymhm.reconstruction3d import reconstruct_darcy_moments_3d
from pymhm.tetrahedral import TetraMesh

path = Path(__file__).resolve().parents[1] / "examples/planar3d_data.py"
spec = importlib.util.spec_from_file_location("planar3d_data", path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
Planar3DData = module.Planar3DData


def test_independent_transmission_and_body_source() -> None:
    """Differences of the pressure and physical flux verify signs and both material sides."""
    data = Planar3DData()
    points = np.array([[0.1, 0.2, 0.3], [0.8, 0.4, 0.7]])
    step = 1e-5
    gradient = np.column_stack(
        [
            (data.pressure(points + step * axis) - data.pressure(points - step * axis)) / (2 * step)
            for axis in np.eye(3)
        ]
    )
    assert_allclose(
        -np.einsum("nij,nj->ni", data.material(points), gradient), data.flux(points), atol=2e-11
    )
    divergence = sum(
        (data.flux(points + step * axis)[:, j] - data.flux(points - step * axis)[:, j]) / (2 * step)
        for j, axis in enumerate(np.eye(3))
    )
    assert_allclose(divergence, data.source, atol=2e-11)
    interface = np.array([[0.63, 0.0, 0.0], [0.31, 0.5, 0.6]])
    assert_allclose(data.pressure(interface), 0.0, atol=2e-16)
    inside = data.material.trace_values(interface, interface - data.normal)
    outside = data.material.trace_values(interface, interface + data.normal)
    assert_allclose(inside, np.broadcast_to(25 * data.tensor, (2, 3, 3)))
    assert_allclose(outside, np.broadcast_to(data.tensor, (2, 3, 3)))


def test_oblique_material_interface_mixed_boundary_patch() -> None:
    """One fitted tetrahedron exercises the public arbitrary-face/local-mesh contract."""
    data = Planar3DData()
    cube = TetraMesh.unit_cube()
    macro = TetraMesh(cube.points, cube.cells[:1])
    skeleton = TriangularSkeleton(
        macro, degree=1, face_partitions=planar_face_partitions(macro, data.material)
    )
    local = fit_planar_material(macro, data.material).mesh
    face = int(macro.boundary_faces[0])
    with threadpool_limits(1):
        solution = solve_darcy_3d(
            macro,
            skeleton=skeleton,
            local_meshes=(local,),
            degree=4,
            permeability=data.material,
            source=data.source,
            dirichlet=data.pressure,
            neumann={face: partial(data.normal_flux, normal=macro.normals[face])},
        )
        assert solution.l2_error(data.pressure) < 1e-11
        assert solution.flux_l2_error(data.flux) < 1e-10
        assert max(abs(solution.conservation_residuals())) < 1e-12
        recovered = reconstruct_darcy_moments_3d(solution, degree=1)
        assert recovered.flux_l2_error(data.flux) < 1e-10
