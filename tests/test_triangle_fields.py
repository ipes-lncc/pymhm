"""Independent physical moments for nonaligned layers and finite-radius forces."""

import pickle
from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose

import pymhm.materials.sources as triangle_fields
from pymhm import PolylineLayerField, RadialDiskLoad, TriangleMesh, material_triangle_quadrature
from pymhm._legacy.models.waves.elastodynamics import ElastodynamicStepper, _make_local
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace
from pymhm.materials.sources import triangle_field_quadrature


def physical_rule(mesh, field, order=4):
    """Read physical measures and samples from the public normalized rule contract."""
    bary, weights, values = material_triangle_quadrature(mesh, field, order)
    points = np.einsum("tqi,tia->tqa", bary, mesh.points[mesh.cells])
    return points, weights * mesh.areas[:, None], values.reshape(*weights.shape, *values.shape[1:])


def test_nonmatching_polyline_tensor_moments_and_endpoint_extension():
    """Integrals match analytic sloping-layer moments for arbitrary Kelvin tensor values."""
    mesh = TriangleMesh.unit_square(2)
    tensor = np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 0.0], [0.0, 0.0, 2.0]])
    field = PolylineLayerField([0.0, 1.0], [[0.25, 0.75]], np.array([tensor, 3 * tensor]))
    points, weights, values = physical_rule(mesh, field)
    assert_allclose(np.einsum("tq,tqab->ab", weights, values), 2 * tensor, atol=4e-15)
    assert_allclose(
        np.einsum("tq,tq,tqab->ab", weights, points[..., 0], values),
        (1.5 - 2 * (0.125 + 1 / 6)) * tensor,
        atol=4e-15,
    )
    assert_allclose(
        np.einsum("tq,tq,tqab->ab", weights, points[..., 1], values),
        (1.5 - (0.0625 + 0.125 + 1 / 12)) * tensor,
        atol=5e-15,
    )
    assert_allclose(
        field([[-1.0, 0.2], [-1.0, 0.3], [2.0, 0.7], [2.0, 0.8], [0.5, 0.5]]),
        [tensor, 3 * tensor, tensor, 3 * tensor, tensor],
    )
    extended = TriangleMesh(
        np.array([[-1.0, 0.0], [2.0, 0.0], [-1.0, 1.0], [2.0, 1.0]]), [[0, 1, 2], [1, 3, 2]]
    )
    _, measure, values = physical_rule(extended, field)
    assert_allclose(np.einsum("tq,tqab->ab", measure, values), 6 * tensor, atol=2e-14)
    assert not field.values.flags.writeable


def test_piecewise_slopes_multiple_layers_and_mesh_independence():
    """A tent horizon and a flat second interface have analytically known layer areas."""
    field = PolylineLayerField([0.0, 0.5, 1.0], [[0.2, 0.4, 0.2], [0.8, 0.8, 0.8]], [2.0, 3.0, 5.0])
    # Exact points on either interface retain the smaller-index layer value.
    assert_allclose(field([[0.5, 0.4], [0.5, 0.8]]), [2.0, 3.0], rtol=0, atol=0)
    for n in (1, 3, 4):
        _, weights, values = physical_rule(TriangleMesh.unit_square(n), field, 3)
        assert_allclose(np.sum(weights * values), 0.3 * 2 + 0.5 * 3 + 0.2 * 5, atol=3e-15)
        assert_allclose(weights.sum(), 1.0, atol=2e-15)


def test_layer_partition_rejects_loss_of_a_positive_area_piece(monkeypatch):
    """Fault injection checks that an incomplete geometric partition cannot reach assembly."""
    split = triangle_fields._split

    def missing_piece(vertices, normal, offset):
        """Emulate a geometry failure that silently drops a positive-area intersection."""
        return split(vertices, normal, offset)[:1]

    monkeypatch.setattr(triangle_fields, "_split", missing_piece)
    field = PolylineLayerField([0.0, 1.0], [[0.2, 0.4]], [1.0, 2.0])
    with pytest.raises(ValueError, match="intersection area"):
        material_triangle_quadrature(TriangleMesh.unit_square(), field, 3)


@pytest.mark.parametrize("center,radius", [([0.5, 0.5], 0.2), ([0.31, 0.44], 0.23)])
def test_radial_force_disk_invariants(center, radius):
    """Zero resultant/torque and exact force moments use the physical force-density units."""
    mesh = TriangleMesh.unit_square(2)
    amplitude = 3.7
    source = RadialDiskLoad(center, radius, amplitude)
    points, weights, values = physical_rule(mesh, source, 4)
    delta = points - center
    assert_allclose(weights.sum(), np.pi * radius**2, atol=2e-14)
    assert_allclose(np.einsum("tq,tqa->a", weights, values), 0.0, atol=3e-14)
    assert_allclose(
        np.einsum("tq,tqa,tqb->ab", weights, delta, values),
        np.eye(2) * amplitude * np.pi * radius**3 / 3,
        atol=2e-14,
    )
    assert_allclose(
        np.sum(weights * np.sum(values * values, axis=2)),
        np.pi * radius**2 * amplitude**2,
        atol=3e-13,
    )
    assert_allclose(
        np.sum(weights * (delta[..., 0] * values[..., 1] - delta[..., 1] * values[..., 0])),
        0.0,
        atol=2e-16,
    )


def test_half_disk_enclosing_disk_and_disjoint_support():
    """A boundary-cut source has its true nonzero resultant; an exterior disk has none."""
    mesh = TriangleMesh.unit_square()
    radius, amplitude = 0.3, 2.0
    _, weights, values = physical_rule(mesh, RadialDiskLoad([0.0, 0.5], radius, amplitude))
    assert_allclose(weights.sum(), np.pi * radius**2 / 2, atol=2e-14)
    assert_allclose(
        np.einsum("tq,tqa->a", weights, values), [amplitude * radius**2, 0.0], atol=2e-14
    )
    _, weights, _ = physical_rule(mesh, RadialDiskLoad([0.5, 0.5], 2.0))
    assert_allclose(weights.sum(), 1.0, atol=2e-14)
    _, weights, _ = physical_rule(mesh, RadialDiskLoad([5.0, 5.0], 0.1))
    assert_allclose(weights, 0.0)
    depth = 1e-4
    _, weights, _ = physical_rule(mesh, RadialDiskLoad([0.5, -radius + depth], radius))
    cap = radius**2 * np.arccos((radius - depth) / radius) - (radius - depth) * np.sqrt(
        2 * radius * depth - depth**2
    )
    assert_allclose(weights.sum(), cap, rtol=0.0, atol=2e-16)
    assert_allclose(
        RadialDiskLoad([0.0, 0.0], 1.0)([[0.0, 0.0], [1.0, 0.0], [0.0, 0.5]]),
        [[0.0, 0.0], [0.0, 0.0], [0.0, 1.0]],
    )


def test_force_load_owner_and_temporal_provider():
    """Original P3 load assembly integrates a disk independently of density quadrature."""
    mesh = TriangleMesh([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], [[0, 1, 2]])
    local = _make_local(
        0,
        mesh,
        SkeletonSpace(mesh, tuple(FaceSpace.uniform(0) for _ in mesh.faces), 2),
        3,
        1,
        4,
        2.0,
        None,
        1.0,
        1.0,
    )
    source = RadialDiskLoad([0.25, 0.25], 0.1, 3.0, time_function=lambda t: 2 * t)
    load = local.load_at_time(source, 0.7).reshape(-1, 2)
    assert_allclose(load.sum(axis=0), 0.0, atol=2e-15)
    assert_allclose(
        (local.nodes - source.center).T @ load, np.eye(2) * 4.2 * np.pi * 0.1**3 / 3, atol=3e-16
    )
    assert_allclose(local.load_at_time(source.at_time(0.7), 0.5), load.ravel(), atol=0.0)
    field = PolylineLayerField([0.0, 1.0], [[0.5, 0.5]], [[1.0, 2.0], [1.0, 2.0]])
    assert_allclose(local.load_at_time(field, 0.5), local.load((1.0, 2.0)), atol=2e-16)
    assert_allclose(local.load(RadialDiskLoad([2.0, 2.0], 0.1)), 0.0)
    with pytest.raises(ValueError, match="density-weighted"):
        local.load(source, density_weighted=True)
    with pytest.raises(ValueError, match="two dimensions"):
        replace(local, nodes=np.zeros((len(local.nodes), 3))).load(source)


@pytest.mark.parametrize("scale", [0.0, -1.4, 1.4])
def test_radial_time_snapshot_preserves_physical_values_and_separable_contract(scale):
    """Frozen time samples are pickleable; replacements reset their integration metadata."""
    source = RadialDiskLoad([0.25, 0.25], 0.1, 3.0, time_function=lambda time: scale)
    snapshot = source.at_time(0.7)
    physical = RadialDiskLoad(source.center, source.radius, source.amplitude * scale)
    points = np.array([[0.25, 0.25], [0.26, 0.27], [0.31, 0.25], [0.9, 0.9]])
    mesh = TriangleMesh([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], [[0, 1, 2]])
    for sample in (snapshot, snapshot.at_time(0.5), pickle.loads(pickle.dumps(snapshot))):
        assert not sample.center.flags.writeable
        assert not sample.spatial_field().center.flags.writeable
        assert sample.amplitude == physical.amplitude
        assert sample.time_scale(0.5) == scale
        assert sample.spatial_field().amplitude == source.amplitude
        np.testing.assert_array_equal(sample(points), physical(points))
        for actual, expected in zip(
            sample.triangle_quadrature(mesh, 3), physical.triangle_quadrature(mesh, 3), strict=True
        ):
            np.testing.assert_array_equal(actual, expected)
    for changed in (
        replace(snapshot, amplitude=7.0),
        replace(snapshot, center=[0.3, 0.3]),
        replace(snapshot, radius=0.2),
    ):
        assert changed.time_scale(0.5) == 1.0
        assert changed.spatial_field().amplitude == changed.amplitude
        np.testing.assert_array_equal(changed.spatial_field()(points), changed(points))
    modulated = replace(snapshot, time_function=lambda time: 2 * time)
    assert modulated.time_scale(0.5) == 1.0
    assert modulated.spatial_field().amplitude == snapshot.amplitude


def test_kelvin_elasticity_and_density_use_their_own_material_partitions():
    """Affine strain energy and rigid inertia reproduce distinct analytical layer areas."""
    mesh = TriangleMesh([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], [[0, 1, 2]])
    skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(0) for _ in mesh.faces), 2)
    tensor = np.array([[4.0, 1.0, 0.3], [1.0, 3.0, -0.2], [0.3, -0.2, 2.0]])
    stiffness = PolylineLayerField([0.0, 1.0], [[0.2, 0.2]], [tensor, 3 * tensor])
    density = PolylineLayerField([0.0, 1.0], [[0.6, 0.6]], [2.0, 3.0])
    local = _make_local(0, mesh, skeleton, 3, 1, 4, density, stiffness, None, None)
    gradient = np.array([[0.2, 0.3], [-0.1, -0.4]])
    displacement = (local.nodes @ gradient.T).ravel()
    strain = np.array(
        [gradient[0, 0], gradient[1, 1], (gradient[0, 1] + gradient[1, 0]) / np.sqrt(2)]
    )
    assert_allclose(
        displacement @ (local.stiffness @ displacement),
        (0.18 + 3 * 0.32) * (strain @ tensor @ strain),
        atol=4e-15,
    )
    translation = np.tile([1.0, 0.0], len(local.nodes))
    assert_allclose(translation @ (local.mass @ translation), 0.42 * 2 + 0.08 * 3, atol=8e-16)
    force = RadialDiskLoad([0.25, 0.25], 0.1)
    assert_allclose(local.load(force).reshape(-1, 2).sum(axis=0), 0.0, atol=2e-16)


@pytest.mark.parametrize("backend", ["thread", "process"])
def test_layered_radial_trajectory_preserves_momentum_and_parallel_semantics(backend):
    """Distinct density/stiffness partitions and circular forcing preserve rigid balances."""
    mesh = TriangleMesh.unit_square()
    skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(1) for _ in mesh.faces), 2)
    tensor = np.array([[3.0, 1.0, 0.0], [1.0, 3.0, 0.0], [0.0, 0.0, 2.0]])
    options = dict(
        time_step=0.01,
        degree=2,
        quadrature_order=4,
        skeleton=skeleton,
        traction={int(face): (0.0, 0.0) for face in mesh.boundary_faces},
        density=PolylineLayerField([0.0, 1.0], [[0.35, 0.45]], [2.0, 3.0]),
        constitutive=PolylineLayerField([0.0, 1.0], [[0.6, 0.7]], [tensor, 2 * tensor]),
    )
    source = RadialDiskLoad([0.5, 0.5], 0.2, 2.0)
    with (
        ElastodynamicStepper(mesh, **options) as serial,
        ElastodynamicStepper(mesh, backend=backend, workers=2, **options) as parallel,
    ):
        serial.initialize()
        parallel.initialize()
        for _ in range(3):
            result = serial.advance(source)
            other = parallel.advance(source)
            assert_allclose(result.trace, other.trace, rtol=2e-13, atol=2e-16)
            moments = np.zeros(3)
            for local, velocity, second, displacement, second_u in zip(
                serial.locals,
                result.velocity,
                other.velocity,
                result.displacement,
                other.displacement,
                strict=True,
            ):
                assert_allclose(velocity, second, rtol=2e-13, atol=2e-16)
                assert_allclose(displacement, second_u, rtol=2e-13, atol=2e-16)
                momentum = (local.mass @ velocity).reshape(-1, 2)
                moments[:2] += momentum.sum(axis=0)
                delta = local.nodes - [0.5, 0.5]
                moments[2] += np.sum(delta[:, 0] * momentum[:, 1] - delta[:, 1] * momentum[:, 0])
            assert_allclose(moments, 0.0, atol=2e-15)


@pytest.mark.parametrize(
    "x,h,v",
    [
        ([0.0], [[0.2]], [1.0, 2.0]),
        ([0.0, 0.0], [[0.2, 0.3]], [1.0, 2.0]),
        ([0.0, 1.0], [[0.2, 0.3], [0.3, 0.1]], [1.0, 2.0, 3.0]),
        ([0.0, 1.0], [[0.2, 0.3]], [1.0]),
        ([0.0, 1.0], [[0.2, np.nan]], [1.0, 2.0]),
        ([0j, 1.0], [[0.2, 0.3]], [1.0, 2.0]),
    ],
)
def test_invalid_layer_contract(x, h, v):
    """Malformed or intersecting horizons cannot silently define another material."""
    with pytest.raises(ValueError):
        PolylineLayerField(x, h, v)


@pytest.mark.parametrize(
    "options",
    [
        {"radius": 0.0},
        {"radius": -1.0},
        {"radius": np.inf},
        {"center": [0.0]},
        {"center": [0j, 0.0]},
        {"tolerance": 0.0},
        {"amplitude": 1j},
        {"time_function": 2},
        {"max_depth": 0},
    ],
)
def test_invalid_disk_contract(options):
    """Invalid geometry, force units or integration controls are rejected."""
    with pytest.raises(ValueError):
        RadialDiskLoad(**{"center": [0.0, 0.0], "radius": 1.0, **options})


@pytest.mark.parametrize("points", [[1.0, 2.0], [[1j, 0.0]], [[np.nan, 0.0]]])
def test_invalid_sample_points(points):
    """Neither coefficient nor force evaluation accepts malformed physical points."""
    for field in (
        PolylineLayerField([0.0, 1.0], [[0.2, 0.3]], [1.0, 2.0]),
        RadialDiskLoad([0.0, 0.0], 1.0),
    ):
        with pytest.raises(ValueError):
            field(points)
    with pytest.raises(ValueError):
        RadialDiskLoad([0.0, 0.0], 1.0).at_time(np.inf)


def test_quadrature_provider_contract_and_unattainable_tolerance():
    """Contract failures and an exhausted adaptive quadrature budget are explicit errors."""
    mesh = TriangleMesh.unit_square()

    class Provider:
        """Small malformed providers exercise the public array contract."""

        def __init__(self, data):
            self.data = data

        def triangle_quadrature(self, mesh, order):
            """Return deliberately supplied data without modifying the geometry."""
            return self.data

    data = [np.full((2, 1, 3), 1 / 3), np.ones((2, 1)), np.ones(2)]
    for modified in (
        [data[0].astype(complex), *data[1:]],
        [data[0], -data[1], data[2]],
        [data[0][0], *data[1:]],
        [data[0], data[1], np.ones(1)],
    ):
        with pytest.raises(ValueError):
            triangle_field_quadrature(mesh, Provider(modified), 3)
    with pytest.raises(ValueError, match="triangular mesh"):
        triangle_field_quadrature(None, Provider(data), 3)
    with pytest.raises(ValueError, match="requested tolerance"):
        RadialDiskLoad([0.31, 0.44], 0.23, tolerance=1e-30, max_depth=1).triangle_quadrature(
            mesh, 4
        )
