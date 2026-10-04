"""Small data-orientation, dimensional and canonical stress-interchange checks."""

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose

ROOT = Path(__file__).resolve().parents[1]


def _example(name: str):
    """Load an original scientific example without modifying the import search path."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "examples" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_material_orientation_and_physical_scaling(tmp_path):
    """Depth samples reverse only the vertical index; gravity scales with L/S."""
    data = _example("hpc4e_data")
    path = tmp_path / "samples.txt"
    path.write_text("2 3\n1 2 3 4 5 6\n")
    values = data.read_samples(path)
    assert_allclose(values, [[3, 2, 1], [6, 5, 4]])
    fields = data.HPC4EData(
        np.full((2, 3), 15e6), np.full((2, 3), 0.49), np.full((2, 3), 1760)
    ).fields()
    lam, mu, body = [field(np.array([[0.1, 0.2]]))[0] for field in fields]
    assert_allclose(mu * data.STRESS_SCALE, 15e6 / (2 * 1.49))
    assert_allclose(lam / mu, 49)
    assert_allclose(body * data.STRESS_SCALE / data.LENGTH_SCALE, [0, -1760 * 9.81])


@pytest.mark.parametrize("content", ["", "2 3 1", "-2 2 1 2", "1 1 nan", "2.0 1 2 3"])
def test_malformed_material_samples_are_rejected(tmp_path, content):
    """File parsing rejects truncation, invalid dimensions and nonfinite material data."""
    path = tmp_path / "bad.txt"
    path.write_text(content)
    with pytest.raises(ValueError):
        _example("hpc4e_data").read_samples(path)


def test_stress_archive_preserves_physical_fields(tmp_path):
    """Canonical RT orientation preserves odd face moments and a quadratic displacement."""
    _example("hpc4e_data")
    archive = _example("solve_hpc4e_mhm").archive
    field_type = _example("hpc4e_fields").RectangularElasticityField
    from pymhm import CartesianMacroMesh
    from pymhm._legacy.models.elasticity.stress_tensor import solve_elasticity_tensor_rt

    mesh = CartesianMacroMesh(2, 1, (0, 1, 0, 0.45))
    solution = solve_elasticity_tensor_rt(
        mesh,
        local_refinement=2,
        enrichment=1,
        source=(-2, 0),
        dirichlet=lambda x: np.column_stack((x[:, 0] ** 2, -2 * x[:, 0] * x[:, 1])),
    )
    path = tmp_path / "field.npz"
    archive(solution, path)
    field = field_type.load(path)
    points = np.array([[0.031, 0.011], [0.6, 0.217], [0.95, 0.449]])
    u, sigma, div, rot = field.evaluate(points)
    x, y = points.T
    assert_allclose(u, np.column_stack((x**2, -2 * x * y)), atol=2e-13)
    target = np.zeros_like(sigma)
    target[:, 0, 0], target[:, 1, 1] = 4 * x, -4 * x
    target[:, 0, 1] = target[:, 1, 0] = -2 * y
    assert_allclose(sigma, target, atol=2e-12)
    assert_allclose(div, np.broadcast_to([2, 0], div.shape), atol=2e-11)
    assert_allclose(rot, y, atol=2e-12)
    material = _example("hpc4e_data").HPC4EData(
        np.full((1, 1), 1e8), np.full((1, 1), 0.25), np.full((1, 1), 1760)
    )
    doubled = replace(
        field,
        displacement=2 * field.displacement,
        stress=2 * field.stress,
        rotation=2 * field.rotation,
    )
    norms = _example("hpc4e_fields").compare_fields(doubled, field, material)
    height = 0.45
    stress_squared = 32 * height / 3 + 8 * height**3 / 3
    for name, squared in zip(
        ("displacement", "stress", "divergence", "rotation", "compliance"),
        (
            height / 5 + 4 * height**3 / 9,
            stress_squared,
            4 * height,
            height**3 / 3,
            stress_squared / 0.8,
        ),
        strict=True,
    ):
        assert_allclose(norms[name + "_reference_norm_dimensionless"] ** 2, squared, rtol=1e-11)
        assert_allclose(norms[name + "_relative"], 1, atol=2e-14)


def test_published_profile_interval_keeps_jump_and_interior_extremum():
    """Pixel uncertainty contains exact one-sided limits and a quadratic stationary point."""
    _example("hpc4e_data")
    _example("hpc4e_fields")
    interval = _example("compare_hpc4e").stress_interval

    def evaluate(points):
        """Provide a broken quadratic stress with an independently known maximum."""
        x = points[:, 0]
        stress = np.zeros((len(x), 2, 2))
        stress[:, 0, 0] = -((x - 0.57) ** 2) + 0.02 * (x >= 0.5)
        return None, stress

    field = SimpleNamespace(degree=1, enrichment=0, nx=4, bounds=(0, 1, 0, 0.45), evaluate=evaluate)
    lower, upper = interval(field, np.array([5000.0]), 2000.0)
    assert_allclose(lower, [-7.29], atol=2e-14)
    assert_allclose(upper, [2], atol=2e-14)


@pytest.mark.parametrize("degree", [1, 2])
def test_classical_archive_uses_unenriched_canonical_contract(tmp_path, degree):
    """The shared plot/norm reader preserves every independent classical polynomial."""
    _example("hpc4e_data")
    original = _example("solve_hpc4e_reference")
    reader = _example("hpc4e_fields").RectangularElasticityField
    rng = np.random.default_rng(42)
    field = original.ReferenceField(
        rng.normal(size=(2, 2 * (degree + 1) * (degree + 2), 2)),
        rng.normal(size=(2, (degree + 1) ** 2, 2)),
        rng.normal(size=(2, (degree + 1) * (degree + 2) // 2)),
        2,
        1,
        degree,
    )
    path = tmp_path / "reference.npz"
    field.save(path)
    loaded = reader.load(path)
    assert loaded.enrichment == 0
    points = rng.uniform(size=(31, 2)) * [1.0, 0.45]
    expected_stress, expected_u, expected_rotation = field.evaluate(points)
    actual_u, actual_stress, _, actual_rotation = loaded.evaluate(points)
    assert_allclose(actual_u, expected_u, atol=2e-14)
    assert_allclose(actual_stress, expected_stress, atol=2e-13)
    assert_allclose(actual_rotation, expected_rotation, atol=2e-14)
